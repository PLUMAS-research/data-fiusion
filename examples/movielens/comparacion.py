"""Head to head between the legacy path and `fit`, on equal footing.

Three protocols predict the genres of the same held-out movies. They
share the splits, the seeds, the hyperparameter grid and the metric, so
the only thing that differs is the method.

  legacy       dfmf_sparse + fold_in_entities + predict_attribute, with
               the clamp applied outside the library
  fold-in      fuse + FusionModel.transform, non negative by construction
  enmascarado  fuse with every movie present and the genre relation masked
               to the labelled rows, so held-out movies get their factor
               from ratings and cast without their label entering the loss

Each protocol runs its own sweep on validation and is reported on test,
over four seeds, with the standard error. The grid extends well below the
values the first sweep chose, because an optimum sitting on the edge of a
grid means the grid is wrong, not that the method lost.

Ejecucion: uv run python examples/movielens/comparacion.py
Toma alrededor de media hora.
"""

# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from datafiusion import (
    Relation, dfmf_sparse, fit, fold_in_entities, normalize_relations,
    predict_attribute,
)

import datos as datos_movielens
from datos import CLAVE_GENERO


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2, 3)
MAX_ITER = 300
TOL = 1e-8
RANKS_FIJOS = {"usuario": 25, "actor": 40, "genero": 15}
GRILLA_RANK = (5, 10, 20, 40)
GRILLA_PESO = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

VISTAS = ("usuario", "actor")
NOMBRES = {"usuario": "vistas_usuario", "actor": "vistas_actor", "genero": "etiquetas"}
ETIQUETAS = NOMBRES["genero"]


# %%
def metricas(scores, Y):
    """hit@1, R-precision y AP media para un ranking multi-etiqueta."""
    orden = np.argsort(-scores, axis=1)
    Y_ordenado = np.take_along_axis(Y, orden, axis=1)
    k = np.maximum(Y.sum(axis=1).astype(int), 1)
    acumulado = np.cumsum(Y_ordenado, axis=1)
    precision = acumulado / np.arange(1, Y.shape[1] + 1)
    return {
        "hit@1": float(Y_ordenado[:, 0].mean()),
        "R-precision": float((acumulado[np.arange(len(k)), k - 1] / k).mean()),
        "AP media": float(((precision * Y_ordenado).sum(axis=1) / k).mean()),
    }


def universo(datos):
    """Full matrices plus the row indices of each split."""
    piezas = {
        clave: sp.vstack([datos.train.R[clave][0], datos.valid.R[clave][0],
                          datos.test.R[clave][0]], format="csr")
        for clave in [("pelicula", "usuario"), CLAVE_GENERO, ("pelicula", "actor")]
    }
    n_tr, n_va = len(datos.train), len(datos.valid)
    indices = {"train": np.arange(n_tr), "valid": np.arange(n_tr, n_tr + n_va),
               "test": np.arange(n_tr + n_va, piezas[CLAVE_GENERO].shape[0])}
    return piezas, indices


def ranks_de(rank):
    salida = {"pelicula": rank, "genero": RANKS_FIJOS["genero"]}
    salida.update({t: RANKS_FIJOS[t] for t in VISTAS})
    return salida


# ---------------------------------------------------------------- protocolos


def protocolo_enmascarado(piezas, etiquetadas, objetivo, rank, peso, semilla):
    """Every movie in the fit; the genre relation masked to `etiquetadas`."""
    R = {ETIQUETAS: Relation(src="pelicula", dst="genero",
                             matrix=piezas[CLAVE_GENERO], rows=etiquetadas)}
    for tipo in VISTAS:
        R[NOMBRES[tipo]] = Relation(src="pelicula", dst=tipo,
                                    matrix=piezas[("pelicula", tipo)])
    modelo = fuse(R, ranks_de(rank), weights={ETIQUETAS: peso}, max_iter=MAX_ITER,
                 tol=TOL, init="nndsvd", random_state=semilla)
    return modelo.predict_proba(target="genero", views=[ETIQUETAS],
                                known={"pelicula": objetivo})


def protocolo_foldin_nuevo(piezas, etiquetadas, objetivo, rank, peso, semilla):
    """Only labelled movies in the fit; the rest projected with transform."""
    R = {ETIQUETAS: Relation(src="pelicula", dst="genero",
                             matrix=piezas[CLAVE_GENERO][etiquetadas])}
    for tipo in VISTAS:
        R[NOMBRES[tipo]] = Relation(src="pelicula", dst=tipo,
                                    matrix=piezas[("pelicula", tipo)][etiquetadas])
    modelo = fuse(R, ranks_de(rank), weights={ETIQUETAS: peso}, max_iter=MAX_ITER,
                 tol=TOL, init="nndsvd", random_state=semilla)
    nuevas = {NOMBRES[t]: Relation(src="pelicula", dst=t,
                                   matrix=piezas[("pelicula", t)][objetivo])
              for t in VISTAS}
    proyectado = modelo.transform(nuevas, target="pelicula")
    return proyectado.predict_proba(target="genero", views=[ETIQUETAS],
                                    known={"pelicula": np.arange(len(objetivo))})


def protocolo_legacy(piezas, etiquetadas, objetivo, rank, peso, semilla):
    """The path the notebooks use today, including the clamp applied outside."""
    R_bruto = {CLAVE_GENERO: [piezas[CLAVE_GENERO][etiquetadas]]}
    for tipo in VISTAS:
        R_bruto[("pelicula", tipo)] = [piezas[("pelicula", tipo)][etiquetadas]]
    R, escalas = normalize_relations(R_bruto, weights={CLAVE_GENERO: peso})

    G, S = dfmf_sparse(R=R, ranks=ranks_de(rank), lambda_G=0.0, lambda_S=1e-2,
                       max_iter=MAX_ITER, init="nndsvd", random_state=semilla)

    R_nuevo = {}
    for tipo in VISTAS:
        clave = ("pelicula", tipo)
        norma = escalas[clave][0]
        M = piezas[clave][objetivo]
        R_nuevo[clave] = [M.multiply(1.0 / norma).tocsr() if norma > 0 else M]
    G_nuevo = np.maximum(
        fold_in_entities(R_nuevo, G, S, target_type="pelicula", lambda_reg=1e-3), 0.0)
    return predict_attribute(
        known={"pelicula": np.arange(len(objetivo))},
        G={**G, "pelicula": G_nuevo}, S=S, target_type="genero",
        view_keys=[CLAVE_GENERO])


PROTOCOLOS = {
    "legacy": protocolo_legacy,
    "fold-in nuevo": protocolo_foldin_nuevo,
    "enmascarado": protocolo_enmascarado,
}


# %%
print("=" * 78)
print("MovieLens: legacy contra `fuse`, mismas semillas y misma grilla")
print("=" * 78)
print(f"semillas: {SEMILLAS}, rank en {GRILLA_RANK}, peso en {GRILLA_PESO}")
print(f"max_iter={MAX_ITER}, tol={TOL}")

filas = []
for semilla in SEMILLAS:
    datos = datos_movielens.cargar(semilla=semilla, verbose=False)
    piezas, indices = universo(datos)
    Y = {parte: np.asarray(piezas[CLAVE_GENERO][idx].todense())
         for parte, idx in indices.items()}
    etiquetadas_finales = np.concatenate([indices["train"], indices["valid"]])

    print(f"\n--- semilla {semilla} ---")
    for nombre, funcion in PROTOCOLOS.items():
        inicio = time.time()
        mejor = (-1.0, None)
        for rank in GRILLA_RANK:
            for peso in GRILLA_PESO:
                scores = funcion(piezas, indices["train"], indices["valid"],
                                 rank, peso, semilla)
                ap = metricas(scores, Y["valid"])["AP media"]
                if ap > mejor[0]:
                    mejor = (ap, (rank, peso))
        rank, peso = mejor[1]
        scores = funcion(piezas, etiquetadas_finales, indices["test"], rank, peso, semilla)
        m = metricas(scores, Y["test"])
        borde = (rank in (GRILLA_RANK[0], GRILLA_RANK[-1])
                 or peso in (GRILLA_PESO[0], GRILLA_PESO[-1]))
        print(f"  {nombre:<14} rank={rank:<3} peso={peso:<5} AP valid={mejor[0]:.3f} "
              f"AP test={m['AP media']:.3f}  ({time.time() - inicio:.0f} s)"
              f"{'  [optimo en el borde de la grilla]' if borde else ''}")
        filas.append({"semilla": semilla, "protocolo": nombre, "rank": rank,
                      "peso": peso, "ap_valid": mejor[0], "en_borde": borde, **m})


# %%
tabla = pd.DataFrame(filas)
agregado = tabla.groupby("protocolo")[["AP media", "hit@1", "R-precision"]].agg(["mean", "sem"])
orden = ["legacy", "fold-in nuevo", "enmascarado"]
agregado = agregado.loc[orden]

print("\n" + "=" * 78)
print(f"resultado sobre {len(SEMILLAS)} particiones")
print("=" * 78)
print(agregado.to_string(float_format=lambda x: f"{x:.3f}"))

media, error = agregado[("AP media", "mean")], agregado[("AP media", "sem")]
print("\ncontra la ruta legacy, sobre las mismas particiones:")
for protocolo in orden[1:]:
    diferencia = media[protocolo] - media["legacy"]
    # Error estandar de la diferencia pareada, que es lo que corresponde
    # aqui porque ambos protocolos ven exactamente los mismos splits.
    pareado = (tabla[tabla.protocolo == protocolo].set_index("semilla")["AP media"]
               - tabla[tabla.protocolo == "legacy"].set_index("semilla")["AP media"])
    sem = pareado.std(ddof=1) / np.sqrt(len(pareado))
    veces = abs(pareado.mean()) / max(sem, 1e-12)
    print(f"  {protocolo:<14} {diferencia:+.3f}  (pareado {pareado.mean():+.3f} "
          f"+- {sem:.3f}, {veces:.1f} errores estandar, "
          f"{'significativo' if veces >= 2 else 'dentro del ruido'})")

en_borde = tabla[tabla.en_borde]
if len(en_borde):
    print(f"\naviso: {len(en_borde)} de {len(tabla)} configuraciones eligieron un "
          f"optimo en el borde de la grilla, asi que la grilla limita el resultado:")
    for _, fila in en_borde.iterrows():
        print(f"  semilla {fila.semilla}, {fila.protocolo}: rank={fila['rank']}, peso={fila.peso}")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
ruta = DIR_SALIDA / "comparacion_legacy_fit.csv"
tabla.to_csv(ruta, index=False)
print(f"\ntabla guardada en {ruta}")
