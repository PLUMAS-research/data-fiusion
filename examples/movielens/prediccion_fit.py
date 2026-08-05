"""Genre prediction with `fit`, against the same protocol as prediccion.py.

Row masks make a regime possible that the previous version could not
express. Every movie enters the fit, including the ones being evaluated,
but the genre relation is masked down to the training rows. The held-out
movies therefore get their factor from ratings and cast alone, and their
genre never enters the loss, so nothing leaks. Predicting them is then a
matter of reading the reconstructed genre view, with no fold-in and so no
estimator mismatch between how the factor was fitted and how it is later
inferred.

Two ways of scoring the same held-out movies are compared:

  masked    every movie in the fit, genre masked to the training rows
  fold-in   only training movies in the fit, held-out ones projected

Hyperparameters are chosen on validation and the winner is reported on
test, four seeds each, with the standard error. Differences below two
standard errors are not claimed.

Ejecucion: uv run python examples/movielens/prediccion_fit.py
"""

# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from datafiusion import Relation, fit

import datos as datos_movielens
from datos import CLAVE_GENERO


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2, 3)
MAX_ITER = 200
RANKS_FIJOS = {"usuario": 25, "actor": 40, "genero": 15}
GRILLA_RANK = (10, 19, 20, 40)
GRILLA_PESO = (0.1, 1.0, 3.0, 10.0, 30.0)

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


def matrices_completas(datos_split):
    """Reassemble the full universe from the three splits, keeping the offsets."""
    piezas = {}
    for clave in [("pelicula", "usuario"), CLAVE_GENERO, ("pelicula", "actor")]:
        piezas[clave] = sp.vstack(
            [datos_split.train.R[clave][0], datos_split.valid.R[clave][0],
             datos_split.test.R[clave][0]], format="csr")
    n_train = len(datos_split.train)
    n_valid = len(datos_split.valid)
    indices = {
        "train": np.arange(n_train),
        "valid": np.arange(n_train, n_train + n_valid),
        "test": np.arange(n_train + n_valid, piezas[CLAVE_GENERO].shape[0]),
    }
    return piezas, indices


def relaciones(piezas, filas_etiquetadas, vistas):
    """Named relations, with the genre relation masked to the labelled rows."""
    salida = {
        ETIQUETAS: Relation(src="pelicula", dst="genero",
                                    matrix=piezas[CLAVE_GENERO],
                                    rows=filas_etiquetadas)
    }
    for tipo in vistas:
        salida[NOMBRES[tipo]] = Relation(src="pelicula", dst=tipo,
                                         matrix=piezas[("pelicula", tipo)])
    return salida


def ajustar_enmascarado(piezas, filas_etiquetadas, rank, peso, semilla, vistas):
    R = relaciones(piezas, filas_etiquetadas, vistas)
    ranks = {"pelicula": rank, "genero": RANKS_FIJOS["genero"]}
    ranks.update({t: RANKS_FIJOS[t] for t in vistas})
    return fit(R, ranks, weights={ETIQUETAS: peso}, max_iter=MAX_ITER,
               tol=1e-6, init="nndsvd", random_state=semilla)


def puntajes(modelo, filas):
    """Reconstructed genre view for the given movie rows."""
    return modelo.predict_proba(target="genero", views=[ETIQUETAS],
                                known={"pelicula": filas})


# %%
print("=" * 78)
print("MovieLens: genero con `fit`, mascara por fila contra fold-in")
print("=" * 78)

VISTAS = ("usuario", "actor")
resumen = []

for semilla in SEMILLAS:
    datos = datos_movielens.cargar(semilla=semilla, verbose=(semilla == SEMILLAS[0]))
    piezas, indices = matrices_completas(datos)
    Y = {parte: np.asarray(piezas[CLAVE_GENERO][indices[parte]].todense())
         for parte in ("train", "valid", "test")}

    # Barrido en validacion: solo las peliculas de train aportan etiqueta.
    mejor = (-1.0, None)
    for rank in GRILLA_RANK:
        for peso in GRILLA_PESO:
            modelo = ajustar_enmascarado(piezas, indices["train"], rank, peso,
                                         semilla, VISTAS)
            ap = metricas(puntajes(modelo, indices["valid"]), Y["valid"])["AP media"]
            if ap > mejor[0]:
                mejor = (ap, (rank, peso))
    rank, peso = mejor[1]
    print(f"\nsemilla {semilla}: elegido rank={rank}, peso={peso} (AP valid {mejor[0]:.3f})")

    # Ajuste final: train y validacion aportan etiqueta, test no.
    etiquetadas = np.concatenate([indices["train"], indices["valid"]])
    inicio = time.time()
    modelo = ajustar_enmascarado(piezas, etiquetadas, rank, peso, semilla, VISTAS)
    segundos = time.time() - inicio
    m = metricas(puntajes(modelo, indices["test"]), Y["test"])
    print(f"  enmascarado: AP {m['AP media']:.3f}, hit@1 {m['hit@1']:.3f}, "
          f"{modelo.n_iter} iter en {segundos:.1f} s ({modelo.stop_reason})")

    # Diagnostico: cuanto memoriza. AP sobre las filas que si aportaron etiqueta.
    ap_in = metricas(puntajes(modelo, etiquetadas),
                     np.asarray(piezas[CLAVE_GENERO][etiquetadas].todense()))["AP media"]
    print(f"  brecha: AP con etiqueta vista {ap_in:.3f}, sin verla {m['AP media']:.3f}, "
          f"diferencia {ap_in - m['AP media']:.3f}")

    fila = {"semilla": semilla, "rank": rank, "peso": peso, "protocolo": "enmascarado",
            "segundos": segundos, "brecha": ap_in - m["AP media"], **m}
    resumen.append(fila)

    # Contraste: mismo split, pero las peliculas de test quedan fuera del ajuste
    # y se proyectan. Es el protocolo de prediccion.py, ahora con fold-in no negativo.
    piezas_train = {c: M[etiquetadas] for c, M in piezas.items()}
    R_train = relaciones(piezas_train, None, VISTAS)
    ranks = {"pelicula": rank, "genero": RANKS_FIJOS["genero"]}
    ranks.update({t: RANKS_FIJOS[t] for t in VISTAS})
    modelo_train = fit(R_train, ranks, weights={ETIQUETAS: peso},
                       max_iter=MAX_ITER, tol=1e-6, init="nndsvd", random_state=semilla)
    nuevas = {NOMBRES[t]: Relation(src="pelicula", dst=t,
                                   matrix=piezas[("pelicula", t)][indices["test"]])
              for t in VISTAS}
    proyectado = modelo_train.transform(nuevas, target="pelicula")
    m_fold = metricas(puntajes(proyectado, np.arange(len(indices["test"]))), Y["test"])
    print(f"  fold-in:     AP {m_fold['AP media']:.3f}, hit@1 {m_fold['hit@1']:.3f}")
    resumen.append({"semilla": semilla, "rank": rank, "peso": peso,
                    "protocolo": "fold-in", "segundos": np.nan,
                    "brecha": np.nan, **m_fold})


# %%
tabla = pd.DataFrame(resumen)
agregado = tabla.groupby("protocolo")[["AP media", "hit@1", "R-precision"]].agg(["mean", "sem"])
print("\n" + "=" * 78)
print(f"resultado sobre {len(SEMILLAS)} particiones")
print("=" * 78)
print(agregado.to_string(float_format=lambda x: f"{x:.3f}"))

REFERENCIAS = {"libreria actual (dfmf_sparse + fold-in)": 0.722,
               "kNN coseno sobre las mismas matrices": 0.749,
               "baseline marginal": 0.550}
media = agregado[("AP media", "mean")]
error = agregado[("AP media", "sem")]
print("\ncontra las referencias medidas en el protocolo anterior:")
for nombre, valor in REFERENCIAS.items():
    for protocolo in media.index:
        delta = media[protocolo] - valor
        veces = abs(delta) / max(error[protocolo], 1e-9)
        marca = "significativo" if veces >= 2 else "dentro del ruido"
        print(f"  {protocolo:<12} contra {nombre:<38} {delta:+.3f} "
              f"({veces:.1f} errores estandar, {marca})")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
ruta = DIR_SALIDA / "prediccion_fit.csv"
tabla.to_csv(ruta, index=False)
print(f"\ntabla guardada en {ruta}")
