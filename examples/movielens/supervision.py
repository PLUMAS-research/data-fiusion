"""Anclar componentes a etiquetas conocidas, al estilo TS-NMF.

Hay dos formas distintas de usar etiquetas parciales, y hacen cosas
distintas.

`Relation.rows` enmascara FILAS DE DATOS: dice cuales observaciones entran
a la perdida. Una pelicula sin etiqueta no aporta nada a la relacion de
generos, y sus componentes latentes quedan libres.

`supervision` enmascara ENTRADAS DEL FACTOR: dice que componentes puede
usar cada entidad. Es la formulacion de TS-NMF (MacMillan y Wilson, 2017),
donde la reconstruccion pasa a ser (G * L) S G^T. Si se sabe que una
pelicula es Comedy, se la obliga a cargar solo en la componente que
representa Comedy.

La segunda usa informacion que la primera descarta. Saber que una pelicula
es Comedy dice EN QUE COLUMNA debe cargar, no solo que la fila existe.

Con 19 componentes ancladas a los 19 generos, la componente j pasa a ser
el genero j, y el grupo de una pelicula se lee sin traducir nada. Este
script mide si eso ademas mejora la prediccion, o si solo hace el modelo
mas legible.

Ejecucion: uv run python examples/movielens/supervision.py
"""

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from datafiusion import Relation, fit

import datos as datos_movielens
from datos import CLAVE_GENERO


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2, 3)
MAX_ITER = 300
RANKS_FIJOS = {"usuario": 25, "actor": 40, "genero": 15}
VISTAS = ("usuario", "actor")
NOMBRES = {"usuario": "vistas_usuario", "actor": "vistas_actor"}
ETIQUETAS = "etiquetas"


# %%
datos = datos_movielens.cargar(semilla=0, verbose=False)
piezas = {c: sp.vstack([datos.train.R[c][0], datos.valid.R[c][0], datos.test.R[c][0]],
                       format="csr")
          for c in [("pelicula", "usuario"), CLAVE_GENERO, ("pelicula", "actor")]}
Y = np.asarray(piezas[CLAVE_GENERO].todense())
n_pel, n_gen = Y.shape
n_train = len(datos.train) + len(datos.valid)
etiquetadas = np.arange(n_train)
reservadas = np.arange(n_train, n_pel)

frecuencia = Y.sum(axis=0)
dominante = np.array([np.flatnonzero(f)[np.argmin(frecuencia[np.flatnonzero(f)])]
                      for f in Y])
print(f"{n_pel} peliculas, {n_gen} generos, {len(etiquetadas)} etiquetadas, "
      f"{len(reservadas)} reservadas")


# %%
def ap_media(scores, Y_verdad):
    orden = np.argsort(-scores, axis=1)
    Yo = np.take_along_axis(Y_verdad, orden, axis=1)
    k = np.maximum(Y_verdad.sum(axis=1).astype(int), 1)
    prec = np.cumsum(Yo, axis=1) / np.arange(1, Y_verdad.shape[1] + 1)
    return float(((prec * Yo).sum(axis=1) / k).mean())


def relaciones():
    R = {ETIQUETAS: Relation(src="pelicula", dst="genero",
                             matrix=piezas[CLAVE_GENERO], rows=etiquetadas)}
    for tipo in VISTAS:
        R[NOMBRES[tipo]] = Relation(src="pelicula", dst=tipo,
                                    matrix=piezas[("pelicula", tipo)])
    return R


def matriz_de_supervision():
    """Las etiquetadas solo pueden cargar en sus generos; el resto, en todos.

    Es la matriz L de TS-NMF: L[i, j] = 1 si la componente j esta permitida
    para la entidad i.
    """
    permitido = np.ones((n_pel, n_gen), dtype=bool)
    permitido[etiquetadas] = Y[etiquetadas] > 0
    return permitido


def evaluar(nombre, modelo, semilla):
    scores = modelo.predict_proba(target="genero", views=[ETIQUETAS],
                                  known={"pelicula": np.arange(n_pel)})
    grupos = modelo.factor("pelicula").argmax(axis=1)
    fila = {
        "metodo": nombre, "semilla": semilla,
        "AP held-out": ap_media(scores[reservadas], Y[reservadas]),
        "ARI held-out": adjusted_rand_score(dominante[reservadas], grupos[reservadas]),
    }
    # Legibilidad: con las componentes ancladas, el grupo j deberia ser el
    # genero j sin necesidad de traducir por el backbone.
    if modelo.ranks["pelicula"] == n_gen:
        directo = [int(grupos[i] == dominante[i]) for i in reservadas]
        fila["grupo = genero"] = float(np.mean(directo))
    return fila


# %%
filas = []
permitido = matriz_de_supervision()
print(f"\ncomponentes permitidas por pelicula etiquetada: "
      f"{permitido[etiquetadas].sum(axis=1).mean():.2f} de {n_gen}")

for semilla in SEMILLAS:
    ranks = {"pelicula": n_gen, "genero": RANKS_FIJOS["genero"]}
    ranks.update({t: RANKS_FIJOS[t] for t in VISTAS})

    # Sin anclaje: las componentes son grupos latentes cualesquiera.
    libre = fit(relaciones(), ranks, weights={ETIQUETAS: 3.0}, max_iter=MAX_ITER,
                tol=1e-7, init="random", random_state=semilla)
    filas.append(evaluar("sin anclaje", libre, semilla))

    # Con anclaje TS-NMF: la componente j representa al genero j.
    anclado = fit(relaciones(), ranks, weights={ETIQUETAS: 3.0},
                  supervision={"pelicula": permitido}, max_iter=MAX_ITER,
                  tol=1e-7, init="random", random_state=semilla)
    filas.append(evaluar("anclado (TS-NMF)", anclado, semilla))

    if semilla == SEMILLAS[0]:
        ceros = (anclado.G["pelicula"][~permitido] == 0).all()
        print(f"las componentes prohibidas quedaron en cero: {ceros}")


# %%
tabla = pd.DataFrame(filas)
resumen = tabla.groupby("metodo")[["AP held-out", "ARI held-out",
                                   "grupo = genero"]].agg(["mean", "sem"])
print("\n" + "=" * 74)
print(f"peliculas reservadas, {len(SEMILLAS)} semillas")
print("=" * 74)
print(resumen.to_string(float_format=lambda x: f"{x:.3f}"))

for metrica in ("AP held-out", "ARI held-out", "grupo = genero"):
    a = tabla[tabla.metodo == "anclado (TS-NMF)"].set_index("semilla")[metrica]
    b = tabla[tabla.metodo == "sin anclaje"].set_index("semilla")[metrica]
    diferencia = a - b
    sem = diferencia.std(ddof=1) / np.sqrt(len(diferencia))
    veces = abs(diferencia.mean()) / max(sem, 1e-12)
    print(f"\n{metrica}: anclado menos libre = {diferencia.mean():+.3f} "
          f"+- {sem:.3f} ({veces:.1f} SE, "
          f"{'significativo' if veces >= 2 else 'dentro del ruido'})")

print("\n'grupo = genero' es la fraccion de peliculas reservadas cuyo grupo")
print("coincide directamente con su genero, sin traducir por el backbone.")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
tabla.to_csv(DIR_SALIDA / "supervision.csv", index=False)
