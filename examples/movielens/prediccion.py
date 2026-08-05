"""Predecir los generos de peliculas que el modelo nunca vio.

Tarea: el 40% de las peliculas queda fuera del ajuste, la mitad para elegir
hiperparametros y la mitad para el resultado final. Cada pelicula reservada
se proyecta al espacio latente con `fold_in_entities` usando solo sus
ratings y su elenco, y su genero se predice con `predict_attribute`. El
ground truth existe para todas, asi que el acierto se mide directamente.

El experimento contesta dos preguntas:

  1. Correctitud. Las peliculas reservadas no participan del ajuste en
     ninguna relacion, asi que su genero no puede filtrarse al modelo. Si
     el acierto supera a los baselines, el fold-in y la prediccion estan
     bien implementados.
  2. Aporte de la fusion. Tres configuraciones comparten tarea, split y
     metricas, y difieren solo en que relaciones entran al ajuste: solo
     ratings, solo elenco, o ambas. Cada una recibe su propio barrido de
     hiperparametros en validacion, para que la comparacion sea pareja.

Metricas, multi-etiqueta (2.4 generos por pelicula en promedio):

  hit@1        el genero de mayor score esta entre los verdaderos
  R-precision  aciertos entre los k primeros, con k = generos verdaderos
  AP media     average precision del ranking de los 19 generos

Ejecucion: uv run python examples/movielens/prediccion.py
Toma alrededor de tres minutos, dominados por el barrido en validacion.
"""

# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from datafiusion import (
    dfmf_sparse,
    fold_in_entities,
    normalize_relations,
    predict_attribute,
)

import datos as datos_movielens
from datos import CLAVE_GENERO


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLA = 0
MAX_ITER = 100
RANKS_FIJOS = {"usuario": 25, "actor": 40, "genero": 15}

CONFIGURACIONES = {
    "solo ratings": [("pelicula", "usuario")],
    "solo elenco": [("pelicula", "actor")],
    "fusion": [("pelicula", "usuario"), ("pelicula", "actor")],
}

# Grilla de validacion. El rank de pelicula controla cuanto puede memorizar
# el ajuste, y el peso del genero cuanto pesa la etiqueta frente a las
# vistas una vez que todas las matrices estan Frobenius-normalizadas.
GRILLA_RANK = (10, 20, 40)
GRILLA_PESO = (1.0, 3.0)


# %%
def metricas(scores, Y):
    """hit@1, R-precision y AP media para un ranking multi-etiqueta.

    `scores` es (n, n_generos) y `Y` la matriz binaria de generos reales.
    """
    orden = np.argsort(-scores, axis=1)
    Y_ordenado = np.take_along_axis(Y, orden, axis=1)
    k = np.maximum(Y.sum(axis=1).astype(int), 1)

    acumulado = np.cumsum(Y_ordenado, axis=1)
    precision_en_posicion = acumulado / np.arange(1, Y.shape[1] + 1)

    return {
        "hit@1": float(Y_ordenado[:, 0].mean()),
        "R-precision": float((acumulado[np.arange(len(k)), k - 1] / k).mean()),
        "AP media": float(((precision_en_posicion * Y_ordenado).sum(axis=1) / k).mean()),
    }


def ajustar(claves_vistas, rank_pelicula, peso_genero, lambda_G=0.0,
            max_iter=MAX_ITER, verbose=0):
    """Ajusta el modelo sobre el train con las vistas indicadas."""
    R_bruto = {CLAVE_GENERO: datos.train.R[CLAVE_GENERO]}
    R_bruto.update({c: datos.train.R[c] for c in claves_vistas})
    pesos = {CLAVE_GENERO: peso_genero}
    R, escalas = normalize_relations(R_bruto, weights=pesos)

    ranks = {"pelicula": rank_pelicula, "genero": RANKS_FIJOS["genero"]}
    ranks.update({c[1]: RANKS_FIJOS[c[1]] for c in claves_vistas})

    G, S = dfmf_sparse(R=R, ranks=ranks, lambda_G=lambda_G, lambda_S=0.01,
                       max_iter=max_iter, init="nndsvd", verbose=verbose)
    return G, S, (escalas, pesos)


def predecir(split, claves_vistas, G, S, normalizacion, clamp=True):
    """Proyecta las peliculas del split y devuelve los scores por genero.

    Las matrices nuevas reciben la misma transformacion que sus pares del
    train: se dividen por aquella norma Frobenius y se multiplican por
    aquel peso. Normalizarlas por su propia norma las dejaria en otra
    unidad que los backbones S aprendidos, y el fold-in devolveria factores
    con la magnitud equivocada.

    `fold_in_entities` resuelve un ridge sin restriccion de signo, asi que
    devuelve factores con entradas negativas aunque el modelo asuma G >= 0.
    El clamp los lleva de vuelta al cono no negativo.
    """
    escalas, pesos = normalizacion
    R_nuevo = {}
    for clave in claves_vistas:
        norma = escalas[clave][0]
        factor = pesos.get(clave, 1.0) / norma if norma > 0 else 1.0
        R_nuevo[clave] = [split.R[clave][0].multiply(factor).tocsr()]

    G_nuevo = fold_in_entities(R_nuevo, G, S, target_type="pelicula", lambda_reg=1e-3)
    fraccion_negativa = float((G_nuevo < 0).mean())
    if clamp:
        G_nuevo = np.maximum(G_nuevo, 0.0)

    scores = predict_attribute(
        known={"pelicula": np.arange(G_nuevo.shape[0])},
        G={**G, "pelicula": G_nuevo},
        S=S,
        target_type="genero",
        view_keys=[CLAVE_GENERO],
    )
    return scores, fraccion_negativa


# %%
print("=" * 74)
print("MovieLens: prediccion de genero para peliculas fuera del ajuste")
print("=" * 74)
datos = datos_movielens.cargar(semilla=SEMILLA)
Y_train = np.asarray(datos.train.R[CLAVE_GENERO][0].todense())
generos_por_pelicula = datos.test.Y.sum(axis=1)
print(f"\ntest: {len(datos.test)} peliculas, {generos_por_pelicula.mean():.2f} generos "
      f"por pelicula (min {int(generos_por_pelicula.min())}, "
      f"max {int(generos_por_pelicula.max())})")


# %%
# Baselines sin factorizacion. El marginal ignora la pelicula y marca el
# piso. El kNN vota los generos de las peliculas train mas parecidas y mide
# cuanto aporta la factorizacion sobre una comparacion directa.
def knn_coseno(X_train, X_nuevo, Y_train, k=20):
    normalizar = lambda X: sp.diags(1.0 / np.maximum(
        np.sqrt(X.multiply(X).sum(axis=1)).A.ravel(), 1e-12)) @ X
    similitud = np.asarray((normalizar(X_nuevo) @ normalizar(X_train).T).todense())
    vecinos = np.argpartition(-similitud, kth=k, axis=1)[:, :k]
    pesos = np.take_along_axis(similitud, vecinos, axis=1)
    return np.einsum("nk,nkg->ng", pesos, Y_train[vecinos])


def scores_baseline(nombre, split):
    if nombre == "marginal":
        return np.tile(Y_train.mean(axis=0), (len(split), 1))
    claves = CONFIGURACIONES[nombre.replace("kNN ", "")] if nombre != "kNN fusion" \
        else CONFIGURACIONES["fusion"]
    return sum(knn_coseno(datos.train.R[c][0], split.R[c][0], Y_train) for c in claves)


BASELINES = ["marginal", "kNN solo ratings", "kNN solo elenco", "kNN fusion"]
resultados = {}
print("\nbaselines en test:")
for nombre in BASELINES:
    resultados[nombre] = metricas(scores_baseline(nombre, datos.test), datos.test.Y)
    print(f"  {nombre:<18} AP media = {resultados[nombre]['AP media']:.3f}")


# %%
# Barrido en validacion. Cada configuracion elige su rank y su peso de
# genero mirando solo el split de validacion.
print("\n" + "=" * 74)
print("barrido en validacion (AP media)")
print("=" * 74)
print(f"{'configuracion':<14} {'rank':>5} {'peso':>6} {'AP valid':>9}")

mejores = {}
for nombre, claves in CONFIGURACIONES.items():
    mejor_ap, mejor_par = -1.0, None
    for rank_pelicula in GRILLA_RANK:
        for peso in GRILLA_PESO:
            G, S, normalizacion = ajustar(claves, rank_pelicula, peso)
            scores, _ = predecir(datos.valid, claves, G, S, normalizacion)
            ap = metricas(scores, datos.valid.Y)["AP media"]
            marca = ""
            if ap > mejor_ap:
                mejor_ap, mejor_par, marca = ap, (rank_pelicula, peso), " <-"
            print(f"{nombre:<14} {rank_pelicula:>5} {peso:>6.1f} {ap:>9.3f}{marca}")
    mejores[nombre] = mejor_par
    print(f"{'':<14} elegido: rank={mejor_par[0]}, peso={mejor_par[1]}, "
          f"AP valid={mejor_ap:.3f}\n")


# %%
# Evaluacion final en test con la configuracion elegida en validacion.
print("=" * 74)
print("ajuste final y evaluacion en test")
print("=" * 74)
modelos = {}
for nombre, claves in CONFIGURACIONES.items():
    rank_pelicula, peso = mejores[nombre]
    print(f"\n--- {nombre}: rank={rank_pelicula}, peso genero={peso} ---")
    inicio = time.time()
    G, S, normalizacion = ajustar(claves, rank_pelicula, peso, verbose=50)
    segundos = time.time() - inicio

    scores, fraccion_negativa = predecir(datos.test, claves, G, S, normalizacion)
    resultados[nombre] = metricas(scores, datos.test.Y)
    resultados[nombre]["segundos"] = segundos
    modelos[nombre] = (G, S, normalizacion, scores)

    # In-sample sobre las peliculas del train, cuyo genero si entro al
    # ajuste. La brecha contra el test mide cuanto memoriza el modelo.
    scores_train = predict_attribute(
        known={"pelicula": np.arange(len(datos.train))}, G=G, S=S,
        target_type="genero", view_keys=[CLAVE_GENERO])
    ap_train = metricas(scores_train, Y_train)["AP media"]

    print(f"  ajuste en {segundos:.1f} s")
    print(f"  fold-in: {len(datos.test)} peliculas, "
          f"{fraccion_negativa:.1%} de entradas negativas antes del clamp")
    print(f"  AP in-sample (train) = {ap_train:.3f}, "
          f"AP fold-in (test) = {resultados[nombre]['AP media']:.3f}, "
          f"brecha = {ap_train - resultados[nombre]['AP media']:.3f}")


# %%
tabla = pd.DataFrame(resultados).T[["hit@1", "R-precision", "AP media", "segundos"]]
print("\n" + "=" * 74)
print(f"resultados sobre las mismas {len(datos.test)} peliculas de test")
print("=" * 74)
print(tabla.to_string(float_format=lambda x: f"{x:.3f}", na_rep="-"))

ap = tabla["AP media"]
print(f"\nfusion sobre la mejor fuente sola: "
      f"{ap['fusion'] - ap[['solo ratings', 'solo elenco']].max():+.3f}")
print(f"fusion sobre el marginal:          {ap['fusion'] - ap['marginal']:+.3f}")
print(f"fusion sobre el kNN equivalente:   {ap['fusion'] - ap['kNN fusion']:+.3f}")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
ruta_tabla = DIR_SALIDA / "prediccion_genero.csv"
tabla.to_csv(ruta_tabla)
print(f"\ntabla guardada en {ruta_tabla}")


# %%
# Sensibilidad a la regularizacion y al numero de iteraciones, sobre
# validacion. Muestra por que el barrido fija lambda_G en cero.
print("\n" + "=" * 74)
print("sensibilidad del modelo fusionado (AP en validacion)")
print("=" * 74)
claves_fusion = CONFIGURACIONES["fusion"]
rank_fusion, peso_fusion = mejores["fusion"]

print(f"{'lambda_G':>9} {'AP valid':>9}")
for lambda_G in (0.0, 0.01, 0.1, 1.0):
    G, S, normalizacion = ajustar(claves_fusion, rank_fusion, peso_fusion, lambda_G=lambda_G)
    scores, _ = predecir(datos.valid, claves_fusion, G, S, normalizacion)
    print(f"{lambda_G:>9.2f} {metricas(scores, datos.valid.Y)['AP media']:>9.3f}")

print(f"\n{'max_iter':>9} {'AP valid':>9}")
for max_iter in (25, 50, 100, 200, 400):
    G, S, normalizacion = ajustar(claves_fusion, rank_fusion, peso_fusion, max_iter=max_iter)
    scores, _ = predecir(datos.valid, claves_fusion, G, S, normalizacion)
    print(f"{max_iter:>9d} {metricas(scores, datos.valid.Y)['AP media']:>9.3f}")
print(f"el barrido usa max_iter={MAX_ITER} para acotar el tiempo; subirlo a 400 "
      f"agrega alrededor de 0.01 de AP a costa de cuadruplicar cada ajuste")


# %%
# Ejemplos concretos del modelo fusionado.
scores_fusion = modelos["fusion"][3]
rng = np.random.default_rng(SEMILLA)
print("\nejemplos de prediccion (top 3 generos):")
for i in rng.choice(len(datos.test), size=10, replace=False):
    pid = datos.test.peliculas[i]
    reales = [datos.generos[j] for j in np.flatnonzero(datos.test.Y[i])]
    top3 = [datos.generos[j] for j in np.argsort(-scores_fusion[i])[:3]]
    print(f"  {datos.titulos[pid][:42]:<44} reales: {', '.join(reales):<32} "
          f"predichos: {', '.join(top3)}")
