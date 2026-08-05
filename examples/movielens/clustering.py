"""Co-clustering: what the factors group, not what the model predicts.

The tri-factorization is a co-clustering method. G_i assigns rows to
groups, G_j assigns columns to groups, and S_ij says how the groups
relate. Reading a factor as a soft cluster assignment is a large part of
why the model is used at all, and it is a different question from whether
the model predicts a held-out attribute well.

This script measures the clustering side on MovieLens, where the genres
give a ground truth for what a good grouping of movies looks like.

Three things are compared:

  1. Whether the movie factor recovers genre structure at all, against
     k-means on the same matrices and against a random partition.
  2. Whether the design decisions taken for the new path help or hurt the
     grouping. The column gauge is the one to watch: it rescales the
     columns of G, and the cluster of a row is the argmax over exactly
     those columns.
  3. Whether the co-clustering is coherent, that is whether the backbone
     S connects movie groups to genre groups in a readable way.

Genres are multi-label, so no single hard partition is the truth. Two
metrics are reported: NMI and ARI against the dominant genre, which is a
convention and not a fact, and a label-free coherence lift, which asks how
much more likely two movies in the same cluster are to share a genre than
two movies picked at random.

Ejecucion: uv run python examples/movielens/clustering.py
"""

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datafiusion import Relation, dfmf_sparse, fit, normalize_relations

import datos as datos_movielens
from datos import CLAVE_GENERO


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))
SEMILLA = 0
MAX_ITER = 300
N_CLUSTERS = 19          # tantos grupos como generos, para poder comparar
RANKS_FIJOS = {"usuario": 25, "actor": 40, "genero": 15}


# %%
def coherencia(etiquetas, Y):
    """Cuanto mas probable es compartir genero dentro de un cluster que al azar.

    No necesita una particion verdadera, asi que no arrastra la convencion
    de elegir un genero dominante. Un valor de 1.0 es el azar.
    """
    comparte = (Y @ Y.T) > 0
    np.fill_diagonal(comparte, False)
    mismo = etiquetas[:, None] == etiquetas[None, :]
    np.fill_diagonal(mismo, False)
    if mismo.sum() == 0:
        return float("nan")
    dentro = comparte[mismo].mean()
    global_ = comparte[~np.eye(len(Y), dtype=bool)].mean()
    return float(dentro / global_) if global_ > 0 else float("nan")


def tamanos(etiquetas):
    """Cuantos clusters quedaron ocupados y que tan desbalanceados estan."""
    _, cuentas = np.unique(etiquetas, return_counts=True)
    return len(cuentas), float(cuentas.max() / cuentas.sum())


def evaluar(nombre, etiquetas, Y, genero_dominante, semilla=SEMILLA):
    ocupados, mayor = tamanos(etiquetas)
    return {
        "metodo": nombre,
        "semilla": semilla,
        "NMI": normalized_mutual_info_score(genero_dominante, etiquetas),
        "ARI": adjusted_rand_score(genero_dominante, etiquetas),
        "coherencia": coherencia(etiquetas, Y),
        "ocupados": ocupados,
        "mayor": mayor,
    }


def legibilidad(modelo, etiquetas, Y):
    """En que fraccion de los grupos el backbone nombra el genero real.

    Es lo que decide si los factores se pueden leer: el backbone conecta
    cada grupo de peliculas con el espacio de generos, y para interpretar
    un grupo se mira ese perfil en vez de contar las etiquetas a mano.
    """
    perfil = modelo.S["etiquetas"] @ modelo.G["genero"].T
    aciertos, evaluados = 0, 0
    for grupo in np.unique(etiquetas):
        miembros = np.flatnonzero(etiquetas == grupo)
        if len(miembros) < 5:
            continue
        evaluados += 1
        aciertos += int(perfil[grupo].argmax() == Y[miembros].sum(axis=0).argmax())
    return aciertos / max(evaluados, 1)


# %%
print("=" * 78)
print("MovieLens: el lado de co-clustering de la factorizacion")
print("=" * 78)

datos = datos_movielens.cargar(semilla=SEMILLA, verbose=False)
piezas = {c: sp.vstack([datos.train.R[c][0], datos.valid.R[c][0], datos.test.R[c][0]],
                       format="csr")
          for c in [("pelicula", "usuario"), CLAVE_GENERO, ("pelicula", "actor")]}
Y = np.asarray(piezas[CLAVE_GENERO].todense())
n_peliculas = Y.shape[0]

# Genero dominante: el menos frecuente de los que tiene la pelicula. El mas
# frecuente seria "Drama" para casi todo y no distinguiria nada.
frecuencia = Y.sum(axis=0)
dominante = np.array([np.flatnonzero(fila)[np.argmin(frecuencia[np.flatnonzero(fila)])]
                      for fila in Y])
print(f"{n_peliculas} peliculas, {Y.shape[1]} generos, "
      f"{Y.sum(axis=1).mean():.2f} generos por pelicula")
print(f"genero dominante (el mas especifico de cada pelicula): "
      f"{len(np.unique(dominante))} valores distintos")


# %%
SEMILLAS = (0, 1, 2, 3)
resultados = []

# Referencias sin factorizacion.
for semilla in SEMILLAS:
    rng = np.random.default_rng(semilla)
    resultados.append(evaluar("azar", rng.integers(0, N_CLUSTERS, n_peliculas),
                              Y, dominante, semilla))
    for nombre, clave in [("k-means ratings", ("pelicula", "usuario")),
                          ("k-means elenco", ("pelicula", "actor"))]:
        etiquetas = KMeans(n_clusters=N_CLUSTERS, random_state=semilla,
                           n_init=10).fit_predict(piezas[clave].toarray())
        resultados.append(evaluar(nombre, etiquetas, Y, dominante, semilla))


# %%
# La factorizacion. El cluster de una pelicula es el argmax de su fila en
# G[pelicula], que es la lectura estandar de un factor no negativo.
def clusters_de(G_pelicula):
    return np.asarray(G_pelicula).argmax(axis=1)


def relaciones(vistas, con_genero=True):
    salida = {}
    if con_genero:
        salida["etiquetas"] = Relation(src="pelicula", dst="genero",
                                       matrix=piezas[CLAVE_GENERO])
    for tipo in vistas:
        salida[f"vistas_{tipo}"] = Relation(src="pelicula", dst=tipo,
                                            matrix=piezas[("pelicula", tipo)])
    return salida


VISTAS = ("usuario", "actor")
ranks = {"pelicula": N_CLUSTERS, "genero": RANKS_FIJOS["genero"]}
ranks.update({t: RANKS_FIJOS[t] for t in VISTAS})

configuraciones = {
    "fit, gauge por columnas": dict(gauge="column"),
    "fit, sin gauge": dict(gauge=None),
    "fit, eta=0.5": dict(gauge="column", eta=0.5),
}
# nndsvd es determinista e ignora random_state, asi que repetirlo con varias
# semillas da el mismo ajuste y un error estandar de cero que no significa
# nada. La parte estadistica usa init="random", que si varia; nndsvd va
# aparte como corrida unica.
legibles = {}
R_bruto = {CLAVE_GENERO: [piezas[CLAVE_GENERO]]}
for tipo in VISTAS:
    R_bruto[("pelicula", tipo)] = [piezas[("pelicula", tipo)]]
R_legacy, _ = normalize_relations(R_bruto)
ranks_ciego = {k: v for k, v in ranks.items() if k != "genero"}

for semilla in SEMILLAS:
    for nombre, opciones in configuraciones.items():
        modelo = fit(relaciones(VISTAS), ranks, max_iter=MAX_ITER, tol=1e-7,
                     init="random", random_state=semilla, **opciones)
        etiquetas = clusters_de(modelo.G["pelicula"])
        resultados.append(evaluar(nombre, etiquetas, Y, dominante, semilla))
        legibles.setdefault(nombre, []).append(legibilidad(modelo, etiquetas, Y))

    G_legacy, _ = dfmf_sparse(R=R_legacy, ranks=ranks, lambda_G=0.0, lambda_S=1e-2,
                              max_iter=MAX_ITER, init="random", random_state=semilla)
    resultados.append(evaluar("dfmf_sparse", clusters_de(G_legacy["pelicula"]),
                              Y, dominante, semilla))

    # Sin la relacion de generos: el clustering sale de ratings y elenco
    # solamente. Es la situacion real cuando no hay etiquetas, y la unica
    # que mide agrupacion no supervisada: con la relacion de generos dentro
    # del ajuste, recuperar generos es en buena parte tautologico.
    modelo_ciego = fit(relaciones(VISTAS, con_genero=False), ranks_ciego,
                       max_iter=MAX_ITER, tol=1e-7, init="random", random_state=semilla)
    resultados.append(evaluar("fit, sin ver generos",
                              clusters_de(modelo_ciego.G["pelicula"]),
                              Y, dominante, semilla))

# Corrida unica con nndsvd, que es el default y no depende de la semilla.
deterministas = []
for nombre, opciones in configuraciones.items():
    modelo = fit(relaciones(VISTAS), ranks, max_iter=MAX_ITER, tol=1e-7,
                 init="nndsvd", **opciones)
    etiquetas = clusters_de(modelo.G["pelicula"])
    fila = evaluar(nombre, etiquetas, Y, dominante)
    fila["legibilidad"] = legibilidad(modelo, etiquetas, Y)
    deterministas.append(fila)
G_nnd, _ = dfmf_sparse(R=R_legacy, ranks=ranks, lambda_G=0.0, lambda_S=1e-2,
                       max_iter=MAX_ITER, init="nndsvd")
deterministas.append(evaluar("dfmf_sparse", clusters_de(G_nnd["pelicula"]), Y, dominante))
modelo_ciego_nnd = fit(relaciones(VISTAS, con_genero=False), ranks_ciego,
                       max_iter=MAX_ITER, tol=1e-7, init="nndsvd")
deterministas.append(evaluar("fit, sin ver generos",
                             clusters_de(modelo_ciego_nnd.G["pelicula"]), Y, dominante))


# %%
crudo = pd.DataFrame(resultados)
orden = ["azar", "k-means ratings", "k-means elenco", "fit, sin ver generos",
         "dfmf_sparse", "fit, sin gauge", "fit, eta=0.5", "fit, gauge por columnas"]
tabla = crudo.groupby("metodo")[["NMI", "ARI", "coherencia"]].agg(["mean", "sem"])
tabla = tabla.loc[[m for m in orden if m in tabla.index]]

print("\n" + "=" * 78)
print(f"agrupacion en {N_CLUSTERS} grupos, init aleatorio, {len(SEMILLAS)} semillas")
print("=" * 78)
print(tabla.to_string(float_format=lambda x: f"{x:.3f}"))

print("\ncon init nndsvd, que es determinista (una sola corrida posible):")
det = pd.DataFrame(deterministas).set_index("metodo")[["NMI", "ARI", "coherencia"]]
print(det.loc[[m for m in orden if m in det.index]].to_string(
    float_format=lambda x: f"{x:.3f}"))
print("\ncoherencia: cuantas veces mas probable es compartir genero dentro de un")
print("cluster que entre dos peliculas al azar. 1.0 es el azar.")

if legibles:
    print("\nfraccion de grupos cuyo genero top en el backbone coincide con el observado:")
    for nombre, valores in legibles.items():
        print(f"  {nombre:<26} {np.mean(valores):.2f}")

# Comparacion pareada por semilla contra la ruta anterior, que es la unica
# forma de saber si la diferencia sobrevive al ruido de inicializacion.
pivote = crudo.pivot_table(index="semilla", columns="metodo", values=["NMI", "ARI"])
print("\ncontra dfmf_sparse, pareado por semilla:")
for metrica in ("NMI", "ARI"):
    for metodo in ("fit, gauge por columnas", "fit, sin gauge"):
        diferencia = pivote[(metrica, metodo)] - pivote[(metrica, "dfmf_sparse")]
        sem = diferencia.std(ddof=1) / np.sqrt(len(diferencia))
        veces = abs(diferencia.mean()) / max(sem, 1e-12)
        print(f"  {metrica} {metodo:<26} {diferencia.mean():+.3f} +- {sem:.3f} "
              f"({veces:.1f} SE, {'significativo' if veces >= 2 else 'ruido'})")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
crudo.to_csv(DIR_SALIDA / "clustering.csv", index=False)


# %%
# El otro lado del co-clustering: que dice el backbone. S[etiquetas] conecta
# los grupos de pelicula con los de genero, asi que cada grupo de peliculas
# deberia mapear a generos reconocibles.
modelo = fit(relaciones(VISTAS), ranks, max_iter=MAX_ITER, tol=1e-7,
             init="nndsvd", random_state=SEMILLA)
etiquetas = clusters_de(modelo.G["pelicula"])
perfil = modelo.S["etiquetas"] @ modelo.G["genero"].T

print("\n" + "=" * 78)
print("que genero describe a cada grupo, segun el backbone")
print("=" * 78)
for grupo in range(N_CLUSTERS):
    miembros = np.flatnonzero(etiquetas == grupo)
    if len(miembros) == 0:
        continue
    top = np.argsort(-perfil[grupo])[:3]
    reales = Y[miembros].sum(axis=0)
    top_reales = np.argsort(-reales)[:3]
    print(f"  grupo {grupo:>2} ({len(miembros):>4} peliculas)  "
          f"backbone: {', '.join(datos.generos[j] for j in top):<34} "
          f"observado: {', '.join(datos.generos[j] for j in top_reales)}")
