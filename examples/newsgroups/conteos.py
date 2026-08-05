"""Transformaciones de conteos bajo la perdida cuadratica.

Los conteos de palabras son sobredispersos: la varianza de las entradas
crece mas rapido que su media, y la perdida cuadratica pesa cada entrada
por su magnitud, asi que los terminos frecuentes dominan el ajuste.

Antes de cambiar la likelihood (Poisson, negative binomial), la
referencia que cualquier likelihood nueva tiene que ganar es una
transformacion estabilizadora de varianza bajo el modelo actual. Este
script mide esa referencia: agrupar documentos con conteos crudos,
sqrt, log1p, Anscombe y TF-IDF, contra las categorias como verdad.

Las transformaciones se aplican solo a las entradas almacenadas. sqrt y
log1p mapean 0 a 0, asi que la esparsidad queda intacta; Anscombe se
desplaza (2 sqrt(x + 3/8) - 2 sqrt(3/8)) por la misma razon.

Ejecucion: uv run python examples/newsgroups/conteos.py
"""

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datafiusion import Relation, fuse


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2)
MAX_ITER = 200
RANK_TERMINO = 50
MIN_DF, MAX_DF, MAX_FEATURES = 5, 0.5, 20_000

TRANSFORMACIONES = ("crudo", "sqrt", "log1p", "anscombe", "tfidf")


# %%
print("cargando 20 Newsgroups...")
corpus = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
vectorizador = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF, stop_words="english",
                               max_features=MAX_FEATURES)
conteos = vectorizador.fit_transform(corpus.data).tocsr()
categoria = np.asarray(corpus.target)

con_texto = np.flatnonzero(conteos.getnnz(axis=1) > 0)
if len(con_texto) < conteos.shape[0]:
    print(f"  {conteos.shape[0] - len(con_texto)} documentos sin terminos se descartan")
    conteos, categoria = conteos[con_texto], categoria[con_texto]
n_docs, n_terminos = conteos.shape
n_categorias = len(corpus.target_names)
print(f"  {n_docs} documentos, {n_terminos} terminos, {n_categorias} categorias")

# La sobredispersion que motiva mirar mas alla de la perdida cuadratica:
# bajo Poisson la varianza igualaria a la media.
datos = conteos.data.astype(np.float64)
print(f"  entradas almacenadas: media {datos.mean():.2f}, varianza {datos.var():.2f}, "
      f"indice de dispersion {datos.var() / datos.mean():.1f}")


# %%
def transformar(M, como):
    salida = M.copy().astype(np.float64)
    if como == "crudo":
        return salida
    if como == "sqrt":
        salida.data = np.sqrt(salida.data)
        return salida
    if como == "log1p":
        salida.data = np.log1p(salida.data)
        return salida
    if como == "anscombe":
        salida.data = 2.0 * (np.sqrt(salida.data + 0.375) - np.sqrt(0.375))
        return salida
    if como == "tfidf":
        return TfidfTransformer().fit_transform(M).tocsr()
    raise ValueError(como)


# %%
filas = []
for como in TRANSFORMACIONES:
    V = transformar(conteos, como)
    for semilla in SEMILLAS:
        R = {"docs": Relation(src="documento", dst="termino", matrix=V)}
        modelo = fuse(R, {"documento": n_categorias, "termino": RANK_TERMINO},
                     max_iter=MAX_ITER, tol=1e-7, init="random",
                     random_state=semilla)
        grupos = modelo.factor("documento").argmax(axis=1)
        filas.append({
            "transformacion": como, "semilla": semilla,
            "ARI": adjusted_rand_score(categoria, grupos),
            "NMI": normalized_mutual_info_score(categoria, grupos),
        })
        print(f"  {como:<9} semilla {semilla}: ARI={filas[-1]['ARI']:.3f}, "
              f"NMI={filas[-1]['NMI']:.3f}")


# %%
tabla = pd.DataFrame(filas)
resumen = tabla.groupby("transformacion")[["ARI", "NMI"]].agg(["mean", "sem"])
resumen = resumen.loc[list(TRANSFORMACIONES)]

print("\n" + "=" * 60)
print(f"agrupamiento contra categorias, {len(SEMILLAS)} semillas")
print("=" * 60)
print(resumen.to_string(float_format=lambda x: f"{x:.3f}"))

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
tabla.to_csv(DIR_SALIDA / "conteos.csv", index=False)
print(f"\ntabla guardada en {DIR_SALIDA / 'conteos.csv'}")
