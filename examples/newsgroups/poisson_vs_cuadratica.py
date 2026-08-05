"""Familia Poisson contra la referencia cuadratica sobre TF-IDF.

`conteos.py` establecio la referencia: la tri-factorizacion cuadratica sobre
TF-IDF llega a ARI 0.185, y ninguna transformacion estabilizadora de
varianza pasa de 0.114. Este script mide si la likelihood de conteo
(KL generalizada, familia Poisson) supera esa referencia.

Criterio de exito, declarado antes de correr: ganarle al ARI 0.185 de la
cuadratica sobre TF-IDF por 2 errores estandar. Si Poisson sobre conteos
crudos no lo logra pero la variante con idf si, la conclusion es que el
reponderado por especificidad es necesario tambien bajo la likelihood
correcta.

Tres variantes:

- crudo: KL sobre los conteos tal cual.
- idf: KL sobre la matriz con columnas escaladas por idf. Es exactamente
  la KL ponderada por columna (el factor se absorbe en el factor de
  terminos), asi que no necesita pesos por entrada.
- tfidf: KL sobre los mismos valores TF-IDF de la referencia cuadratica.

Ejecucion: uv run python examples/newsgroups/poisson_vs_cuadratica.py
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

REFERENCIA_CUADRATICA = ("tfidf", 0.185, 0.013)   # de output/conteos.csv


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

idf = TfidfTransformer(norm=None).fit(conteos).idf_
VARIANTES = {
    "poisson crudo": conteos.astype(np.float64),
    "poisson idf": conteos.multiply(idf[None, :]).tocsr(),
    "poisson tfidf": TfidfTransformer().fit_transform(conteos).tocsr(),
}


# %%
filas = []
for nombre_variante, V in VARIANTES.items():
    for semilla in SEMILLAS:
        R = {"docs": Relation(src="documento", dst="termino", matrix=V,
                              family="poisson")}
        modelo = fuse(R, {"documento": n_categorias, "termino": RANK_TERMINO},
                     max_iter=MAX_ITER, tol=1e-6, init="random",
                     random_state=semilla)
        grupos = modelo.factor("documento").argmax(axis=1)
        filas.append({
            "variante": nombre_variante, "semilla": semilla,
            "ARI": adjusted_rand_score(categoria, grupos),
            "NMI": normalized_mutual_info_score(categoria, grupos),
            "desviacion": modelo.rel_error["docs"],
            "iteraciones": modelo.n_iter,
        })
        print(f"  {nombre_variante:<14} semilla {semilla}: "
              f"ARI={filas[-1]['ARI']:.3f}, NMI={filas[-1]['NMI']:.3f}, "
              f"desviacion={filas[-1]['desviacion']:.3f} "
              f"({filas[-1]['iteraciones']} iter)")


# %%
tabla = pd.DataFrame(filas)
resumen = tabla.groupby("variante")[["ARI", "NMI"]].agg(["mean", "sem"])
resumen = resumen.loc[list(VARIANTES)]

print("\n" + "=" * 64)
print(f"agrupamiento contra categorias, {len(SEMILLAS)} semillas")
print("=" * 64)
print(resumen.to_string(float_format=lambda x: f"{x:.3f}"))

nombre_ref, ari_ref, sem_ref = REFERENCIA_CUADRATICA
print(f"\nreferencia cuadratica ({nombre_ref}): ARI {ari_ref:.3f} +- {sem_ref:.3f}")
for variante in VARIANTES:
    media = resumen.loc[variante, ("ARI", "mean")]
    sem = resumen.loc[variante, ("ARI", "sem")]
    diferencia = media - ari_ref
    se_comb = float(np.sqrt(sem ** 2 + sem_ref ** 2))
    veredicto = "SUPERA" if diferencia > 2 * se_comb else \
        ("empata" if abs(diferencia) <= 2 * se_comb else "pierde")
    print(f"  {variante:<14} {media:+.3f} contra la referencia: "
          f"{diferencia:+.3f} ({diferencia / max(se_comb, 1e-12):+.1f} SE) -> {veredicto}")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
tabla.to_csv(DIR_SALIDA / "poisson.csv", index=False)
print(f"\ntabla guardada en {DIR_SALIDA / 'poisson.csv'}")
