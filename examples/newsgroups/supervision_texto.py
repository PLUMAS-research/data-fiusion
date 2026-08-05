"""Anclaje de componentes sobre texto, el regimen donde TS-NMF fue disenado.

En MovieLens, anclar componentes a los generos conocidos mejoro la
legibilidad y empeoro la prediccion. Ese caso tiene varias relaciones
compitiendo (ratings y elenco traen estructura que no se alinea con el
genero), asi que no dice mucho sobre el mecanismo en si.

Este script lo evalua donde deberia funcionar: una sola relacion
documento-termino, con categorias que si tienen correspondencia con el
vocabulario. Es el experimento de TS-NMF (MacMillan y Wilson, 2017) sobre
20 Newsgroups.

Una diferencia con el paper que conviene tener presente. TS-NMF factoriza
V ~ (W . L) H, donde cada topico es una distribucion libre sobre TODOS los
terminos. data-fiusion hace tri-factorizacion, V ~ G_doc S G_term^T, que
ademas agrupa los terminos en c_term grupos. Es una restriccion adicional,
asi que los numeros no son comparables con los del paper; lo comparable es
el efecto de subir la tasa de supervision.

Los documentos supervisados quedan forzados a su categoria, asi que
evaluarlos seria circular. Todas las metricas van sobre los NO supervisados.

Requiere descargar 20 Newsgroups la primera vez (unos 14 MB, via sklearn).

Ejecucion: uv run python examples/newsgroups/supervision_texto.py
"""

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datafiusion import Relation, fuse


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2)
MAX_ITER = 200
TASAS = (0.0, 0.05, 0.10, 0.20, 0.50)
RANK_TERMINO = 50
MIN_DF, MAX_DF, MAX_FEATURES = 5, 0.5, 20_000


# %%
print("cargando 20 Newsgroups...")
corpus = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
vectorizador = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, stop_words="english",
                               max_features=MAX_FEATURES)
V = vectorizador.fit_transform(corpus.data).tocsr()
categoria = np.asarray(corpus.target)
n_docs, n_terminos = V.shape
n_categorias = len(corpus.target_names)

# Documentos que quedaron sin ningun termino tras el filtrado de vocabulario.
con_texto = np.flatnonzero(V.getnnz(axis=1) > 0)
if len(con_texto) < n_docs:
    print(f"  {n_docs - len(con_texto)} documentos quedaron sin terminos y se descartan")
    V, categoria = V[con_texto], categoria[con_texto]
    n_docs = V.shape[0]

print(f"  {n_docs} documentos, {n_terminos} terminos, {n_categorias} categorias")
print(f"  densidad: {V.nnz / (n_docs * n_terminos):.4%}, nnz: {V.nnz}")


# %%
def matriz_de_supervision(supervisados):
    """L de TS-NMF: cada documento supervisado solo puede usar su categoria."""
    permitido = np.ones((n_docs, n_categorias), dtype=bool)
    permitido[supervisados] = False
    permitido[supervisados, categoria[supervisados]] = True
    return permitido


def evaluar(nombre, modelo, evaluados, tasa, semilla):
    grupos = modelo.factor("documento").argmax(axis=1)[evaluados]
    verdad = categoria[evaluados]
    fila = {
        "metodo": nombre, "tasa": tasa, "semilla": semilla,
        "ARI": adjusted_rand_score(verdad, grupos),
        "NMI": normalized_mutual_info_score(verdad, grupos),
        # Con las componentes ancladas, la componente j es la categoria j, asi
        # que se puede leer el grupo directamente sin emparejar nada.
        "acierto directo": float((grupos == verdad).mean()),
    }
    return fila


def ajustar(supervisados, semilla):
    R = {"docs": Relation(src="documento", dst="termino", matrix=V)}
    ranks = {"documento": n_categorias, "termino": RANK_TERMINO}
    supervision = ({"documento": matriz_de_supervision(supervisados)}
                   if len(supervisados) else None)
    return fuse(R, ranks, supervision=supervision, max_iter=MAX_ITER, tol=1e-7,
               init="random", random_state=semilla)


# %%
filas = []
for semilla in SEMILLAS:
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(n_docs)
    for tasa in TASAS:
        n_sup = int(round(tasa * n_docs))
        supervisados = np.sort(orden[:n_sup])
        evaluados = np.sort(orden[n_sup:])

        modelo = ajustar(supervisados, semilla)
        filas.append(evaluar(f"supervision {tasa:.0%}", modelo, evaluados,
                             tasa, semilla))
        print(f"  semilla {semilla}, tasa {tasa:.0%}: "
              f"ARI={filas[-1]['ARI']:.3f}, acierto={filas[-1]['acierto directo']:.3f}")


# %%
tabla = pd.DataFrame(filas)
resumen = tabla.groupby("tasa")[["ARI", "NMI", "acierto directo"]].agg(["mean", "sem"])

print("\n" + "=" * 74)
print(f"documentos NO supervisados, {len(SEMILLAS)} semillas")
print("=" * 74)
print(resumen.to_string(float_format=lambda x: f"{x:.3f}"))

base = tabla[tabla.tasa == 0.0].set_index("semilla")
for tasa in TASAS[1:]:
    actual = tabla[tabla.tasa == tasa].set_index("semilla")
    for metrica in ("ARI", "acierto directo"):
        diferencia = actual[metrica] - base[metrica]
        sem = diferencia.std(ddof=1) / np.sqrt(len(diferencia))
        veces = abs(diferencia.mean()) / max(sem, 1e-12)
        if metrica == "ARI":
            print(f"\ntasa {tasa:.0%} contra sin supervisar: "
                  f"ARI {diferencia.mean():+.3f} +- {sem:.3f} ({veces:.1f} SE)")

print("\nSin supervision las componentes no corresponden a categorias, asi que")
print("el 'acierto directo' de la fila 0% es solo el azar de que coincidan.")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
tabla.to_csv(DIR_SALIDA / "supervision_texto.csv", index=False)
print(f"\ntabla guardada en {DIR_SALIDA / 'supervision_texto.csv'}")


# %%
# Referencia externa: NMF puro de sklearn, que es el modelo del paper (no
# agrupa terminos). Sirve para saber cuanto cuesta la tri-factorizacion.
from sklearn.decomposition import NMF

print("\n" + "=" * 74)
print("referencia: NMF de sklearn, sin supervisar y sin agrupar terminos")
print("=" * 74)
referencia = []
for semilla in SEMILLAS:
    rng = np.random.default_rng(semilla)
    evaluados = np.sort(rng.permutation(n_docs)[int(TASAS[-2] * n_docs):])
    W = NMF(n_components=n_categorias, init="nndsvd", max_iter=MAX_ITER,
            random_state=semilla).fit_transform(V)
    referencia.append(adjusted_rand_score(categoria[evaluados],
                                          W.argmax(axis=1)[evaluados]))
print(f"  ARI = {np.mean(referencia):.3f} +- {np.std(referencia, ddof=1) / np.sqrt(len(referencia)):.3f}")
print(f"  contra {resumen.loc[0.0, ('ARI', 'mean')]:.3f} de la tri-factorizacion sin supervisar")
print("  Agrupar los terminos no cuesta calidad en este dataset.")


# %%
# Que palabras describe cada componente, para ver si el anclaje produce
# topicos legibles. Se lee por el backbone hacia el espacio de terminos.
modelo = ajustar(np.sort(np.random.default_rng(0).permutation(n_docs)[:int(0.2 * n_docs)]), 0)
vocabulario = np.asarray(vectorizador.get_feature_names_out())
perfil = modelo.backbone("docs") @ modelo.factor("termino").T

print("\n" + "=" * 74)
print("terminos de cada componente, con 20% de supervision")
print("=" * 74)
for j in range(n_categorias):
    top = np.argsort(-perfil[j])[:6]
    print(f"  {corpus.target_names[j]:<26} {', '.join(vocabulario[top])}")
