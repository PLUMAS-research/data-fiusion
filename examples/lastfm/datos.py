"""Carga de Last.fm (HetRec 2011) como relaciones para fusion.

Descarga el zip de GroupLens la primera vez (unos 2.5 MB) al directorio
data/ junto a este script, o al que indique la variable de entorno
LASTFM_DIR. Uso no comercial segun el README del dataset.

Construye dos relaciones con el artista como tipo compartido:

- escuchas (artista x usuario): conteos de reproduccion por usuario, la
  relacion de conteos sobredispersos.
- etiquetas (artista x tag): binaria sobre los TOP_TAGS tags mas usados;
  un artista lleva un tag si al menos MIN_USUARIOS_TAG usuarios
  distintos se lo aplicaron, lo que filtra el ruido de la folksonomia.

El universo de artistas es el de los que tienen escuchas. La red de
amistades del dataset queda fuera de esta etapa.
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
DIR_DATOS = Path(os.environ.get("LASTFM_DIR", Path(__file__).parent / "data"))

TOP_TAGS = 100
MIN_USUARIOS_TAG = 2


def _descargar():
    if (DIR_DATOS / "user_artists.dat").exists():
        return
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    print(f"descargando {URL} ...")
    with urllib.request.urlopen(URL) as respuesta:
        contenido = respuesta.read()
    zipfile.ZipFile(io.BytesIO(contenido)).extractall(DIR_DATOS)
    print(f"  datos en {DIR_DATOS}")


def cargar():
    _descargar()
    escuchas_df = pd.read_csv(DIR_DATOS / "user_artists.dat", sep="\t")
    tags_df = pd.read_csv(DIR_DATOS / "user_taggedartists.dat", sep="\t")
    nombres_tags = pd.read_csv(DIR_DATOS / "tags.dat", sep="\t",
                               encoding="latin-1").set_index("tagID")["tagValue"]

    artista_ids = np.sort(escuchas_df["artistID"].unique())
    usuario_ids = np.sort(escuchas_df["userID"].unique())
    indice_artista = pd.Series(np.arange(len(artista_ids)), index=artista_ids)
    indice_usuario = pd.Series(np.arange(len(usuario_ids)), index=usuario_ids)

    escuchas = sp.csr_matrix(
        (escuchas_df["weight"].astype(np.float64),
         (indice_artista[escuchas_df["artistID"]].to_numpy(),
          indice_usuario[escuchas_df["userID"]].to_numpy())),
        shape=(len(artista_ids), len(usuario_ids)))
    escuchas.sum_duplicates()

    tags_df = tags_df[tags_df["artistID"].isin(indice_artista.index)]
    por_par = (tags_df.groupby(["artistID", "tagID"])["userID"].nunique()
               .reset_index(name="usuarios"))
    por_par = por_par[por_par["usuarios"] >= MIN_USUARIOS_TAG]
    top = (por_par.groupby("tagID").size().sort_values(ascending=False)
           .head(TOP_TAGS).index)
    por_par = por_par[por_par["tagID"].isin(top)]
    indice_tag = pd.Series(np.arange(len(top)), index=top)

    etiquetas = sp.csr_matrix(
        (np.ones(len(por_par)),
         (indice_artista[por_par["artistID"]].to_numpy(),
          indice_tag[por_par["tagID"]].to_numpy())),
        shape=(len(artista_ids), len(top)))

    etiquetados = np.flatnonzero(etiquetas.getnnz(axis=1) > 0)
    return dict(
        escuchas=escuchas, etiquetas=etiquetas, etiquetados=etiquetados,
        nombres_tags=nombres_tags.reindex(top).to_numpy(),
    )


if __name__ == "__main__":
    d = cargar()
    escuchas, etiquetas = d["escuchas"], d["etiquetas"]
    print(f"escuchas: {escuchas.shape[0]} artistas x {escuchas.shape[1]} usuarios, "
          f"nnz={escuchas.nnz}")
    datos = escuchas.data
    print(f"  conteos: media {datos.mean():.1f}, maximo {datos.max():.0f}, "
          f"indice de dispersion {datos.var() / datos.mean():.0f}")
    print(f"etiquetas: {etiquetas.shape[1]} tags, {len(d['etiquetados'])} artistas "
          f"con al menos una, nnz={etiquetas.nnz}")
    print(f"  tags mas frecuentes: {', '.join(d['nombres_tags'][:8])}")
