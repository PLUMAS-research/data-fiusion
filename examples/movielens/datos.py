"""Carga de MovieLens como relaciones sparse con pelicula como tipo fuente.

El dataset viene con scikit-fusion (~6.5 MB en disco, no requiere descarga).
Tres tipos acompanan a la pelicula: usuario (quien la califico), genero
(la etiqueta a predecir) y actor (quien aparece en ella).

Las tres relaciones se construyen con `pelicula` como fuente:

    (pelicula, usuario)  ratings, valores 0.5 a 5.0
    (pelicula, genero)   binaria, multi-etiqueta
    (pelicula, actor)    binaria

Esa orientacion permite proyectar peliculas nuevas con `fold_in_entities`,
que exige que el tipo objetivo sea la fuente de todas las relaciones nuevas.

El split separa peliculas completas en tres partes. Las de validacion
sirven para elegir hiperparametros y las de test se reservan para el
resultado final, asi que ninguna de las dos participa del ajuste y su
genero no se filtra al modelo.

La ruta al dataset se controla con la variable de entorno MOVIELENS_DIR.
"""

import csv
import gzip
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def ruta_por_defecto():
    """Locate the MovieLens copy that ships with scikit-fusion, if installed.

    The package is located rather than imported: importing it executes
    code that relies on numpy names retired years ago, so an import here
    would fail for reasons that have nothing to do with finding the data.
    """
    import importlib.util

    especificacion = importlib.util.find_spec("skfusion")
    if especificacion is None or not especificacion.origin:
        return None
    return Path(especificacion.origin).parent / "datasets" / "data" / "movielens"

GENERO_VACIO = "(no genres listed)"

CLAVE_GENERO = ("pelicula", "genero")
CLAVES_VISTA = [("pelicula", "usuario"), ("pelicula", "actor")]


@dataclass
class Split:
    """Un subconjunto de peliculas con sus relaciones y su ground truth."""

    R: dict                # dict[(src, dst)] -> list[csr_matrix], incluye genero
    Y: np.ndarray          # (n, n_generos) binaria
    peliculas: list        # movieId en el orden de las filas

    def vistas(self, claves):
        """Relaciones para fold-in: las claves pedidas, sin el genero."""
        return {c: self.R[c] for c in claves}

    def __len__(self):
        return len(self.peliculas)


@dataclass
class DatosMovieLens:
    train: Split
    valid: Split
    test: Split
    generos: list
    usuarios: list
    actores: list
    titulos: dict

    @property
    def n_generos(self):
        return len(self.generos)


def _leer_ratings(ruta):
    """Devuelve (usuario, pelicula, rating) como tres listas paralelas."""
    usuarios, peliculas, valores = [], [], []
    with gzip.open(ruta / "ratings.csv.gz", "rt", encoding="utf-8") as f:
        f.readline()
        for linea in f:
            u, m, r, _ = linea.strip().split(",")
            usuarios.append(int(u))
            peliculas.append(int(m))
            valores.append(float(r))
    return usuarios, peliculas, valores


def _leer_lista_por_pelicula(ruta, archivo):
    """Lee movies.csv.gz o actors.csv.gz: movieId -> (titulo, lista de items)."""
    salida = {}
    with gzip.open(ruta / archivo, "rt", encoding="utf-8") as f:
        f.readline()
        for fila in csv.reader(f):
            items = [x for x in fila[2].split("|") if x and x != GENERO_VACIO]
            salida[int(fila[0])] = (fila[1], items)
    return salida


def _matriz_binaria(peliculas, mapa_items, indice_item):
    """Matriz (n_peliculas, n_items) con 1 donde la pelicula contiene el item."""
    filas, columnas = [], []
    for i, pid in enumerate(peliculas):
        for item in mapa_items.get(pid, ("", []))[1]:
            j = indice_item.get(item)
            if j is not None:
                filas.append(i)
                columnas.append(j)
    datos = np.ones(len(filas), dtype=np.float64)
    return sp.csr_matrix(
        (datos, (filas, columnas)), shape=(len(peliculas), len(indice_item))
    )


def cargar(min_peliculas_actor=5, min_ratings_pelicula=5, frac_valid=0.2,
           frac_test=0.2, semilla=0, ruta=None, verbose=True):
    """Construye las relaciones de MovieLens y los tres splits de peliculas.

    Parameters
    ----------
    min_peliculas_actor : int
        Un actor entra al modelo si aparece en al menos esta cantidad de
        peliculas del universo. Controla el ancho de (pelicula, actor).
    min_ratings_pelicula : int
        Descarta peliculas con muy pocos ratings: sin senal de usuarios el
        fold-in queda determinado solo por el elenco.
    frac_valid, frac_test : float
        Fraccion de peliculas para elegir hiperparametros y para el
        resultado final.
    semilla : int
    ruta : Path or None
        Directorio con ratings.csv.gz, movies.csv.gz y actors.csv.gz.
        Por defecto MOVIELENS_DIR, o la copia que trae scikit-fusion.
    """
    ruta = ruta or os.environ.get("MOVIELENS_DIR") or ruta_por_defecto()
    if ruta is None:
        raise FileNotFoundError(
            "No encuentro MovieLens. Instala scikit-fusion, que lo trae incluido "
            "(uv pip install -e <ruta a scikit-fusion>), o define MOVIELENS_DIR "
            "apuntando a un directorio con ratings.csv.gz, movies.csv.gz y actors.csv.gz."
        )
    ruta = Path(ruta)
    if not (ruta / "ratings.csv.gz").exists():
        raise FileNotFoundError(
            f"No encuentro ratings.csv.gz en {ruta}. "
            "Define MOVIELENS_DIR apuntando al directorio del dataset."
        )
    if verbose:
        print(f"leyendo MovieLens desde {ruta}")

    r_usuarios, r_peliculas, r_valores = _leer_ratings(ruta)
    generos_por_pelicula = _leer_lista_por_pelicula(ruta, "movies.csv.gz")
    actores_por_pelicula = _leer_lista_por_pelicula(ruta, "actors.csv.gz")
    if verbose:
        print(f"  ratings: {len(r_valores)}")
        print(f"  peliculas con metadatos: {len(generos_por_pelicula)}")

    # Universo: peliculas con suficientes ratings, con genero y con elenco.
    ratings_por_pelicula = Counter(r_peliculas)
    peliculas = sorted(
        pid for pid, n in ratings_por_pelicula.items()
        if n >= min_ratings_pelicula
        and generos_por_pelicula.get(pid, ("", []))[1]
        and actores_por_pelicula.get(pid, ("", []))[1]
    )
    if verbose:
        print(f"  peliculas en el universo: {len(peliculas)} "
              f"(>= {min_ratings_pelicula} ratings, con genero y elenco)")

    conteo_actores = Counter()
    for pid in peliculas:
        for actor in actores_por_pelicula[pid][1]:
            conteo_actores[actor] += 1
    actores = sorted(a for a, n in conteo_actores.items() if n >= min_peliculas_actor)
    generos = sorted({g for pid in peliculas for g in generos_por_pelicula[pid][1]})
    usuarios = sorted(set(r_usuarios))
    if verbose:
        print(f"  usuarios: {len(usuarios)}, generos: {len(generos)}, "
              f"actores: {len(actores)} (de {len(conteo_actores)} con >= "
              f"{min_peliculas_actor} peliculas)")

    idx_pelicula = {pid: i for i, pid in enumerate(peliculas)}
    idx_usuario = {uid: i for i, uid in enumerate(usuarios)}
    idx_genero = {g: i for i, g in enumerate(generos)}
    idx_actor = {a: i for i, a in enumerate(actores)}

    filas, columnas, datos = [], [], []
    for u, m, v in zip(r_usuarios, r_peliculas, r_valores):
        i = idx_pelicula.get(m)
        if i is not None:
            filas.append(i)
            columnas.append(idx_usuario[u])
            datos.append(v)
    X = {
        ("pelicula", "usuario"): sp.csr_matrix(
            (datos, (filas, columnas)), shape=(len(peliculas), len(usuarios))),
        CLAVE_GENERO: _matriz_binaria(peliculas, generos_por_pelicula, idx_genero),
        ("pelicula", "actor"): _matriz_binaria(peliculas, actores_por_pelicula, idx_actor),
    }

    if verbose:
        for (_, tipo), M in X.items():
            densidad = M.nnz / (M.shape[0] * M.shape[1])
            denso_mb = M.shape[0] * M.shape[1] * 8 / 1e6
            print(f"  (pelicula, {tipo}): shape={M.shape}, nnz={M.nnz}, "
                  f"densidad={densidad:.4%}, densificada={denso_mb:.1f} MB")

    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(peliculas))
    n_valid = int(round(frac_valid * len(peliculas)))
    n_test = int(round(frac_test * len(peliculas)))
    cortes = {
        "valid": np.sort(orden[:n_valid]),
        "test": np.sort(orden[n_valid:n_valid + n_test]),
        "train": np.sort(orden[n_valid + n_test:]),
    }
    if verbose:
        print("  split: " + ", ".join(f"{k}={len(v)}" for k, v in cortes.items()))

    splits = {
        nombre: Split(
            R={clave: [M[idx]] for clave, M in X.items()},
            Y=np.asarray(X[CLAVE_GENERO][idx].todense()),
            peliculas=[peliculas[i] for i in idx],
        )
        for nombre, idx in cortes.items()
    }
    return DatosMovieLens(
        train=splits["train"],
        valid=splits["valid"],
        test=splits["test"],
        generos=generos,
        usuarios=usuarios,
        actores=actores,
        titulos={pid: generos_por_pelicula[pid][0] for pid in peliculas},
    )


if __name__ == "__main__":
    datos = cargar()
    print(f"\ngeneros: {datos.generos}")
    print(f"generos por pelicula en test: media={datos.test.Y.sum(axis=1).mean():.2f}")
