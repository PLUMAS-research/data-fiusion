"""Tiempo y memoria de dfmf_sparse frente a scikit-fusion sobre MovieLens.

Las dos implementaciones resuelven el mismo problema con la misma cantidad
de iteraciones y los mismos rangos. La diferencia esta en el
almacenamiento: `dfmf_sparse` mantiene las relaciones en `scipy.sparse` y
scikit-fusion las densifica.

Dos barridos, porque el costo de densificar crece por dos ejes distintos:

  filas     mas peliculas, con el ancho de las matrices fijo
  columnas  mas actores (bajando el minimo de peliculas por actor), con
            todas las peliculas. La matriz se ensancha y se vuelve mas
            dispersa a la vez, que es el regimen donde vive el caso real.

Cada corrida se ejecuta en un subproceso propio para que el pico de
memoria sea atribuible y para que un desvio de consumo mate solo esa
corrida. Antes de lanzarla se estima la huella densa y se compara contra
la memoria disponible; las que no caben se informan y se saltan. Mientras
corre, el proceso padre sondea MemAvailable y termina al hijo si baja del
umbral.

Variables de entorno: MOVIELENS_DIR, DIR_SALIDA, UMBRAL_MEMORIA_GB.
Ejecucion: uv run python examples/movielens/escala.py
"""

# %%
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))
DIR_CACHE = DIR_SALIDA / "cache_escala"
UMBRAL_MEMORIA_GB = float(os.environ.get("UMBRAL_MEMORIA_GB", 6.0))

RANKS = {"pelicula": 20, "usuario": 25, "actor": 40, "genero": 15}
MAX_ITER = 20
SEMILLA = 0

# Margen sobre la estimacion densa antes de declarar que una corrida no
# cabe. scikit-fusion mantiene varias matrices intermedias del tamano de
# las relaciones, asi que el pico supera al de los datos de entrada.
FACTOR_COPIAS_DENSAS = 4.0
MARGEN_SEGURIDAD = 1.5

CLAVES = [("pelicula", "usuario"), ("pelicula", "genero"), ("pelicula", "actor")]


def memoria_disponible_gb():
    with open("/proc/meminfo") as f:
        for linea in f:
            if linea.startswith("MemAvailable:"):
                return int(linea.split()[1]) / 1e6
    raise RuntimeError("No pude leer MemAvailable de /proc/meminfo")


# %%
# ----------------------------- worker --------------------------------
# Corre una sola configuracion y reporta tiempo y pico de memoria propio.

def cargar_matrices(dir_cache, min_actor):
    return {
        clave: sp.load_npz(dir_cache / f"min{min_actor}_{clave[1]}.npz")
        for clave in CLAVES
    }


def pico_mb():
    """Maximo RSS del proceso hasta ahora, en MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_actual_mb():
    """RSS instantaneo, en MB. Sirve de linea base antes de factorizar.

    En las configuraciones chicas el neto que resulta de restar esta base
    al pico queda por debajo del uso real: el maximo historico del proceso
    se alcanza al importar numpy, scipy y joblib, y el trabajo posterior
    reutiliza esa memoria sin superarla. Las corridas del eje de columnas
    son las que reflejan el costo de densificar.
    """
    with open("/proc/self/status") as f:
        for linea in f:
            if linea.startswith("VmRSS:"):
                return int(linea.split()[1]) / 1024
    raise RuntimeError("No pude leer VmRSS de /proc/self/status")


def correr_sparse(matrices, ranks, max_iter):
    from datafiusion import dfmf_sparse

    R = {clave: [M] for clave, M in matrices.items()}
    base = rss_actual_mb()
    inicio = time.time()
    dfmf_sparse(R=R, ranks=ranks, lambda_G=0.0, lambda_S=0.01,
                max_iter=max_iter, init="random", random_state=SEMILLA)
    return time.time() - inicio, base


def _parchar_skfusion():
    """Repara el filtro `entry != []` de _par_bdot, que numpy >= 1.25 rechaza.

    `entry` puede ser un ndarray y la comparacion contra la lista vacia ya
    no se difunde a un arreglo de booleanos. Mismo parche que usa
    tests/test_compare_skfusion.py.
    """
    from joblib import Parallel, delayed
    import skfusion.fusion.decomposition._dfmf as _dfmf

    bdot = next((getattr(_dfmf, n) for n in dir(_dfmf) if n.endswith("__bdot")), None)
    if bdot is None:
        raise RuntimeError("No encontre __bdot dentro de skfusion._dfmf")

    def _par_bdot(A, B, obj_types, verbose, n_jobs):
        parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e3, verbose=verbose,
                                backend="multiprocessing")
        entries = parallelizer(
            delayed(bdot)(A, B, i, j, obj_types) for i in obj_types for j in obj_types)
        return {(i, j): entry for i, j, entry in entries
                if not (isinstance(entry, list) and len(entry) == 0)}

    _dfmf._par_bdot = _par_bdot


def correr_skfusion(matrices, ranks, max_iter):
    # scikit-fusion todavia usa nombres que numpy y collections retiraron.
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    import collections
    import collections.abc
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable

    from skfusion import fusion as skf
    _parchar_skfusion()

    tipos = {nombre: skf.ObjectType(nombre, ranks[nombre]) for nombre in ranks}
    # La linea base se toma antes de densificar, asi que el neto que reporta
    # la tabla incluye el costo de convertir las relaciones a ndarray.
    base = rss_actual_mb()
    relaciones = [
        skf.Relation(np.asarray(M.todense()), tipos[clave[0]], tipos[clave[1]])
        for clave, M in matrices.items()
    ]
    grafo = skf.FusionGraph(relaciones)
    inicio = time.time()
    skf.Dfmf(max_iter=max_iter, init_type="random",
             random_state=np.random.RandomState(SEMILLA), n_jobs=1).fuse(grafo)
    return time.time() - inicio, base


def worker(args):
    matrices = cargar_matrices(Path(args.dir_cache), args.min_actor)
    if args.n_peliculas < matrices[CLAVES[0]].shape[0]:
        rng = np.random.default_rng(SEMILLA)
        filas = np.sort(rng.permutation(
            matrices[CLAVES[0]].shape[0])[:args.n_peliculas])
        matrices = {clave: M[filas] for clave, M in matrices.items()}

    ranks = {"pelicula": RANKS["pelicula"]}
    ranks.update({clave[1]: RANKS[clave[1]] for clave in CLAVES})

    if args.backend == "sparse":
        segundos, base = correr_sparse(matrices, ranks, MAX_ITER)
    else:
        segundos, base = correr_skfusion(matrices, ranks, MAX_ITER)

    pico = pico_mb()
    print("RESULTADO " + json.dumps({
        "segundos": segundos,
        "pico_mb": pico,
        "neto_mb": pico - base,
        "n_peliculas": matrices[CLAVES[0]].shape[0],
        "n_actores": matrices[("pelicula", "actor")].shape[1],
        "nnz": int(sum(M.nnz for M in matrices.values())),
    }))


# %%
# ----------------------------- padre ---------------------------------

def preparar_cache(min_actores):
    """Escribe las matrices completas una vez por cada minimo de actores."""
    import datos as datos_movielens

    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    informacion = {}
    for min_actor in min_actores:
        rutas = {c: DIR_CACHE / f"min{min_actor}_{c[1]}.npz" for c in CLAVES}
        if all(r.exists() for r in rutas.values()):
            matrices = cargar_matrices(DIR_CACHE, min_actor)
            print(f"  min_actor={min_actor}: cache reutilizada, "
                  f"{matrices[('pelicula', 'actor')].shape[1]} actores")
        else:
            datos = datos_movielens.cargar(
                min_peliculas_actor=min_actor, frac_valid=0.0, frac_test=0.0,
                verbose=False)
            matrices = {c: datos.train.R[c][0] for c in CLAVES}
            for clave, M in matrices.items():
                sp.save_npz(rutas[clave], M)
            print(f"  min_actor={min_actor}: {matrices[('pelicula', 'actor')].shape[1]} "
                  f"actores, cache escrita")
        informacion[min_actor] = {
            "n_peliculas": matrices[CLAVES[0]].shape[0],
            "n_actores": matrices[("pelicula", "actor")].shape[1],
            "columnas": sum(M.shape[1] for M in matrices.values()),
            "nnz": int(sum(M.nnz for M in matrices.values())),
        }
    return informacion


def estimar_gb(n_peliculas, columnas, nnz, backend):
    """Huella esperada de los datos de entrada, en GB."""
    if backend == "skfusion":
        return n_peliculas * columnas * 8 * FACTOR_COPIAS_DENSAS / 1e9
    # csr: valor float64 mas indice int32, mas indptr por fila.
    return (nnz * 12 + n_peliculas * 8) / 1e9


def ejecutar(backend, n_peliculas, min_actor, umbral_gb):
    """Lanza una corrida en subproceso, vigilando la memoria de la maquina."""
    comando = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--backend", backend, "--n-peliculas", str(n_peliculas),
        "--min-actor", str(min_actor), "--dir-cache", str(DIR_CACHE),
    ]
    proceso = subprocess.Popen(comando, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    minimo_visto = memoria_disponible_gb()
    matado = False
    while proceso.poll() is None:
        time.sleep(0.5)
        disponible = memoria_disponible_gb()
        minimo_visto = min(minimo_visto, disponible)
        if disponible < umbral_gb:
            print(f"    guardian: MemAvailable bajo a {disponible:.1f} GB, "
                  f"terminando la corrida")
            proceso.terminate()
            try:
                proceso.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proceso.kill()
            matado = True
            break
    salida, error = proceso.communicate()

    if matado:
        return {"estado": "cortado por memoria"}
    if proceso.returncode != 0:
        return {"estado": f"fallo (codigo {proceso.returncode})",
                "detalle": error.strip().splitlines()[-1] if error.strip() else ""}
    for linea in salida.splitlines():
        if linea.startswith("RESULTADO "):
            resultado = json.loads(linea[len("RESULTADO "):])
            resultado["estado"] = "ok"
            resultado["min_disponible_gb"] = minimo_visto
            return resultado
    return {"estado": "sin resultado"}


def main():
    import pandas as pd

    print("=" * 78)
    print("MovieLens: costo de dfmf_sparse frente a scikit-fusion")
    print("=" * 78)
    print(f"iteraciones por corrida: {MAX_ITER}, rangos: {RANKS}")
    print(f"memoria disponible: {memoria_disponible_gb():.1f} GB, "
          f"umbral del guardian: {UMBRAL_MEMORIA_GB:.1f} GB")

    print("\npreparando matrices")
    min_actores = [5, 2, 1]
    informacion = preparar_cache(min_actores)

    n_total = informacion[5]["n_peliculas"]
    corridas = [("filas", n, 5) for n in (250, 500, 1000, 2000, n_total)]
    corridas += [("columnas", n_total, m) for m in (2, 1)]

    filas_tabla = []
    for eje, n_peliculas, min_actor in corridas:
        info = informacion[min_actor]
        # Las columnas del cache corresponden al universo completo; al
        # submuestrear filas el ancho no cambia.
        columnas = info["columnas"]
        nnz = int(info["nnz"] * n_peliculas / info["n_peliculas"])
        print(f"\n--- eje {eje}: {n_peliculas} peliculas, {info['n_actores']} actores, "
              f"{columnas} columnas totales ---")

        for backend in ("sparse", "skfusion"):
            estimado = estimar_gb(n_peliculas, columnas, nnz, backend)
            disponible = memoria_disponible_gb()
            if estimado * MARGEN_SEGURIDAD > disponible:
                print(f"  {backend:<9} estimado {estimado:.1f} GB por "
                      f"{MARGEN_SEGURIDAD:.1f} no cabe en {disponible:.1f} GB, se salta")
                filas_tabla.append({
                    "eje": eje, "backend": backend, "peliculas": n_peliculas,
                    "actores": info["n_actores"], "estado": "no cabe",
                    "estimado_gb": estimado,
                })
                continue

            resultado = ejecutar(backend, n_peliculas, min_actor, UMBRAL_MEMORIA_GB)
            fila = {
                "eje": eje, "backend": backend, "peliculas": n_peliculas,
                "actores": info["n_actores"], "estado": resultado["estado"],
                "estimado_gb": estimado,
            }
            if resultado["estado"] == "ok":
                fila["segundos"] = resultado["segundos"]
                fila["pico_mb"] = resultado["pico_mb"]
                fila["neto_mb"] = resultado["neto_mb"]
                print(f"  {backend:<9} {resultado['segundos']:7.1f} s, "
                      f"pico {resultado['pico_mb']:8.0f} MB, "
                      f"neto {resultado['neto_mb']:8.0f} MB "
                      f"(estimado {estimado * 1000:.0f} MB)")
            else:
                print(f"  {backend:<9} {resultado['estado']} "
                      f"{resultado.get('detalle', '')}")
            filas_tabla.append(fila)

    tabla = pd.DataFrame(filas_tabla)
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = DIR_SALIDA / "escala.csv"
    tabla.to_csv(ruta, index=False)

    print("\n" + "=" * 78)
    print("resumen")
    print("=" * 78)
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="-"))

    completas = tabla[tabla["estado"] == "ok"]
    if {"sparse", "skfusion"}.issubset(set(completas["backend"])):
        comparables = completas.pivot_table(
            index=["eje", "peliculas", "actores"], columns="backend",
            values=["segundos", "neto_mb"])
        razon = pd.DataFrame({
            "tiempo skfusion / sparse":
                comparables[("segundos", "skfusion")] / comparables[("segundos", "sparse")],
            "memoria skfusion / sparse":
                comparables[("neto_mb", "skfusion")] / comparables[("neto_mb", "sparse")],
        })
        print("\nrazon entre implementaciones, sobre la memoria neta "
              "(mayor que 1 favorece a sparse)")
        print(razon.to_string(float_format=lambda x: f"{x:.2f}", na_rep="-"))

    print(f"\ntabla guardada en {ruta}")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("sparse", "skfusion"))
    parser.add_argument("--n-peliculas", type=int)
    parser.add_argument("--min-actor", type=int)
    parser.add_argument("--dir-cache", type=str)
    argumentos = parser.parse_args()

    if argumentos.worker:
        worker(argumentos)
    else:
        main()
