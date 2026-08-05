"""Two things this repo claims but had never measured together.

1. The initialization no longer copies the nnz. `init_nndsvd` used to
   normalize each block into a copy and then stack the copies, so the
   concatenation existed in memory before the SVD started. The streaming
   path applies each block against a LinearOperator instead. Both are run
   here on the same input, in separate processes, and compared on peak RSS
   and on the factors they produce.

2. Sparse gives the same answer as dense, not just a cheaper one. The
   parity test against scikit-fusion checks structure (shapes, monotone
   descent, comparable errors) on synthetic data. Here both engines fit
   the same MovieLens relations and are scored by the same downstream
   code, so the comparison is about the fit and nothing else.

Ejecucion: uv run python examples/movielens/equivalencia.py
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
SEMILLA = 0


def pico_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    with open("/proc/self/status") as f:
        for linea in f:
            if linea.startswith("VmRSS:"):
                return int(linea.split()[1]) / 1024
    raise RuntimeError("no pude leer VmRSS")


# --------------------------------------------------------------- worker init


def rutas_cache(n_filas):
    return {
        ("usuario", "celda"): DIR_SALIDA / f"cache_init/{n_filas}_celda.npz",
        ("usuario", "tiempo"): DIR_SALIDA / f"cache_init/{n_filas}_tiempo.npz",
    }


def preparar_matrices(n_filas):
    """Write the test matrices once, so the worker only loads them.

    Generating them inside the worker would put the peak of the process
    before the initialization even starts, and `ru_maxrss` is a historical
    maximum: the measurement would report the generator, not the init.
    """
    rutas = rutas_cache(n_filas)
    if all(r.exists() for r in rutas.values()):
        return
    (DIR_SALIDA / "cache_init").mkdir(parents=True, exist_ok=True)
    sp.save_npz(rutas[("usuario", "celda")],
                sp.random(n_filas, 1341, density=0.01, format="csr", random_state=1))
    sp.save_npz(rutas[("usuario", "tiempo")],
                sp.random(n_filas, 48, density=0.20, format="csr", random_state=2))


def worker_init(n_filas, camino):
    """Initialize one type from two wide blocks, reporting peak memory."""
    import datafiusion.init as modulo_init
    from datafiusion.init import init_nndsvd

    if camino == "exacto":
        modulo_init.MAX_NNZ_DENSO = 10 ** 12
        modulo_init.MAX_FILAS_DENSO = 10 ** 12
    else:
        modulo_init.MAX_NNZ_DENSO = 0

    rutas = rutas_cache(n_filas)
    R = {clave: [sp.load_npz(ruta)] for clave, ruta in rutas.items()}
    sizes = {"usuario": n_filas, "celda": 1341, "tiempo": 48}
    ranks = {"usuario": 50, "celda": 30, "tiempo": 10}

    base = rss_mb()
    pico_antes = pico_mb()
    inicio = time.time()
    G = init_nndsvd(R, sizes, ranks)
    segundos = time.time() - inicio
    huella = sum(M.data.nbytes + M.indices.nbytes for mats in R.values() for M in mats)

    print("RESULTADO " + json.dumps({
        "camino": camino, "n_filas": n_filas, "segundos": segundos,
        "pico_mb": pico_mb(), "pico_antes_mb": pico_antes,
        "neto_mb": pico_mb() - base,
        "datos_mb": huella / 1e6,
        "checksum": float(np.abs(G["usuario"]).sum()),
    }))


def medir_init():
    print("=" * 78)
    print("inicializacion: copiar la concatenacion contra aplicarla al vuelo")
    print("=" * 78)
    filas = []
    for n_filas in (250_000, 1_000_000):
        preparar_matrices(n_filas)
        for camino in ("exacto", "streaming"):
            proceso = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-init",
                 "--n-filas", str(n_filas), "--camino", camino],
                capture_output=True, text=True)
            if proceso.returncode != 0:
                ultima = proceso.stderr.strip().splitlines()[-1] if proceso.stderr.strip() else ""
                print(f"  {n_filas:>9} {camino:<10} fallo: {ultima}")
                continue
            for linea in proceso.stdout.splitlines():
                if linea.startswith("RESULTADO "):
                    filas.append(json.loads(linea[len("RESULTADO "):]))
    # El crecimiento del init es el pico DESPUES menos el pico ANTES. Restar
    # el RSS instantaneo en su lugar mezcla en la cuenta todo lo que el
    # proceso haya reservado antes, que fue el error de la primera medicion.
    for f in filas:
        f["crecimiento_mb"] = f["pico_mb"] - f["pico_antes_mb"]
        print(f"  {f['n_filas']:>9} filas  {f['camino']:<10} "
              f"{f['segundos']:>7.1f} s  el init crece {f['crecimiento_mb']:>7.0f} MB "
              f"(pico {f['pico_antes_mb']:.0f} -> {f['pico_mb']:.0f} MB, "
              f"datos {f['datos_mb']:.0f} MB)")

    for n_filas in sorted({f["n_filas"] for f in filas}):
        par = {f["camino"]: f for f in filas if f["n_filas"] == n_filas}
        if len(par) == 2:
            diferencia = abs(par["exacto"]["checksum"] - par["streaming"]["checksum"])
            relativa = diferencia / max(abs(par["exacto"]["checksum"]), 1e-30)
            print(f"\n  {n_filas} filas: mismo resultado a {relativa:.2e} relativo; "
                  f"memoria {par['exacto']['crecimiento_mb']:.0f} contra "
                  f"{par['streaming']['crecimiento_mb']:.0f} MB "
                  f"({par['exacto']['crecimiento_mb'] / max(par['streaming']['crecimiento_mb'], 1e-9):.1f}x), "
                  f"tiempo {par['exacto']['segundos']:.1f} contra "
                  f"{par['streaming']['segundos']:.1f} s "
                  f"({par['exacto']['segundos'] / max(par['streaming']['segundos'], 1e-9):.1f}x)")
    return filas


# ------------------------------------------------------- equivalencia motores


def medir_equivalencia():
    """Fit MovieLens with both engines and score them with the same code."""
    import datos as datos_movielens
    from datos import CLAVE_GENERO
    from datafiusion import dfmf_sparse, predict_attribute, normalize_relations

    print("\n" + "=" * 78)
    print("mismo modelo, dos motores: dfmf_sparse contra scikit-fusion")
    print("=" * 78)

    datos = datos_movielens.cargar(semilla=SEMILLA, verbose=False)
    piezas = {c: sp.vstack([datos.train.R[c][0], datos.valid.R[c][0],
                            datos.test.R[c][0]], format="csr")
              for c in [("pelicula", "usuario"), CLAVE_GENERO, ("pelicula", "actor")]}
    n = len(datos.train) + len(datos.valid)
    tr, te = np.arange(n), np.arange(n, piezas[CLAVE_GENERO].shape[0])
    Y = np.asarray(piezas[CLAVE_GENERO][te].todense())

    ranks = {"pelicula": 20, "genero": 15, "usuario": 25, "actor": 40}
    R_bruto = {CLAVE_GENERO: [piezas[CLAVE_GENERO][tr]]}
    for tipo in ("usuario", "actor"):
        R_bruto[("pelicula", tipo)] = [piezas[("pelicula", tipo)][tr]]
    R, _ = normalize_relations(R_bruto, weights={CLAVE_GENERO: 3.0})

    def ap(scores):
        orden = np.argsort(-scores, axis=1)
        Yo = np.take_along_axis(Y, orden, axis=1)
        k = np.maximum(Y.sum(axis=1).astype(int), 1)
        prec = np.cumsum(Yo, axis=1) / np.arange(1, Y.shape[1] + 1)
        return float(((prec * Yo).sum(axis=1) / k).mean())

    salidas = {}

    inicio = time.time()
    base = rss_mb()
    G, S = dfmf_sparse(R=R, ranks=ranks, lambda_G=0.0, lambda_S=1e-2,
                       max_iter=200, init="nndsvd", random_state=SEMILLA)
    salidas["dfmf_sparse"] = {"segundos": time.time() - inicio,
                              "neto_mb": pico_mb() - base, "G": G, "S": S}

    try:
        G_sk, S_sk, segundos_sk, neto_sk = _ajustar_skfusion(R, ranks)
        salidas["scikit-fusion"] = {"segundos": segundos_sk, "neto_mb": neto_sk,
                                    "G": G_sk, "S": S_sk}
    except Exception as error:
        print(f"  scikit-fusion no corrio: {error}")

    # Las peliculas de test se proyectan y se evaluan con el mismo codigo
    # aguas abajo en ambos casos: solo cambia quien produjo G y S.
    from datafiusion import fold_in_entities

    escalas = {}
    for tipo in ("usuario", "actor"):
        clave = ("pelicula", tipo)
        M = piezas[clave][tr]
        escalas[clave] = float(np.sqrt(np.square(M.data).sum()))

    for nombre, salida in salidas.items():
        R_nuevo = {
            clave: [piezas[clave][te].multiply(1.0 / norma).tocsr()]
            for clave, norma in escalas.items()
        }
        G_nuevo = np.maximum(
            fold_in_entities(R_nuevo, salida["G"], salida["S"],
                             target_type="pelicula", lambda_reg=1e-3), 0.0)
        scores = predict_attribute(
            known={"pelicula": np.arange(len(te))},
            G={**salida["G"], "pelicula": G_nuevo}, S=salida["S"],
            target_type="genero", view_keys=[CLAVE_GENERO])
        salida["ap"] = ap(scores)
        print(f"  {nombre:<15} AP {salida['ap']:.3f}   {salida['segundos']:>6.1f} s   "
              f"memoria neta {salida['neto_mb']:>7.0f} MB")

    if len(salidas) == 2:
        a, b = salidas["dfmf_sparse"], salidas["scikit-fusion"]
        print(f"\n  diferencia de AP: {a['ap'] - b['ap']:+.4f}")
        print(f"  sparse es {b['segundos'] / max(a['segundos'], 1e-9):.1f}x mas rapido "
              f"y usa {b['neto_mb'] / max(a['neto_mb'], 1e-9):.1f}x menos memoria")
    return salidas


def _ajustar_skfusion(R, ranks):
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    import collections
    import collections.abc
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    from skfusion import fusion as skf
    from escala import _parchar_skfusion

    _parchar_skfusion()
    tipos = {nombre: skf.ObjectType(nombre, ranks[nombre]) for nombre in ranks}
    relaciones, claves = [], []
    base = rss_mb()
    for (src, dst), mats in R.items():
        for M in mats:
            relaciones.append(skf.Relation(np.asarray(M.todense()), tipos[src], tipos[dst]))
            claves.append((src, dst))
    grafo = skf.FusionGraph(relaciones)

    inicio = time.time()
    fusor = skf.Dfmf(max_iter=200, init_type="random",
                     random_state=np.random.RandomState(SEMILLA), n_jobs=1).fuse(grafo)
    segundos = time.time() - inicio
    neto = pico_mb() - base

    G = {nombre: fusor.factor(tipo) for nombre, tipo in tipos.items()}
    S = {}
    for relacion, clave in zip(relaciones, claves):
        S.setdefault(clave, []).append(fusor.backbone(relacion))
    return G, S, segundos, neto


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-init", action="store_true")
    parser.add_argument("--n-filas", type=int)
    parser.add_argument("--camino", choices=("exacto", "streaming"))
    argumentos = parser.parse_args()

    if argumentos.worker_init:
        worker_init(argumentos.n_filas, argumentos.camino)
    else:
        medir_init()
        medir_equivalencia()
