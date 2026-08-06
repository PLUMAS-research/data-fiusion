# %%
"""Ajuste completo en CPU contra GPU, sobre instancias sinteticas.

Corre `fuse` con los mismos datos, semilla e hiperparametros en ambos
dispositivos y reporta tiempo por ajuste, aceleracion y el desvio
relativo de la trayectoria de perdida. Los datos van en float32, la
precision recomendada para GPU.

Uso: uv run python examples/gpu/comparacion.py
Requiere el extra gpu: uv sync --extra gpu
"""
import time

import numpy as np
import scipy.sparse as sp

from datafiusion import Relation, fuse, gpu

INSTANCIAS = [
    ("chica", 200_000, 5_000, 5_000_000),
    ("grande", 2_000_000, 10_000, 50_000_000),
]
RANKS = {"a": 20, "b": 15}
MAX_ITER = 10
SEED = 7

if not gpu.available():
    raise SystemExit(
        "No hay GPU utilizable. Instala el extra (uv sync --extra gpu) y "
        "verifica el driver con nvidia-smi; el detalle de requisitos esta "
        "en el README del proyecto.")

# %% Calentamiento: la primera corrida GPU compila kernels una sola vez
# por proceso, y ese costo no es del ajuste que queremos medir.
_tiny = sp.random(200, 50, density=0.1, random_state=0,
                  format="csr").astype(np.float32)
fuse({"w": Relation("a", "b", _tiny)}, {"a": 3, "b": 3}, max_iter=2,
     tol=None, init="random", random_state=0, device="gpu")
print("kernels compilados")

# %% Instancias
for nombre, n_src, n_dst, nnz in INSTANCIAS:
    print(f"\n=== instancia {nombre}: {n_src}x{n_dst}, {nnz/1e6:.0f}M nnz, "
          f"rangos 20/15, float32, {MAX_ITER} iteraciones ===")
    rng = np.random.default_rng(SEED)
    M = sp.csr_matrix((rng.random(nnz, dtype=np.float32),
                       (rng.integers(0, n_src, size=nnz),
                        rng.integers(0, n_dst, size=nnz))),
                      shape=(n_src, n_dst))
    M.sum_duplicates()
    print(f"matriz lista: nnz efectivo {M.nnz/1e6:.1f}M")

    tiempos = {}
    historias = {}
    for device in ("cpu", "gpu"):
        t0 = time.perf_counter()
        modelo = fuse({"r": Relation("a", "b", M)}, RANKS, max_iter=MAX_ITER,
                      tol=None, init="random", random_state=SEED,
                      device=device)
        tiempos[device] = time.perf_counter() - t0
        historias[device] = modelo.history
        print(f"{device}: {tiempos[device]:6.2f} s, "
              f"perdida final {modelo.history[-1]:.6f}")

    desvio = np.abs(historias["gpu"] - historias["cpu"]) / historias["cpu"]
    print(f"aceleracion: {tiempos['cpu']/tiempos['gpu']:.1f}x")
    print(f"desvio relativo maximo de la perdida: {desvio.max():.2e}")
