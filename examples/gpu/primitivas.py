# %%
"""Primitivas de una iteracion de fuse en CPU (scipy) contra GPU (CuPy).

Mide por separado las operaciones que componen una iteracion: SpMM
directo M @ G_dst, SpMM transpuesto (transpuesta materializada como CSR
y variante cusparse con transa), grams G^T G, GEMMs de tamano factor y
el update elementwise. Esta vista por operacion es la que ubica el
cuello; el numero de punta a punta esta en comparacion.py.

Uso: uv run python examples/gpu/primitivas.py
Requiere el extra gpu: uv sync --extra gpu
"""
import time

import numpy as np
import scipy.sparse as sp

from datafiusion import gpu

INSTANCIAS = [
    ("chica", 200_000, 5_000, 5_000_000),
    ("grande", 2_000_000, 10_000, 50_000_000),
]
C_SRC, C_DST = 20, 15
REPS, WARMUP = 5, 2

if not gpu.available():
    raise SystemExit(
        "No hay GPU utilizable. Instala el extra (uv sync --extra gpu) y "
        "verifica el driver con nvidia-smi.")

import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx import cusparse


def instancia(n_src, n_dst, nnz, seed=7):
    rng = np.random.default_rng(seed)
    M = sp.csr_matrix((rng.random(nnz, dtype=np.float32),
                       (rng.integers(0, n_src, size=nnz),
                        rng.integers(0, n_dst, size=nnz))),
                      shape=(n_src, n_dst))
    M.sum_duplicates()
    return M


def cronometrar(fn, sync=None, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        tiempos.append(time.perf_counter() - t0)
    return min(tiempos)


def medir_backend(xp, M, MT, G_src, G_dst, S, sync=None):
    """Tiempos por operacion de una iteracion (una relacion), en segundos."""
    tiempos = {}
    tiempos["spmm_directo"] = cronometrar(lambda: M @ G_dst, sync)
    tiempos["spmm_transpuesto"] = cronometrar(lambda: MT @ G_src, sync)
    P = M @ G_dst
    Q = MT @ G_src
    tiempos["gram_src"] = cronometrar(lambda: G_src.T @ G_src, sync)
    tiempos["gram_dst"] = cronometrar(lambda: G_dst.T @ G_dst, sync)
    tiempos["gemm_P_St"] = cronometrar(lambda: P @ S.T, sync)
    B = S @ (G_dst.T @ G_dst) @ S.T
    tiempos["gemm_G_B"] = cronometrar(lambda: G_src @ B, sync)
    tiempos["gemm_Q_S"] = cronometrar(lambda: Q @ S, sync)
    N = xp.abs(G_src @ B)
    D = N + xp.float32(1.0)
    eps = 2.220446049250313e-16
    tiempos["update_src"] = cronometrar(
        lambda: G_src * xp.sqrt(xp.maximum(N, 0.0) / xp.maximum(D, eps)), sync)
    return tiempos, P


# %% Recorre instancias
for nombre, n_src, n_dst, nnz in INSTANCIAS:
    print(f"\n=== instancia {nombre}: {n_src}x{n_dst}, {nnz/1e6:.0f}M nnz, "
          f"rangos {C_SRC}/{C_DST}, float32 ===")
    M = instancia(n_src, n_dst, nnz)
    MT = M.T.tocsr()
    rng = np.random.default_rng(0)
    G_src = rng.random((n_src, C_SRC), dtype=np.float32)
    G_dst = rng.random((n_dst, C_DST), dtype=np.float32)
    S = rng.standard_normal((C_SRC, C_DST)).astype(np.float32)

    t_cpu, P_cpu = medir_backend(np, M, MT, G_src, G_dst, S)

    dev = cp.cuda.Device()
    sync = dev.synchronize

    t0 = time.perf_counter()
    M_g = csp.csr_matrix(M)
    MT_g = csp.csr_matrix(MT)
    G_src_g = cp.asarray(G_src)
    G_dst_g = cp.asarray(G_dst)
    S_g = cp.asarray(S)
    sync()
    t_transfer = time.perf_counter() - t0

    t_gpu, P_gpu = medir_backend(cp, M_g, MT_g, G_src_g, G_dst_g, S_g,
                                 sync=sync)

    # La pasada transpuesta sin materializar la transpuesta: dispersa
    # sobre la salida chica en vez de recolectar del factor grande. Es la
    # variante que usa fuse(device="gpu").
    G_src_f = cp.asfortranarray(G_src_g)
    t_transa = cronometrar(
        lambda: cusparse.spmm(M_g, G_src_f, transa=True), sync)
    Q_ref = MT_g @ G_src_g
    Q_tr = cusparse.spmm(M_g, G_src_f, transa=True)
    dif_q = float(cp.abs(Q_tr - Q_ref).max() / cp.abs(Q_ref).max())

    dif = float(np.abs(cp.asnumpy(P_gpu) - P_cpu).max() / np.abs(P_cpu).max())
    usado = cp.get_default_memory_pool().used_bytes() / 1e9
    libre, total = dev.mem_info
    print(f"transferencia H2D (CSR x2 + factores): {t_transfer*1e3:.0f} ms")
    print(f"VRAM: pool {usado:.2f} GB, libre {libre/1e9:.2f} de {total/1e9:.2f} GB")
    print(f"diferencia relativa maxima del SpMM: {dif:.2e}")
    print(f"spmm transa: {t_transa*1e3:.2f} ms "
          f"(contra {t_gpu['spmm_transpuesto']*1e3:.2f} ms del CSR "
          f"transpuesto), dif rel {dif_q:.2e}")
    print(f"\n{'operacion':<18} {'CPU (ms)':>10} {'GPU (ms)':>10} {'x':>8}")
    total_cpu = total_gpu = 0.0
    for op in t_cpu:
        total_cpu += t_cpu[op]
        total_gpu += t_gpu[op]
        print(f"{op:<18} {t_cpu[op]*1e3:>10.2f} {t_gpu[op]*1e3:>10.2f} "
              f"{t_cpu[op]/t_gpu[op]:>8.1f}")
    print(f"{'compuesto':<18} {total_cpu*1e3:>10.2f} {total_gpu*1e3:>10.2f} "
          f"{total_cpu/total_gpu:>8.1f}")
