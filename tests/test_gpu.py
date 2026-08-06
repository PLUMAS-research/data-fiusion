"""The GPU backend returns the same model as the CPU, in numpy.

These tests skip themselves on machines without CuPy or without a CUDA
GPU, so the suite stays runnable everywhere. What they pin: the loss
trajectory matches the CPU fit within the float32 noise floor, every
output of the model is host-resident numpy, masks and supervision work,
and the unsupported combinations raise instead of degrading.

Run with: uv run pytest tests/test_gpu.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import Relation, fuse

cupy = pytest.importorskip("cupy")
try:
    _n_gpus = cupy.cuda.runtime.getDeviceCount()
except Exception:
    _n_gpus = 0
pytestmark = pytest.mark.skipif(_n_gpus == 0, reason="no CUDA GPU detected")


RANKS = {"t1": 5, "t2": 4, "t3": 3}


def instancia(dtype=np.float32, semilla=0, n=(80, 60, 30)):
    rng = np.random.default_rng(semilla)
    G = [rng.exponential(1.0, (n[i], RANKS[f"t{i+1}"])) for i in range(3)]
    relaciones = {}
    for (a, b), densidad in [((0, 1), 0.25), ((0, 2), 0.35), ((1, 2), 0.30)]:
        S = rng.standard_normal((G[a].shape[1], G[b].shape[1]))
        denso = G[a] @ S @ G[b].T
        mascara = rng.random(denso.shape) < densidad
        relaciones[f"r{a}{b}"] = Relation(
            src=f"t{a+1}", dst=f"t{b+1}",
            matrix=sp.csr_matrix(np.where(mascara, denso, 0.0).astype(dtype)))
    return relaciones


COMUN = dict(ranks=RANKS, max_iter=15, tol=None, init="random", random_state=0)


def test_trayectoria_igual_a_cpu():
    m_cpu = fuse(instancia(), **COMUN)
    m_gpu = fuse(instancia(), device="gpu", **COMUN)
    np.testing.assert_allclose(m_gpu.history, m_cpu.history, rtol=1e-4)


def test_salidas_en_numpy():
    modelo = fuse(instancia(), device="gpu", **COMUN)
    for tipo in RANKS:
        assert isinstance(modelo.G[tipo], np.ndarray), tipo
        assert modelo.G[tipo].dtype == np.float32
    for nombre, S_r in modelo.S.items():
        assert isinstance(S_r, np.ndarray), nombre
    for idx in modelo.dead_columns.values():
        assert isinstance(idx, np.ndarray)


def test_mascara_por_filas():
    filas = np.arange(0, 80, 2)
    m_cpu = fuse(instancia(), masks={"r01": filas}, **COMUN)
    m_gpu = fuse(instancia(), masks={"r01": filas}, device="gpu", **COMUN)
    np.testing.assert_allclose(m_gpu.history, m_cpu.history, rtol=1e-4)


def test_supervision():
    permitido = np.zeros((80, 5), dtype=bool)
    permitido[:, :2] = True
    modelo = fuse(instancia(), supervision={"t1": permitido},
                  device="gpu", **COMUN)
    assert np.all(modelo.G["t1"][:, 2:] == 0.0)
    assert np.isfinite(modelo.history).all()


def test_grafos():
    rng = np.random.default_rng(3)
    W = sp.random(80, 80, density=0.05, random_state=3, format="csr")
    W = W + W.T
    m_cpu = fuse(instancia(), graphs={"t1": W}, alpha_graph=0.5, **COMUN)
    m_gpu = fuse(instancia(), graphs={"t1": W}, alpha_graph=0.5,
                 device="gpu", **COMUN)
    np.testing.assert_allclose(m_gpu.history, m_cpu.history, rtol=1e-4)


def test_float64_tambien_corre():
    modelo = fuse(instancia(np.float64), device="gpu", **COMUN)
    assert modelo.G["t1"].dtype == np.float64
    assert np.isfinite(modelo.history).all()


def test_pesos_por_entrada_rechazados():
    relaciones = instancia()
    r = relaciones["r01"]
    relaciones["r01"] = Relation(src=r.src, dst=r.dst, matrix=r.matrix,
                                 entry_weights=np.ones(r.matrix.nnz),
                                 background=0.5)
    with pytest.raises(ValueError, match="entry weights"):
        fuse(relaciones, device="gpu", **COMUN)


def test_poisson_rechazado():
    conteos = sp.random(40, 30, density=0.2, random_state=1, format="csr")
    conteos.data = np.ceil(conteos.data * 5)
    relaciones = {"c": Relation("a", "b", conteos, family="poisson")}
    with pytest.raises(ValueError, match="poisson"):
        fuse(relaciones, {"a": 3, "b": 3}, device="gpu", max_iter=3)


def test_device_desconocido():
    with pytest.raises(ValueError, match="device"):
        fuse(instancia(), device="tpu", **COMUN)


def test_resume_conserva_el_device():
    modelo = fuse(instancia(), device="gpu", **COMUN)
    assert modelo.params["device"] == "gpu"
    seguido = modelo.resume(instancia(), max_iter=3)
    assert seguido.params["device"] == "gpu"
    assert isinstance(seguido.G["t1"], np.ndarray)
    assert seguido.n_iter == modelo.n_iter + 3
