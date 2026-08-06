"""The working precision follows the data.

With every relation in float32 the factors, backbones and buffers are
float32; the loss and the rank-size solves accumulate in float64, so the
history stays finite and non negative and lands close to the float64 fit.
A dtype mix falls back to float64, and a tol under the float32 noise
floor warns.

Run with: uv run pytest tests/test_float32.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import Relation, fuse


RANKS = {"t1": 5, "t2": 4, "t3": 3}


def instancia(dtype, semilla=0, n=(80, 60, 30)):
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


def test_todo_float32_produce_factores_float32():
    modelo = fuse(instancia(np.float32), RANKS, max_iter=20, tol=None,
                  init="random", random_state=0)
    for tipo in RANKS:
        assert modelo.G[tipo].dtype == np.float32, tipo
    for nombre, S_r in modelo.S.items():
        assert S_r.dtype == np.float32, nombre
    assert np.isfinite(modelo.history).all()
    assert (modelo.history >= 0.0).all()


def test_perdida_cercana_a_float64():
    comun = dict(ranks=RANKS, max_iter=20, tol=None, init="random",
                 random_state=0)
    m32 = fuse(instancia(np.float32), **comun)
    m64 = fuse(instancia(np.float64), **comun)
    assert m64.G["t1"].dtype == np.float64
    np.testing.assert_allclose(m32.history, m64.history, rtol=1e-3)


def test_mezcla_de_dtypes_cae_a_float64():
    relaciones = instancia(np.float32)
    r = relaciones["r01"]
    relaciones["r01"] = Relation(src=r.src, dst=r.dst,
                                 matrix=r.matrix.astype(np.float64))
    modelo = fuse(relaciones, RANKS, max_iter=5, tol=None, init="random",
                  random_state=0)
    for tipo in RANKS:
        assert modelo.G[tipo].dtype == np.float64, tipo


def test_mascara_por_filas_mantiene_float32():
    relaciones = instancia(np.float32)
    filas = np.arange(0, 80, 2)
    modelo = fuse(relaciones, RANKS, max_iter=10, tol=None, init="random",
                  random_state=0, masks={"r01": filas})
    for tipo in RANKS:
        assert modelo.G[tipo].dtype == np.float32, tipo
    assert np.isfinite(modelo.history).all()


def test_tol_bajo_el_piso_advierte():
    with pytest.warns(UserWarning, match="float32"):
        fuse(instancia(np.float32), RANKS, max_iter=2, tol=1e-5,
             init="random", random_state=0)


def test_tol_sobre_el_piso_no_advierte():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fuse(instancia(np.float32), RANKS, max_iter=2, tol=1e-4,
             init="random", random_state=0)
