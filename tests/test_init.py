"""The streaming initialization must agree with the exact one, and stay small.

`init_nndsvd` used to build the horizontal concatenation of every block a
type takes part in, which copies the whole nnz twice (once to normalize
each block, once to stack them) before the SVD even starts. Measured at
5M rows that came to 16.5 GB, more than the fit it was preparing.

The streaming path applies each block in turn against a LinearOperator.
These tests pin that it computes the same thing.

Run with: uv run pytest tests/test_init.py -v
"""

import resource

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion.init import (
    _abs_mean_streaming, _block_scales, _left_factors_streaming, init_nndsvd,
)
import datafiusion.init as modulo_init


def relaciones(n1=400, n2=250, n3=60, densidad=0.05, semilla=0):
    rng = np.random.default_rng(semilla)
    return {
        ("t1", "t2"): [sp.random(n1, n2, density=densidad, format="csr", random_state=1)],
        ("t1", "t3"): [sp.random(n1, n3, density=densidad * 2, format="csr", random_state=2)],
        ("t2", "t3"): [sp.random(n2, n3, density=densidad * 2, format="csr", random_state=3)],
    }


@pytest.mark.parametrize("ranks", [
    {"t1": 6, "t2": 5, "t3": 4},
    {"t1": 12, "t2": 10, "t3": 8},
])
def test_streaming_coincide_con_el_camino_exacto(ranks, monkeypatch):
    """Both paths must produce the same factors, up to SVD precision."""
    R = relaciones()
    sizes = {"t1": 400, "t2": 250, "t3": 60}

    exacto = init_nndsvd(R, sizes, ranks)
    # Forzar el camino streaming bajando los umbrales.
    monkeypatch.setattr(modulo_init, "MAX_NNZ_DENSO", 0)
    streaming = init_nndsvd(R, sizes, ranks)

    for tipo in sizes:
        escala = max(float(np.abs(exacto[tipo]).max()), 1e-30)
        desvio = float(np.abs(exacto[tipo] - streaming[tipo]).max()) / escala
        assert desvio < 1e-8, f"{tipo}: desvio relativo {desvio:.3e}"


def test_streaming_es_determinista(monkeypatch):
    """Two runs must agree bit for bit, including the sign of each column."""
    R = relaciones()
    sizes = {"t1": 400, "t2": 250, "t3": 60}
    ranks = {"t1": 6, "t2": 5, "t3": 4}
    monkeypatch.setattr(modulo_init, "MAX_NNZ_DENSO", 0)

    primero = init_nndsvd(R, sizes, ranks)
    np.random.seed(12345)  # ensuciar el estado global, que ARPACK solia leer
    segundo = init_nndsvd(R, sizes, ranks)
    for tipo in sizes:
        assert np.array_equal(primero[tipo], segundo[tipo]), tipo


def test_rank_igual_al_numero_de_niveles(monkeypatch):
    """A type with as many latent dims as entities must still initialize.

    That is the case of an attribute used as its own basis, like 19 genres
    at rank 19: there is no room for a truncated SVD, so the eigen path of
    M M^T takes over.
    """
    R = relaciones(n3=19)
    sizes = {"t1": 400, "t2": 250, "t3": 19}
    ranks = {"t1": 6, "t2": 5, "t3": 19}
    monkeypatch.setattr(modulo_init, "MAX_NNZ_DENSO", 0)

    G = init_nndsvd(R, sizes, ranks)
    assert G["t3"].shape == (19, 19)
    assert (G["t3"] >= 0).all()
    assert np.isfinite(G["t3"]).all()


def test_factores_no_negativos_y_finitos(monkeypatch):
    R = relaciones()
    sizes = {"t1": 400, "t2": 250, "t3": 60}
    ranks = {"t1": 6, "t2": 5, "t3": 4}
    monkeypatch.setattr(modulo_init, "MAX_NNZ_DENSO", 0)
    for tipo, factor in init_nndsvd(R, sizes, ranks).items():
        assert (factor >= 0).all(), tipo
        assert np.isfinite(factor).all(), tipo


def test_abs_mean_coincide():
    bloques = [sp.random(200, 90, density=0.1, format="csr", random_state=4),
               sp.random(200, 40, density=0.2, format="csr", random_state=5)]
    escalas = _block_scales(bloques)
    normalizados = sp.hstack([b * s for b, s in zip(bloques, escalas)], format="csr")
    esperado = float(np.abs(normalizados.data).sum()) / (200 * 130)
    assert abs(_abs_mean_streaming(bloques, escalas) - esperado) < 1e-12


def test_no_copia_el_nnz():
    """Memory must not grow with the size of the concatenation.

    The blocks here hold 24M nonzeros, so materializing the concatenation
    normalized and stacked would cost around 570 MB before the SVD starts.
    """
    n, m = 300_000, 400
    bloques = [sp.random(n, m, density=0.1, format="csr", random_state=6),
               sp.random(n, m, density=0.1, format="csr", random_state=7)]
    escalas = _block_scales(bloques)

    antes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    U, sigma = _left_factors_streaming(bloques, escalas, k=8)
    despues = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    crecimiento_mb = (despues - antes) / 1024

    assert U.shape == (n, 8)
    assert (sigma >= 0).all()
    assert crecimiento_mb < 150, (
        f"crecio {crecimiento_mb:.0f} MB; concatenar costaria unos 570 MB")
