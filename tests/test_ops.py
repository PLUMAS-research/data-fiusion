"""SDDMM against the dense reference, on random patterns and blocks.

Run with: uv run pytest tests/test_ops.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion.ops import product_at, sddmm


@pytest.mark.parametrize("block_nnz", [1_000_000, 7])
def test_sddmm_coincide_con_el_producto_denso(block_nnz):
    rng = np.random.default_rng(0)
    patron = sp.random(40, 25, density=0.2, random_state=1, format="csr")
    A = rng.standard_normal((40, 6))
    B = rng.standard_normal((25, 6))
    resultado = sddmm(patron, A, B, block_nnz=block_nnz)
    denso = A @ B.T
    assert resultado.shape == patron.shape
    assert np.array_equal(resultado.indices, patron.indices)
    esperado = denso[patron.nonzero()]
    assert np.allclose(resultado.data, esperado, atol=1e-12)


def test_product_at_en_coordenadas_arbitrarias():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 5))
    B = rng.standard_normal((20, 5))
    filas = rng.integers(0, 30, size=50)
    columnas = rng.integers(0, 20, size=50)
    valores = product_at(A, B, filas, columnas, block_nnz=13)
    denso = A @ B.T
    assert np.allclose(valores, denso[filas, columnas], atol=1e-12)


def test_sddmm_patron_vacio():
    patron = sp.csr_matrix((10, 8))
    resultado = sddmm(patron, np.ones((10, 3)), np.ones((8, 3)))
    assert resultado.nnz == 0
