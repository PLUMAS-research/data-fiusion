"""Smoke of the fusion schema diagram.

Run with: uv run pytest tests/test_diagram.py -v
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import scipy.sparse as sp

from datafiusion import Relation, fuse
from datafiusion.diagram import fusion_diagram


def _relaciones():
    A = sp.random(40, 30, density=0.2, random_state=1, format="csr")
    A.data = np.abs(A.data)
    B = sp.random(40, 5, density=0.4, random_state=2, format="csr")
    B.data = np.abs(B.data)
    C = sp.random(30, 5, density=0.3, random_state=3, format="csr")
    C.data = np.abs(C.data)
    return {
        "una": Relation(src="a", dst="b", matrix=A, preprocess="log1p"),
        "otra": Relation(src="a", dst="c", matrix=B, rows=np.arange(20)),
        "conteos": Relation(src="b", dst="c", matrix=C, family="poisson"),
    }


def test_diagrama_desde_relaciones():
    fig, ax = fusion_diagram(_relaciones(), ranks={"a": 5, "b": 4, "c": 3})
    textos = " ".join(t.get_text() for t in ax.texts)
    for esperado in ("una", "otra", "conteos", "a", "n = 40", "c = 5",
                     "log1p", "observed rows"):
        assert esperado in textos, esperado
    # Con dos familias presentes hay leyenda.
    assert ax.get_legend() is not None
    plt.close(fig)


def test_diagrama_desde_modelo():
    relaciones = {n: r for n, r in _relaciones().items() if r.family == "gaussian"}
    modelo = fuse(relaciones, {"a": 3, "b": 3, "c": 2}, max_iter=5, tol=None,
                 random_state=0)
    fig, ax = fusion_diagram(modelo)
    textos = " ".join(t.get_text() for t in ax.texts)
    assert "una" in textos and "c = 3" in textos and "log1p" in textos
    assert ax.get_legend() is None
    plt.close(fig)


def test_posiciones_personalizadas():
    posiciones = {"a": (0, 0), "b": (2, 0), "c": (1, 1.5)}
    fig, ax = fusion_diagram(_relaciones(), positions=posiciones)
    assert ax.get_xlim()[1] > 2
    plt.close(fig)
