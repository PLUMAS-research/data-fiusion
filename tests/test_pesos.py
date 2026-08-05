"""Entry weights: the properties the weighted loss promises.

Uniform weights reduce exactly to the classic path; a held-out entry
cannot influence the fit through any route (loss, scaling or
initialization); a zero background does not divide by zero; and the
reconstruction at held-out coordinates recovers low-rank structure.

Run with: uv run pytest tests/test_pesos.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import Relation, fuse, holdout_entries
from test_fit import RANKS, instancia


def _con_pesos(base, nombre, **kwargs):
    salida = dict(base)
    r = base[nombre]
    salida[nombre] = Relation(src=r.src, dst=r.dst, matrix=r.matrix, **kwargs)
    return salida


def test_pesos_uniformes_equivalen_a_la_ruta_clasica():
    """entry_weights=c, background=c is the classic loss with weight c."""
    base = instancia()
    nnz = base["r01"].matrix.nnz
    ponderada = _con_pesos(base, "r01", entry_weights=np.full(nnz, 2.0),
                           background=2.0)
    comun = dict(ranks=RANKS, max_iter=25, tol=None, random_state=0)
    m1 = fuse(ponderada, **comun)
    m2 = fuse(base, weights={"r01": 2.0}, **comun)
    for tipo in m1.G:
        assert np.array_equal(m1.G[tipo], m2.G[tipo]), tipo


def test_entrada_retenida_no_influye_en_el_ajuste():
    """Changing the VALUE of a weight-zero entry must change nothing."""
    base = instancia()
    M = base["r01"].matrix
    pesos, _ = holdout_entries(M, fraction=0.2, random_state=1)

    contaminada = M.copy()
    contaminada.data = contaminada.data.copy()
    contaminada.data[pesos == 0.0] = 999.0

    comun = dict(ranks=RANKS, max_iter=25, tol=None, random_state=0)
    m1 = fuse(_con_pesos(base, "r01", entry_weights=pesos), **comun)
    base2 = dict(base)
    base2["r01"] = Relation(src="t1", dst="t2", matrix=contaminada,
                            entry_weights=pesos)
    m2 = fuse(base2, **comun)
    for tipo in m1.G:
        desvio = np.abs(m1.G[tipo] - m2.G[tipo]).max()
        assert desvio < 1e-12, f"{tipo}: {desvio:.3e}"


def test_fondo_cero_no_da_nan_y_la_perdida_baja():
    """The regime that made the naive sign split diverge to NaN."""
    base = instancia()
    solo_observadas = _con_pesos(base, "r01", background=0.0)
    modelo = fuse(solo_observadas, RANKS, max_iter=40, tol=None, random_state=0)
    assert np.isfinite(modelo.history).all()
    assert modelo.history[-1] < modelo.history[0]
    for factor in modelo.G.values():
        assert np.isfinite(factor).all()


def test_pesos_generales_ajustan_sin_diverger():
    base = instancia()
    rng = np.random.default_rng(3)
    nnz = base["r01"].matrix.nnz
    pesos = rng.uniform(0.5, 2.0, size=nnz)
    ponderada = _con_pesos(base, "r01", entry_weights=pesos, background=0.3)
    modelo = fuse(ponderada, RANKS, max_iter=40, tol=None, random_state=0)
    assert np.isfinite(modelo.history).all()
    assert modelo.history[-1] < modelo.history[0]


def test_entradas_retenidas_se_recuperan():
    """The stored entries of instancia() follow a low-rank model, so with
    background=0 (only stored entries in the loss) the reconstruction at
    held-out coordinates must beat predicting zero."""
    base = instancia()
    M = base["r01"].matrix
    pesos, (filas, columnas) = holdout_entries(M, fraction=0.1, random_state=2)
    modelo = fuse(_con_pesos(base, "r01", entry_weights=pesos, background=0.0),
                 RANKS, max_iter=80, tol=None, random_state=0)
    pred = modelo.reconstruct_entries("r01", filas, columnas)
    verdad = np.asarray(M[filas, columnas]).ravel()
    rmse = float(np.sqrt(np.mean(np.square(pred - verdad))))
    rmse_cero = float(np.sqrt(np.mean(np.square(verdad))))
    assert rmse < 0.8 * rmse_cero, f"rmse {rmse:.3f} contra {rmse_cero:.3f}"


def test_reconstruct_entries_en_unidades_originales():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=20, tol=None, random_state=0)
    filas = np.array([0, 3, 7])
    columnas = np.array([1, 2, 5])
    pred = modelo.reconstruct_entries("r01", filas, columnas)
    denso = (modelo.G["t1"] @ modelo.S["r01"] @ modelo.G["t2"].T) / modelo.scale["r01"]
    assert np.allclose(pred, denso[filas, columnas], atol=1e-10)


def test_validaciones_de_pesos():
    M = sp.csr_matrix(np.eye(4))
    with pytest.raises(ValueError, match="align"):
        Relation(src="a", dst="b", matrix=M, entry_weights=np.ones(3))
    with pytest.raises(ValueError, match="non negative"):
        Relation(src="a", dst="b", matrix=M, entry_weights=-np.ones(4))
    with pytest.raises(ValueError, match="not"):
        Relation(src="a", dst="b", matrix=M, entry_weights=np.ones(4),
                 rows=[0, 1])
    with pytest.raises(ValueError, match="sparse"):
        Relation(src="a", dst="b", matrix=np.eye(4), entry_weights=np.ones(4))
    with pytest.raises(ValueError, match="zero"):
        fuse({"r": Relation(src="a", dst="b", matrix=M,
                           entry_weights=np.zeros(4), background=0.0)},
            {"a": 2, "b": 2}, max_iter=2)


def test_transform_rechaza_relaciones_ponderadas():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=15, tol=None, random_state=0)
    M_nueva = base["r01"].matrix[:10]
    nuevas = {"r01": Relation(src="t1", dst="t2", matrix=M_nueva,
                              entry_weights=np.ones(M_nueva.nnz))}
    with pytest.raises(ValueError, match="entry weights"):
        modelo.transform(nuevas, target="t1")
