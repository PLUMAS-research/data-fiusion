"""Preprocess as part of the model: the contract it promises.

Fitting with preprocess equals fitting on the hand-transformed matrix;
transform and loss reapply the fit-time chain to raw incoming data; the
idf state comes from training, never recomputed on the new batch; the
caller's matrix is never modified; and everything round-trips through
save/load.

Run with: uv run pytest tests/test_preprocesamiento.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import FusionModel, Relation, fit
from test_fit import RANKS, instancia
from test_poisson import _bloques


COMUN = dict(ranks=RANKS, max_iter=20, tol=None, random_state=0)


def _base_no_negativa():
    """instancia() with absolute values: the transforms target count-like
    data and reject negatives."""
    salida = {}
    for nombre, r in instancia().items():
        M = r.matrix.copy()
        M.data = np.abs(M.data)
        salida[nombre] = Relation(src=r.src, dst=r.dst, matrix=M)
    return salida


def _log1p(M):
    salida = M.copy().astype(np.float64)
    salida.data = np.log1p(salida.data)
    return salida


def _idf_de(M):
    df = M.getnnz(axis=0)
    return np.log((1.0 + M.shape[0]) / (1.0 + df)) + 1.0


def _con(base, nombre, **kwargs):
    salida = dict(base)
    r = base[nombre]
    salida[nombre] = Relation(src=r.src, dst=r.dst, matrix=kwargs.pop("matrix", r.matrix),
                              **kwargs)
    return salida


def test_equivale_a_transformar_a_mano():
    base = _base_no_negativa()
    con_param = _con(base, "r01", preprocess="log1p")
    a_mano = _con(base, "r01", matrix=_log1p(base["r01"].matrix))
    m1, m2 = fit(con_param, **COMUN), fit(a_mano, **COMUN)
    for tipo in m1.G:
        assert np.array_equal(m1.G[tipo], m2.G[tipo]), tipo
    # loss recibe las relaciones crudas y aplica la cadena del fit.
    assert m1.loss(con_param) == m2.loss(a_mano)


def test_idf_coincide_con_la_formula():
    base = _base_no_negativa()
    M = base["r01"].matrix
    idf = _idf_de(M)
    con_param = _con(base, "r01", preprocess="idf")
    a_mano = _con(base, "r01", matrix=M.multiply(idf[None, :]).tocsr())
    m1, m2 = fit(con_param, **COMUN), fit(a_mano, **COMUN)
    assert np.allclose(m1.G["t1"], m2.G["t1"], atol=1e-12)
    assert np.allclose(m1.params["idf"]["r01"], idf)


def test_transform_aplica_el_preprocesamiento_del_fit():
    base = _base_no_negativa()
    nueva = base["r01"].matrix[:10]
    m1 = fit(_con(base, "r01", preprocess="log1p"), **COMUN)
    m2 = fit(_con(base, "r01", matrix=_log1p(base["r01"].matrix)), **COMUN)
    d1 = m1.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva,
                                       preprocess="log1p")}, target="t1")
    d2 = m2.transform({"r01": Relation(src="t1", dst="t2",
                                       matrix=_log1p(nueva))}, target="t1")
    assert np.allclose(d1.factor("t1"), d2.factor("t1"), atol=1e-12)


def test_idf_en_transform_usa_el_estado_del_entrenamiento():
    base = _base_no_negativa()
    M = base["r01"].matrix
    nueva = M[:10]
    idf_entrenamiento = _idf_de(M)
    m1 = fit(_con(base, "r01", preprocess="idf"), **COMUN)
    m2 = fit(_con(base, "r01", matrix=M.multiply(idf_entrenamiento[None, :]).tocsr()),
             **COMUN)
    d1 = m1.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva)},
                      target="t1")
    d2 = m2.transform({"r01": Relation(
        src="t1", dst="t2",
        matrix=nueva.multiply(idf_entrenamiento[None, :]).tocsr())}, target="t1")
    assert np.allclose(d1.factor("t1"), d2.factor("t1"), atol=1e-10)
    # El idf recomputado sobre el lote nuevo es distinto: el test distingue.
    assert not np.allclose(_idf_de(nueva), idf_entrenamiento)


def test_declaracion_inconsistente_se_rechaza():
    base = _base_no_negativa()
    nueva = base["r01"].matrix[:10]
    sin_prep = fit(base, **COMUN)
    with pytest.raises(ValueError, match="declares"):
        sin_prep.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva,
                                            preprocess="log1p")}, target="t1")
    con_prep = fit(_con(base, "r01", preprocess="log1p"), **COMUN)
    with pytest.raises(ValueError, match="declares"):
        con_prep.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva,
                                            preprocess="sqrt")}, target="t1")


def test_idf_no_aplica_a_columnas_nuevas():
    base = _base_no_negativa()
    modelo = fit(_con(base, "r01", preprocess="idf"), **COMUN)
    nueva = base["r01"].matrix[:, :10]
    with pytest.raises(ValueError, match="column side"):
        modelo.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva)},
                         target="t2")


def test_save_load_conserva_el_estado(tmp_path):
    base = _base_no_negativa()
    nueva = base["r01"].matrix[:10]
    modelo = fit(_con(base, "r01", preprocess=("log1p", "idf")), **COMUN)
    modelo.save(tmp_path / "modelo")
    leido = FusionModel.load(tmp_path / "modelo")
    assert leido.params["preprocess"] == {"r01": ["log1p", "idf"]}
    d1 = modelo.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva)},
                          target="t1")
    d2 = leido.transform({"r01": Relation(src="t1", dst="t2", matrix=nueva)},
                         target="t1")
    assert np.allclose(d1.factor("t1"), d2.factor("t1"), atol=1e-12)


def test_no_muta_los_datos_del_usuario():
    base = _base_no_negativa()
    matriz = base["r01"].matrix.copy()
    datos_antes = matriz.data.copy()
    fit(_con(base, "r01", matrix=matriz, preprocess="log1p"), **COMUN)
    assert np.array_equal(matriz.data, datos_antes)


def test_reconstruct_entries_invierte_la_cadena():
    base = _base_no_negativa()
    modelo = fit(_con(base, "r01", preprocess="log1p"), **COMUN)
    filas = np.array([0, 3, 7])
    columnas = np.array([1, 2, 5])
    transformado = modelo.reconstruct_entries("r01", filas, columnas)
    original = modelo.reconstruct_entries("r01", filas, columnas, original=True)
    assert np.allclose(original, np.expm1(transformado), atol=1e-12)


def test_resume_reaprende_el_preprocesamiento():
    base = _base_no_negativa()
    con_param = _con(base, "r01", preprocess="log1p")
    modelo = fit(con_param, ranks=RANKS, max_iter=10, tol=None, random_state=0)
    reanudado = modelo.resume(con_param, max_iter=5)
    assert reanudado.n_iter == modelo.n_iter + 5
    assert reanudado.params["preprocess"] == {"r01": ["log1p"]}


def test_poisson_con_preprocesamiento_idf():
    X, grupo_fila = _bloques()
    R = {"conteos": Relation(src="fila", dst="columna", matrix=X,
                             family="poisson", preprocess="idf")}
    modelo = fit(R, {"fila": 3, "columna": 3}, max_iter=30, tol=None,
                 random_state=0)
    assert np.isfinite(modelo.history).all()
    assert modelo.history[-1] < modelo.history[0]
    assert modelo.params["preprocess"] == {"conteos": ["idf"]}


def test_validaciones():
    M = sp.csr_matrix(np.eye(4))
    with pytest.raises(ValueError, match="unknown preprocess"):
        Relation(src="a", dst="b", matrix=M, preprocess="tfidf")
    with pytest.raises(ValueError, match="sparse"):
        Relation(src="a", dst="b", matrix=np.eye(4), preprocess="log1p")
    negativa = sp.csr_matrix(np.array([[1.0, -2.0], [0.0, 3.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        fit({"r": Relation(src="a", dst="b", matrix=negativa, preprocess="sqrt")},
            {"a": 2, "b": 2}, max_iter=2)
