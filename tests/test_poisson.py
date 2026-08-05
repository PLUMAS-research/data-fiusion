"""Poisson family: the properties the KL path promises.

The deviance ratio decreases monotonically, planted count structure is
recovered, factors and backbones stay non-negative, unsupported
combinations fail loudly, and persistence round-trips with resume.

Run with: uv run pytest tests/test_poisson.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from datafiusion import FusionModel, Relation, fuse


def _bloques(semilla=0, n_filas=90, n_cols=60, grupos=3, alto=8.0, bajo=0.2):
    """Counts drawn from a planted block-diagonal rate matrix."""
    rng = np.random.default_rng(semilla)
    grupo_fila = np.repeat(np.arange(grupos), n_filas // grupos)
    grupo_col = np.repeat(np.arange(grupos), n_cols // grupos)
    tasa = np.where(grupo_fila[:, None] == grupo_col[None, :], alto, bajo)
    X = sp.csr_matrix(rng.poisson(tasa).astype(np.float64))
    return X, grupo_fila


def _relacion(X, **kwargs):
    return {"conteos": Relation(src="fila", dst="columna", matrix=X,
                                family="poisson", **kwargs)}


def test_la_desviacion_baja_monotonamente():
    X, _ = _bloques()
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=40,
                 tol=None, init="random", random_state=0)
    h = modelo.history
    subidas = np.diff(h) > h[:-1] * 1e-8 + 1e-12
    assert not subidas.any(), f"{int(subidas.sum())} subidas en la traza"
    assert h[-1] < 1.0, "no le gana al modelo nulo de tasa constante"


def test_recupera_los_bloques_plantados():
    X, grupo_fila = _bloques()
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=100,
                 random_state=0)
    grupos = modelo.factor("fila").argmax(axis=1)
    assert adjusted_rand_score(grupo_fila, grupos) > 0.95
    assert modelo.rel_error["conteos"] < 0.5


def test_factores_y_backbone_no_negativos():
    X, _ = _bloques()
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=30,
                 init="random", random_state=1)
    for factor in modelo.G.values():
        assert (factor >= 0).all()
    assert (modelo.S["conteos"] >= 0).all()
    assert np.isfinite(modelo.S["conteos"]).all()


def test_fusion_de_dos_relaciones_poisson():
    X, grupo_fila = _bloques(semilla=1)
    Y, _ = _bloques(semilla=2, n_cols=30)
    R = {"a": Relation(src="fila", dst="c1", matrix=X, family="poisson"),
         "b": Relation(src="fila", dst="c2", matrix=Y, family="poisson")}
    modelo = fuse(R, {"fila": 3, "c1": 3, "c2": 3}, max_iter=60, random_state=0)
    assert np.isfinite(modelo.history).all()
    assert modelo.history[-1] < modelo.history[0]
    grupos = modelo.factor("fila").argmax(axis=1)
    assert adjusted_rand_score(grupo_fila, grupos) > 0.9


def test_supervision_ancla_componentes():
    X, grupo_fila = _bloques()
    permitido = np.ones((X.shape[0], 3), dtype=bool)
    permitido[:30] = False
    permitido[:30, 0] = True
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3},
                 supervision={"fila": permitido}, max_iter=30, random_state=0)
    assert (modelo.G["fila"][:30, 1:] == 0).all()


def _etiquetas_de(grupos, n_clases=3):
    Y = np.zeros((len(grupos), n_clases))
    Y[np.arange(len(grupos)), grupos] = 1.0
    return sp.csr_matrix(Y)


def test_fusion_mixta_poisson_y_gaussiana():
    """A count relation and a masked gaussian label relation share the
    row factor; the labels of the unlabeled half are recovered from the
    counts through the shared factor."""
    X, grupo_fila = _bloques()
    Y = _etiquetas_de(grupo_fila)
    # Filas intercaladas, para que las tres clases tengan ejemplos.
    R = {"conteos": Relation(src="fila", dst="columna", matrix=X,
                             family="poisson"),
         "etiquetas": Relation(src="fila", dst="clase", matrix=Y,
                               rows=np.arange(0, 90, 2))}
    modelo = fuse(R, {"fila": 3, "columna": 3, "clase": 3}, max_iter=80,
                 tol=None, random_state=0)
    assert modelo.params["family"] == "mixed"
    assert np.isfinite(modelo.history).all()
    assert modelo.history[-1] < modelo.history[0]
    proba = modelo.predict_proba(target="clase", views=["etiquetas"])
    sin_etiqueta = np.arange(1, 90, 2)
    acierto = float((proba[sin_etiqueta].argmax(axis=1)
                     == grupo_fila[sin_etiqueta]).mean())
    assert acierto > 0.8, f"acierto {acierto:.2f}"


def test_mascara_gaussiana_en_fit_mixto_no_filtra():
    """Garbage in the masked label rows must not move the mixed fit."""
    X, grupo_fila = _bloques()
    Y = _etiquetas_de(grupo_fila)
    Y_sucia = Y.copy()
    Y_sucia = Y_sucia.tolil()
    Y_sucia[60:] = 999.0
    Y_sucia = Y_sucia.tocsr()
    comun = dict(ranks={"fila": 3, "columna": 3, "clase": 3}, max_iter=25,
                 tol=None, random_state=0)
    base = {"conteos": Relation(src="fila", dst="columna", matrix=X,
                                family="poisson"),
            "etiquetas": Relation(src="fila", dst="clase", matrix=Y,
                                  rows=np.arange(60))}
    sucia = dict(base)
    sucia["etiquetas"] = Relation(src="fila", dst="clase", matrix=Y_sucia,
                                  rows=np.arange(60))
    m1 = fuse(base, **comun)
    m2 = fuse(sucia, **comun)
    for tipo in m1.G:
        assert np.abs(m1.G[tipo] - m2.G[tipo]).max() < 1e-12, tipo


def test_el_peso_entre_familias_responde():
    """The KL gradient is normalized by its null deviance, so the label
    weight moves a mixed fit instead of being drowned by the count mass."""
    X, grupo_fila = _bloques()
    Y = _etiquetas_de(grupo_fila)

    def con_peso(w):
        R = {"conteos": Relation(src="fila", dst="columna", matrix=X,
                                 family="poisson"),
             "etiquetas": Relation(src="fila", dst="clase", matrix=Y)}
        return fuse(R, {"fila": 3, "columna": 3, "clase": 3},
                   weights={"etiquetas": w}, max_iter=20, tol=None,
                   random_state=0)

    chico, grande = con_peso(0.01), con_peso(100.0)
    assert np.abs(chico.G["fila"] - grande.G["fila"]).max() > 1e-3


def test_resume_de_fit_mixto():
    X, grupo_fila = _bloques()
    Y = _etiquetas_de(grupo_fila)
    R = {"conteos": Relation(src="fila", dst="columna", matrix=X,
                             family="poisson"),
         "etiquetas": Relation(src="fila", dst="clase", matrix=Y)}
    modelo = fuse(R, {"fila": 3, "columna": 3, "clase": 3}, max_iter=15,
                 tol=None, random_state=0)
    reanudado = modelo.resume(R, max_iter=5)
    assert reanudado.n_iter == modelo.n_iter + 5
    with pytest.raises(ValueError, match="poisson"):
        modelo.transform(R, target="fila")


def test_combinaciones_no_soportadas():
    X, _ = _bloques()
    with pytest.raises(ValueError, match="row mask"):
        fuse(_relacion(X, rows=np.arange(10)), {"fila": 3, "columna": 3}, max_iter=2)
    with pytest.raises(ValueError, match="entry weights"):
        fuse(_relacion(X, entry_weights=np.ones(X.nnz)),
            {"fila": 3, "columna": 3}, max_iter=2)
    with pytest.raises(ValueError, match="sparse"):
        fuse({"c": Relation(src="fila", dst="columna", matrix=np.ones((6, 4)),
                           family="poisson")}, {"fila": 2, "columna": 2}, max_iter=2)
    negativa = sp.csr_matrix(-np.eye(5))
    with pytest.raises(ValueError, match="negative"):
        fuse({"c": Relation(src="fila", dst="columna", matrix=negativa,
                           family="poisson")}, {"fila": 2, "columna": 2}, max_iter=2)
    with pytest.raises(ValueError, match="not supported"):
        fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=2,
            graphs={"fila": sp.eye(90)}, alpha_graph=0.5)


def test_save_load_resume_poisson(tmp_path):
    X, _ = _bloques()
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=15,
                 tol=None, random_state=0)
    assert modelo.params["family"] == "poisson"
    modelo.save(tmp_path / "modelo")
    leido = FusionModel.load(tmp_path / "modelo")
    assert leido.params["family"] == "poisson"
    reanudado = leido.resume(_relacion(X), max_iter=5)
    assert reanudado.n_iter == modelo.n_iter + 5
    assert reanudado.history[-1] <= modelo.history[-1] + 1e-12


def test_transform_se_rechaza_en_poisson():
    X, _ = _bloques()
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=10,
                 random_state=0)
    with pytest.raises(ValueError, match="poisson"):
        modelo.transform(_relacion(X), target="fila")


def test_reconstruct_entries_aproxima_los_conteos():
    X, _ = _bloques()
    modelo = fuse(_relacion(X), {"fila": 3, "columna": 3}, max_iter=100,
                 random_state=0)
    filas, columnas = X.nonzero()
    pred = modelo.reconstruct_entries("conteos", filas, columnas)
    assert (pred >= 0).all()
    verdad = np.asarray(X[filas, columnas]).ravel()
    correlacion = np.corrcoef(pred, verdad)[0, 1]
    assert correlacion > 0.7, f"correlacion {correlacion:.2f}"
