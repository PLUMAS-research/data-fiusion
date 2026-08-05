"""Persistence of the fit configuration: supervision, masks and graphs.

save() writes them as optional subdirectories next to meta.json, load()
restores them into params, and resume() after load() reapplies them
without the caller passing them again. A directory saved before these
subdirectories existed still loads.

Run with: uv run pytest tests/test_persistencia.py -v
"""

import numpy as np

from datafiusion import FusionModel, fuse
from test_fit import RANKS, instancia, _grafo_anillo


def test_save_load_con_supervision(tmp_path):
    base = instancia()
    permitido = np.ones((80, RANKS["t1"]), dtype=bool)
    permitido[:40, 0] = False
    modelo = fuse(base, RANKS, supervision={"t1": permitido}, max_iter=15,
                 tol=None, random_state=0)
    modelo.save(tmp_path / "modelo")
    leido = FusionModel.load(tmp_path / "modelo")
    assert np.array_equal(leido.params["supervision"]["t1"], permitido)
    reanudado = leido.resume(base, max_iter=5)
    assert reanudado.n_iter == modelo.n_iter + 5
    assert (reanudado.G["t1"][:40, 0] == 0).all()


def test_save_load_con_masks_y_graphs(tmp_path):
    base = instancia()
    mascara = np.arange(0, 80, 2)
    grafo = {"t1": _grafo_anillo(80)}
    modelo = fuse(base, RANKS, masks={"r01": mascara}, graphs=grafo,
                 alpha_graph=0.5, max_iter=15, tol=None, random_state=0)
    modelo.resume(base, max_iter=2)   # antes fallaba: params no registraba graphs
    modelo.save(tmp_path / "modelo")
    leido = FusionModel.load(tmp_path / "modelo")
    assert np.array_equal(leido.params["masks"]["r01"], mascara)
    assert (leido.params["graphs"]["t1"] != grafo["t1"]).nnz == 0
    reanudado = leido.resume(base, max_iter=5)
    assert reanudado.n_iter == modelo.n_iter + 5


def test_save_sin_extras_mantiene_el_layout(tmp_path):
    modelo = fuse(instancia(), RANKS, max_iter=10, random_state=0)
    modelo.save(tmp_path / "modelo")
    contenido = {p.name for p in (tmp_path / "modelo").iterdir()}
    assert contenido == {"factors", "backbones", "history.npy", "meta.json"}
    leido = FusionModel.load(tmp_path / "modelo")
    assert leido.rel == modelo.rel
    assert leido.params["masks"] is None
    assert leido.params["graphs"] is None


def test_save_sobre_directorio_reutilizado_limpia_lo_anterior(tmp_path):
    """A stale masks/ from a previous save must not override meta.json."""
    base = instancia()
    con_extras = fuse(base, RANKS, masks={"r01": np.arange(0, 80, 2)},
                     max_iter=10, tol=None, random_state=0)
    con_extras.save(tmp_path / "modelo")
    sin_extras = fuse(base, RANKS, max_iter=10, tol=None, random_state=0)
    sin_extras.save(tmp_path / "modelo")
    contenido = {p.name for p in (tmp_path / "modelo").iterdir()}
    assert contenido == {"factors", "backbones", "history.npy", "meta.json"}
    assert FusionModel.load(tmp_path / "modelo").params["masks"] is None


def test_save_no_conserva_archivos_huerfanos_del_save_anterior(tmp_path):
    base = instancia()
    dos = fuse(base, RANKS, masks={"r01": np.arange(0, 80, 2),
                                  "r12": np.arange(0, 60, 3)},
              max_iter=10, tol=None, random_state=0)
    dos.save(tmp_path / "modelo")
    una = fuse(base, RANKS, masks={"r01": np.arange(0, 80, 2)},
              max_iter=10, tol=None, random_state=0)
    una.save(tmp_path / "modelo")
    leido = FusionModel.load(tmp_path / "modelo")
    assert set(leido.params["masks"]) == {"r01"}


def test_escalares_numpy_en_params_sobreviven_el_save(tmp_path):
    """np.float32 and np.int64 are not subclasses of float and int, so
    without conversion these params would vanish from meta.json in silence."""
    modelo = fuse(instancia(), RANKS, weights={"r01": np.float32(30.0)},
                 max_iter=10, tol=None, random_state=np.int64(7))
    modelo.save(tmp_path / "modelo")
    leido = FusionModel.load(tmp_path / "modelo")
    assert leido.params["weights"] == {"r01": 30.0}
    assert leido.params["random_state"] == 7
