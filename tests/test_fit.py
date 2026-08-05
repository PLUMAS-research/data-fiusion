"""Properties the new fitting path promises, checked against definitions.

These are not opinion metrics. Each test pins a claim the design makes:
the gauge is a change of variables, the calibration is invariant to the
scale of the data, a full mask equals no mask, the fold-in returns non
negative factors close to the NNLS optimum, and the graph term preserves
mass instead of shrinking low-degree nodes.

Run with: uv run pytest tests/test_fit.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import Relation, dfmf_sparse, fuse
from datafiusion.model import _graph_laplacian


RANKS = {"t1": 5, "t2": 4, "t3": 3}


def instancia(semilla=0, n=(80, 60, 30)):
    rng = np.random.default_rng(semilla)
    G = [rng.exponential(1.0, (n[i], RANKS[f"t{i+1}"])) for i in range(3)]
    relaciones = {}
    for (a, b), densidad in [((0, 1), 0.25), ((0, 2), 0.35), ((1, 2), 0.30)]:
        S = rng.standard_normal((G[a].shape[1], G[b].shape[1]))
        denso = G[a] @ S @ G[b].T
        mascara = rng.random(denso.shape) < densidad
        relaciones[f"r{a}{b}"] = Relation(
            src=f"t{a+1}", dst=f"t{b+1}",
            matrix=sp.csr_matrix(np.where(mascara, denso, 0.0)))
    return relaciones


def test_baja_la_perdida():
    modelo = fuse(instancia(), RANKS, max_iter=60, tol=None, random_state=0)
    assert modelo.history[-1] < modelo.history[0]
    assert (modelo.history >= 0).all()


def test_factores_no_negativos():
    modelo = fuse(instancia(), RANKS, max_iter=40, random_state=0)
    for tipo, factor in modelo.G.items():
        assert (factor >= 0).all(), f"G[{tipo}] tiene entradas negativas"


def test_gauge_fija_la_norma():
    """The column gauge must leave ||G_t||_F^2 exactly equal to c_t."""
    modelo = fuse(instancia(), RANKS, max_iter=30, gauge="column", random_state=0)
    for tipo, factor in modelo.G.items():
        assert abs(np.square(factor).sum() - RANKS[tipo]) < 1e-9


def test_calibracion_invariante_a_la_escala_de_los_datos():
    """Fitting on R and on 3.7*R must follow the same trajectory.

    With Frobenius normalization and alphas calibrated against the data
    energy, multiplying every relation by a constant cannot change the
    fit. An absolute penalty would fail this.
    """
    base = instancia()
    escalada = {n: Relation(src=r.src, dst=r.dst, matrix=r.matrix * 3.7)
                for n, r in base.items()}
    grafo = {"t1": _grafo_anillo(80)}
    comun = dict(ranks=RANKS, max_iter=25, tol=None, random_state=0,
                 alpha_graph=0.5, graphs=grafo)
    m1 = fuse(base, **comun)
    m2 = fuse(escalada, **comun)
    for tipo in m1.G:
        desvio = np.abs(m1.G[tipo] - m2.G[tipo]).max() / max(np.abs(m1.G[tipo]).max(), 1e-30)
        assert desvio < 1e-9, f"{tipo}: desvio {desvio:.3e}"
    assert abs(m1.history[-1] - m2.history[-1]) < 1e-12


def test_mascara_completa_equivale_a_no_tener_mascara():
    base = instancia()
    con_mascara = {
        n: Relation(src=r.src, dst=r.dst, matrix=r.matrix,
                    rows=np.arange(r.shape[0]))
        for n, r in base.items()
    }
    m1 = fuse(base, RANKS, max_iter=30, tol=None, random_state=0)
    m2 = fuse(con_mascara, RANKS, max_iter=30, tol=None, random_state=0)
    for tipo in m1.G:
        assert np.abs(m1.G[tipo] - m2.G[tipo]).max() < 1e-12


def test_mascara_parcial_ignora_el_contenido_oculto():
    """What sits in a masked row must not affect the fit at all.

    This is the property that separates "not observed" from "a measured
    zero". Two fits whose hidden rows carry completely different content
    must agree exactly. Without a mask they would not: every zero, and
    every fill value, would be taught as a real observation.
    """
    base = instancia()
    relacion = base["r01"]
    observadas = np.arange(0, relacion.shape[0], 2)
    ocultas = np.setdiff1d(np.arange(relacion.shape[0]), observadas)

    def con_basura(semilla):
        denso = relacion.matrix.toarray().copy()
        rng = np.random.default_rng(semilla)
        denso[ocultas] = rng.exponential(50.0, (len(ocultas), denso.shape[1]))
        enmascarada = dict(base)
        enmascarada["r01"] = Relation(src=relacion.src, dst=relacion.dst,
                                      matrix=sp.csr_matrix(denso), rows=observadas)
        return fuse(enmascarada, RANKS, max_iter=25, tol=None, random_state=0)

    m1, m2 = con_basura(11), con_basura(22)
    for tipo in m1.G:
        assert np.abs(m1.G[tipo] - m2.G[tipo]).max() < 1e-12, tipo
    assert abs(m1.history[-1] - m2.history[-1]) < 1e-12


def test_pesos_cambian_el_ajuste():
    """A relation weighted 30x must end up better reconstructed than at 0.03x."""
    base = instancia()
    alto = fuse(base, RANKS, weights={"r01": 30.0}, max_iter=40, tol=None, random_state=0)
    bajo = fuse(base, RANKS, weights={"r01": 0.03}, max_iter=40, tol=None, random_state=0)
    assert alto.rel_error["r01"] < bajo.rel_error["r01"]


def test_tol_detiene_y_lo_reporta():
    modelo = fuse(instancia(), RANKS, max_iter=500, tol=1e-4, random_state=0)
    assert modelo.converged
    assert modelo.stop_reason == "tol"
    assert modelo.n_iter < 500


def test_callback_puede_detener():
    visto = []

    def parar(iteracion, perdida, G):
        visto.append((iteracion, perdida))
        return iteracion >= 7

    modelo = fuse(instancia(), RANKS, max_iter=100, tol=None, callback=parar, random_state=0)
    assert modelo.stop_reason == "callback"
    assert modelo.n_iter == 7
    assert len(visto) == 7


# ------------------------------------------------------------------ fold-in


def test_transform_devuelve_factores_no_negativos():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=40, random_state=0)
    nuevas = _relaciones_nuevas(base, n_new=25, semilla=3)
    derivado = modelo.transform(nuevas, target="t1")
    assert (derivado.G["t1"] >= 0).all()
    assert derivado.G["t1"].shape == (25, RANKS["t1"])


def test_transform_cerca_del_optimo_nnls():
    """The non negative fold-in must land within 0.5% of the NNLS optimum."""
    from scipy.optimize import nnls

    base = instancia()
    modelo = fuse(base, RANKS, max_iter=40, random_state=0)
    nuevas = _relaciones_nuevas(base, n_new=40, semilla=5)
    G_new = modelo.transform(nuevas, target="t1", max_iter=500, tol=1e-10).G["t1"]

    # Problema equivalente denso, fila por fila, para tener el optimo exacto.
    bloques, objetivos = [], []
    for nombre, relacion in nuevas.items():
        S_r = modelo.S[nombre]
        s, beta = modelo.scale[nombre], modelo.weight[nombre]
        D = S_r @ modelo.G[relacion.dst].T
        bloques.append(np.sqrt(beta) * D)
        objetivos.append(np.sqrt(beta) * s * relacion.matrix.toarray())
    D = np.hstack(bloques)
    Y = np.hstack(objetivos)

    peor = 0.0
    for i in range(Y.shape[0]):
        exacto, _ = nnls(D.T, Y[i])
        obj_exacto = np.square(Y[i] - exacto @ D).sum()
        obj_nuestro = np.square(Y[i] - G_new[i] @ D).sum()
        peor = max(peor, (obj_nuestro - obj_exacto) / max(obj_exacto, 1e-30))
    assert peor < 0.005, f"objetivo {peor:.2%} sobre el optimo NNLS"


def test_transform_rechaza_columnas_permutadas():
    """A permuted destination must raise, not score silently.

    Measured on MovieLens, permuting the columns of the destination type
    ran without any warning and dropped AP from 0.714 to 0.459, below the
    marginal baseline of 0.550.
    """
    base = instancia()
    etiquetas = np.array([f"c{i}" for i in range(60)])
    con_etiquetas = {
        n: Relation(src=r.src, dst=r.dst, matrix=r.matrix,
                    col_labels=etiquetas if r.dst == "t2" else None)
        for n, r in base.items()
    }
    modelo = fuse(con_etiquetas, RANKS, max_iter=20, random_state=0)

    nuevas = _relaciones_nuevas(base, n_new=15, semilla=8)
    permutadas = {}
    for nombre, relacion in nuevas.items():
        if relacion.dst == "t2":
            orden = np.random.default_rng(0).permutation(60)
            permutadas[nombre] = Relation(src=relacion.src, dst=relacion.dst,
                                          matrix=relacion.matrix[:, orden],
                                          col_labels=etiquetas[orden])
        else:
            permutadas[nombre] = relacion
    with pytest.raises(ValueError, match="different order"):
        modelo.transform(permutadas, target="t1")


def test_transform_rechaza_forma_incompatible():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=20, random_state=0)
    nuevas = _relaciones_nuevas(base, n_new=10, semilla=2)
    rota = dict(nuevas)
    nombre = next(iter(nuevas))
    rota[nombre] = Relation(src=nuevas[nombre].src, dst=nuevas[nombre].dst,
                            matrix=nuevas[nombre].matrix[:, :-5])
    with pytest.raises(ValueError, match="was fitted with"):
        modelo.transform(rota, target="t1")


def test_transform_con_mascara_excluye_y_reporta():
    """Una entidad nueva fuera de la mascara no puede salir del fold-in.

    Su factor queda en cero, queda registrada en empty_rows y hay aviso.
    Las observadas tienen que dar lo mismo que transformar la submatriz
    sin mascara: el solve por patrones no puede mover el resultado.
    """
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=30, random_state=0)
    nuevas = _relaciones_nuevas(base, n_new=10, semilla=6)
    observadas = np.setdiff1d(np.arange(10), [3])
    con_mascara = {"r01": Relation(src="t1", dst="t2",
                                   matrix=nuevas["r01"].matrix, rows=observadas)}
    with pytest.warns(UserWarning, match="transform"):
        derivado = modelo.transform(con_mascara, target="t1")
    G = derivado.G["t1"]
    assert np.array_equal(G[3], np.zeros(RANKS["t1"]))
    assert np.array_equal(derivado.empty_rows["t1"], np.array([3]))

    sub = {"r01": Relation(src="t1", dst="t2",
                           matrix=nuevas["r01"].matrix[observadas])}
    esperado = modelo.transform(sub, target="t1")
    assert np.abs(G[observadas] - esperado.G["t1"]).max() < 1e-12


def test_transform_mascara_en_una_relacion_usa_la_otra():
    """Una entidad excluida de una relacion se resuelve solo con la otra.

    Se compara con nonneg=False porque el solve es exacto; el refinado
    multiplicativo acopla filas distintas a traves de su criterio de
    parada global.
    """
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=30, random_state=0)
    nuevas = _relaciones_nuevas(base, n_new=12, semilla=7)
    excluidas = np.array([2, 5])
    observadas = np.setdiff1d(np.arange(12), excluidas)
    con_mascara = {
        "r01": Relation(src="t1", dst="t2", matrix=nuevas["r01"].matrix,
                        rows=observadas),
        "r02": nuevas["r02"],
    }
    derivado = modelo.transform(con_mascara, target="t1", nonneg=False)
    solo_otra = modelo.transform({"r02": nuevas["r02"]}, target="t1", nonneg=False)
    assert np.abs(derivado.G["t1"][excluidas]
                  - solo_otra.G["t1"][excluidas]).max() < 1e-12


def test_transform_propaga_empty_rows_del_padre():
    base = instancia()
    vacias_t2 = [4, 9]
    ajustadas = {}
    for nombre, r in base.items():
        M = r.matrix.toarray()
        if r.src == "t2":
            M[vacias_t2] = 0.0
        if r.dst == "t2":
            M[:, vacias_t2] = 0.0
        ajustadas[nombre] = Relation(src=r.src, dst=r.dst, matrix=sp.csr_matrix(M))
    with pytest.warns(UserWarning, match="no observation"):
        modelo = fuse(ajustadas, RANKS, max_iter=20, tol=None, random_state=0)
    derivado = modelo.transform(_relaciones_nuevas(base, n_new=8, semilla=9),
                                target="t1")
    assert np.array_equal(derivado.empty_rows["t2"], np.array(vacias_t2))
    assert "t1" not in derivado.empty_rows


def test_transform_compone_con_predict_proba():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=30, random_state=0)
    nuevas = _relaciones_nuevas(base, n_new=12, semilla=4)
    proba = modelo.transform(nuevas, target="t1").predict_proba(
        target="t3", views=["r02"], known={"t1": np.arange(12)})
    assert proba.shape == (12, 30)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ----------------------------------------------------------------- prediccion


def test_predict_proba_top_k():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=20, random_state=0)
    completo = modelo.predict_proba("t3", views=["r02"], known={"t1": np.arange(80)})
    indices, puntajes = modelo.predict_proba(
        "t3", views=["r02"], known={"t1": np.arange(80)}, top_k=3)
    assert indices.shape == (80, 3)
    assert np.array_equal(indices[:, 0], completo.argmax(axis=1))
    assert np.allclose(puntajes[:, 0], completo.max(axis=1))


def test_predict_proba_rechaza_salida_gigante():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=10, random_state=0)
    with pytest.raises(ValueError, match="top_k"):
        modelo.predict_proba("t3", views=["r02"], known={"t1": np.arange(80)},
                             max_bytes=100)


def test_lotes_no_cambian_el_resultado():
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=20, random_state=0)
    entero = modelo.predict_proba("t3", views=["r02", "r12"],
                                  known={"t1": np.arange(80), "t2": np.arange(80) % 60})
    por_lotes = modelo.predict_proba("t3", views=["r02", "r12"],
                                     known={"t1": np.arange(80), "t2": np.arange(80) % 60},
                                     batch_rows=7)
    assert np.abs(entero - por_lotes).max() < 1e-12


# --------------------------------------------------------------------- grafo


def _grafo_anillo(n):
    """Ring adjacency: every node has degree 2, except none. Uniform by design."""
    filas = np.concatenate([np.arange(n), np.arange(n)])
    columnas = np.concatenate([(np.arange(n) + 1) % n, (np.arange(n) - 1) % n])
    return sp.csr_matrix((np.ones(len(filas)), (filas, columnas)), shape=(n, n))


def _grafo_cadena(n):
    """Path adjacency: the two endpoints have degree 1, the rest degree 2."""
    filas = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    columnas = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    return sp.csr_matrix((np.ones(len(filas)), (filas, columnas)), shape=(n, n))


def test_laplaciano_conserva_masa():
    """The constant vector must be in the kernel for any degree distribution.

    That is what makes smoothing not shrink low-degree nodes. The usual
    I - W_sym fails this: its kernel is D^1/2 1, not 1.
    """
    W_sym, grado = _graph_laplacian(_grafo_cadena(50))
    constante = np.ones((50, 1))
    numerador = W_sym @ constante
    denominador = grado[:, None] * constante
    assert abs(numerador.sum() / denominador.sum() - 1.0) < 1e-12
    assert np.abs(numerador - denominador).max() < 1e-12


def test_grafo_no_encoge_los_extremos():
    """Smoothing a path graph must not systematically shrink its endpoints."""
    n = 60
    relaciones = {
        "r01": Relation(src="a", dst="b",
                        matrix=sp.csr_matrix(np.abs(np.random.default_rng(0).standard_normal((n, 20)))))
    }
    modelo = fuse(relaciones, {"a": 4, "b": 3}, graphs={"a": _grafo_cadena(n)},
                 alpha_graph=2.0, max_iter=200, tol=None, random_state=0)
    normas = np.linalg.norm(modelo.G["a"], axis=1)
    extremos = normas[[0, -1]].mean()
    interior = normas[1:-1].mean()
    assert 0.9 < extremos / interior < 1.1, f"razon extremos/interior {extremos/interior:.3f}"


def test_grafo_sin_alpha_levanta_error():
    with pytest.raises(ValueError, match="alpha_graph"):
        fuse(instancia(), RANKS, graphs={"t1": _grafo_anillo(80)}, alpha_graph=0.0)


def test_alpha_sin_grafo_levanta_error():
    with pytest.raises(ValueError, match="no graphs"):
        fuse(instancia(), RANKS, alpha_graph=1.0)


# --------------------------------------------------------------- persistencia


def test_save_load_ida_y_vuelta(tmp_path):
    modelo = fuse(instancia(), RANKS, max_iter=25, random_state=0)
    modelo.save(tmp_path / "modelo")
    from datafiusion import FusionModel
    leido = FusionModel.load(tmp_path / "modelo")
    for tipo in modelo.G:
        assert np.array_equal(modelo.G[tipo], leido.G[tipo])
    for nombre in modelo.S:
        assert np.array_equal(modelo.S[nombre], leido.S[nombre])
    assert leido.rel == modelo.rel
    assert leido.scale == modelo.scale
    assert leido.stop_reason == modelo.stop_reason


def test_load_con_mmap(tmp_path):
    """Factors must stay on disk, which .npz cannot do."""
    modelo = fuse(instancia(), RANKS, max_iter=10, random_state=0)
    modelo.save(tmp_path / "modelo")
    from datafiusion import FusionModel
    leido = FusionModel.load(tmp_path / "modelo", mmap=True)
    assert isinstance(leido.G["t1"], np.memmap)


def test_resume_equivale_a_correr_de_corrido():
    base = instancia()
    entero = fuse(base, RANKS, max_iter=40, tol=None, random_state=0)
    parcial = fuse(base, RANKS, max_iter=20, tol=None, random_state=0)
    reanudado = parcial.resume(base, max_iter=20)
    for tipo in entero.G:
        desvio = np.abs(entero.G[tipo] - reanudado.G[tipo]).max()
        assert desvio < 1e-10, f"{tipo}: desvio {desvio:.3e}"
    assert reanudado.n_iter == 40
    assert len(reanudado.history) == 40


# ------------------------------------------------------------------- entrada


def test_acepta_el_dict_legacy():
    R = {("t1", "t2"): [sp.random(30, 20, density=0.3, format="csr", random_state=0)]}
    modelo = fuse(R, {"t1": 4, "t2": 3}, max_iter=10, random_state=0)
    assert "t1~t2" in modelo.S


def test_rechaza_auto_relacion():
    with pytest.raises(ValueError, match="self-relation"):
        Relation(src="t1", dst="t1", matrix=sp.eye(5, format="csr"))


def test_rechaza_rank_mayor_que_entidades():
    R = {("t1", "t2"): [sp.random(10, 20, density=0.3, format="csr", random_state=0)]}
    with pytest.raises(ValueError, match="exceeds"):
        fuse(R, {"t1": 40, "t2": 3}, max_iter=5)


def test_rechaza_nombres_de_relacion_invalidos():
    M = sp.random(30, 20, density=0.3, format="csr", random_state=0)
    for nombre in ("a/b", "", ".."):
        with pytest.raises(ValueError, match="save"):
            fuse({nombre: Relation(src="t1", dst="t2", matrix=M)},
                {"t1": 4, "t2": 3}, max_iter=2)


def test_rechaza_nombre_de_tipo_invalido():
    with pytest.raises(ValueError, match="save"):
        Relation(src="a/b", dst="t2",
                 matrix=sp.random(5, 4, density=0.5, format="csr", random_state=0))


def test_desempaqueta_como_la_ruta_vieja():
    """`G, S = model` must keep working for code written against dfmf_sparse."""
    modelo = fuse(instancia(), RANKS, max_iter=10, random_state=0)
    G, S = modelo
    assert set(G) == {"t1", "t2", "t3"}
    assert set(S) == {"r01", "r02", "r12"}


def _relaciones_nuevas(base, n_new, semilla):
    """New rows of type t1, with the same destination sides as the fit."""
    rng = np.random.default_rng(semilla)
    salida = {}
    for nombre, relacion in base.items():
        if relacion.src != "t1":
            continue
        densa = np.abs(rng.standard_normal((n_new, relacion.shape[1])))
        mascara = rng.random(densa.shape) < 0.3
        salida[nombre] = Relation(src=relacion.src, dst=relacion.dst,
                                  matrix=sp.csr_matrix(np.where(mascara, densa, 0.0)))
    return salida


# ------------------------------------------------- supervision de factores


def test_supervision_fija_las_componentes_prohibidas_en_cero():
    """TS-NMF: una entidad solo puede cargar en las componentes permitidas."""
    base = instancia()
    n, c = 80, RANKS["t1"]
    permitido = np.zeros((n, c), dtype=bool)
    permitido[:40, :2] = True        # la primera mitad solo usa 2 componentes
    permitido[40:, 2:] = True        # la segunda mitad, las otras 3

    modelo = fuse(base, RANKS, supervision={"t1": permitido}, max_iter=50,
                 tol=None, random_state=0)
    G = modelo.G["t1"]
    assert (G[~permitido] == 0).all(), "hay carga en componentes prohibidas"
    assert (G[permitido] > 0).any(), "todo quedo en cero"


def test_supervision_no_impide_que_baje_la_perdida():
    base = instancia()
    permitido = np.ones((80, RANKS["t1"]), dtype=bool)
    permitido[:40, 0] = False
    modelo = fuse(base, RANKS, supervision={"t1": permitido}, max_iter=60,
                 tol=None, random_state=0)
    assert modelo.history[-1] < modelo.history[0]


def test_supervision_todo_permitido_equivale_a_no_pasarla():
    base = instancia()
    todo = np.ones((80, RANKS["t1"]), dtype=bool)
    m1 = fuse(base, RANKS, max_iter=30, tol=None, random_state=0)
    m2 = fuse(base, RANKS, supervision={"t1": todo}, max_iter=30, tol=None,
             random_state=0)
    for tipo in m1.G:
        assert np.abs(m1.G[tipo] - m2.G[tipo]).max() < 1e-12


def test_supervision_rechaza_forma_incorrecta():
    base = instancia()
    with pytest.raises(ValueError, match="expected"):
        fuse(base, RANKS, supervision={"t1": np.ones((80, 99), dtype=bool)}, max_iter=5)


def test_supervision_rechaza_entidad_sin_componentes():
    base = instancia()
    permitido = np.ones((80, RANKS["t1"]), dtype=bool)
    permitido[7] = False
    with pytest.raises(ValueError, match="no allowed component"):
        fuse(base, RANKS, supervision={"t1": permitido}, max_iter=5)


# --------------------------------------------- entidades sin observaciones


def _con_filas_vacias(indices_vacios):
    """Instancia donde ciertas entidades de t1 no tienen ningun dato."""
    base = instancia()
    salida = {}
    for nombre, relacion in base.items():
        M = relacion.matrix.toarray()
        if relacion.src == "t1":
            M[indices_vacios] = 0.0
        if relacion.dst == "t1":
            M[:, indices_vacios] = 0.0
        salida[nombre] = Relation(src=relacion.src, dst=relacion.dst,
                                  matrix=sp.csr_matrix(M))
    return salida


def test_detecta_entidades_sin_datos():
    """Una entidad sin ninguna observacion tiene que quedar registrada."""
    vacias = [10, 11, 37]
    with pytest.warns(UserWarning, match="no observation"):
        modelo = fuse(_con_filas_vacias(vacias), RANKS, max_iter=30, tol=None,
                     random_state=0)
    assert np.array_equal(modelo.empty_rows["t1"], np.array(vacias))


def test_sin_entidades_vacias_no_avisa():
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        modelo = fuse(instancia(), RANKS, max_iter=20, tol=None, random_state=0)
    assert modelo.empty_rows == {}


@pytest.mark.parametrize("init", ["nndsvd", "random"])
def test_el_factor_de_una_entidad_sin_datos_no_dice_nada(init):
    """Documenta por que hay que avisar.

    Sin observaciones, el factor queda donde lo dejo la inicializacion:
    despreciable frente al resto y, con nndsvd, identico para todas, asi que
    argmax las manda a todas al mismo grupo sin que nada lo indique.
    """
    vacias = [10, 11]
    with pytest.warns(UserWarning):
        modelo = fuse(_con_filas_vacias(vacias), RANKS, max_iter=50, tol=None,
                     init=init, random_state=0)
    G = modelo.G["t1"]
    normas = np.linalg.norm(G, axis=1)
    tipica = np.median(np.delete(normas, vacias))
    assert normas[vacias].max() < 0.05 * tipica, "el factor no es despreciable"
    if init == "nndsvd":
        assert np.allclose(G[vacias[0]], G[vacias[1]]), "deberian ser identicos"
        assert len(set(G[vacias].argmax(axis=1))) == 1


def test_una_fila_enmascarada_no_cuenta_como_observacion():
    """`rows` y la deteccion de filas vacias tienen que ser coherentes."""
    base = instancia()
    relacion = base["r01"]
    # t1 solo aparece como fuente en r01 y r02; se enmascara en ambas.
    observadas = np.arange(40)
    enmascarada = {}
    for nombre, r in base.items():
        rows = observadas if r.src == "t1" else None
        enmascarada[nombre] = Relation(src=r.src, dst=r.dst, matrix=r.matrix, rows=rows)
    with pytest.warns(UserWarning, match="no observation"):
        modelo = fuse(enmascarada, RANKS, max_iter=20, tol=None, random_state=0)
    assert set(modelo.empty_rows["t1"]) == set(range(40, 80))


def test_mascara_de_filas_y_supervision_conviven():
    """Los dos mecanismos son independientes y se pueden usar juntos."""
    base = instancia()
    etiquetadas = np.arange(0, 80, 2)
    relaciones = dict(base)
    relaciones["r01"] = Relation(src="t1", dst="t2", matrix=base["r01"].matrix,
                                 rows=etiquetadas)
    permitido = np.ones((80, RANKS["t1"]), dtype=bool)
    permitido[etiquetadas, 2:] = False       # las etiquetadas usan 2 componentes

    modelo = fuse(relaciones, RANKS, supervision={"t1": permitido},
                 max_iter=40, tol=None, random_state=0)
    G = modelo.G["t1"]
    assert (G[etiquetadas, 2:] == 0).all(), "la supervision no se respeto"
    assert (G[etiquetadas, :2] > 0).any(), "las componentes permitidas quedaron vacias"
    assert modelo.history[-1] < modelo.history[0]


# ----------------------------------------------------------------- multistart


def test_n_runs_elige_la_mejor_corrida():
    """El modelo devuelto es el de menor perdida entre las corridas.

    Las semillas salen de np.random.SeedSequence(random_state).spawn, asi
    que las corridas se pueden reproducir una a una.
    """
    base = instancia()
    modelo = fuse(base, RANKS, n_runs=3, init="random", random_state=0,
                 max_iter=15, tol=None)
    semillas = np.random.SeedSequence(0).spawn(3)
    perdidas = [fuse(base, RANKS, init="random", random_state=s,
                    max_iter=15, tol=None).history[-1] for s in semillas]
    assert modelo.history[-1] == min(perdidas)
    assert modelo.params["n_runs"] == 3
    assert modelo.params["best_run"] == int(np.argmin(perdidas))
    assert np.allclose(modelo.params["run_losses"], perdidas)
    assert modelo.params["random_state"] == 0


def test_n_runs_exige_init_random():
    with pytest.raises(ValueError, match="init='random'"):
        fuse(instancia(), RANKS, n_runs=2, init="nndsvd", max_iter=2)


def test_n_runs_rechaza_generator():
    with pytest.raises(ValueError, match="Generator"):
        fuse(instancia(), RANKS, n_runs=2, init="random",
            random_state=np.random.default_rng(0), max_iter=2)


def test_resume_de_un_multistart_corre():
    base = instancia()
    modelo = fuse(base, RANKS, n_runs=2, init="random", random_state=1,
                 max_iter=10, tol=None)
    reanudado = modelo.resume(base, max_iter=5)
    assert reanudado.n_iter == modelo.n_iter + 5
    assert "n_runs" not in reanudado.params


def test_mascara_con_duplicados_se_deduplica():
    """The mask is a set of observed rows; with duplicates kept, the block
    accumulation in transform would depend on block_rows."""
    M = sp.csr_matrix(np.eye(4))
    r = Relation(src="a", dst="b", matrix=M, rows=[2, 0, 2, 1])
    assert np.array_equal(r.rows, [0, 1, 2])


def test_mascara_flotante_no_vacia_se_rechaza():
    M = sp.csr_matrix(np.eye(4))
    with pytest.raises(ValueError, match="boolean or integer"):
        Relation(src="a", dst="b", matrix=M, rows=[0.5, 1.0])


def test_mascara_entera_vacia_equivale_a_booleana_toda_falsa():
    """rows=[] used to crash in transform: np.asarray([]) is float64."""
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=20, tol=None, random_state=0)
    M_nueva = base["r01"].matrix[:10]
    d = {}
    for clave, mascara in [("lista", []), ("bool", np.zeros(10, dtype=bool))]:
        nuevas = {"r01": Relation(src="t1", dst="t2", matrix=M_nueva,
                                  rows=mascara)}
        with pytest.warns(UserWarning, match="no observation"):
            d[clave] = modelo.transform(nuevas, target="t1")
        assert (d[clave].factor("t1") == 0).all()
        assert np.array_equal(d[clave].empty_rows["t1"], np.arange(10))
    assert np.array_equal(d["lista"].factor("t1"), d["bool"].factor("t1"))


def test_resume_de_un_transformado_se_rechaza():
    """The derived factor belongs to folded-in entities; resuming from it
    would warm-start the fit with factors of different entities."""
    base = instancia()
    modelo = fuse(base, RANKS, max_iter=15, tol=None, random_state=0)
    nuevas = {"r01": Relation(src="t1", dst="t2", matrix=base["r01"].matrix[:10])}
    derivado = modelo.transform(nuevas, target="t1")
    with pytest.raises(ValueError, match="transform"):
        derivado.resume(base, max_iter=2)
