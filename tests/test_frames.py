"""Coordinate frames to aligned relations: the contract.

The vocabulary of a shared type is resolved once and every relation
comes out aligned; categories outside a fixed vocabulary fail with
their names (or extend it, or drop rows with a warning); fixed
categories without data stay as empty rows or raise; duplicates sum;
and the labels attached make the fold-in validate order downstream.

Run with: uv run pytest tests/test_frames.py -v
"""

import numpy as np
import pandas as pd
import pytest

from datafiusion import (Relation, fuse, relation_from_frame,
                         relations_from_frames)


def _frames():
    viajes = pd.DataFrame({
        "usuario": ["u1", "u2", "u1", "u3", "u1"],
        "zona": ["b", "a", "b", "c", "a"],
        "n": [2.0, 1.0, 3.0, 4.0, 5.0],
    })
    horarios = pd.DataFrame({
        "usuario": ["u2", "u4", "u2"],
        "hora": ["h1", "h2", "h1"],
    })
    return viajes, horarios


def test_construccion_y_alineacion():
    viajes, horarios = _frames()
    relaciones = relations_from_frames({
        "viajes": dict(frame=viajes, src="usuario", dst="zona", value="n"),
        "horarios": dict(frame=horarios, src="usuario", dst="hora"),
    })
    r_v, r_h = relaciones["viajes"], relaciones["horarios"]
    # El vocabulario de usuario es la union ordenada de ambos frames.
    assert list(r_v.row_labels) == ["u1", "u2", "u3", "u4"]
    assert np.array_equal(r_v.row_labels, r_h.row_labels)
    # Duplicados sumados: u1 tiene dos viajes a b (2 + 3) y sin value
    # se cuentan ocurrencias (u2 en h1 dos veces).
    assert r_v.matrix[0, list(r_v.col_labels).index("b")] == 5.0
    assert r_h.matrix[1, 0] == 2.0
    # u4 no viaja: fila vacia en viajes, no un error.
    assert r_v.matrix[3].nnz == 0
    modelo = fuse(relaciones, {"usuario": 2, "zona": 2, "hora": 2},
                  max_iter=5, tol=None, random_state=0)
    assert list(modelo.index["usuario"]) == ["u1", "u2", "u3", "u4"]


def test_categoria_fuera_del_vocabulario_falla_con_nombre():
    viajes, _ = _frames()
    with pytest.raises(ValueError, match="u3"):
        relations_from_frames(
            {"viajes": dict(frame=viajes, src="usuario", dst="zona", value="n")},
            vocabularies={"usuario": ["u1", "u2"]})


def test_on_unknown_add_extiende_una_sola_vez():
    viajes, horarios = _frames()
    relaciones = relations_from_frames(
        {"viajes": dict(frame=viajes, src="usuario", dst="zona", value="n"),
         "horarios": dict(frame=horarios, src="usuario", dst="hora")},
        vocabularies={"usuario": ["u2", "u1"]}, on_unknown="add")
    # El orden fijado se respeta y las nuevas se agregan al final, igual
    # en todas las relaciones del tipo.
    assert list(relaciones["viajes"].row_labels) == ["u2", "u1", "u3", "u4"]
    assert np.array_equal(relaciones["viajes"].row_labels,
                          relaciones["horarios"].row_labels)


def test_on_unknown_drop_descarta_con_aviso():
    viajes, _ = _frames()
    with pytest.warns(UserWarning, match="dropped"):
        relaciones = relations_from_frames(
            {"viajes": dict(frame=viajes, src="usuario", dst="zona", value="n")},
            vocabularies={"usuario": ["u1", "u2"]}, on_unknown="drop")
    r = relaciones["viajes"]
    assert r.shape[0] == 2
    assert r.matrix.sum() == 2.0 + 3.0 + 5.0 + 1.0   # los viajes de u3 no estan


def test_on_missing():
    viajes, _ = _frames()
    spec = {"viajes": dict(frame=viajes, src="usuario", dst="zona", value="n")}
    relaciones = relations_from_frames(
        spec, vocabularies={"zona": ["a", "b", "c", "d"]})
    assert relaciones["viajes"].matrix[:, 3].nnz == 0
    with pytest.raises(ValueError, match="no observation in any frame"):
        relations_from_frames(spec, vocabularies={"zona": ["a", "b", "c", "d"]},
                              on_missing="error")


def test_rows_como_etiquetas():
    viajes, _ = _frames()
    relaciones = relations_from_frames({
        "viajes": dict(frame=viajes, src="usuario", dst="zona", value="n",
                       rows=["u1", "u3"]),
    })
    assert np.array_equal(relaciones["viajes"].rows, [0, 2])
    with pytest.raises(ValueError, match="u9"):
        relations_from_frames({
            "viajes": dict(frame=viajes, src="usuario", dst="zona", value="n",
                           rows=["u1", "u9"]),
        })


def test_tipos_renombrados_comparten_vocabulario():
    viajes, _ = _frames()
    otro = pd.DataFrame({"id_usuario": ["u1", "u5"], "atributo": ["x", "y"]})
    relaciones = relations_from_frames({
        "viajes": dict(frame=viajes, src="usuario", dst="zona", value="n"),
        "perfil": dict(frame=otro, src="id_usuario", dst="atributo",
                       src_type="usuario"),
    })
    assert list(relaciones["perfil"].row_labels) == ["u1", "u2", "u3", "u5"]
    assert np.array_equal(relaciones["viajes"].row_labels,
                          relaciones["perfil"].row_labels)


def test_kwargs_de_relation_pasan():
    viajes, _ = _frames()
    r = relation_from_frame(viajes, src="usuario", dst="zona", value="n",
                            preprocess="log1p", family="poisson")
    assert r.family == "poisson"
    assert r.preprocess == ("log1p",)


def test_errores_de_datos_rotos():
    con_nan = pd.DataFrame({"usuario": ["u1", None], "zona": ["a", "b"]})
    with pytest.raises(ValueError, match="missing value"):
        relation_from_frame(con_nan, src="usuario", dst="zona")
    valor_nan = pd.DataFrame({"usuario": ["u1"], "zona": ["a"], "n": [np.nan]})
    with pytest.raises(ValueError, match="value column"):
        relation_from_frame(valor_nan, src="usuario", dst="zona", value="n")
    viajes, _ = _frames()
    with pytest.raises(ValueError, match="not in the frame"):
        relation_from_frame(viajes, src="usuario", dst="comuna")


def test_fold_in_con_vocabulario_del_modelo():
    """El patron de proyeccion: las columnas del lote nuevo se construyen
    con el vocabulario del modelo, y una categoria nueva falla claro."""
    viajes, horarios = _frames()
    relaciones = relations_from_frames({
        "viajes": dict(frame=viajes, src="usuario", dst="zona", value="n"),
        "horarios": dict(frame=horarios, src="usuario", dst="hora"),
    })
    modelo = fuse(relaciones, {"usuario": 2, "zona": 2, "hora": 2},
                  max_iter=10, tol=None, random_state=0)

    nuevos = pd.DataFrame({"usuario": ["n1", "n2"], "zona": ["a", "c"],
                           "n": [1.0, 2.0]})
    lote = relations_from_frames(
        {"viajes": dict(frame=nuevos, src="usuario", dst="zona", value="n")},
        vocabularies={"zona": modelo.index["zona"]})
    derivado = modelo.transform(lote, target="usuario")
    assert derivado.factor("usuario").shape == (2, 2)

    con_zona_nueva = pd.DataFrame({"usuario": ["n1"], "zona": ["z9"],
                                   "n": [1.0]})
    with pytest.raises(ValueError, match="z9"):
        relations_from_frames(
            {"viajes": dict(frame=con_zona_nueva, src="usuario", dst="zona",
                            value="n")},
            vocabularies={"zona": modelo.index["zona"]})
