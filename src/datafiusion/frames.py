"""Coordinate DataFrames to aligned relations.

Most real data arrives as long tables (entity, entity, value), not as
matrices. The pain this module closes is alignment: when a type appears
in several relations, its entities must occupy the same positions in the
same order everywhere. Doing that with matrices means reindexing
everything against everything; here the vocabulary of each type is
resolved once (the union of what appears, sorted, or a fixed list given
by the caller) and every relation comes out aligned, with row and column
labels attached so the fold-in can keep validating order downstream.

Mismatches fail with the offending categories named instead of silently
misaligning: a category outside a fixed vocabulary raises by default
(`on_unknown` offers "add" and "drop" as remedies), and a fixed category
with no observation stays as an empty row by default (`on_missing`
offers "error" for universes that must be complete).

Duplicate (src, dst) pairs are summed, so with value=None the matrix
counts occurrences. Pre-aggregate the frame for any other statistic.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .model import Relation


def relation_from_frame(frame, src, dst, value=None, src_type=None,
                        dst_type=None, src_labels=None, dst_labels=None,
                        on_unknown="error", **relation_kwargs):
    """One Relation from a coordinate DataFrame.

    Parameters
    ----------
    frame : DataFrame
        One row per observation.
    src, dst : str
        Column names with the source and destination entities.
    value : str, optional
        Column with the observed value; None counts one per row.
        Duplicate pairs are summed.
    src_type, dst_type : str, optional
        Type names; the column names by default.
    src_labels, dst_labels : sequence, optional
        Fixed vocabulary (and order) for each side. Without it, the
        sorted unique values of the column.
    on_unknown : {"error", "add", "drop"}
        What to do with categories outside a fixed vocabulary.
    **relation_kwargs
        Passed to `Relation` (family, preprocess, entry_weights, ...).

    Returns
    -------
    Relation, with row_labels and col_labels set.
    """
    nombre = "relacion"
    spec = {nombre: dict(frame=frame, src=src, dst=dst, value=value,
                         src_type=src_type, dst_type=dst_type,
                         **relation_kwargs)}
    vocabularios = {}
    if src_labels is not None:
        vocabularios[src_type or src] = src_labels
    if dst_labels is not None:
        vocabularios[dst_type or dst] = dst_labels
    return relations_from_frames(spec, vocabularies=vocabularios or None,
                                 on_unknown=on_unknown)[nombre]


def relations_from_frames(specs, vocabularies=None, on_unknown="error",
                          on_missing="zero"):
    """Aligned relations from several coordinate DataFrames.

    Parameters
    ----------
    specs : dict[str, dict]
        One entry per relation. Each spec carries `frame`, `src`, `dst`
        (column names), and optionally `value` (column with the observed
        value; None counts one per row), `src_type` / `dst_type` (type
        names when they differ from the column names, or when two frames
        name the same type differently), `rows` (observed src entities,
        given as LABELS and mapped to the vocabulary), and any other
        `Relation` argument (family, preprocess, ...).
    vocabularies : dict[str, sequence], optional
        Fixed vocabulary (and order) per type. Types not listed get the
        union of the values that appear across frames, sorted.
    on_unknown : {"error", "add", "drop"}
        Categories that appear in a frame but not in that type's FIXED
        vocabulary: raise naming them (default), append them at the end,
        or drop those frame rows (with a warning; never silently).
    on_missing : {"zero", "error"}
        Fixed-vocabulary categories with no observation in any frame:
        leave their rows or columns empty (default) or raise.

    Returns
    -------
    dict[str, Relation], aligned per type and with labels attached.
    """
    if on_unknown not in ("error", "add", "drop"):
        raise ValueError(f"unknown on_unknown {on_unknown!r}; "
                         "supported: 'error', 'add', 'drop'")
    if on_missing not in ("zero", "error"):
        raise ValueError(f"unknown on_missing {on_missing!r}; "
                         "supported: 'zero', 'error'")
    vocabularies = dict(vocabularies or {})

    especificaciones = {}
    for nombre, spec in specs.items():
        spec = dict(spec)
        frame = spec.pop("frame")
        col_src, col_dst = spec.pop("src"), spec.pop("dst")
        especificaciones[nombre] = dict(
            frame=frame, col_src=col_src, col_dst=col_dst,
            value=spec.pop("value", None),
            tipo_src=spec.pop("src_type", None) or col_src,
            tipo_dst=spec.pop("dst_type", None) or col_dst,
            rows=spec.pop("rows", None), extra=spec)
        for col in (col_src, col_dst, especificaciones[nombre]["value"]):
            if col is not None and col not in frame.columns:
                raise ValueError(
                    f"relation {nombre!r}: column {col!r} is not in the frame")

    # Serie de cada tipo en cada relacion, con los NaN rechazados: un NaN
    # en una columna de entidades no es una categoria, es un dato roto.
    series_por_tipo = {}
    for nombre, e in especificaciones.items():
        for tipo, col in ((e["tipo_src"], e["col_src"]),
                          (e["tipo_dst"], e["col_dst"])):
            serie = e["frame"][col]
            nulos = int(serie.isna().sum())
            if nulos:
                raise ValueError(
                    f"relation {nombre!r}: {nulos} rows have a missing value "
                    f"in the {tipo!r} column {col!r}")
            series_por_tipo.setdefault(tipo, []).append(serie)

    # Vocabulario por tipo: fijo si se dio, union ordenada si no. Con
    # on_unknown="add" la extension se resuelve aca, una sola vez, para
    # que todas las relaciones del tipo compartan el mismo orden.
    vocabulario = {}
    for tipo, series in series_por_tipo.items():
        presentes = pd.Index([])
        for serie in series:
            presentes = presentes.union(pd.Index(serie.unique()))
        if tipo not in vocabularies:
            vocabulario[tipo] = presentes
            continue
        fijo = pd.Index(vocabularies[tipo])
        if fijo.has_duplicates:
            raise ValueError(f"the vocabulary of type {tipo!r} has duplicates")
        desconocidas = presentes.difference(fijo)
        if len(desconocidas) and on_unknown == "error":
            muestra = ", ".join(map(str, desconocidas[:5].tolist()))
            raise ValueError(
                f"{len(desconocidas)} categories of type {tipo!r} are not in "
                f"its vocabulary (e.g. {muestra}); pass on_unknown='add' to "
                "extend it or 'drop' to ignore those rows")
        if len(desconocidas) and on_unknown == "add":
            fijo = fijo.append(desconocidas)
        faltantes = fijo.difference(presentes)
        if len(faltantes) and on_missing == "error":
            muestra = ", ".join(map(str, faltantes[:5].tolist()))
            raise ValueError(
                f"{len(faltantes)} categories of the {tipo!r} vocabulary have "
                f"no observation in any frame (e.g. {muestra}); with "
                "on_missing='zero' their rows stay empty")
        vocabulario[tipo] = fijo

    relaciones = {}
    for nombre, e in especificaciones.items():
        vocab_src = vocabulario[e["tipo_src"]]
        vocab_dst = vocabulario[e["tipo_dst"]]
        filas = vocab_src.get_indexer(e["frame"][e["col_src"]])
        columnas = vocab_dst.get_indexer(e["frame"][e["col_dst"]])
        validas = (filas >= 0) & (columnas >= 0)
        if not validas.all():
            descartadas = int((~validas).sum())
            warnings.warn(
                f"relation {nombre!r}: {descartadas} rows dropped because "
                "their category is outside a fixed vocabulary "
                "(on_unknown='drop')", stacklevel=2)
            filas, columnas = filas[validas], columnas[validas]
        if e["value"] is not None:
            datos = e["frame"][e["value"]].to_numpy(dtype=np.float64)
            if np.isnan(datos).any():
                raise ValueError(
                    f"relation {nombre!r}: the value column {e['value']!r} "
                    "has missing values")
            datos = datos[validas] if not validas.all() else datos
        else:
            datos = np.ones(len(filas))
        matriz = sp.csr_matrix((datos, (filas, columnas)),
                               shape=(len(vocab_src), len(vocab_dst)))
        matriz.sum_duplicates()

        extra = dict(e["extra"])
        if e["rows"] is not None:
            indices = vocab_src.get_indexer(pd.Index(e["rows"]))
            if (indices < 0).any():
                perdidas = pd.Index(e["rows"])[indices < 0]
                muestra = ", ".join(map(str, perdidas[:5].tolist()))
                raise ValueError(
                    f"relation {nombre!r}: {len(perdidas)} entities in `rows` "
                    f"are not in the {e['tipo_src']!r} vocabulary "
                    f"(e.g. {muestra})")
            extra["rows"] = indices
        relaciones[nombre] = Relation(
            src=e["tipo_src"], dst=e["tipo_dst"], matrix=matriz,
            row_labels=vocab_src.to_numpy(), col_labels=vocab_dst.to_numpy(),
            **extra)
    return relaciones
