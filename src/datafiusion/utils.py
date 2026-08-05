import numpy as np
import pandas as pd
import scipy.sparse as sp


def merge_relations(*relations_dicts):
    """Combine relation dicts from multiple sources or builders.

    Each input is ``dict[(src, dst)] -> list[sparse]``. Matrices under the
    same key are concatenated. Common pattern: write one builder function
    per data source returning its own dict, then ``merge_relations(*dicts)``
    before passing to ``dfmf_sparse``.
    """
    out = {}
    for d in relations_dicts:
        for key, mats in d.items():
            out.setdefault(key, []).extend(mats)
    return out


def normalize_relations(R, weights=None):
    """Frobenius-normalize every matrix in every relation, with optional
    per-relation scalar weight applied after normalization.

    With many relations of very different magnitudes (e.g., a 30M-trip OD
    matrix and a 50-entry purpose x gender pivot), the unweighted squared
    Frobenius loss is dominated by the largest matrix. Frobenius
    normalization (each matrix divided by its norm) puts every relation on
    unit loss contribution. The optional ``weights`` lets you down- or
    up-weight specific relations after normalization (e.g., halve the
    influence of ``(usuario, celda)`` relative to ``(usuario, tiempo)``).

    Parameters
    ----------
    R : dict[(src, dst)] -> list[scipy.sparse]
    weights : dict[(src, dst)] -> float, optional
        Per-relation scalar multiplier applied after normalization. Missing
        keys default to 1.0.

    Returns
    -------
    R_norm : dict same shape as ``R``
    scales : dict[(src, dst)] -> list[float]
        Frobenius norms of the original matrices, for later reconstruction
        to absolute scale.
    """
    weights = weights or {}
    out = {}
    scales = {}
    for key, mats in R.items():
        normed = []
        norms = []
        for M in mats:
            norm = float(sp.linalg.norm(M))
            if norm > 0:
                normed.append(M.multiply(1.0 / norm).tocsr())
            else:
                normed.append(M.copy())
            norms.append(norm)
        w = weights.get(key, 1.0)
        if w != 1.0:
            normed = [m.multiply(w).tocsr() for m in normed]
        out[key] = normed
        scales[key] = norms
    return out, scales


def labels_to_relation(labels, classes=None, fill=0.0):
    """One-hot encode a label Series into a (entity, class) relation.

    Rows where `labels` is NaN are written as `fill`. `fill=0.0` reads as
    "no class membership", which biases the factorization away from any
    class for unlabeled rows. `fill=1.0/n_classes` reads as a uniform prior.
    Neither is the right thing semantically. For proper unlabeled handling,
    use `fit` with a row observation mask (`Relation.rows` or the `masks`
    argument) so unlabeled rows drop out of the loss. Still, for sparse
    labeling the bias is often dominated by signal from other relations.

    Parameters
    ----------
    labels : pd.Series
        Index labels the entities; values are class names (NaN if unknown).
    classes : sequence or None
        Class names in the order they should appear as columns. If None,
        uses sorted unique non-null values from `labels`.
    fill : float
        Value for unlabeled rows. Default 0.

    Returns
    -------
    pd.DataFrame of shape (len(labels), n_classes).
    """
    Y = pd.get_dummies(labels, dtype=float)
    if classes is not None:
        Y = Y.reindex(columns=list(classes), fill_value=0.0)
    unlabeled = labels.isna()
    if unlabeled.any() and fill != 0.0:
        Y.loc[unlabeled] = fill
    return Y


def holdout_entries(matrix, fraction=0.1, random_state=None):
    """Pick a random subset of stored entries to hold out of a fit.

    Returns ``(weights, (rows, cols))``: `weights` is aligned with
    ``matrix.data`` and is 1.0 everywhere except 0.0 at the held-out
    entries, ready for ``Relation(entry_weights=weights)``; the
    coordinate pair locates the held-out entries so they can be scored
    with ``FusionModel.reconstruct_entries`` after the fit.

    The matrix is canonicalized in place (duplicates summed) so the
    weights stay aligned with its data.
    """
    M = matrix.tocsr()
    M.sum_duplicates()
    rng = np.random.default_rng(random_state)
    n_out = int(round(fraction * M.nnz))
    elegidos = rng.choice(M.nnz, size=n_out, replace=False)
    pesos = np.ones(M.nnz)
    pesos[elegidos] = 0.0
    filas = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))[elegidos]
    return pesos, (filas, M.indices[elegidos].copy())


def ensure_columns(df, columns, fill_value=np.nan):
    df = df.copy()

    if isinstance(columns, pd.Series):
        values = columns.values
    elif isinstance(columns, pd.DataFrame):
        values = columns.columns.values
    else:
        values = list(columns)

    for col in values:
        if not col in df.columns:
            df[col] = fill_value

    df = df[values].copy()
    return df


def ensure_index(df, index):
    if isinstance(index, pd.Series):
        values = index.values
    elif isinstance(index, pd.DataFrame):
        values = index.index.values
    else:
        values = list(index)

    return pd.DataFrame(index=values).join(df, how="left").copy()
