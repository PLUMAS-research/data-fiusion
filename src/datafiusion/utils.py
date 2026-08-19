import numpy as np
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
