"""Sparse non-negative matrix tri-factorization for data fusion.

Each relation matrix R_ij of shape (n_i, n_j) is factorized as

    R_ij ~ G_i S_ij G_j^T

with non-negative G_i, G_j and arbitrary-sign S_ij. R matrices are
scipy.sparse; factors are dense ndarrays (low rank).

Supported regularization terms:

  - L2 on each G_i, weight `lambda_G`
  - L2 on each S_ij, weight `lambda_S` (Tikhonov in the closed-form solve)
  - Graph Laplacian per type via `theta`, a dict mapping type name to a
    square sparse matrix (typically D - W for adjacency W). Positive and
    negative parts are handled separately so signed Laplacians work.
"""

import numpy as np
import scipy.sparse as sp

from .init import init_random, init_nndsvd


EPS = np.finfo(np.float64).eps


def _split_signs(M):
    """Return (M_pos, M_neg) with M = M_pos - M_neg, both non-negative."""
    pos = np.where(M > 0.0, M, 0.0)
    neg = np.where(M < 0.0, -M, 0.0)
    return pos, neg


def _solve_S(G_i, G_j, R_ij, lambda_S):
    """Closed-form S_ij = (G_i^T G_i + lambda I)^{-1} G_i^T R_ij G_j (G_j^T G_j + lambda I)^{-1}."""
    c_i = G_i.shape[1]
    c_j = G_j.shape[1]
    A = G_i.T @ G_i + lambda_S * np.eye(c_i)
    B = G_j.T @ G_j + lambda_S * np.eye(c_j)
    middle = G_i.T @ (R_ij @ G_j)
    X = np.linalg.solve(A, middle)
    S_ij = np.linalg.solve(B.T, X.T).T
    return S_ij


def _split_theta(theta):
    """Split each sparse theta into (theta_pos, theta_neg) preserving sparsity."""
    theta_pos, theta_neg = {}, {}
    for t, T in theta.items():
        T = sp.csr_matrix(T)
        data_pos = np.maximum(T.data, 0.0)
        data_neg = np.maximum(-T.data, 0.0)
        theta_pos[t] = sp.csr_matrix((data_pos, T.indices.copy(), T.indptr.copy()),
                                     shape=T.shape)
        theta_neg[t] = sp.csr_matrix((data_neg, T.indices.copy(), T.indptr.copy()),
                                     shape=T.shape)
    return theta_pos, theta_neg


def fold_in_entities(R_new, G, S, target_type, lambda_reg=1e-3,
                     nonneg=True, max_iter=100, tol=1e-4):
    """Project new entities into the latent space of an already-fit model.

    Given relations whose source type is ``target_type`` and whose
    destination types are already present in ``G``, find ``G_new`` of shape
    ``(n_new, c_target)`` that best reconstructs the new rows through the
    trained backbones and target factors. Solved in closed form (ridge
    least squares) so the cost is ``O(n_new * sum_dst(c_target * n_dst))``.

    Use cases include classifying a held-out cohort without refitting and
    projecting millions of entities given a model trained on a sample.

    Parameters
    ----------
    R_new : dict[(target_type, dst), list[scipy.sparse]]
        New relations to fold in. Source must be ``target_type``; each
        matrix shape ``(n_new, n_dst)`` with ``n_dst == G[dst].shape[0]``.
    G : dict[type, ndarray]
    S : dict[(src, dst), list[ndarray]]
    target_type : str
    lambda_reg : float
        Ridge regularization for the solve.
    nonneg : bool
        Keep the projected factors non-negative, like the ones the fit
        produces. The ridge solve alone has no sign constraint and returns
        24% to 38% negative entries in practice, which contradicts the
        model the backbones were learned under and measurably degrades
        anything computed downstream. Pass ``nonneg=False`` to recover the
        old unconstrained behaviour.
    max_iter, tol : int, float
        Refinement budget when ``nonneg`` is set.

    Returns
    -------
    G_new : ndarray of shape ``(n_new, G[target_type].shape[1])``
    """
    c_target = G[target_type].shape[1]
    decoders = []
    blocks = []
    for (src, dst), mats in R_new.items():
        if src != target_type:
            raise ValueError(f"R_new key {(src, dst)!r} must have source == target_type {target_type!r}")
        S_list = S[(src, dst)]
        if len(S_list) != len(mats):
            raise ValueError(
                f"R_new[{(src, dst)!r}] has {len(mats)} matrices but S has {len(S_list)}; "
                "the new relations must use the same per-slot S as the trained model"
            )
        for M, S_k in zip(mats, S_list):
            decoders.append(S_k @ G[dst].T)
            blocks.append(M)
    if not decoders:
        raise ValueError("R_new is empty; nothing to fold in")
    D = np.hstack(decoders)
    R_stacked = sp.hstack(blocks, format="csr") if any(sp.issparse(b) for b in blocks) else np.hstack(blocks)
    gram = D @ D.T + lambda_reg * np.eye(c_target)
    rhs = R_stacked @ D.T
    X = np.linalg.solve(gram, rhs.T).T
    if not nonneg:
        return X
    return nonneg_refine(X, gram, rhs, lam=0.0, max_iter=max_iter, tol=tol)


def nonneg_refine(X, Q, rhs, lam=0.0, max_iter=100, tol=1e-4):
    """Turn a ridge solution into a non-negative one of the same problem.

    Minimizes ||Y - X D||^2 + lam ||X||^2 subject to X >= 0, given
    Q = D D^T and rhs = Y D^T, with multiplicative updates.

    The starting point is floored at a small positive value rather than
    hard-clamped. A multiplicative update cannot change a sign, so from
    the raw ridge solution the negative entries produce NaN, and from a
    hard clamp the zeroed entries can never move again, which freezes the
    support and leaves the objective measurably above the optimum.
    """
    Q_pos, Q_neg = _split_signs(Q)
    rhs_pos, rhs_neg = _split_signs(rhs)

    floor = 1e-6 * np.maximum(np.abs(X).max(axis=1, keepdims=True), EPS)
    out = np.maximum(X, floor)
    for _ in range(max_iter):
        numerator = rhs_pos + out @ Q_neg
        denominator = rhs_neg + out @ Q_pos + lam * out
        previous = out
        out = out * (numerator / np.maximum(denominator, EPS))
        change = np.abs(out - previous).max() / max(np.abs(previous).max(), EPS)
        if change < tol:
            break
    return out


def predict_attribute(known, G, S, target_type, view_keys,
                      combine="geometric_mean", eps=1e-3):
    """Predict a soft distribution over ``target_type`` from several views.

    Given indices for one or more "known" types, each pair ``(src, target)``
    in ``view_keys`` produces a per-row predicted distribution over
    ``target_type``. The views are combined into a single distribution.

    Used for classifying entities by an attribute they do not directly
    observe (e.g., predicting trip purpose from origin, destination and
    time of day after training on a labeled subset).

    Parameters
    ----------
    known : dict[type, array of int]
        Indices into ``G[type]`` for each known type. All arrays must have
        the same length n.
    view_keys : list[(src, target_type)]
        Each must point to a relation in ``S``. ``src`` must appear in
        ``known``. ``target_type`` must be the second element.
    combine : {"geometric_mean", "product", "sum"}
        How to merge the per-view distributions. Geometric mean is the
        smoothed equivalent of Naive Bayes (multiplicative without one
        view dominating); product is hard Naive Bayes; sum is additive
        evidence.
    eps : float
        Smoothing added to each view before normalization to avoid zero
        probabilities collapsing the product.

    Returns
    -------
    proba : ndarray of shape ``(n, G[target_type].shape[0])``
        Each row is a probability distribution over the target levels.
    """
    n = len(next(iter(known.values())))
    views = []
    for src, dst in view_keys:
        if dst != target_type:
            raise ValueError(f"view key {(src, dst)!r} must end in target_type {target_type!r}")
        if src not in known:
            raise ValueError(f"view key {(src, dst)!r} requires indices for {src!r} in `known`")
        v = G[src][known[src]] @ S[(src, dst)][0] @ G[target_type].T
        v = np.maximum(v, 0.0) + eps
        v = v / v.sum(axis=1, keepdims=True)
        views.append(v)
    if combine == "geometric_mean":
        stacked = np.stack(views, axis=0)
        combined = np.exp(np.mean(np.log(stacked), axis=0))
    elif combine == "product":
        combined = np.prod(np.stack(views, axis=0), axis=0)
    elif combine == "sum":
        combined = np.sum(np.stack(views, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown combine {combine!r}")
    return combined / combined.sum(axis=1, keepdims=True)


def _canonical(M):
    """Return M with duplicate entries summed, so sum(data**2) is its norm.

    A CSR built from coordinate triplets keeps duplicates until asked to
    merge them. Reading `data` directly before that overcounts the squared
    norm, which lands right on the scale where a convergence tolerance
    lives.
    """
    if sp.issparse(M) and hasattr(M, "sum_duplicates"):
        if not getattr(M, "has_canonical_format", False):
            M.sum_duplicates()
    return M


def _squared_error(M, G_i, S_ij, G_j):
    """Return (||M - G_i S G_j^T||_F^2, ||M||_F^2) without densifying.

    Expanding the square leaves only quantities of rank size:

        ||M - G_i S G_j^T||^2 = ||M||^2
                                - 2 <G_i^T M G_j, S>
                                + <(G_i^T G_i) S (G_j^T G_j), S>

    with <A, B> = sum(A * B). The largest intermediate is (c_i, c_j), so
    the cost is O(nnz * c_j) flops and no allocation of size n_i x n_j.

    The subtraction cancels against terms of order ||M||^2, so precision
    degrades once the relative error drops below about 1e-8. Accumulation
    is forced to float64 for that reason, and the result is floored at
    zero because rounding can push a near-perfect fit slightly negative.
    """
    G_i = np.asarray(G_i, dtype=np.float64)
    G_j = np.asarray(G_j, dtype=np.float64)
    S_ij = np.asarray(S_ij, dtype=np.float64)

    if sp.issparse(M):
        M = _canonical(M)
        norm_sq = float(np.square(M.data.astype(np.float64, copy=False)).sum())
    else:
        M = np.asarray(M, dtype=np.float64)
        norm_sq = float(np.square(M).sum())

    middle = G_i.T @ (M @ G_j)
    gram_i = G_i.T @ G_i
    gram_j = G_j.T @ G_j
    cross = float(np.sum(middle * S_ij))
    quad = float(np.sum((gram_i @ S_ij @ gram_j) * S_ij))
    return max(norm_sq - 2.0 * cross + quad, 0.0), norm_sq


def reconstruction_error(R, G, S):
    """Sum of relative Frobenius errors across all relations."""
    total = 0.0
    for (src, dst), mats in R.items():
        for k, M in enumerate(mats):
            err_sq, norm_sq = _squared_error(M, G[src], S[(src, dst)][k], G[dst])
            total += np.sqrt(err_sq) / (np.sqrt(norm_sq) + EPS)
    return total


def dfmf_sparse(
    R,
    ranks,
    theta=None,
    lambda_G=0.0,
    lambda_S=0.0,
    max_iter=100,
    init="random",
    random_state=None,
    verbose=0,
):
    """Sparse data fusion by non-negative matrix tri-factorization.

    Parameters
    ----------
    R : dict[(str, str), list[scipy.sparse.csr_matrix]]
        Relation matrices keyed by (src_type, dst_type). Multiple matrices
        between the same pair share factors but get their own S_ij.
        Self-relations (src == dst) are not supported here; use `theta`.
    ranks : dict[str, int]
        Rank c_i for each object type.
    theta : dict[str, scipy.sparse.csr_matrix] or None
        Optional graph-Laplacian constraints per type. Each matrix must be
        square with side equal to n_i.
    lambda_G, lambda_S : float
        L2 regularization weights on factors and backbones.
    max_iter : int
        Number of multiplicative-update iterations.
    init : {"random", "nndsvd"}
        Initialization strategy. "random" samples uniform [0, 1). "nndsvd"
        concatenates the relations where each type participates, takes a
        truncated SVD per type, and keeps the dominant-sign part of each
        singular vector (NNDSVDa). NNDSVD is deterministic, so the
        `random_state` argument is ignored when `init='nndsvd'`.
    random_state : int, np.random.Generator, or None
    verbose : int
        Print reconstruction error every `verbose` iterations (0 disables).

    Returns
    -------
    G : dict[str, np.ndarray]
        Factor of shape (n_i, c_i) per type.
    S : dict[(str, str), list[np.ndarray]]
        Backbones, parallel to R.
    """
    if init not in ("random", "nndsvd"):
        raise ValueError(f"Unknown init {init!r}; supported: 'random', 'nndsvd'")

    sizes = {}
    for (src, dst), mats in R.items():
        if src == dst:
            raise ValueError(
                f"Self-relation ({src}, {dst}) not supported in R. "
                "Use the `theta` argument for type-level constraints."
            )
        for M in mats:
            if not sp.issparse(M):
                raise TypeError(
                    f"R[({src!r}, {dst!r})] must contain scipy.sparse matrices"
                )
            if src in sizes and sizes[src] != M.shape[0]:
                raise ValueError(
                    f"Inconsistent row count for type {src!r}: "
                    f"got {M.shape[0]}, expected {sizes[src]}"
                )
            if dst in sizes and sizes[dst] != M.shape[1]:
                raise ValueError(
                    f"Inconsistent column count for type {dst!r}: "
                    f"got {M.shape[1]}, expected {sizes[dst]}"
                )
            sizes[src] = M.shape[0]
            sizes[dst] = M.shape[1]

    missing = set(ranks) - set(sizes)
    if missing:
        raise ValueError(f"Types declared in ranks but absent from R: {sorted(missing)}")

    if init == "random":
        G = init_random(R, sizes, ranks, random_state=random_state)
    else:
        G = init_nndsvd(R, sizes, ranks)

    theta = theta or {}
    for t, T in theta.items():
        if t not in sizes:
            raise ValueError(f"theta[{t!r}] given but type not present in R")
        if T.shape != (sizes[t], sizes[t]):
            raise ValueError(
                f"theta[{t!r}] has shape {T.shape}, expected "
                f"({sizes[t]}, {sizes[t]})"
            )
    theta_pos, theta_neg = _split_theta(theta)

    S = {}
    for it in range(max_iter):
        for (src, dst), mats in R.items():
            S[(src, dst)] = [_solve_S(G[src], G[dst], M, lambda_S) for M in mats]

        G_enum = {t: np.zeros_like(G[t]) for t in G}
        G_denom = {t: np.zeros_like(G[t]) for t in G}

        for (src, dst), mats in R.items():
            for k, M in enumerate(mats):
                S_ij = S[(src, dst)][k]

                tmp1 = M @ (G[dst] @ S_ij.T)
                tmp2 = S_ij @ (G[dst].T @ G[dst]) @ S_ij.T
                t1p, t1n = _split_signs(tmp1)
                t2p, t2n = _split_signs(tmp2)
                G_enum[src] += t1p + G[src] @ t2n
                G_denom[src] += t1n + G[src] @ t2p

                tmp4 = M.T @ (G[src] @ S_ij)
                tmp5 = S_ij.T @ (G[src].T @ G[src]) @ S_ij
                t4p, t4n = _split_signs(tmp4)
                t5p, t5n = _split_signs(tmp5)
                G_enum[dst] += t4p + G[dst] @ t5n
                G_denom[dst] += t4n + G[dst] @ t5p

        for t, Tp in theta_pos.items():
            G_denom[t] += Tp @ G[t]
        for t, Tn in theta_neg.items():
            G_enum[t] += Tn @ G[t]

        if lambda_G > 0.0:
            for t in G:
                G_denom[t] += lambda_G * G[t]

        for t in G:
            num = np.maximum(G_enum[t], 0.0)
            den = np.maximum(G_denom[t], EPS)
            G[t] = G[t] * np.sqrt(num / den)

        if verbose and (it + 1) % verbose == 0:
            err = reconstruction_error(R, G, S)
            print(f"iter {it + 1:4d} / {max_iter}: error = {err:.6f}")

    return G, S
