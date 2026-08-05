"""Factor initialization strategies for sparse data fusion.

  - random : uniform [0, 1) per entry. Default. Reproducible, ignores R.
  - nndsvd : non-negative double SVD (Boutsidis and Gallopoulos, 2008)
    adapted to multi-relation factorization. For each type t, we
    concatenate every relation block where t participates (transposing
    blocks where t is the column type), normalize each block by its
    Frobenius norm so scales do not dominate, take a truncated SVD, and
    keep the dominant-sign part of each left singular vector scaled by
    sqrt(sigma_j). Exact zeros are replaced by a small epsilon so that
    multiplicative updates can move them.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def init_random(R, sizes, ranks, random_state):
    rng = np.random.default_rng(random_state)
    return {t: rng.random((sizes[t], ranks[t])) for t in sizes}


def _gather_blocks(R, type_name):
    """Return blocks where `type_name` participates, transposing the ones
    where it is the column type so all blocks share row count n_t."""
    blocks = []
    for (src, dst), mats in R.items():
        if src == type_name:
            for M in mats:
                blocks.append(M)
        if dst == type_name and src != dst:
            for M in mats:
                blocks.append(M.T)
    return blocks


def _normalize_block(M):
    norm = sp.linalg.norm(M) if sp.issparse(M) else np.linalg.norm(M, "fro")
    if norm > 0:
        return M / norm
    return M


def _svd_flip_u(U):
    """Pin the sign of each column of U so the entry with largest absolute
    value is positive. Singular vectors are only defined up to sign, so
    different SVD backends or ARPACK iterations can return the same factor
    with arbitrary sign per column. Canonicalizing here makes NNDSVD
    reproducible across processes (the dominant-sign choice in `_nndsvda`
    depends on which half of u is positive).
    """
    max_abs_rows = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_rows, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs


def _truncated_svd_left(M, k):
    """Return (U, sigma) of shape ((M.shape[0], k), (k,)) sorted by descending sigma.

    `spla.svds` uses ARPACK with a random starting vector by default, which
    leaks global numpy random state into the result. We pin v0 to a uniform
    deterministic vector and then sign-flip U so two runs in different
    processes (or after other libraries seed numpy) produce identical factors.
    """
    m, n = M.shape
    if k >= min(m, n):
        dense = M.toarray() if sp.issparse(M) else np.asarray(M)
        U_full, sigma_full, _ = np.linalg.svd(dense, full_matrices=False)
        return _svd_flip_u(U_full[:, :k]), sigma_full[:k]
    v0 = np.full(min(m, n), 1.0 / np.sqrt(min(m, n)))
    U, sigma = _svds_con_reintento(M, k, v0)
    order = np.argsort(-sigma)
    return _svd_flip_u(U[:, order]), sigma[order]


def _svds_con_reintento(operador, k, v0):
    """ARPACK with a retry on its sporadic 'no shifts' failure.

    With the default Krylov basis ARPACK sometimes fails with error 3
    even on well conditioned inputs; enlarging ncv resolves it. The
    retry keeps the pinned v0, so the result stays deterministic.
    """
    try:
        U, sigma, _ = spla.svds(operador, k=k, v0=v0)
    except spla.ArpackError:
        m, n = operador.shape
        ncv = min(min(m, n) - 1, max(4 * k + 1, 25))
        U, sigma, _ = spla.svds(operador, k=k, v0=v0, ncv=ncv)
    return U, sigma


def _nndsvda(U, sigma, fill, eps):
    """NNDSVDa: pick the dominant-sign part of each left singular vector,
    scale to preserve the contribution sqrt(sigma_j) * norm_chosen, then
    replace below-eps entries with `fill` (usually mean of the source
    matrix) so multiplicative updates can move them away from the floor.

    The first component uses |u_0| because the leading singular vector
    typically captures a DC component with no sign ambiguity.
    """
    G = np.zeros_like(U)
    for j in range(U.shape[1]):
        u = U[:, j]
        if j == 0:
            chosen = np.abs(u)
        else:
            u_pos = np.maximum(u, 0.0)
            u_neg = np.maximum(-u, 0.0)
            chosen = u_pos if np.linalg.norm(u_pos) >= np.linalg.norm(u_neg) else u_neg
        norm_c = np.linalg.norm(chosen)
        if norm_c > 0:
            G[:, j] = np.sqrt(max(sigma[j], 0.0) * norm_c) * chosen / norm_c
    G[G < eps] = fill
    return G


def _abs_mean(M):
    """Mean of |entries|, sparse-safe."""
    if sp.issparse(M):
        if M.nnz == 0:
            return 0.0
        return float(np.abs(M.data).sum()) / (M.shape[0] * M.shape[1])
    return float(np.abs(M).mean())


# Above either of these the horizontal concatenation is not materialized and
# the SVD runs against a LinearOperator instead. Below both, the exact path
# is kept: it is what tests/test_golden.py pins, and at that size copying the
# blocks costs nothing worth avoiding.
MAX_NNZ_DENSO = 5_000_000
MAX_FILAS_DENSO = 100_000

# A Gram matrix of this side is formed explicitly, which costs side^2 * 8
# bytes: 5000 columns come to 200 MB. Beyond both limits ARPACK is the only
# option left, and its Krylov basis is what makes the initialization
# expensive on tall inputs.
MAX_COLUMNAS_GRAM = 5_000
MAX_FILAS_GRAM = 5_000


def _block_scales(blocks):
    """Frobenius scale of each block, computed without copying it."""
    escalas = []
    for b in blocks:
        if sp.issparse(b):
            norma = float(np.sqrt(np.square(b.data.astype(np.float64, copy=False)).sum()))
        else:
            norma = float(np.linalg.norm(b, "fro"))
        escalas.append(1.0 / norma if norma > 0 else 1.0)
    return escalas


def _stacked_operator(blocks, escalas):
    """The horizontal concatenation as a LinearOperator, never materialized.

    `init_nndsvd` only ever needs products against the concatenation, so
    building it costs a full copy of the nnz for nothing. Applying each
    block in turn and scaling on the fly gives the same operator with no
    allocation beyond the result.
    """
    n = blocks[0].shape[0]
    anchos = [b.shape[1] for b in blocks]
    total = int(sum(anchos))
    cortes = np.cumsum([0] + anchos)

    def matvec(x):
        x = np.asarray(x).reshape(total, -1)
        salida = np.zeros((n, x.shape[1]))
        for b, s, inicio, fin in zip(blocks, escalas, cortes[:-1], cortes[1:]):
            salida += s * (b @ x[inicio:fin])
        return salida

    def rmatvec(y):
        y = np.asarray(y).reshape(n, -1)
        return np.vstack([s * (b.T @ y) for b, s in zip(blocks, escalas)])

    return spla.LinearOperator(
        (n, total), matvec=matvec, rmatvec=rmatvec, matmat=matvec, rmatmat=rmatvec,
        dtype=np.float64,
    )


def _left_factors_via_gram(blocks, escalas, k):
    """(U, sigma) from the eigendecomposition of M M^T, of size (n, n).

    Used when the requested rank leaves no room for a truncated SVD, which
    happens for types with few entities (a 19-level attribute asked for
    rank 19). Forming M M^T costs O(nnz * n) and stays small precisely
    because n is small in that case.
    """
    n = blocks[0].shape[0]
    gram = np.zeros((n, n))
    for b, s in zip(blocks, escalas):
        producto = (b @ b.T) * (s * s)
        gram += producto.toarray() if sp.issparse(producto) else np.asarray(producto)
    autovalores, autovectores = np.linalg.eigh(gram)
    orden = np.argsort(-autovalores)[:k]
    sigma = np.sqrt(np.maximum(autovalores[orden], 0.0))
    return _svd_flip_u(autovectores[:, orden]), sigma


def _left_factors_via_column_gram(blocks, escalas, k):
    """(U, sigma) through the eigendecomposition of M^T M, of size (total, total).

    This is the path that matters at scale. A type with millions of
    entities is almost always related to a handful of small ones, so the
    concatenation is very tall and narrow, and every quantity the SVD
    needs can be obtained from the small side:

        M^T M = V diag(sigma^2) V^T,    U = M V diag(1 / sigma)

    Peak memory is O(total^2 + n k) instead of the O(n * ncv) that ARPACK
    needs for its Krylov basis, which is what actually dominated: at 1M
    rows the Krylov basis alone is an order of magnitude larger than the
    data. It is also deterministic, so no starting vector has to be pinned.
    """
    total = int(sum(b.shape[1] for b in blocks))
    gram = np.zeros((total, total))
    cortes = np.cumsum([0] + [b.shape[1] for b in blocks])
    for i, (b_i, s_i) in enumerate(zip(blocks, escalas)):
        for j, (b_j, s_j) in enumerate(zip(blocks, escalas)):
            if j < i:
                continue
            producto = (b_i.T @ b_j) * (s_i * s_j)
            bloque = producto.toarray() if sp.issparse(producto) else np.asarray(producto)
            gram[cortes[i]:cortes[i + 1], cortes[j]:cortes[j + 1]] = bloque
            if j > i:
                gram[cortes[j]:cortes[j + 1], cortes[i]:cortes[i + 1]] = bloque.T

    autovalores, autovectores = np.linalg.eigh(gram)
    orden = np.argsort(-autovalores)[:k]
    sigma = np.sqrt(np.maximum(autovalores[orden], 0.0))
    V = autovectores[:, orden]

    # U = M V / sigma, una sola pasada por cada bloque. Las direcciones con
    # sigma nulo no aportan y se dejan en cero en vez de dividir por cero.
    n = blocks[0].shape[0]
    U = np.zeros((n, k))
    for b, s, inicio, fin in zip(blocks, escalas, cortes[:-1], cortes[1:]):
        U += s * (b @ V[inicio:fin])
    utiles = sigma > sigma.max() * 1e-12 if sigma.size and sigma.max() > 0 else np.zeros(k, bool)
    U[:, utiles] /= sigma[utiles]
    U[:, ~utiles] = 0.0
    return _svd_flip_u(U), sigma


def _left_factors_streaming(blocks, escalas, k):
    """(U, sigma) without ever concatenating the blocks.

    Three routes, chosen by which dimension is small enough to form a
    Gram matrix of. Only when both sides are large is ARPACK unavoidable.
    """
    n = blocks[0].shape[0]
    total = int(sum(b.shape[1] for b in blocks))

    if total <= MAX_COLUMNAS_GRAM:
        return _left_factors_via_column_gram(blocks, escalas, k)
    if k >= min(n, total) or n <= MAX_FILAS_GRAM:
        if n > MAX_FILAS_DENSO:
            raise ValueError(
                f"rank {k} leaves no room for a truncated SVD on a type with "
                f"{n} entities and {total} columns; lower the rank"
            )
        return _left_factors_via_gram(blocks, escalas, k)
    operador = _stacked_operator(blocks, escalas)
    v0 = np.full(min(n, total), 1.0 / np.sqrt(min(n, total)))
    U, sigma = _svds_con_reintento(operador, k, v0)
    orden = np.argsort(-sigma)
    return _svd_flip_u(U[:, orden]), sigma[orden]


def _abs_mean_streaming(blocks, escalas):
    """Mean of |entries| over the concatenation, without building it."""
    n = blocks[0].shape[0]
    total = int(sum(b.shape[1] for b in blocks))
    acumulado = 0.0
    for b, s in zip(blocks, escalas):
        datos = b.data if sp.issparse(b) else np.asarray(b)
        acumulado += s * float(np.abs(datos).sum())
    return acumulado / (n * total)


def init_nndsvd(R, sizes, ranks, eps=1e-6):
    """NNDSVDa initialization adapted to multi-relation factorization.

    For each type t, concatenate every relation block where t participates
    (transposing those where t is the column type), normalize each block by
    its Frobenius norm so scales do not dominate, take a truncated SVD, and
    apply NNDSVDa to the left singular vectors.

    Parameters
    ----------
    R : dict[(str, str), list[scipy.sparse.csr_matrix]]
    sizes : dict[str, int]
        Row counts per type.
    ranks : dict[str, int]
        Latent rank per type.
    eps : float
        Cutoff below which an entry is treated as zero and replaced by the
        mean of the concatenated source matrix.
    """
    G = {}
    for t in sizes:
        blocks = _gather_blocks(R, t)
        if not blocks:
            raise ValueError(f"Type {t!r} appears in no relation")

        nnz = sum(b.nnz if sp.issparse(b) else b.size for b in blocks)
        cabe = nnz <= MAX_NNZ_DENSO and sizes[t] <= MAX_FILAS_DENSO
        if cabe and all(sp.issparse(b) for b in blocks):
            normalizados = [_normalize_block(b) for b in blocks]
            M_t = sp.hstack(normalizados, format="csr")
            U, sigma = _truncated_svd_left(M_t, ranks[t])
            fill = _abs_mean(M_t)
        elif cabe:
            normalizados = [_normalize_block(b) for b in blocks]
            M_t = np.hstack(
                [b.toarray() if sp.issparse(b) else np.asarray(b) for b in normalizados]
            )
            U, sigma = _truncated_svd_left(M_t, ranks[t])
            fill = _abs_mean(M_t)
        else:
            escalas = _block_scales(blocks)
            U, sigma = _left_factors_streaming(blocks, escalas, ranks[t])
            fill = _abs_mean_streaming(blocks, escalas)
        G[t] = _nndsvda(U, sigma, fill=fill, eps=eps)
    return G
