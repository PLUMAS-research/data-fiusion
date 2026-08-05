"""Sparse numerical primitives that scipy does not provide.

The one that matters is the SDDMM (sampled dense-dense matrix multiply):
evaluate a low-rank product A @ B^T only at the stored entries of a sparse
pattern, in O(nnz * c) flops and O(block * c) extra memory. It is what
makes entry-level weights and count likelihoods expressible without ever
materializing a matrix of relation size.
"""

import numpy as np
import scipy.sparse as sp


def product_at(A, B, rows, cols, block_nnz=1_000_000):
    """Entries of A @ B^T at the given coordinates.

    Parameters
    ----------
    A : ndarray of shape (m, c)
    B : ndarray of shape (n, c)
    rows, cols : ndarray of int
        Coordinate lists of equal length.
    block_nnz : int
        Coordinates per block; bounds the gather buffer at block_nnz * c.

    Returns
    -------
    ndarray of shape (len(rows),) with (A @ B^T)[rows[k], cols[k]].
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    salida = np.empty(rows.shape[0])
    for inicio in range(0, rows.shape[0], block_nnz):
        bloque = slice(inicio, min(inicio + block_nnz, rows.shape[0]))
        salida[bloque] = np.einsum("ij,ij->i", A[rows[bloque]], B[cols[bloque]])
    return salida


def sddmm(pattern, A, B, block_nnz=1_000_000):
    """A @ B^T sampled at the stored entries of `pattern`.

    Parameters
    ----------
    pattern : scipy.sparse matrix of shape (m, n)
        Only its sparsity structure is used, not its values. Must be in
        canonical form (no duplicate entries).
    A : ndarray of shape (m, c)
    B : ndarray of shape (n, c)

    Returns
    -------
    csr_matrix with the indptr and indices of `pattern` and data
    (A @ B^T)[i, j] at each stored entry.
    """
    pattern = pattern.tocsr()
    filas = np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr))
    datos = product_at(A, B, filas, pattern.indices, block_nnz=block_nnz)
    return sp.csr_matrix((datos, pattern.indices.copy(), pattern.indptr.copy()),
                         shape=pattern.shape)
