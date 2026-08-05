"""The trace identity must agree with the dense computation it replaces.

`reconstruction_error` used to build G_i S G_j^T as a dense array and
convert it to CSR, which costs O(n_i * n_j) memory and made the error
unaffordable on real relations. The replacement expands the square and
keeps only rank-sized terms. These tests pin it against the definition.

Run with: uv run pytest tests/test_reconstruction_error.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import reconstruction_error
from datafiusion.core import _squared_error


def error_denso(R, G, S):
    """The definition, computed densely. Only usable on small instances."""
    total = 0.0
    for (src, dst), mats in R.items():
        for k, M in enumerate(mats):
            denso = M.toarray() if sp.issparse(M) else np.asarray(M)
            M_hat = G[src] @ S[(src, dst)][k] @ G[dst].T
            total += (np.linalg.norm(denso - M_hat, "fro")
                      / np.linalg.norm(denso, "fro"))
    return total


def instancia(semilla, densidad, n=(60, 40, 25), c=(5, 4, 3)):
    rng = np.random.default_rng(semilla)
    G = {f"t{i+1}": rng.exponential(1.0, (n[i], c[i])) for i in range(3)}
    S, R = {}, {}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        clave = (f"t{a+1}", f"t{b+1}")
        S[clave] = [rng.standard_normal((c[a], c[b]))]
        denso = rng.exponential(1.0, (n[a], n[b]))
        mascara = rng.random(denso.shape) < densidad
        R[clave] = [sp.csr_matrix(np.where(mascara, denso, 0.0))]
    return R, G, S


@pytest.mark.parametrize("semilla,densidad", [(0, 0.10), (1, 0.35), (2, 1.00)])
def test_coincide_con_el_calculo_denso(semilla, densidad):
    R, G, S = instancia(semilla, densidad)
    esperado = error_denso(R, G, S)
    obtenido = reconstruction_error(R, G, S)
    assert abs(obtenido - esperado) / esperado < 1e-12


def test_matrices_densas():
    """Dense relation matrices must keep working."""
    R, G, S = instancia(3, 0.30)
    R_denso = {k: [M.toarray() for M in mats] for k, mats in R.items()}
    assert abs(reconstruction_error(R_denso, G, S) - error_denso(R, G, S)) < 1e-10


def test_duplicados_en_el_csr():
    """A CSR with unmerged duplicates must not overcount the squared norm.

    Building from coordinate triplets merges duplicates already, so the
    case only shows up when indptr and indices are supplied directly,
    which is what hand-built matrices do.
    """
    datos = np.array([2.0, 3.0, 1.5, 0.5, 0.5, 4.0])
    indices = np.array([1, 1, 0, 2, 2, 3])
    indptr = np.array([0, 2, 3, 6])
    con_duplicados = sp.csr_matrix((datos, indices, indptr), shape=(3, 4))
    assert not con_duplicados.has_canonical_format

    rng = np.random.default_rng(7)
    G_i, G_j = rng.exponential(1.0, (3, 2)), rng.exponential(1.0, (4, 2))
    S_ij = rng.standard_normal((2, 2))

    err_sq, norm_sq = _squared_error(con_duplicados, G_i, G_j=G_j, S_ij=S_ij)
    denso = con_duplicados.toarray()
    assert abs(norm_sq - np.square(denso).sum()) < 1e-12
    esperado = np.square(denso - G_i @ S_ij @ G_j.T).sum()
    assert abs(err_sq - esperado) / esperado < 1e-12


def test_ajuste_perfecto_no_da_negativo():
    """Cancellation must not push an exact fit below zero.

    The expanded form subtracts quantities of order ||M||^2, so the
    absolute accuracy floor is about eps * ||M||^2. That caps the
    relative Frobenius error it can resolve at roughly 1e-8, which is
    well below any tolerance worth stopping on.
    """
    rng = np.random.default_rng(11)
    G_i, G_j = rng.exponential(1.0, (30, 4)), rng.exponential(1.0, (20, 3))
    S_ij = rng.standard_normal((4, 3))
    M = sp.csr_matrix(G_i @ S_ij @ G_j.T)
    err_sq, norm_sq = _squared_error(M, G_i, S_ij, G_j)
    assert err_sq >= 0.0
    assert np.sqrt(err_sq / norm_sq) < 1e-6


def test_no_materializa_la_reconstruccion():
    """Memory must not scale with n_i * n_j.

    A 40k x 5k relation would need 1.6 GB densified. The identity should
    add nothing measurable over the sparse inputs themselves.
    """
    import resource

    n_i, n_j, c = 40_000, 5_000, 8
    rng = np.random.default_rng(5)
    M = sp.random(n_i, n_j, density=0.001, format="csr", random_state=1)
    G_i, G_j = rng.exponential(1.0, (n_i, c)), rng.exponential(1.0, (n_j, c))
    S_ij = rng.standard_normal((c, c))

    antes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _squared_error(M, G_i, S_ij, G_j)
    despues = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    crecimiento_mb = (despues - antes) / 1024
    assert crecimiento_mb < 100, f"creció {crecimiento_mb:.0f} MB; densa serían 1600 MB"


def test_fold_in_devuelve_factores_no_negativos():
    """The legacy fold-in must respect the constraint the fit enforces."""
    from datafiusion import fold_in_entities

    rng = np.random.default_rng(3)
    G = {"u": rng.exponential(1.0, (50, 6)), "c": rng.exponential(1.0, (30, 4))}
    S = {("u", "c"): [rng.standard_normal((6, 4))]}
    M = sp.csr_matrix(np.abs(rng.standard_normal((20, 30))) * (rng.random((20, 30)) < 0.3))
    R_new = {("u", "c"): [M]}

    con_signo = fold_in_entities(R_new, G, S, "u", nonneg=False)
    assert (con_signo < 0).any(), "el caso de prueba deberia producir negativos"

    G_new = fold_in_entities(R_new, G, S, "u")
    assert (G_new >= 0).all()

    # Y debe reconstruir al menos tan bien como el clamp duro, que es lo que
    # habia que hacer a mano antes.
    D = S[("u", "c")][0] @ G["c"].T
    objetivo = np.asarray(M.todense())
    error = lambda X: np.square(objetivo - X @ D).sum()
    assert error(G_new) <= error(np.maximum(con_signo, 0.0))
