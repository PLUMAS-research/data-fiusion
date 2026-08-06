"""Compare datafiusion.dfmf_sparse against stored scikit-fusion traces.

The parity with scikit-fusion was validated live while both libraries
were installed side by side. What survives here is the reference:
`tests/data/referencia_skfusion.json` stores the error traces that
scikit-fusion's Dfmf produced on the synthetic instances below, so the
comparison runs without the dependency. To regenerate the reference
(only needed if the instances change), install a scikit-fusion checkout
and run `tests/data/generar_referencia_skfusion.py`.

The two implementations are not numerically identical: the random
streams differ (np.random.default_rng vs RandomState) and S is solved
with np.linalg.solve instead of pinv. The comparison therefore checks
structural properties and qualitative parity:

  - Factor and backbone dimensions are as declared.
  - The datafiusion error trace decreases monotonically (allowing small
    violations); the stored scikit-fusion trace did too.
  - Final reconstruction errors are within a factor of 2 of each other.

Run as a script for the full report:

    uv run python tests/test_compare_skfusion.py

Run under pytest for the reduced parity check:

    uv run pytest tests/test_compare_skfusion.py
"""

import json
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

from datafiusion.core import dfmf_sparse, reconstruction_error


REFERENCIA = pathlib.Path(__file__).with_name("data") / "referencia_skfusion.json"

INSTANCIAS = {
    "reducida": dict(n=(24, 18, 12), c=(4, 3, 2), noise=0.05, seed=0,
                     max_iter=50, paso=10),
    "completa": dict(n=(40, 30, 20), c=(5, 4, 3), noise=0.05, seed=0,
                     max_iter=100, paso=10),
}


def referencia_skfusion(nombre):
    """Stored scikit-fusion trace for one instance, validated against INSTANCIAS."""
    todo = json.loads(REFERENCIA.read_text())
    guardado = dict(todo["instancias"][nombre], generado_con=todo["generado_con"])
    esperado = INSTANCIAS[nombre]
    for clave in ("n", "c", "noise", "seed", "max_iter", "paso"):
        if list(np.atleast_1d(esperado[clave])) != list(np.atleast_1d(guardado[clave])):
            raise RuntimeError(
                f"la referencia guardada no corresponde a la instancia "
                f"{nombre!r} ({clave}: {guardado[clave]!r} contra "
                f"{esperado[clave]!r}); regenerar con "
                "tests/data/generar_referencia_skfusion.py")
    return guardado


def make_low_rank_data(n1, n2, n3, c1, c2, c3, noise, seed):
    """Generate a heavy-tailed low-rank ground truth plus Gaussian noise.

    Uses exponential(1) for G_true rather than uniform so that init='random'
    does not gain an artificial head start: real-world relation matrices
    almost never have uniformly distributed factors, and a uniform ground
    truth makes the comparison meaningless because uniform init is
    coincidentally near the optimum.
    """
    rng = np.random.default_rng(seed)
    G1 = rng.exponential(1.0, (n1, c1))
    G2 = rng.exponential(1.0, (n2, c2))
    G3 = rng.exponential(1.0, (n3, c3))
    S12 = rng.standard_normal((c1, c2))
    S13 = rng.standard_normal((c1, c3))
    S23 = rng.standard_normal((c2, c3))
    R12 = G1 @ S12 @ G2.T + noise * rng.standard_normal((n1, n2))
    R13 = G1 @ S13 @ G3.T + noise * rng.standard_normal((n1, n3))
    R23 = G2 @ S23 @ G3.T + noise * rng.standard_normal((n2, n3))
    noise_norm = (
        np.linalg.norm(R12 - G1 @ S12 @ G2.T, "fro") / np.linalg.norm(R12, "fro")
        + np.linalg.norm(R13 - G1 @ S13 @ G3.T, "fro") / np.linalg.norm(R13, "fro")
        + np.linalg.norm(R23 - G2 @ S23 @ G3.T, "fro") / np.linalg.norm(R23, "fro")
    )
    return R12, R13, R23, noise_norm


def relative_frobenius(A, B):
    return np.linalg.norm(A - B, "fro") / (np.linalg.norm(A, "fro") + np.finfo(float).eps)


def run_datafiusion(R12, R13, R23, ranks, max_iter, seed, init="random"):
    R = {
        ("t1", "t2"): [sp.csr_matrix(R12)],
        ("t1", "t3"): [sp.csr_matrix(R13)],
        ("t2", "t3"): [sp.csr_matrix(R23)],
    }
    types = {"t1": ranks[0], "t2": ranks[1], "t3": ranks[2]}

    errors = []
    G, S = None, None
    step = 10
    for chunk_start in range(0, max_iter, step):
        chunk = min(step, max_iter - chunk_start)
        if chunk_start == 0:
            G, S = dfmf_sparse(
                R=R, ranks=types, max_iter=chunk,
                lambda_G=0.0, lambda_S=0.0,
                init=init, random_state=seed,
            )
        else:
            G, S = _resume_dfmf_sparse(R, types, G, chunk)
        errors.append(reconstruction_error(R, G, S))
    return G, S, errors


def _resume_dfmf_sparse(R, ranks, G_init, max_iter):
    """Continue dfmf_sparse from given G factors. Not exposed publicly because
    it would require restructuring the API. Reimplemented here just for the
    convergence-tracking part of this test."""
    from datafiusion.core import _solve_S, _split_signs, EPS

    G = {t: G_init[t].copy() for t in G_init}
    S = {}
    for _ in range(max_iter):
        for (src, dst), mats in R.items():
            S[(src, dst)] = [_solve_S(G[src], G[dst], M, 0.0) for M in mats]
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
        for t in G:
            G[t] = G[t] * np.sqrt(np.maximum(G_enum[t], 0.0) / np.maximum(G_denom[t], EPS))
    return G, S


def check_monotone(errors, tol_violations=2):
    diffs = np.diff(errors)
    violations = int((diffs > 1e-6).sum())
    return violations <= tol_violations, violations


def _first_under(trace, threshold, step=10):
    """Return the iteration index at which the trace first drops below threshold."""
    for i, v in enumerate(trace):
        if v < threshold:
            return f"{(i + 1) * step}"
    return f">{len(trace) * step}"


def test_paridad_reducida():
    """Reduced parity check against the stored scikit-fusion trace.

    Asserts declared factor and backbone dimensions, monotone descent of
    the datafiusion trace, and a final error within a factor of 2 of the
    stored scikit-fusion result on the same instance.
    """
    params = INSTANCIAS["reducida"]
    ref = referencia_skfusion("reducida")

    n1, n2, n3 = params["n"]
    c1, c2, c3 = params["c"]
    R12, R13, R23, _ = make_low_rank_data(
        n1, n2, n3, c1, c2, c3, noise=params["noise"], seed=params["seed"]
    )
    G_new, S_new, err_new = run_datafiusion(
        R12, R13, R23, params["c"], params["max_iter"], params["seed"]
    )

    for t, n, c in (("t1", n1, c1), ("t2", n2, c2), ("t3", n3, c3)):
        assert G_new[t].shape == (n, c)
    for par, dims in (
        (("t1", "t2"), (c1, c2)),
        (("t1", "t3"), (c1, c3)),
        (("t2", "t3"), (c2, c3)),
    ):
        assert S_new[par][0].shape == dims

    monotona_new, viol_new = check_monotone(err_new)
    assert monotona_new, f"traza datafiusion no desciende ({viol_new} violaciones)"
    monotona_ref, viol_ref = check_monotone(ref["traza"])
    assert monotona_ref, f"traza skfusion guardada no desciende ({viol_ref} violaciones)"

    razon = err_new[-1] / ref["traza"][-1]
    assert 0.5 <= razon <= 2.0, (
        f"errores finales fuera de factor 2: {err_new[-1]:.4f} contra "
        f"{ref['traza'][-1]:.4f} de la referencia"
    )


def main():
    params = INSTANCIAS["completa"]
    try:
        ref = referencia_skfusion("completa")
    except FileNotFoundError:
        print(f"No existe {REFERENCIA}.")
        print("Regenerar con un checkout de scikit-fusion instalado:")
        print("  uv run python tests/data/generar_referencia_skfusion.py")
        sys.exit(2)

    n1, n2, n3 = params["n"]
    c1, c2, c3 = params["c"]
    max_iter, seed = params["max_iter"], params["seed"]

    print("Generating low-rank synthetic data:")
    R12, R13, R23, noise_floor = make_low_rank_data(
        n1, n2, n3, c1, c2, c3, noise=params["noise"], seed=seed
    )
    print(f"  shapes: R12={R12.shape}, R13={R13.shape}, R23={R23.shape}")
    print(f"  ranks: t1={c1}, t2={c2}, t3={c3}")
    print(f"  noise_floor (sum of relative noise norms): {noise_floor:.4f}\n")

    print("=" * 60)
    print("datafiusion.dfmf_sparse  (init='random')")
    print("=" * 60)
    G_new, S_new, err_new_trace = run_datafiusion(
        R12, R13, R23, params["c"], max_iter, seed, init="random"
    )
    print(f"  factor dims: t1={G_new['t1'].shape}, t2={G_new['t2'].shape}, t3={G_new['t3'].shape}")
    print(f"  backbone dims: t1-t2={S_new[('t1','t2')][0].shape}, "
          f"t1-t3={S_new[('t1','t3')][0].shape}, t2-t3={S_new[('t2','t3')][0].shape}")
    print(f"  error trace (every 10 iter): {[f'{e:.3f}' for e in err_new_trace]}")
    print(f"  final error: {err_new_trace[-1]:.4f}")
    new_monotone, new_violations = check_monotone(err_new_trace)
    print(f"  monotone descent: {'PASS' if new_monotone else 'FAIL'} "
          f"({new_violations} small increases tolerated)\n")

    print("=" * 60)
    print("datafiusion.dfmf_sparse  (init='nndsvd')")
    print("=" * 60)
    G_svd, S_svd, err_svd_trace = run_datafiusion(
        R12, R13, R23, params["c"], max_iter, seed, init="nndsvd"
    )
    print(f"  factor dims: t1={G_svd['t1'].shape}, t2={G_svd['t2'].shape}, t3={G_svd['t3'].shape}")
    print(f"  error trace (every 10 iter): {[f'{e:.3f}' for e in err_svd_trace]}")
    print(f"  final error: {err_svd_trace[-1]:.4f}")
    svd_monotone, svd_violations = check_monotone(err_svd_trace)
    print(f"  monotone descent: {'PASS' if svd_monotone else 'FAIL'} "
          f"({svd_violations} small increases tolerated)\n")

    print("=" * 60)
    print("skfusion.fusion.Dfmf (stored reference)")
    print("=" * 60)
    err_old_trace = ref["traza"]
    print(f"  generated with: scikit-fusion {ref['generado_con']['skfusion']} "
          f"(tests/data/referencia_skfusion.json)")
    print(f"  error trace (every 10 iter): {[f'{e:.3f}' for e in err_old_trace]}")
    print(f"  final error: {err_old_trace[-1]:.4f}")
    old_monotone, old_violations = check_monotone(err_old_trace)
    print(f"  monotone descent: {'PASS' if old_monotone else 'FAIL'} "
          f"({old_violations} small increases tolerated)\n")

    print("=" * 60)
    print("Structural parity (must pass)")
    print("=" * 60)

    failures = []
    for t, c in (("t1", c1), ("t2", c2), ("t3", c3)):
        n_expected = G_new[t].shape[0]
        ok_new = G_new[t].shape == (n_expected, c)
        status = "PASS" if ok_new else "FAIL"
        print(f"  factor[{t}] shape: {status} ({G_new[t].shape})")
        if not ok_new:
            failures.append(f"shape mismatch for {t}")

    print(f"  datafiusion (random) monotone: "
          f"{'PASS' if new_monotone else 'FAIL'} ({new_violations} violations)")
    print(f"  datafiusion (nndsvd) monotone: "
          f"{'PASS' if svd_monotone else 'FAIL'} ({svd_violations} violations)")
    print(f"  skfusion (stored) monotone:    "
          f"{'PASS' if old_monotone else 'FAIL'} ({old_violations} violations)")
    if not new_monotone:
        failures.append(f"datafiusion random not monotone ({new_violations} violations)")
    if not svd_monotone:
        failures.append(f"datafiusion nndsvd not monotone ({svd_violations} violations)")
    if not old_monotone:
        failures.append(f"stored skfusion trace not monotone ({old_violations} violations)")

    print()
    print("=" * 60)
    print("Convergence comparison (informational)")
    print("=" * 60)
    print(f"  noise floor:                  {noise_floor:.4f}")
    print(f"  datafiusion (random) final:   {err_new_trace[-1]:.4f}  "
          f"(gap: {err_new_trace[-1] - noise_floor:+.4f})")
    print(f"  datafiusion (nndsvd) final:   {err_svd_trace[-1]:.4f}  "
          f"(gap: {err_svd_trace[-1] - noise_floor:+.4f})")
    print(f"  skfusion (stored) final:      {err_old_trace[-1]:.4f}  "
          f"(gap: {err_old_trace[-1] - noise_floor:+.4f})")
    print()
    threshold = max(0.10, 2 * noise_floor)
    print(f"  iterations to reach {threshold:.3f} (datafiusion random): "
          f"{_first_under(err_new_trace, threshold)}")
    print(f"  iterations to reach {threshold:.3f} (datafiusion nndsvd): "
          f"{_first_under(err_svd_trace, threshold)}")
    print(f"  iterations to reach {threshold:.3f} (skfusion, stored):   "
          f"{_first_under(err_old_trace, threshold)}")
    print()
    print("  Note: different errors at fixed max_iter reflect convergence speed,")
    print("  not algorithm correctness. Random init streams differ between")
    print("  implementations (default_rng vs RandomState). NNDSVD is deterministic")
    print("  and should reach a comparable local minimum in fewer iterations.")

    print()
    if failures:
        print("OVERALL: FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("OVERALL: PASS (structural checks)")


if __name__ == "__main__":
    main()
