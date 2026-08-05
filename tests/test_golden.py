"""Golden test: freeze the current numerical behaviour of dfmf_sparse.

The public entry points that existing notebooks import (`dfmf_sparse` and
`reconstruction_error`) must keep producing the same numbers while the
internals are restructured. This test pins G, S and the reconstruction
error for both initializations against a stored reference.

The reference is generated on first run and versioned afterwards. To
regenerate it on purpose, delete tests/data/golden_*.npz and run again;
any other diff means the legacy path changed and the notebooks in
notebooks/ would no longer reproduce their published numbers.

Run with: uv run pytest tests/test_golden.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from datafiusion import dfmf_sparse, reconstruction_error


DIR_REFERENCIA = Path(__file__).parent / "data"
TOLERANCIA = 1e-10
RANKS = {"t1": 6, "t2": 5, "t3": 4}
MAX_ITER = 100


def instancia():
    """Deterministic sparse instance with structure in it, not pure noise."""
    rng = np.random.default_rng(20260803)
    n1, n2, n3 = 120, 90, 40
    G1 = rng.exponential(1.0, (n1, RANKS["t1"]))
    G2 = rng.exponential(1.0, (n2, RANKS["t2"]))
    G3 = rng.exponential(1.0, (n3, RANKS["t3"]))
    S12 = rng.standard_normal((RANKS["t1"], RANKS["t2"]))
    S13 = rng.standard_normal((RANKS["t1"], RANKS["t3"]))
    S23 = rng.standard_normal((RANKS["t2"], RANKS["t3"]))

    def dispersa(dense, densidad, semilla):
        mascara = np.random.default_rng(semilla).random(dense.shape) < densidad
        return sp.csr_matrix(np.where(mascara, dense, 0.0))

    return {
        ("t1", "t2"): [dispersa(G1 @ S12 @ G2.T, 0.20, 1)],
        ("t1", "t3"): [dispersa(G1 @ S13 @ G3.T, 0.30, 2)],
        ("t2", "t3"): [dispersa(G2 @ S23 @ G3.T, 0.25, 3)],
    }


def ajustar(init):
    R = instancia()
    G, S = dfmf_sparse(
        R=R, ranks=RANKS, lambda_G=0.01, lambda_S=0.01,
        max_iter=MAX_ITER, init=init, random_state=0,
    )
    return R, G, S


def aplanar(G, S, error):
    """Pack factors, backbones and error into one flat dict of arrays."""
    salida = {f"G::{t}": G[t] for t in sorted(G)}
    for (src, dst), mats in sorted(S.items()):
        for k, M in enumerate(mats):
            salida[f"S::{src}::{dst}::{k}"] = M
    salida["error"] = np.asarray(error)
    return salida


@pytest.mark.parametrize("init", ["random", "nndsvd"])
def test_golden(init):
    R, G, S = ajustar(init)
    actual = aplanar(G, S, reconstruction_error(R, G, S))

    DIR_REFERENCIA.mkdir(parents=True, exist_ok=True)
    ruta = DIR_REFERENCIA / f"golden_{init}.npz"
    if not ruta.exists():
        np.savez_compressed(ruta, **actual)
        pytest.skip(f"referencia generada en {ruta}; vuelve a correr para comparar")

    referencia = np.load(ruta)
    assert sorted(referencia.files) == sorted(actual)
    for clave in sorted(actual):
        esperado, obtenido = referencia[clave], actual[clave]
        assert esperado.shape == obtenido.shape, clave
        escala = max(float(np.abs(esperado).max()), 1e-300)
        desvio = float(np.abs(obtenido - esperado).max()) / escala
        assert desvio < TOLERANCIA, f"{clave}: desvio relativo {desvio:.3e}"


def test_no_negatividad_de_G():
    """The multiplicative updates must keep every factor non negative."""
    _, G, _ = ajustar("nndsvd")
    for t, factor in G.items():
        assert (factor >= 0).all(), f"G[{t}] tiene entradas negativas"


def test_error_baja():
    """The error at 100 iterations must be below the error at 5."""
    R = instancia()
    errores = []
    for max_iter in (5, 100):
        G, S = dfmf_sparse(R=R, ranks=RANKS, lambda_G=0.01, lambda_S=0.01,
                           max_iter=max_iter, init="nndsvd", random_state=0)
        errores.append(reconstruction_error(R, G, S))
    assert errores[1] < errores[0], f"el error no bajo: {errores}"
