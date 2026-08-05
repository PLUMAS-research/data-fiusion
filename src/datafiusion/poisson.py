"""Count (Poisson/KL) and mixed-family tri-factorization.

`fuse` dispatches here when at least one relation carries
family="poisson". Poisson relations contribute the generalized KL
divergence

    KL(X || R) = sum_ab [ x_ab log(x_ab / r_ab) - x_ab + r_ab ]

whose mass term collapses to rank size through the column sums and whose
log term only needs R at the stored entries (product_at). Gaussian
relations contribute the same squared loss as the classic path, through
the same machinery (_FitState), including row masks and entry weights.

The shared factors G_t receive one joint update. Each family provides
the numerator and denominator of its multiplicative rule (the quadratic
sign-split N/D for gaussian, the KL num/den for poisson), and the joint
step minimizes the SUM of both auxiliary majorizers exactly: setting the
derivative (D_g / g_old) g - N_g + den_p - num_p g_old / g to zero gives
the positive root of a quadratic in g,

    g = (sqrt(b^2 + 4 a c) - b) / (2 a),  a = D_g / g_old,
                                          b = den_p - N_g,
                                          c = num_p g_old.

With no poisson term this is exactly g_old N_g / D_g (the classic rule);
with no gaussian term it is exactly g_old num_p / den_p (the KL rule).

The backbones do not interact across families: gaussian S keeps its
closed-form (or weighted fixed-point) solve, poisson S is non-negative
and multiplicative, and the column gauge is compensated only in the
poisson S (gaussian S is re-solved anyway).

The loss reported is the mean over relations of a normalized quantity:
relative squared error for gaussian, deviance ratio KL(X||R)/KL(X||null)
for poisson. Both are 0 at a perfect fit. So that `weights` reads as a
loss share ACROSS families too, the KL gradient of each poisson relation
is normalized by its null deviance, mirroring the Frobenius
normalization of the gaussian side; without this, a count relation with
large mass dominates every gaussian term numerically and no reasonable weight can
compensate (measured on Last.fm: the label weight had no effect at all
between 0.3 and 10). Pure poisson fits are unaffected: the
multiplicative ratio is invariant to scaling numerator and denominator
together.
"""

import warnings

import numpy as np
import scipy.sparse as sp

from .model import (EPS, FLOOR, FusionModel, _FitState, _collect_labels,
                    _entities_without_data, _error_de, _frobenius_scales,
                    _infer_sizes, _initialize, _validate_supervision)
from .ops import product_at


def fit_families(relations, ranks, weights=None, supervision=None,
                 gauge="column", lambda_S=1e-2, normalize="frobenius",
                 max_iter=100, tol=1e-5, init="nndsvd", random_state=None,
                 block_rows=200_000, verbose=0, callback=None,
                 warm_start=None, extra_params=None):
    """Fit relations of mixed (or pure poisson) families. Called through `fuse`."""
    gaussianas = {n: r for n, r in relations.items() if r.family == "gaussian"}
    poissons = {n: r for n, r in relations.items() if r.family == "poisson"}

    sizes = _infer_sizes(relations)
    faltantes = set(ranks) - set(sizes)
    if faltantes:
        raise ValueError(f"types in ranks but absent from relations: {sorted(faltantes)}")
    for tipo, n in sizes.items():
        if tipo not in ranks:
            raise ValueError(f"type {tipo!r} appears in relations but has no rank")
        if ranks[tipo] > n:
            raise ValueError(f"rank {ranks[tipo]} exceeds the {n} entities of type {tipo!r}")
    if gauge not in ("column", None):
        raise ValueError(f"unknown gauge {gauge!r}; supported: 'column', None")
    for nombre, relacion in poissons.items():
        if relacion.rows is not None:
            raise ValueError(
                f"relation {nombre!r} has a row mask; not supported with "
                "family='poisson' yet (masks on gaussian relations do work)")
        if relacion.weighted:
            raise ValueError(
                f"relation {nombre!r} carries entry weights; with "
                "family='poisson' express row or column weights by scaling "
                "the data (a column-weighted KL equals plain KL on the "
                "column-scaled matrix)")
        if not sp.issparse(relacion.matrix):
            raise ValueError(
                f"relation {nombre!r}: family='poisson' requires sparse data")
        if relacion.matrix.nnz and relacion.matrix.data.min() < 0:
            raise ValueError(
                f"relation {nombre!r} has negative entries; poisson requires "
                "non-negative data")

    vacias = _entities_without_data(relations, sizes)
    if vacias:
        detalle = ", ".join(f"{len(idx)} de tipo {tipo!r}" for tipo, idx in vacias.items())
        warnings.warn(
            f"there are entities with no observation in any relation ({detalle}). "
            "Their factors collapse to zero and argmax assigns all of them to "
            "component 0. See FusionModel.empty_rows; drop them or handle them "
            "explicitly downstream.",
            stacklevel=3,
        )

    weight = {n: float((weights or {}).get(n, 1.0)) for n in relations}
    scale = {n: 1.0 for n in poissons}
    if gaussianas:
        scale.update(_frobenius_scales(gaussianas) if normalize == "frobenius"
                     else {n: 1.0 for n in gaussianas})
    estado = (_FitState(gaussianas, sizes, ranks, scale, weight, block_rows)
              if gaussianas else None)
    supervision = _validate_supervision(supervision, sizes, ranks)

    datos = {}
    for nombre, relacion in poissons.items():
        X = relacion.matrix.tocsr()
        con_valor = X.data > 0
        total = float(X.data.sum())
        x_log_x = float(np.sum(X.data[con_valor] * np.log(X.data[con_valor])))
        tasa_nula = total / (X.shape[0] * X.shape[1])
        kl_nulo = max(x_log_x - total * np.log(max(tasa_nula, EPS)), EPS)
        datos[nombre] = dict(
            X=X, total=total, x_log_x=x_log_x, kl_nulo=kl_nulo,
            filas=np.repeat(np.arange(X.shape[0]), np.diff(X.indptr)))

    if warm_start is not None:
        G = {t: np.array(f, dtype=np.float64) for t, f in warm_start.G.items()}
        S = {n: np.array(warm_start.S[n], dtype=np.float64) for n in relations}
        historia_previa = list(np.asarray(warm_start.history).ravel())
        iteraciones_previas = warm_start.n_iter
    else:
        G = _initialize(relations, sizes, ranks, init, random_state, scale)
        # Un cero exacto no puede revivir bajo el update multiplicativo.
        for t in G:
            np.maximum(G[t], EPS, out=G[t])
        # S poisson parte constante, con la masa de R igual a la de X; el
        # primer update lo reajusta por celda. S gaussiano se resuelve en
        # el loop.
        S = {}
        for nombre, relacion in poissons.items():
            a = G[relacion.src].sum(axis=0)
            b = G[relacion.dst].sum(axis=0)
            escala0 = datos[nombre]["total"] / max(a.sum() * b.sum(), EPS)
            S[nombre] = np.full((ranks[relacion.src], ranks[relacion.dst]), escala0)
        historia_previa, iteraciones_previas = [], 0

    for tipo, permitido in supervision.items():
        G[tipo] = G[tipo] * permitido

    def razon(nombre, relacion):
        """Sparse X / R on the stored pattern, and R at that pattern."""
        d = datos[nombre]
        r_pat = product_at(G[relacion.src] @ S[nombre], G[relacion.dst],
                           d["filas"], d["X"].indices)
        r_pat = np.maximum(r_pat, EPS)
        z = d["X"].data / r_pat
        return sp.csr_matrix((z, d["X"].indices, d["X"].indptr),
                             shape=d["X"].shape), r_pat

    def desviacion(nombre, relacion):
        """KL(X || R) / KL(X || constant), current factors."""
        d = datos[nombre]
        _, r_pat = razon(nombre, relacion)
        total_R = float(G[relacion.src].sum(axis=0) @ S[nombre]
                        @ G[relacion.dst].sum(axis=0))
        kl = (d["x_log_x"] - float(np.sum(d["X"].data * np.log(r_pat)))
              - d["total"] + total_R)
        return max(kl, 0.0) / d["kl_nulo"]

    def perdida_actual():
        terminos = []
        for nombre, relacion in gaussianas.items():
            M, filas = relacion.observed()
            G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
            err_sq, norm_sq = _error_de(relacion, M, G_src, S[nombre],
                                        G[relacion.dst], scale[nombre])
            terminos.append(err_sq / max(norm_sq, EPS))
        terminos += [desviacion(n, r) for n, r in poissons.items()]
        return float(np.mean(terminos))

    history = list(historia_previa)
    muertas = {}
    razon_parada = "max_iter"
    convergio = False
    iteracion = 0

    for iteracion in range(1, max_iter + 1):
        for nombre in gaussianas:
            S[nombre] = estado.solve_backbone(nombre, G, lambda_S,
                                              S_prev=S.get(nombre))
        for nombre, relacion in poissons.items():
            Z, _ = razon(nombre, relacion)
            G_i, G_j = G[relacion.src], G[relacion.dst]
            num = G_i.T @ (Z @ G_j)
            den = np.outer(G_i.sum(axis=0), G_j.sum(axis=0))
            S[nombre] = S[nombre] * (num / np.maximum(den, EPS))

        if estado is not None:
            N_g, D_g, _ = estado.accumulate(G, S)
        else:
            N_g = {t: 0.0 for t in G}
            D_g = {t: 0.0 for t in G}
        num_p = {t: np.zeros_like(G[t]) for t in G}
        den_p = {t: np.zeros_like(G[t]) for t in G}
        for nombre, relacion in poissons.items():
            # El gradiente KL se normaliza por la desviacion nula de su
            # relacion, para que `weights` lea como fraccion de perdida
            # entre familias (el espejo de la normalizacion Frobenius del
            # lado gaussiano). En un fit Poisson puro esto no cambia nada:
            # la regla multiplicativa es invariante a escalar numerador y
            # denominador juntos.
            beta = weight[nombre] / datos[nombre]["kl_nulo"]
            S_r = S[nombre]
            G_i, G_j = G[relacion.src], G[relacion.dst]
            Z, _ = razon(nombre, relacion)
            num_p[relacion.src] += beta * (Z @ (G_j @ S_r.T))
            den_p[relacion.src] += beta * (S_r @ G_j.sum(axis=0))[None, :]
            num_p[relacion.dst] += beta * (Z.T @ (G_i @ S_r))
            den_p[relacion.dst] += beta * (S_r.T @ G_i.sum(axis=0))[None, :]

        for t in G:
            N_t = np.maximum(np.asarray(N_g[t]), 0.0)
            a = np.asarray(D_g[t]) / np.maximum(G[t], EPS)
            b = den_p[t] - N_t
            c = num_p[t] * G[t]
            raiz = np.sqrt(b * b + 4.0 * a * c)
            G[t] = np.where(np.asarray(a) > 0,
                            (raiz - b) / np.maximum(2.0 * a, EPS),
                            c / np.maximum(b, EPS))

        for tipo, permitido in supervision.items():
            G[tipo] *= permitido

        muertas = {}
        if gauge == "column":
            escala_col = {}
            for t in G:
                normas = np.linalg.norm(G[t], axis=0)
                colapsadas = normas < FLOOR
                if colapsadas.any():
                    muertas[t] = np.flatnonzero(colapsadas)
                escala_col[t] = np.maximum(normas, FLOOR)
                G[t] /= escala_col[t]
            for nombre, relacion in poissons.items():
                S[nombre] = (escala_col[relacion.src][:, None] * S[nombre]
                             * escala_col[relacion.dst][None, :])

        perdida = perdida_actual()
        history.append(perdida)

        if verbose and iteracion % verbose == 0:
            print(f"iter {iteracion:4d} / {max_iter}: loss = {perdida:.6f}")
        if callback is not None and callback(iteracion, perdida, G):
            razon_parada = "callback"
            break
        if len(history) > 1 and tol is not None:
            previa = history[-2]
            if previa > 0 and abs(previa - perdida) / previa < tol:
                razon_parada, convergio = "tol", True
                break

    for nombre in gaussianas:
        S[nombre] = estado.solve_backbone(nombre, G, lambda_S,
                                          S_prev=S.get(nombre))

    rel_error = {}
    for nombre, relacion in gaussianas.items():
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        err_sq, norm_sq = _error_de(relacion, M, G_src, S[nombre],
                                    G[relacion.dst], scale[nombre])
        rel_error[nombre] = float(np.sqrt(err_sq / max(norm_sq, EPS)))
    for nombre, relacion in poissons.items():
        rel_error[nombre] = float(desviacion(nombre, relacion))

    params = dict(
        weights=weights, family="poisson" if not gaussianas else "mixed",
        supervision={t: m.astype(bool) for t, m in supervision.items()} or None,
        gauge=gauge, lambda_S=lambda_S, normalize=normalize,
        max_iter=max_iter, tol=tol, init=init, random_state=random_state,
        block_rows=block_rows, **(extra_params or {}),
    )
    return FusionModel(
        G=G, S=S, rel={n: (r.src, r.dst) for n, r in relations.items()},
        ranks=dict(ranks), sizes=sizes, scale=scale, weight=weight,
        index=_collect_labels(relations),
        history=np.asarray(history), rel_error=rel_error,
        n_iter=iteraciones_previas + iteracion, converged=convergio,
        stop_reason=razon_parada, dead_columns=muertas, empty_rows=vacias,
        params=params,
    )
