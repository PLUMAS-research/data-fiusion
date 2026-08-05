"""Fitting API for sparse data fusion, built for relations that do not fit densely.

The entry point is `fit`, which takes named relations and returns a
`FusionModel` carrying everything needed downstream: factors, backbones,
the scaling applied at fit time, the loss trace and the hyperparameters.
Carrying the scaling is what makes `model.transform(...)` correct by
construction: new entities are put in the same units as the training
matrices without the caller having to remember how.

What regularizes and what does not
----------------------------------
An L2 penalty on G is not offered, because it cannot work here. After the
closed-form solve of S the identity <G_t, N_t> = <G_t, D_t> holds exactly,
so a penalty that only feeds the denominator has no fixed point and drives
G to zero. Under the column gauge the same penalty is a constant on the
feasible set, so it cannot move the minimizer either. The levers that do
change the result are:

  - `weights`: how much each relation counts in the loss.
  - `masks` (Relation.rows): which rows are observed at all, which is what
    makes a semi-supervised regime possible.
  - `alpha_graph`: smoothing over a neighbourhood graph per type.
  - the rank of each type, which is what actually controls overfitting.

Every alpha is dimensionless and calibrated against the data gradient
energy of its own type, e_t = <G_t, D_t^data>, taken from the previous
iteration. An absolute penalty is not comparable across types: measured on
a real multi-source case, the data energy varies by a factor of 1100 between
types, so one absolute lambda regularizes one type 620 times harder than
another.

Scale gauge
-----------
The loss is invariant under (G_i, G_j, S) -> (a G_i, a G_j, S / a^2), so
without a gauge the factors drift (||G||_F reaching 7587 was measured) and
no penalty on G is well defined. Normalizing the columns of G after each
update fixes ||G_t||_F^2 = c_t and makes lambda_S readable as a fraction
of the per-column energy.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .core import nonneg_refine
from .init import init_random, init_nndsvd


EPS = np.finfo(np.float64).eps
FLOOR = 1e-12


# --------------------------------------------------------------------------
# entrada


@dataclass
class Relation:
    """One relation matrix, its two endpoints and its row observation mask.

    Parameters
    ----------
    src, dst : str
        Type names. A relation from a type to itself is not allowed; pass
        neighbourhood structure through `graphs` instead.
    matrix : scipy.sparse matrix or ndarray
        Shape (n_src, n_dst).
    rows : ndarray of int or bool, optional
        Which rows carry an observation. Rows outside the mask contribute
        nothing to the loss, which is what separates "no data" from "a
        measured zero". Without a mask every zero is an observation.
    row_labels, col_labels : ndarray, optional
        Entity labels. When present they are checked at fold-in time, so a
        column permutation raises instead of silently scoring nonsense.
    """

    src: str
    dst: str
    matrix: object
    rows: np.ndarray | None = None
    row_labels: np.ndarray | None = None
    col_labels: np.ndarray | None = None

    def __post_init__(self):
        if self.src == self.dst:
            raise ValueError(
                f"self-relation ({self.src}, {self.dst}) is not supported; "
                "pass neighbourhood structure through `graphs`"
            )
        if sp.issparse(self.matrix):
            if not getattr(self.matrix, "has_canonical_format", True):
                self.matrix.sum_duplicates()
        else:
            self.matrix = np.asarray(self.matrix)
        if self.rows is not None:
            rows = np.asarray(self.rows)
            if rows.dtype == bool:
                if rows.shape[0] != self.matrix.shape[0]:
                    raise ValueError(
                        f"boolean mask of length {rows.shape[0]} does not match "
                        f"{self.matrix.shape[0]} rows"
                    )
                rows = np.flatnonzero(rows)
            self.rows = np.sort(rows)
            if self.rows.size and (self.rows[-1] >= self.matrix.shape[0]
                                   or self.rows[0] < 0):
                raise ValueError("row mask indexes outside the matrix")

    @property
    def shape(self):
        """Shape of the underlying matrix."""
        return self.matrix.shape

    def observed(self):
        """Matrix restricted to the observed rows, and those row indices."""
        if self.rows is None:
            return self.matrix, None
        return self.matrix[self.rows], self.rows


def _as_relations(relations):
    """Accept named relations, or the legacy dict keyed by (src, dst)."""
    if all(isinstance(v, Relation) for v in relations.values()):
        return dict(relations)
    salida = {}
    for clave, valor in relations.items():
        if not (isinstance(clave, tuple) and len(clave) == 2):
            raise TypeError(
                "relations must be dict[str, Relation] or the legacy "
                "dict[(src, dst), list[sparse]]"
            )
        src, dst = clave
        mats = valor if isinstance(valor, (list, tuple)) else [valor]
        for k, M in enumerate(mats):
            nombre = f"{src}~{dst}" if len(mats) == 1 else f"{src}~{dst}#{k}"
            salida[nombre] = Relation(src=src, dst=dst, matrix=M)
    return salida


# --------------------------------------------------------------------------
# modelo


@dataclass
class FusionModel:
    """A fitted model: factors, backbones, the fit-time scaling and the trace."""

    G: dict
    S: dict
    rel: dict
    ranks: dict
    sizes: dict
    scale: dict
    weight: dict
    index: dict = field(default_factory=dict)
    history: np.ndarray = field(default_factory=lambda: np.empty(0))
    rel_error: dict = field(default_factory=dict)
    n_iter: int = 0
    converged: bool = False
    stop_reason: str = "max_iter"
    dead_columns: dict = field(default_factory=dict)
    empty_rows: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def __iter__(self):
        """Yield (G, S) so `G, S = model` keeps working."""
        return iter((self.G, self.S))

    def factor(self, type_name, as_frame=False):
        """Return G_t, optionally indexed by the entity labels of that type."""
        factor = self.G[type_name]
        if not as_frame:
            return factor
        import pandas as pd
        return pd.DataFrame(factor, index=self.index.get(type_name))

    def backbone(self, name, as_frame=False):
        """Return S_r for the relation called `name`."""
        backbone = self.S[name]
        if not as_frame:
            return backbone
        import pandas as pd
        return pd.DataFrame(backbone)

    def loss(self, relations=None, per_relation=False, relative=True):
        """Loss by the trace identity, never materializing G_i S G_j^T.

        With relative=True each term is divided by the squared norm of its
        own matrix and the aggregate is the MEAN, so models fitted on a
        different number of relations stay comparable.
        """
        relations = self._relations_for_loss(relations)
        por_relacion = {}
        for nombre, relacion in relations.items():
            S_r = self.S[nombre]
            s, beta = self.scale[nombre], self.weight[nombre]
            M, filas = relacion.observed()
            G_src = self.G[relacion.src]
            G_src = G_src if filas is None else G_src[filas]
            err_sq, norm_sq = _squared_error_scaled(M, G_src, S_r, self.G[relacion.dst], s)
            por_relacion[nombre] = err_sq / norm_sq if relative else beta * err_sq
        if per_relation:
            return por_relacion
        valores = list(por_relacion.values())
        return float(np.mean(valores)) if relative else float(np.sum(valores))

    def _relations_for_loss(self, relations):
        if relations is None:
            raise ValueError("pass the relations the model was fitted on")
        return _as_relations(relations)

    # ---------------------------------------------------------------- fold-in

    def transform(self, relations, target, nonneg=True, alpha=1e-3,
                  max_iter=100, tol=1e-4, block_rows=200_000, verbose=0):
        """Fold new entities of type `target` in, returning a DERIVED model.

        The returned model shares S, scale and weight and replaces
        G[target], so `model.transform(...).predict_proba(...)` composes
        without going back to raw dicts.

        The fit-time scale and weight are reapplied here, not by the
        caller. That is the point: applying the new matrices' own norm
        instead of the training norm puts them in different units than the
        learned backbones, which silently produces the wrong factors
        (cosine 0.44 against the correct fold-in, measured).

        With nonneg=True the solution is refined with multiplicative
        updates from a strictly positive floor. The floor is mandatory: a
        multiplicative update cannot change a sign, so starting from the
        raw ridge yields NaN and starting from a hard clamp freezes the
        support at zero.
        """
        relations = _as_relations(relations)
        self._validate_fold_in(relations, target)

        c = self.ranks[target]
        n_new = _rows_of_target(relations, target)

        Q = np.zeros((c, c))
        rhs = np.zeros((n_new, c))
        for nombre, relacion in relations.items():
            S_r = self.S[nombre]
            s, beta = self.scale[nombre], self.weight[nombre]
            otro = relacion.dst if relacion.src == target else relacion.src
            G_otro = self.G[otro]
            gram = G_otro.T @ G_otro
            if relacion.src == target:
                Q += beta * (S_r @ gram @ S_r.T)
            else:
                Q += beta * (S_r.T @ gram @ S_r)
            M = relacion.matrix if relacion.src == target else relacion.matrix.T
            for inicio in range(0, n_new, block_rows):
                bloque = slice(inicio, min(inicio + block_rows, n_new))
                producto = M[bloque] @ G_otro
                rhs[bloque] += (beta * s) * (producto @ (S_r.T if relacion.src == target else S_r))

        lam = alpha * max(np.trace(Q) / c, EPS)
        X = np.linalg.solve(Q + lam * np.eye(c), rhs.T).T
        if not nonneg:
            G_new = X
        else:
            G_new = nonneg_refine(X, Q, rhs, lam, max_iter=max_iter, tol=tol)

        derivado = FusionModel(
            G={**self.G, target: G_new}, S=self.S, rel=self.rel, ranks=self.ranks,
            sizes={**self.sizes, target: n_new}, scale=self.scale, weight=self.weight,
            index={**self.index, target: _labels_of_target(relations, target)},
            history=self.history, rel_error=self.rel_error, n_iter=self.n_iter,
            converged=self.converged, stop_reason=self.stop_reason,
            dead_columns=self.dead_columns, params={**self.params, "transformed": target},
        )
        return derivado

    def _validate_fold_in(self, relations, target):
        if target not in self.G:
            raise ValueError(f"unknown target type {target!r}; fitted types: {sorted(self.G)}")
        if not relations:
            raise ValueError("no relations given to fold in")
        for nombre, relacion in relations.items():
            if nombre not in self.rel:
                raise ValueError(
                    f"relation {nombre!r} was not part of the fit; known: {sorted(self.rel)}"
                )
            src, dst = self.rel[nombre]
            if (relacion.src, relacion.dst) != (src, dst):
                raise ValueError(
                    f"relation {nombre!r} was fitted as {(src, dst)} but got "
                    f"{(relacion.src, relacion.dst)}"
                )
            if target not in (src, dst):
                raise ValueError(
                    f"relation {nombre!r} does not touch target type {target!r}"
                )
            otro = dst if src == target else src
            eje = 1 if src == target else 0
            if relacion.shape[eje] != self.sizes[otro]:
                raise ValueError(
                    f"relation {nombre!r} has {relacion.shape[eje]} entries on the "
                    f"{otro!r} side but the model was fitted with {self.sizes[otro]}"
                )
            etiquetas_modelo = self.index.get(otro)
            etiquetas_nuevas = relacion.col_labels if src == target else relacion.row_labels
            if etiquetas_modelo is not None and etiquetas_nuevas is not None:
                if not np.array_equal(np.asarray(etiquetas_modelo), np.asarray(etiquetas_nuevas)):
                    raise ValueError(
                        f"relation {nombre!r} has the {otro!r} entities in a different "
                        "order than the fit; reindex before folding in"
                    )

    # -------------------------------------------------------------- prediccion

    def predict_proba(self, target, views, known=None, combine="geometric_mean",
                      eps=1e-3, top_k=None, max_bytes=256 << 20, batch_rows=None):
        """Distribution over `target` combining several views, in row batches.

        Views are relation names and may point either way: when `target`
        is the source, S^T is used. Without `top_k` this refuses to build
        an output larger than `max_bytes`, because the full matrix is
        (n, n_target) and that is the shape that stops fitting first.
        `batch_rows` overrides the working batch size, which is otherwise
        derived from `max_bytes`; the result does not depend on it.
        """
        if isinstance(views, str):
            views = [views]
        for nombre in views:
            if nombre not in self.rel:
                raise ValueError(f"unknown relation {nombre!r}; known: {sorted(self.rel)}")
            if target not in self.rel[nombre]:
                raise ValueError(f"relation {nombre!r} does not touch {target!r}")

        fuentes = {}
        for nombre in views:
            src, dst = self.rel[nombre]
            fuentes[nombre] = dst if src == target else src
        n = _known_length(known, fuentes, self.sizes)
        n_target = self.sizes[target]

        salida_bytes = n * n_target * 8
        if top_k is None and salida_bytes > max_bytes:
            raise ValueError(
                f"the full output would be {salida_bytes / 1e9:.1f} GB; pass top_k "
                f"or raise max_bytes"
            )

        lote = int(batch_rows) if batch_rows else max(1, int(max_bytes // max(2 * n_target * 8, 1)))
        if top_k is None:
            resultado = np.empty((n, n_target))
        else:
            top_k = min(top_k, n_target)
            indices_out = np.empty((n, top_k), dtype=np.int64)
            scores_out = np.empty((n, top_k))

        for inicio in range(0, n, lote):
            fin = min(inicio + lote, n)
            acumulado = None
            for nombre in views:
                filas = _known_rows(known, fuentes[nombre], self.index, inicio, fin)
                v = self._view_scores(nombre, target, filas, eps)
                acumulado = _combine_into(acumulado, v, combine)
            bloque = _finish_combine(acumulado, combine, len(views))
            if top_k is None:
                resultado[inicio:fin] = bloque
            else:
                parcial = np.argpartition(-bloque, top_k - 1, axis=1)[:, :top_k]
                puntajes = np.take_along_axis(bloque, parcial, axis=1)
                orden = np.argsort(-puntajes, axis=1)
                indices_out[inicio:fin] = np.take_along_axis(parcial, orden, axis=1)
                scores_out[inicio:fin] = np.take_along_axis(puntajes, orden, axis=1)

        return resultado if top_k is None else (indices_out, scores_out)

    def _view_scores(self, nombre, target, filas, eps):
        src, dst = self.rel[nombre]
        S_r = self.S[nombre]
        if src == target:
            v = self.G[dst][filas] @ S_r.T @ self.G[target].T
        else:
            v = self.G[src][filas] @ S_r @ self.G[target].T
        np.maximum(v, 0.0, out=v)
        v += eps
        v /= v.sum(axis=1, keepdims=True)
        return v

    # ------------------------------------------------------------ persistencia

    def save(self, path):
        """Write the model as a directory of .npy files plus meta.json.

        A directory of .npy rather than a single .npz because .npz cannot
        be memory mapped: np.load(..., mmap_mode='r') on an .npz returns
        ordinary arrays, so a large model could not be opened without
        loading it whole.
        """
        path = Path(path)
        (path / "factors").mkdir(parents=True, exist_ok=True)
        (path / "backbones").mkdir(parents=True, exist_ok=True)
        for tipo, factor in self.G.items():
            np.save(path / "factors" / f"{tipo}.npy", factor)
        for nombre, backbone in self.S.items():
            np.save(path / "backbones" / f"{nombre}.npy", backbone)
        np.save(path / "history.npy", self.history)
        meta = {
            "rel": {k: list(v) for k, v in self.rel.items()},
            "ranks": self.ranks, "sizes": self.sizes,
            "scale": self.scale, "weight": self.weight,
            "rel_error": self.rel_error, "n_iter": self.n_iter,
            "converged": self.converged, "stop_reason": self.stop_reason,
            "dead_columns": {k: np.asarray(v).tolist() for k, v in self.dead_columns.items()},
            "empty_rows": {k: np.asarray(v).tolist() for k, v in self.empty_rows.items()},
            "params": {k: v for k, v in self.params.items() if _jsonable(v)},
            "index": {k: np.asarray(v).tolist() for k, v in self.index.items() if v is not None},
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path, mmap=False):
        """Read a model written by `save`. With mmap=True factors stay on disk."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        modo = "r" if mmap else None
        G = {p.stem: np.load(p, mmap_mode=modo) for p in sorted((path / "factors").glob("*.npy"))}
        S = {p.stem: np.load(p) for p in sorted((path / "backbones").glob("*.npy"))}
        return cls(
            G=G, S=S, rel={k: tuple(v) for k, v in meta["rel"].items()},
            ranks=meta["ranks"], sizes=meta["sizes"], scale=meta["scale"],
            weight=meta["weight"], index={k: np.asarray(v) for k, v in meta["index"].items()},
            history=np.load(path / "history.npy"), rel_error=meta["rel_error"],
            n_iter=meta["n_iter"], converged=meta["converged"],
            stop_reason=meta["stop_reason"],
            dead_columns={k: np.asarray(v) for k, v in meta["dead_columns"].items()},
            empty_rows={k: np.asarray(v) for k, v in meta.get("empty_rows", {}).items()},
            params=meta["params"],
        )

    def resume(self, relations, max_iter, **overrides):
        """Continue the fit from the current factors, reusing scale and weight.

        The stored scale and weight are reused rather than recomputed, so
        resuming on a subset of the data cannot silently change the units.
        `history` and `n_iter` accumulate.
        """
        params = {**self.params, **overrides, "max_iter": max_iter}
        params.pop("transformed", None)
        return fit(relations, ranks=self.ranks, _warm_start=self, **params)


# --------------------------------------------------------------------------
# ajuste


def fit(relations, ranks, weights=None, normalize="frobenius", masks=None,
        supervision=None,
        eta=1.0, gauge="column", lambda_S=1e-2, lambda_G=0.0,
        alpha_graph=0.0, graphs=None, theta=None,
        max_iter=100, tol=1e-5, init="nndsvd", random_state=None,
        block_rows=200_000, verbose=0, callback=None, _warm_start=None):
    """Fit a data fusion model over named relations.

    Parameters
    ----------
    relations : dict[str, Relation] or dict[(src, dst), list[sparse]]
        Named relations. The legacy keyed form is accepted and named
        automatically, but naming them explicitly is what lets `transform`
        and `predict_proba` refer to a specific matrix. Two matrices
        between the same pair of types are indistinguishable otherwise.
    ranks : dict[str, int]
        Latent rank per type. This is the lever that controls overfitting.
    weights : dict[str, float], optional
        How much each relation counts in the loss. Multiplies the whole
        term, both the linear and the quadratic part. Folded in as a
        scalar, so it costs no copy of the data.
    normalize : {"frobenius", None}
        With "frobenius" each matrix enters with unit norm, so `weights`
        reads directly as its share of the loss budget. The scaling is
        stored in the model and reapplied by `transform`.
    masks : dict[str, ndarray], optional
        Observed rows per relation, as an alternative to Relation.rows.
        This says which OBSERVATIONS enter the loss.
    supervision : dict[str, ndarray], optional
        Boolean matrix of shape (n_t, c_t) per type, saying which latent
        components each entity is allowed to use. This is a different
        thing from `masks`: it constrains the FACTOR, not the data.

        Following TS-NMF (MacMillan and Wilson 2017), a relation is then
        reconstructed as ``(G_i * L_i) S G_j^T``, so an entity whose label
        is known can be pinned to the components that stand for that
        label. Components then carry a fixed meaning instead of being
        latent groups that the backbone has to translate.

        Entities with no supervision get a row of all True. The update
        needs no special case: a multiplicative step cannot turn a zero
        into a non-zero, so applying the mask once after initialization
        keeps the forbidden entries at zero forever.
    eta : float
        Exponent of the multiplicative step. 1.0 is the plain scaled
        gradient step and is safe here: the relevant Hessian blocks are
        PSD, so the step descends. 0.5 is the legacy half step, which
        needs twice the iterations to travel the same distance.
    gauge : {"column", None}
        Column normalization of G after each update, fixing ||G_t||_F^2 =
        c_t. Without it the factors drift and no penalty on G is well
        defined.
    lambda_S : float
        Tikhonov weight in the closed-form solve of S. It penalizes
        lambda_S (||G_i S||^2 + ||S G_j^T||^2) + lambda_S^2 ||S||^2, not
        lambda_S ||S||^2.
    lambda_G : float
        Absolute L2 on G. Kept only for the legacy path and defaults to
        zero: after the closed-form solve of S it has no fixed point and
        drives G to zero, and under the column gauge it is inert. Prefer
        the rank and `alpha_graph`.
    alpha_graph : float or dict[str, float]
        Graph smoothing, as a fraction of the data gradient energy of its
        own type. Dimensionless and comparable across types.
    graphs : dict[str, sparse], optional
        Adjacency W per type. The Laplacian used is diag(W_sym 1) - W_sym
        with W_sym = D^-1/2 W D^-1/2, which keeps the constant vector in
        its kernel, so smoothing does not shrink low-degree nodes.
    theta : dict[str, sparse], optional
        Legacy raw Laplacian per type, uncalibrated. Prefer `graphs`.
    max_iter, tol : int, float
        Stop when the relative change in the loss falls below `tol`.
    init : {"nndsvd", "random"}
    block_rows : int
        Row block size for the sparse passes.
    verbose : int
        Print the loss every `verbose` iterations.
    callback : callable, optional
        Called as callback(iteration, loss, model_state). Returning True
        stops the fit.

    Returns
    -------
    FusionModel
    """
    relations = _as_relations(relations)
    if masks:
        for nombre, filas in masks.items():
            if nombre not in relations:
                raise ValueError(f"mask for unknown relation {nombre!r}")
            relations[nombre] = Relation(
                src=relations[nombre].src, dst=relations[nombre].dst,
                matrix=relations[nombre].matrix, rows=filas,
                row_labels=relations[nombre].row_labels,
                col_labels=relations[nombre].col_labels)

    sizes = _infer_sizes(relations)
    faltantes = set(ranks) - set(sizes)
    if faltantes:
        raise ValueError(f"types in ranks but absent from relations: {sorted(faltantes)}")
    for tipo, n in sizes.items():
        if tipo not in ranks:
            raise ValueError(f"type {tipo!r} appears in relations but has no rank")
        if ranks[tipo] > n:
            raise ValueError(f"rank {ranks[tipo]} exceeds the {n} entities of type {tipo!r}")

    graphs = graphs or {}
    if graphs and _all_zero(alpha_graph, graphs):
        raise ValueError("graphs given but alpha_graph is 0; set alpha_graph > 0")
    if _any_nonzero(alpha_graph) and not graphs:
        raise ValueError("alpha_graph > 0 but no graphs given")
    for tipo, W in graphs.items():
        if tipo not in sizes:
            raise ValueError(f"graph for unknown type {tipo!r}")
        if W.shape != (sizes[tipo], sizes[tipo]):
            raise ValueError(
                f"graph for {tipo!r} has shape {W.shape}, expected "
                f"({sizes[tipo]}, {sizes[tipo]})")

    vacias = _entities_without_data(relations, sizes)
    if vacias:
        detalle = ", ".join(f"{len(idx)} de tipo {tipo!r}" for tipo, idx in vacias.items())
        warnings.warn(
            f"there are entities with no observation in any relation ({detalle}). "
            "Their factors collapse to zero and argmax assigns all of them to "
            "component 0. See FusionModel.empty_rows; drop them or handle them "
            "explicitly downstream.",
            stacklevel=2,
        )

    scale = _frobenius_scales(relations) if normalize == "frobenius" \
        else {n: 1.0 for n in relations}
    weight = {n: float((weights or {}).get(n, 1.0)) for n in relations}

    if _warm_start is not None:
        G = {t: np.array(f, dtype=np.float64) for t, f in _warm_start.G.items()}
        historia_previa = list(np.asarray(_warm_start.history).ravel())
        iteraciones_previas = _warm_start.n_iter
    else:
        G = _initialize(relations, sizes, ranks, init, random_state, scale)
        historia_previa, iteraciones_previas = [], 0

    supervision = _validate_supervision(supervision, sizes, ranks)
    for tipo, permitido in supervision.items():
        G[tipo] = G[tipo] * permitido

    laplacianos = {t: _graph_laplacian(W) for t, W in graphs.items()}
    theta_split = _split_theta(theta or {}, sizes)

    norms_sq = {n: _observed_norm_sq(r, scale[n]) for n, r in relations.items()}
    estado = _FitState(relations, sizes, ranks, scale, weight, block_rows)

    S = {}
    history = list(historia_previa)
    energia_previa = {t: None for t in G}
    razon = "max_iter"
    convergio = False
    iteracion = 0

    for iteracion in range(1, max_iter + 1):
        for nombre, relacion in relations.items():
            S[nombre] = estado.solve_backbone(nombre, G, lambda_S)

        N, D, energia = estado.accumulate(G, S)

        for tipo, L in laplacianos.items():
            alpha = _alpha_for(alpha_graph, tipo)
            if alpha > 0.0:
                W_sym, grado = L
                denominador = float(np.sum(G[tipo] * (grado[:, None] * G[tipo])))
                base = energia_previa[tipo] if energia_previa[tipo] is not None else energia[tipo]
                gamma = alpha * base / max(denominador, EPS)
                N[tipo] += gamma * (W_sym @ G[tipo])
                D[tipo] += gamma * (grado[:, None] * G[tipo])

        for tipo, (T_pos, T_neg) in theta_split.items():
            D[tipo] += T_pos @ G[tipo]
            N[tipo] += T_neg @ G[tipo]

        if lambda_G > 0.0:
            for tipo in G:
                D[tipo] += lambda_G * G[tipo]

        for tipo in G:
            razon_paso = np.maximum(N[tipo], 0.0) / np.maximum(D[tipo], EPS)
            if eta == 1.0:
                G[tipo] *= razon_paso
            elif eta == 0.5:
                G[tipo] *= np.sqrt(razon_paso)
            else:
                G[tipo] *= np.power(razon_paso, eta)

        # La supervision se reaplica antes del gauge, para que la norma de
        # columna se calcule sobre las entradas que de verdad quedan vivas.
        for tipo, permitido in supervision.items():
            G[tipo] *= permitido

        muertas = {}
        if gauge == "column":
            for tipo in G:
                normas = np.linalg.norm(G[tipo], axis=0)
                colapsadas = normas < FLOOR
                if colapsadas.any():
                    muertas[tipo] = np.flatnonzero(colapsadas)
                G[tipo] /= np.maximum(normas, FLOOR)

        energia_previa = energia
        perdida = _loss_now(relations, G, S, scale, weight, norms_sq, estado)
        history.append(perdida)

        if verbose and iteracion % verbose == 0:
            print(f"iter {iteracion:4d} / {max_iter}: loss = {perdida:.6f}")

        if callback is not None:
            if callback(iteracion, perdida, G):
                razon = "callback"
                break

        if len(history) > 1 and tol is not None:
            previa = history[-2]
            if previa > 0 and abs(previa - perdida) / previa < tol:
                razon, convergio = "tol", True
                break

    # S is re-solved against the factors that are actually returned.
    # Without this the backbones belong to the previous iterate, so
    # `transform` and `predict_proba` would assume an optimality that the
    # returned S does not satisfy.
    for nombre in relations:
        S[nombre] = estado.solve_backbone(nombre, G, lambda_S)

    rel_error = {}
    for nombre, relacion in relations.items():
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        err_sq, norm_sq = _squared_error_scaled(M, G_src, S[nombre], G[relacion.dst],
                                                scale[nombre])
        rel_error[nombre] = float(np.sqrt(err_sq / max(norm_sq, EPS)))

    params = dict(
        weights=weights, normalize=normalize, eta=eta, gauge=gauge,
        supervision={t: m.astype(bool) for t, m in supervision.items()} or None,
        lambda_S=lambda_S, lambda_G=lambda_G, alpha_graph=alpha_graph,
        max_iter=max_iter, tol=tol, init=init, random_state=random_state,
        block_rows=block_rows,
    )
    return FusionModel(
        G=G, S=S, rel={n: (r.src, r.dst) for n, r in relations.items()},
        ranks=dict(ranks), sizes=sizes, scale=scale, weight=weight,
        index=_collect_labels(relations),
        history=np.asarray(history), rel_error=rel_error,
        n_iter=iteraciones_previas + iteracion, converged=convergio,
        stop_reason=razon, dead_columns=muertas, empty_rows=vacias, params=params,
    )


# --------------------------------------------------------------------------
# maquinaria interna


class _FitState:
    """Holds the per-iteration work buffers and the Gram cache.

    The Gram of a type is cached per (type, mask) rather than per type:
    with a row mask the destination side needs the MASKED Gram, and using
    the unmasked one changes the resulting factor by 70% in relative norm.
    """

    def __init__(self, relations, sizes, ranks, scale, weight, block_rows):
        self.relations = relations
        self.sizes = sizes
        self.ranks = ranks
        self.scale = scale
        self.weight = weight
        self.block_rows = max(1, int(block_rows))

    def _gram(self, cache, tipo, G, filas):
        clave = (tipo, None if filas is None else filas.tobytes())
        if clave not in cache:
            factor = G[tipo] if filas is None else G[tipo][filas]
            cache[clave] = factor.T @ factor
        return cache[clave]

    def solve_backbone(self, nombre, G, lambda_S):
        """Closed-form S with Tikhonov, on the observed rows only."""
        relacion = self.relations[nombre]
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        G_dst = G[relacion.dst]
        c_i, c_j = G_src.shape[1], G_dst.shape[1]
        A = G_src.T @ G_src + lambda_S * np.eye(c_i)
        B = G_dst.T @ G_dst + lambda_S * np.eye(c_j)
        middle = G_src.T @ (M @ G_dst) * self.scale[nombre]
        X = np.linalg.solve(A, middle)
        return np.linalg.solve(B.T, X.T).T

    def accumulate(self, G, S):
        """Numerator, denominator and data energy per type, in row blocks.

        Both sides of every relation are accumulated before any factor
        moves, so the step stays simultaneous across types and matches the
        legacy trajectory.
        """
        N = {t: np.zeros_like(G[t]) for t in G}
        D = {t: np.zeros_like(G[t]) for t in G}
        cache = {}

        for nombre, relacion in self.relations.items():
            S_r = S[nombre]
            s, beta = self.scale[nombre], self.weight[nombre]
            M, filas = relacion.observed()
            src, dst = relacion.src, relacion.dst
            G_dst = G[dst]

            gram_dst = self._gram(cache, dst, G, None)
            B_src = beta * (S_r @ gram_dst @ S_r.T)
            B_src_pos, B_src_neg = _split_signs(B_src)

            G_src_obs = G[src] if filas is None else G[src][filas]
            gram_src = self._gram(cache, src, G, filas)
            B_dst = beta * (S_r.T @ gram_src @ S_r)
            B_dst_pos, B_dst_neg = _split_signs(B_dst)

            n_obs = M.shape[0]
            acc_dst_pos = np.zeros((self.sizes[dst], self.ranks[src]))
            for inicio in range(0, n_obs, self.block_rows):
                bloque = slice(inicio, min(inicio + self.block_rows, n_obs))
                M_b = M[bloque]
                G_src_b = G_src_obs[bloque]

                A_b = (beta * s) * ((M_b @ G_dst) @ S_r.T)
                A_pos, A_neg = _split_signs(A_b)
                aporte_N = A_pos + G_src_b @ B_src_neg
                aporte_D = A_neg + G_src_b @ B_src_pos
                if filas is None:
                    N[src][bloque] += aporte_N
                    D[src][bloque] += aporte_D
                else:
                    N[src][filas[bloque]] += aporte_N
                    D[src][filas[bloque]] += aporte_D

                acc_dst_pos += M_b.T @ G_src_b

            A_dst = (beta * s) * (acc_dst_pos @ S_r)
            A_dst_pos, A_dst_neg = _split_signs(A_dst)
            N[dst] += A_dst_pos + G_dst @ B_dst_neg
            D[dst] += A_dst_neg + G_dst @ B_dst_pos

        energia = {t: float(np.sum(G[t] * D[t])) for t in G}
        return N, D, energia


def _entities_without_data(relations, sizes):
    """Entities that have no observation in any relation they take part in.

    Their factor stays at whatever the initialization left it, which is
    negligible next to any fitted row, and with `nndsvd` it is the same
    value for all of them. Nothing downstream notices: `argmax` sends every
    one of them to the same component, silently inflating that group. On
    real data this happens whenever a filter leaves some entities without
    rows.

    Counts stored entries, so an explicitly stored zero counts as an
    observation.
    """
    conteo = {tipo: np.zeros(n, dtype=np.int64) for tipo, n in sizes.items()}
    for relacion in relations.values():
        M, filas = relacion.observed()
        if sp.issparse(M):
            por_fila = M.getnnz(axis=1)
            por_columna = M.getnnz(axis=0)
        else:
            no_cero = np.asarray(M) != 0
            por_fila = no_cero.sum(axis=1)
            por_columna = no_cero.sum(axis=0)
        if filas is None:
            conteo[relacion.src] += por_fila
        else:
            conteo[relacion.src][filas] += por_fila
        conteo[relacion.dst] += por_columna
    return {tipo: np.flatnonzero(c == 0) for tipo, c in conteo.items()
            if not c.all()}


def _validate_supervision(supervision, sizes, ranks):
    """Check the per-type factor masks and return them as float arrays."""
    salida = {}
    for tipo, permitido in (supervision or {}).items():
        if tipo not in sizes:
            raise ValueError(f"supervision given for unknown type {tipo!r}")
        permitido = np.asarray(permitido)
        esperado = (sizes[tipo], ranks[tipo])
        if permitido.shape != esperado:
            raise ValueError(
                f"supervision[{tipo!r}] has shape {permitido.shape}, expected {esperado}")
        vacias = ~permitido.any(axis=1)
        if vacias.any():
            raise ValueError(
                f"supervision[{tipo!r}] leaves {int(vacias.sum())} entities with no "
                "allowed component; every row needs at least one")
        salida[tipo] = permitido.astype(np.float64)
    return salida


def _split_signs(M):
    """Return (M_pos, M_neg) with M = M_pos - M_neg, both non negative."""
    pos = np.where(M > 0.0, M, 0.0)
    neg = np.where(M < 0.0, -M, 0.0)
    return pos, neg


def _split_theta(theta, sizes):
    """Split each legacy raw Laplacian into non negative parts."""
    salida = {}
    for tipo, T in theta.items():
        if tipo not in sizes:
            raise ValueError(f"theta given for unknown type {tipo!r}")
        if T.shape != (sizes[tipo], sizes[tipo]):
            raise ValueError(
                f"theta[{tipo!r}] has shape {T.shape}, expected "
                f"({sizes[tipo]}, {sizes[tipo]})")
        T = sp.csr_matrix(T)
        pos = sp.csr_matrix((np.maximum(T.data, 0.0), T.indices.copy(), T.indptr.copy()),
                            shape=T.shape)
        neg = sp.csr_matrix((np.maximum(-T.data, 0.0), T.indices.copy(), T.indptr.copy()),
                            shape=T.shape)
        salida[tipo] = (pos, neg)
    return salida


def _graph_laplacian(W):
    """Return (W_sym, degree) for the mass-preserving Laplacian.

    L = diag(W_sym 1) - W_sym with W_sym = D^-1/2 W D^-1/2. The constant
    vector lies in the kernel of this L for any degree distribution, so
    smoothing does not systematically shrink low-degree nodes. The usual
    I - W_sym does not have that property: its kernel is D^1/2 1, and it
    was measured shrinking boundary cells by 17.5%.
    """
    W = sp.csr_matrix(W)
    grado = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(grado, EPS))
    escala = sp.diags(inv_sqrt)
    W_sym = (escala @ W @ escala).tocsr()
    grado_sym = np.asarray(W_sym.sum(axis=1)).ravel()
    return W_sym, grado_sym


def _alpha_for(alpha, tipo):
    if isinstance(alpha, dict):
        return float(alpha.get(tipo, 0.0))
    return float(alpha)


def _any_nonzero(alpha):
    if isinstance(alpha, dict):
        return any(v > 0 for v in alpha.values())
    return alpha > 0


def _all_zero(alpha, graphs):
    return not any(_alpha_for(alpha, t) > 0 for t in graphs)


def _infer_sizes(relations):
    sizes = {}
    for nombre, relacion in relations.items():
        for tipo, n in ((relacion.src, relacion.shape[0]), (relacion.dst, relacion.shape[1])):
            if tipo in sizes and sizes[tipo] != n:
                raise ValueError(
                    f"type {tipo!r} has {sizes[tipo]} entities elsewhere but "
                    f"{n} in relation {nombre!r}")
            sizes[tipo] = n
    return sizes


def _frobenius_scales(relations):
    escalas = {}
    for nombre, relacion in relations.items():
        M, _ = relacion.observed()
        norma = float(np.sqrt(_norm_sq(M)))
        escalas[nombre] = 1.0 / norma if norma > 0 else 1.0
    return escalas


def _norm_sq(M):
    if sp.issparse(M):
        return float(np.square(M.data.astype(np.float64, copy=False)).sum())
    return float(np.square(np.asarray(M, dtype=np.float64)).sum())


def _observed_norm_sq(relacion, escala):
    M, _ = relacion.observed()
    return _norm_sq(M) * escala * escala


def _squared_error_scaled(M, G_src, S_r, G_dst, escala):
    """(||s M - G_i S G_j^T||^2, ||s M||^2), by the trace identity."""
    G_src = np.asarray(G_src, dtype=np.float64)
    G_dst = np.asarray(G_dst, dtype=np.float64)
    norm_sq = _norm_sq(M) * escala * escala
    middle = (G_src.T @ (M @ G_dst)) * escala
    gram_i = G_src.T @ G_src
    gram_j = G_dst.T @ G_dst
    cross = float(np.sum(middle * S_r))
    quad = float(np.sum((gram_i @ S_r @ gram_j) * S_r))
    return max(norm_sq - 2.0 * cross + quad, 0.0), norm_sq


def _loss_now(relations, G, S, scale, weight, norms_sq, estado):
    """Mean relative squared error over relations, by the trace identity."""
    total = 0.0
    for nombre, relacion in relations.items():
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        err_sq, norm_sq = _squared_error_scaled(M, G_src, S[nombre], G[relacion.dst],
                                                scale[nombre])
        total += err_sq / max(norm_sq, EPS)
    return float(total / len(relations))


def _initialize(relations, sizes, ranks, init, random_state, scale):
    """Build the legacy keyed dict the initializers expect.

    Masked rows are zeroed rather than dropped: dropping them would make
    the same type have different row counts across its relations, and
    keeping their content would leak into the initialization exactly what
    the mask is meant to hide.
    """
    R = {}
    for nombre, relacion in relations.items():
        M = relacion.matrix
        if relacion.rows is not None:
            indicador = np.zeros(M.shape[0])
            indicador[relacion.rows] = 1.0
            M = sp.diags(indicador) @ M if sp.issparse(M) else indicador[:, None] * M
        R.setdefault((relacion.src, relacion.dst), []).append(M)
    if init == "random":
        return init_random(R, sizes, ranks, random_state=random_state)
    if init == "nndsvd":
        return init_nndsvd(R, sizes, ranks)
    raise ValueError(f"unknown init {init!r}; supported: 'nndsvd', 'random'")


def _collect_labels(relations):
    etiquetas = {}
    for relacion in relations.values():
        if relacion.row_labels is not None:
            etiquetas.setdefault(relacion.src, np.asarray(relacion.row_labels))
        if relacion.col_labels is not None:
            etiquetas.setdefault(relacion.dst, np.asarray(relacion.col_labels))
    return etiquetas


def _rows_of_target(relations, target):
    tamanos = {r.shape[0] if r.src == target else r.shape[1] for r in relations.values()}
    if len(tamanos) != 1:
        raise ValueError(f"the new relations disagree on how many {target!r} entities "
                         f"there are: {sorted(tamanos)}")
    return tamanos.pop()


def _labels_of_target(relations, target):
    for relacion in relations.values():
        etiquetas = relacion.row_labels if relacion.src == target else relacion.col_labels
        if etiquetas is not None:
            return np.asarray(etiquetas)
    return None


def _known_length(known, fuentes, sizes):
    if known:
        longitudes = {len(np.asarray(v)) for v in known.values()}
        if len(longitudes) != 1:
            raise ValueError(f"the arrays in `known` have different lengths: {sorted(longitudes)}")
        return longitudes.pop()
    tamanos = {sizes[t] for t in fuentes.values()}
    if len(tamanos) != 1:
        raise ValueError("the views have different numbers of rows; pass `known`")
    return tamanos.pop()


def _known_rows(known, tipo, index, inicio, fin):
    if known and tipo in known:
        return np.asarray(known[tipo])[inicio:fin]
    return slice(inicio, fin)


def _combine_into(acumulado, v, combine):
    if combine == "geometric_mean":
        aporte = np.log(v)
    elif combine == "product":
        aporte = np.log(v)
    elif combine == "sum":
        aporte = v
    else:
        raise ValueError(f"unknown combine {combine!r}")
    return aporte if acumulado is None else acumulado + aporte


def _finish_combine(acumulado, combine, n_views):
    if combine == "geometric_mean":
        salida = np.exp(acumulado / n_views)
    elif combine == "product":
        salida = np.exp(acumulado)
    else:
        salida = acumulado
    return salida / salida.sum(axis=1, keepdims=True)


def _jsonable(v):
    return isinstance(v, (str, int, float, bool, type(None), list, dict))
