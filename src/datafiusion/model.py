"""Fitting API for sparse data fusion, built for relations that do not fit densely.

The entry point is `fuse`, which takes named relations and returns a
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
from .ops import product_at
from .preprocess import apply_chain, invert_chain, validate_names


# Python float en vez de escalar numpy: bajo NEP 50 un escalar np.float64
# subiria a float64 cualquier buffer float32 que toque.
EPS = float(np.finfo(np.float64).eps)
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
        Shape (n_src, n_dst). A sparse matrix that is not in canonical
        format is canonicalized with `sum_duplicates()`, which modifies
        the caller's matrix in place; a copy would be too expensive at
        the scale this library targets.
    rows : ndarray of int or bool, optional
        Which rows carry an observation. Rows outside the mask contribute
        nothing to the loss, which is what separates "no data" from "a
        measured zero". Without a mask every zero is an observation.
    entry_weights : ndarray, optional
        Weight per STORED entry, aligned with `matrix.data` after
        canonicalization. The loss term becomes
        sum_stored w_ab (target - r_ab)^2 + background * sum_rest r_ab^2.
        A weight of zero hides that entry from the fit entirely (its value
        does not enter the loss, the scaling or the initialization), which
        is what makes held-out entry validation possible. Weights above
        one read as importance (implicit-feedback style). Requires a
        sparse matrix and is not combinable with `rows` yet.
    background : float, optional
        Weight of the entries that are NOT stored, whose target is zero.
        1.0 (the default when `entry_weights` is given) keeps the usual
        semantics where every zero is an observation; 0.0 ignores
        unstored entries; values in between read zeros as weak negatives
        (Hu, Koren and Volinsky). Giving only `background` implies unit
        weights on the stored entries.
    preprocess : str or sequence of str, optional
        Named value transforms applied to the stored entries at fit
        time, in order, from a closed registry: "log1p", "sqrt",
        "anscombe" (shifted so 0 maps to 0) and "idf" (column scaling
        learned from the training data). The transforms are part of the
        model: their state travels with it, `FusionModel.transform` and
        `loss` reapply them to incoming raw data, and the caller's
        matrix is never modified. Requires a sparse matrix.
    family : {"gaussian", "poisson"}
        Loss family. "gaussian" is the squared Frobenius loss of the
        classic path. "poisson" fits the generalized KL divergence, the
        likelihood for count data; it requires non-negative sparse data
        and makes that relation's backbone S non-negative. Families can
        be mixed in one fit (a count relation next to gaussian ones,
        sharing factors); the balance between families has no natural
        unit, so their relative `weights` must be validated. Weighting a
        poisson relation by row or column (for example idf per term) is
        expressed by scaling the data: a column-weighted KL equals plain
        KL on the column-scaled matrix.
    row_labels, col_labels : ndarray, optional
        Entity labels. When present they are checked at fold-in time, so a
        column permutation raises instead of silently scoring nonsense.
    """

    src: str
    dst: str
    matrix: object
    rows: np.ndarray | None = None
    entry_weights: np.ndarray | None = None
    background: float | None = None
    family: str = "gaussian"
    preprocess: object = None
    row_labels: np.ndarray | None = None
    col_labels: np.ndarray | None = None

    def __post_init__(self):
        _check_name(self.src, "type")
        _check_name(self.dst, "type")
        if self.family not in ("gaussian", "poisson"):
            raise ValueError(
                f"unknown family {self.family!r}; supported: 'gaussian', 'poisson'")
        if self.preprocess is not None:
            if not sp.issparse(self.matrix):
                raise ValueError("preprocess requires a sparse matrix")
            self.preprocess = validate_names(self.preprocess)
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
            elif rows.size == 0:
                rows = rows.astype(np.int64)
            elif not np.issubdtype(rows.dtype, np.integer):
                raise ValueError(
                    f"row mask must be boolean or integer, got dtype {rows.dtype}")
            # La mascara es un conjunto de filas observadas: con duplicados,
            # la acumulacion por bloques daria un resultado dependiente de
            # block_rows.
            self.rows = np.unique(rows)
            if self.rows.size and (self.rows[-1] >= self.matrix.shape[0]
                                   or self.rows[0] < 0):
                raise ValueError("row mask indexes outside the matrix")
        if self.entry_weights is not None or self.background is not None:
            if not sp.issparse(self.matrix):
                raise ValueError("entry weights require a sparse matrix")
            if self.rows is not None:
                raise ValueError(
                    "entry_weights and rows on the same relation are not "
                    "supported; hide whole rows by giving their entries "
                    "weight zero")
            if self.entry_weights is None:
                self.entry_weights = np.ones(self.matrix.nnz)
            else:
                pesos = np.asarray(self.entry_weights, dtype=np.float64)
                if pesos.shape != (self.matrix.nnz,):
                    raise ValueError(
                        f"entry_weights must align with matrix.data: expected "
                        f"shape ({self.matrix.nnz},), got {pesos.shape}")
                if pesos.size and pesos.min() < 0:
                    raise ValueError("entry weights must be non negative")
                self.entry_weights = pesos
            self.background = 1.0 if self.background is None else float(self.background)
            if self.background < 0:
                raise ValueError("background weight must be non negative")
            tope = float(self.entry_weights.max()) if self.entry_weights.size else 0.0
            if max(tope, self.background) <= 0.0:
                raise ValueError(
                    "every entry weight and the background are zero; nothing "
                    "would enter the loss")

    @property
    def weighted(self):
        """Whether this relation carries per-entry weights."""
        return self.entry_weights is not None

    @property
    def shape(self):
        """Shape of the underlying matrix."""
        return self.matrix.shape

    def observed(self):
        """Matrix restricted to the observed rows, and those row indices."""
        if self.rows is None:
            return self.matrix, None
        return self.matrix[self.rows], self.rows


def _check_name(nombre, papel):
    """Reject names that save() could not use as a file name."""
    invalido = (not isinstance(nombre, str) or not nombre
                or nombre in (".", "..")
                or any(c in nombre for c in ("/", "\\", "\x00")))
    if invalido:
        raise ValueError(
            f"{papel} name {nombre!r} is used as a file name by save(), so it "
            "must be a non-empty str without '/', '\\' or NUL, and distinct "
            "from '.' and '..'; use '~' or '_' as a separator"
        )


def _as_relations(relations):
    """Accept named relations, or the legacy dict keyed by (src, dst)."""
    if all(isinstance(v, Relation) for v in relations.values()):
        for nombre in relations:
            _check_name(nombre, "relation")
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
            err_sq, norm_sq = _error_de(relacion, M, G_src, S_r,
                                        self.G[relacion.dst], s)
            por_relacion[nombre] = err_sq / norm_sq if relative else beta * err_sq
        if per_relation:
            return por_relacion
        valores = list(por_relacion.values())
        return float(np.mean(valores)) if relative else float(np.sum(valores))

    def _relations_for_loss(self, relations):
        if relations is None:
            raise ValueError("pass the relations the model was fitted on")
        return self._con_preprocesamiento(_as_relations(relations))

    def _con_preprocesamiento(self, relations):
        """Incoming raw relations with the fit-time transforms reapplied.

        The stateful transforms reuse the state learned at fit time (the
        stored idf), never one recomputed from the incoming batch: that
        is the whole point of carrying the preprocessing in the model.
        """
        cadenas = self.params.get("preprocess") or {}
        estados = self.params.get("idf") or {}
        salida = dict(relations)
        for nombre, relacion in relations.items():
            cadena = cadenas.get(nombre)
            propia = list(relacion.preprocess) if relacion.preprocess else None
            if propia is not None and propia != list(cadena or []):
                raise ValueError(
                    f"relation {nombre!r} declares preprocess {propia} but the "
                    f"model was fitted with {list(cadena) if cadena else None}; "
                    "pass the raw matrix and let the model apply its own")
            if not cadena:
                continue
            estado = ({"idf": np.asarray(estados[nombre])}
                      if nombre in estados else {})
            matriz, _ = apply_chain(relacion.matrix, tuple(cadena), estado=estado)
            salida[nombre] = Relation(
                src=relacion.src, dst=relacion.dst, matrix=matriz,
                rows=relacion.rows, entry_weights=relacion.entry_weights,
                background=relacion.background, family=relacion.family,
                row_labels=relacion.row_labels, col_labels=relacion.col_labels)
        return salida

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

        Relation.rows is honoured, with the same semantics as in `fuse`:
        it marks the observed rows of the MATRIX. When the source is
        `target` it says which new entities carry an observation in that
        relation, and each entity is solved from the relations it is
        observed in. When the destination is `target` it restricts the
        fitted side, and every new entity counts as observed there. New
        entities with no observation anywhere get a zero factor, are
        listed under `target` in the derived model's `empty_rows` and
        trigger a warning. The derived model keeps the parent's
        `empty_rows` for the other types.

        With nonneg=True the solution is refined with multiplicative
        updates from a strictly positive floor. The floor is mandatory: a
        multiplicative update cannot change a sign, so starting from the
        raw ridge yields NaN and starting from a hard clamp freezes the
        support at zero.
        """
        if self.params.get("family") in ("poisson", "mixed"):
            raise ValueError(
                "transform implements the quadratic ridge fold-in, which "
                "does not apply to a fit with poisson relations; not "
                "supported yet")
        relations = _as_relations(relations)
        self._validate_fold_in(relations, target)
        for nombre in relations:
            cadena = (self.params.get("preprocess") or {}).get(nombre) or ()
            if "idf" in cadena and self.rel[nombre][1] == target:
                raise ValueError(
                    f"relation {nombre!r} was fitted with idf over its "
                    f"columns; that state cannot apply to new {target!r} "
                    "entities on the column side")
        relations = self._con_preprocesamiento(relations)

        c = self.ranks[target]
        n_new = _rows_of_target(relations, target)

        nombres = list(relations)
        Q_por_relacion = []
        # Patron por entidad nueva: en que relaciones src-direccion esta
        # observada. Las relaciones sin mascara y las dst-direccion cuentan
        # para todas, asi que sin mascaras src no hace falta el patron y el
        # fold-in masivo no paga arreglos auxiliares de tamano n_new.
        con_mascara_src = any(
            relations[n].src == target and relations[n].rows is not None
            for n in nombres)
        observada = (np.ones((n_new, len(nombres)), dtype=bool)
                     if con_mascara_src else None)
        rhs = np.zeros((n_new, c))
        conteo = np.zeros(n_new, dtype=np.int64)

        for r, nombre in enumerate(nombres):
            relacion = relations[nombre]
            S_r = self.S[nombre]
            s, beta = self.scale[nombre], self.weight[nombre]
            M, filas = relacion.observed()
            if relacion.src == target:
                G_otro = self.G[relacion.dst]
                gram = G_otro.T @ G_otro
                Q_por_relacion.append(beta * (S_r @ gram @ S_r.T))
                if filas is not None:
                    observada[:, r] = False
                    observada[filas, r] = True
                por_entidad = M.getnnz(axis=1) if sp.issparse(M) \
                    else (np.asarray(M) != 0).sum(axis=1)
                n_obs = M.shape[0]
                for inicio in range(0, n_obs, block_rows):
                    bloque = slice(inicio, min(inicio + block_rows, n_obs))
                    aporte = (beta * s) * ((M[bloque] @ G_otro) @ S_r.T)
                    if filas is None:
                        rhs[bloque] += aporte
                    else:
                        rhs[filas[bloque]] += aporte
                if filas is None:
                    conteo += por_entidad
                else:
                    conteo[filas] += por_entidad
            else:
                G_otro = self.G[relacion.src]
                if filas is not None:
                    G_otro = G_otro[filas]
                gram = G_otro.T @ G_otro
                Q_por_relacion.append(beta * (S_r.T @ gram @ S_r))
                conteo += M.getnnz(axis=0) if sp.issparse(M) \
                    else (np.asarray(M) != 0).sum(axis=0)
                # M.T es una vista CSC y scipy multiplica CSC @ denso en
                # forma nativa, asi que no hace falta materializar la
                # transpuesta; asociar (G_otro @ S_r) primero deja el
                # intermedio mas grande en el tamano de rhs.
                rhs += (beta * s) * (M.T @ (G_otro @ S_r))

        if observada is None:
            Q = sum(Q_por_relacion)
            lam = alpha * max(np.trace(Q) / c, EPS)
            G_new = np.linalg.solve(Q + lam * np.eye(c), rhs.T).T
            if nonneg:
                G_new = nonneg_refine(G_new, Q, rhs, lam, max_iter=max_iter, tol=tol)
        else:
            patrones, inverso = np.unique(observada, axis=0, return_inverse=True)
            inverso = np.asarray(inverso).ravel()
            G_new = np.zeros((n_new, c))
            for k, patron in enumerate(patrones):
                if not patron.any():
                    continue
                idx = np.flatnonzero(inverso == k)
                Q_p = sum(Q_por_relacion[r] for r in np.flatnonzero(patron))
                lam_p = alpha * max(np.trace(Q_p) / c, EPS)
                rhs_p = rhs[idx]
                X_p = np.linalg.solve(Q_p + lam_p * np.eye(c), rhs_p.T).T
                if nonneg:
                    X_p = nonneg_refine(X_p, Q_p, rhs_p, lam_p,
                                        max_iter=max_iter, tol=tol)
                G_new[idx] = X_p

        sin_datos = np.flatnonzero(conteo == 0)
        if sin_datos.size:
            warnings.warn(
                f"transform: {sin_datos.size} new entities of type {target!r} "
                "have no observation in any relation. Their factors collapse "
                "to zero and argmax assigns all of them to component 0. See "
                "the derived model's empty_rows; drop them or handle them "
                "explicitly downstream.",
                stacklevel=2,
            )

        vacios = {t: v for t, v in self.empty_rows.items() if t != target}
        if sin_datos.size:
            vacios[target] = sin_datos

        derivado = FusionModel(
            G={**self.G, target: G_new}, S=self.S, rel=self.rel, ranks=self.ranks,
            sizes={**self.sizes, target: n_new}, scale=self.scale, weight=self.weight,
            index={**self.index, target: _labels_of_target(relations, target)},
            history=self.history, rel_error=self.rel_error, n_iter=self.n_iter,
            converged=self.converged, stop_reason=self.stop_reason,
            dead_columns=self.dead_columns, empty_rows=vacios,
            params={**self.params, "transformed": target},
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
            if relacion.weighted:
                raise ValueError(
                    f"relation {nombre!r} carries entry weights; transform "
                    "does not support them yet"
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

    def reconstruct_entries(self, name, rows, cols, original=False):
        """Reconstructed values of one relation at the given coordinates.

        The fit-time Frobenius scaling is undone, so the values come in
        the units the relation entered the fit with (after `preprocess`,
        if any). With original=True the preprocess chain is inverted as
        well; note that inverting a reconstruction of non-linear
        transforms is a readable approximation, not the conditional mean
        in original units. Cost is O(len(rows) * rank), nothing of
        relation size is materialized. This is how held-out entries are
        scored: hide them with `holdout_entries`, fit, then compare
        `reconstruct_entries` at their coordinates against their values.
        """
        if name not in self.rel:
            raise ValueError(f"unknown relation {name!r}; known: {sorted(self.rel)}")
        src, dst = self.rel[name]
        valores = product_at(self.G[src] @ self.S[name], self.G[dst],
                             np.asarray(rows), np.asarray(cols))
        valores = valores / self.scale[name]
        if original:
            cadena = (self.params.get("preprocess") or {}).get(name)
            if cadena:
                estados = self.params.get("idf") or {}
                estado = ({"idf": np.asarray(estados[name])}
                          if name in estados else {})
                valores = invert_chain(valores, cadena, estado, cols)
        return valores

    # ------------------------------------------------------------ persistencia

    def save(self, path):
        """Write the model as a directory of .npy files plus meta.json.

        A directory of .npy rather than a single .npz because .npz cannot
        be memory mapped: np.load(..., mmap_mode='r') on an .npz returns
        ordinary arrays, so a large model could not be opened without
        loading it whole.

        Supervision and mask arrays and graph matrices from `params` go
        to optional subdirectories; meta.json holds only what json can
        serialize.
        """
        path = Path(path)
        # Un directorio reutilizado no debe conservar artefactos de un modelo
        # anterior: load() los preferiria sobre el null de meta.json y
        # resume() reaplicaria en silencio una configuracion ajena.
        for carpeta in ("factors", "backbones", "supervision", "masks",
                        "graphs", "theta"):
            ruta = path / carpeta
            if ruta.is_dir():
                for archivo in list(ruta.glob("*.npy")) + list(ruta.glob("*.npz")):
                    archivo.unlink()
                if carpeta not in ("factors", "backbones"):
                    try:
                        ruta.rmdir()
                    except OSError:
                        pass
        (path / "factors").mkdir(parents=True, exist_ok=True)
        (path / "backbones").mkdir(parents=True, exist_ok=True)
        for tipo, factor in self.G.items():
            np.save(path / "factors" / f"{tipo}.npy", factor)
        for nombre, backbone in self.S.items():
            np.save(path / "backbones" / f"{nombre}.npy", backbone)
        np.save(path / "history.npy", self.history)
        for carpeta in ("supervision", "masks", "idf"):
            arrays = self.params.get(carpeta)
            if arrays:
                (path / carpeta).mkdir(exist_ok=True)
                for clave, arr in arrays.items():
                    np.save(path / carpeta / f"{clave}.npy", np.asarray(arr))
        for carpeta in ("graphs", "theta"):
            matrices = self.params.get(carpeta)
            if matrices:
                (path / carpeta).mkdir(exist_ok=True)
                for clave, W in matrices.items():
                    sp.save_npz(path / carpeta / f"{clave}.npz", sp.csr_matrix(W))
        meta = {
            "rel": {k: list(v) for k, v in self.rel.items()},
            "ranks": self.ranks, "sizes": self.sizes,
            "scale": self.scale, "weight": self.weight,
            "rel_error": self.rel_error, "n_iter": self.n_iter,
            "converged": self.converged, "stop_reason": self.stop_reason,
            "dead_columns": {k: np.asarray(v).tolist() for k, v in self.dead_columns.items()},
            "empty_rows": {k: np.asarray(v).tolist() for k, v in self.empty_rows.items()},
            "params": _params_json(self.params),
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
        params = dict(meta["params"])
        for carpeta in ("supervision", "masks", "idf"):
            if (path / carpeta).is_dir():
                arrays = {p.stem: np.load(p, mmap_mode=modo)
                          for p in sorted((path / carpeta).glob("*.npy"))}
                if arrays:
                    params[carpeta] = arrays
        for carpeta in ("graphs", "theta"):
            if (path / carpeta).is_dir():
                matrices = {p.stem: sp.load_npz(p)
                            for p in sorted((path / carpeta).glob("*.npz"))}
                if matrices:
                    params[carpeta] = matrices
        return cls(
            G=G, S=S, rel={k: tuple(v) for k, v in meta["rel"].items()},
            ranks=meta["ranks"], sizes=meta["sizes"], scale=meta["scale"],
            weight=meta["weight"], index={k: np.asarray(v) for k, v in meta["index"].items()},
            history=np.load(path / "history.npy"), rel_error=meta["rel_error"],
            n_iter=meta["n_iter"], converged=meta["converged"],
            stop_reason=meta["stop_reason"],
            dead_columns={k: np.asarray(v) for k, v in meta["dead_columns"].items()},
            empty_rows={k: np.asarray(v) for k, v in meta.get("empty_rows", {}).items()},
            params=params,
        )

    def resume(self, relations, max_iter, **overrides):
        """Continue the fit from the current factors.

        The stored hyperparameters (including masks, graphs and
        supervision) are reapplied, so resuming with the same relations
        reproduces the original scaling and configuration without the
        caller having to repeat them. `history` and `n_iter` accumulate.

        A model derived by `transform` cannot be resumed: its target
        factor belongs to the folded-in entities, not to the entities the
        fit was trained on.
        """
        if "transformed" in self.params:
            raise ValueError(
                "this model is a transform() projection over new entities; "
                "resume the original fitted model instead")
        params = {**self.params, **overrides, "max_iter": max_iter}
        # "family" es informativa (viaja en las relaciones) y el estado de
        # preprocesamiento se reaprende de las relaciones entregadas.
        for clave in ("n_runs", "run_losses", "best_run", "family",
                      "preprocess", "idf"):
            params.pop(clave, None)
        return fuse(relations, ranks=self.ranks, _warm_start=self, **params)


# --------------------------------------------------------------------------
# ajuste


def fuse(relations, ranks, weights=None, normalize="frobenius", masks=None,
        supervision=None,
        eta=1.0, gauge="column", lambda_S=1e-2, lambda_G=0.0,
        alpha_graph=0.0, graphs=None, theta=None,
        max_iter=100, tol=1e-5, init="nndsvd", random_state=None, n_runs=1,
        block_rows=200_000, device="cpu", verbose=0, callback=None,
        _warm_start=None):
    """Fit a data fusion model over named relations.

    The working precision follows the data: when every relation matrix is
    float32, the factors, backbones and iteration buffers are float32
    (half the memory and half the memory traffic of the sparse passes),
    while the loss and the rank-size solves still accumulate in float64.
    The loss then carries relative noise around 4e-5, so a `tol` below
    1e-4 sits under that floor and a warning says so. Any other dtype mix
    fits in float64.

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
    n_runs : int
        Random restarts. With n_runs > 1, `init` must be "random" and
        `random_state` must be an int or None; each run draws its seed
        from np.random.SeedSequence(random_state). The model with the
        lowest final loss is returned, with "run_losses" and "best_run"
        recorded in its params.
    block_rows : int
        Row block size for the sparse passes.
    device : {"cpu", "gpu"}
        Where the iteration loop runs. "gpu" uploads the relations once,
        runs the sparse passes through cuSPARSE and the factor algebra
        through cuBLAS (CuPy), and returns the fitted model in numpy, so
        nothing downstream changes. It needs the optional dependency
        `data-fiusion[gpu]` and a CUDA GPU; float32 data is the sensible
        choice on consumer GPUs, whose float64 units run at a fraction
        of the float32 rate. Not supported on GPU yet: poisson relations
        and entry weights. Under device="gpu" the callback receives
        device-resident factors.
    verbose : int
        Print the loss every `verbose` iterations.
    callback : callable, optional
        Called as callback(iteration, loss, model_state). Returning True
        stops the fit.

    Returns
    -------
    FusionModel
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"unknown device {device!r}; supported: 'cpu', 'gpu'")
    if n_runs > 1:
        if init != "random":
            raise ValueError(
                "n_runs > 1 requires init='random': 'nndsvd' is deterministic, "
                "so every run would return the same model")
        if _warm_start is not None:
            raise ValueError("n_runs > 1 cannot be combined with a warm start")
        if not (random_state is None or isinstance(random_state, (int, np.integer))):
            raise ValueError(
                "n_runs > 1 requires an int or None random_state, not a "
                "Generator: each run draws its own seed from SeedSequence")
        semillas = np.random.SeedSequence(random_state).spawn(n_runs)
        comunes = dict(
            weights=weights, normalize=normalize, masks=masks,
            supervision=supervision, eta=eta, gauge=gauge, lambda_S=lambda_S,
            lambda_G=lambda_G, alpha_graph=alpha_graph, graphs=graphs,
            theta=theta, max_iter=max_iter, tol=tol, init=init,
            block_rows=block_rows, device=device, verbose=verbose,
            callback=callback)
        # Solo se retiene la mejor corrida hasta el momento: mantener las k
        # a la vez multiplicaria por k la memoria de los factores.
        modelo, mejor, perdidas = None, 0, []
        for i in range(n_runs):
            corrida = fuse(relations, ranks, random_state=semillas[i], **comunes)
            perdidas.append(float(corrida.history[-1]))
            if modelo is None or perdidas[i] < perdidas[mejor]:
                modelo, mejor = corrida, i
        modelo.params.update(random_state=random_state, n_runs=n_runs,
                             run_losses=perdidas, best_run=mejor)
        return modelo

    relations = _as_relations(relations)
    if masks:
        for nombre, filas in masks.items():
            if nombre not in relations:
                raise ValueError(f"mask for unknown relation {nombre!r}")
            relations[nombre] = Relation(
                src=relations[nombre].src, dst=relations[nombre].dst,
                matrix=relations[nombre].matrix, rows=filas,
                entry_weights=relations[nombre].entry_weights,
                background=relations[nombre].background,
                family=relations[nombre].family,
                preprocess=relations[nombre].preprocess,
                row_labels=relations[nombre].row_labels,
                col_labels=relations[nombre].col_labels)

    # El preprocesamiento se aplica una vez aca: todo lo aguas abajo
    # (escala, inicializacion, perdida) ve la matriz transformada, y el
    # estado aprendido (idf) queda en el modelo para que transform y loss
    # lo reapliquen a datos nuevos.
    prep_nombres = {}
    prep_estado = {}
    for nombre in list(relations):
        relacion = relations[nombre]
        if relacion.preprocess is None:
            continue
        matriz, estado = apply_chain(relacion.matrix, relacion.preprocess)
        prep_nombres[nombre] = list(relacion.preprocess)
        if estado:
            prep_estado[nombre] = estado["idf"]
        relations[nombre] = Relation(
            src=relacion.src, dst=relacion.dst, matrix=matriz,
            rows=relacion.rows, entry_weights=relacion.entry_weights,
            background=relacion.background, family=relacion.family,
            row_labels=relacion.row_labels, col_labels=relacion.col_labels)
    extra_params = dict(preprocess=prep_nombres or None,
                        idf=prep_estado or None)

    if any(r.family == "poisson" for r in relations.values()):
        if graphs or theta or _any_nonzero(alpha_graph) or lambda_G > 0.0:
            raise ValueError(
                "graphs, theta, alpha_graph and lambda_G are not supported "
                "with family='poisson' yet")
        if device == "gpu":
            raise ValueError(
                "family='poisson' is not supported with device='gpu' yet; "
                "fit it on device='cpu'")
        from .poisson import fit_families
        return fit_families(
            relations, ranks, weights=weights, supervision=supervision,
            gauge=gauge, lambda_S=lambda_S, normalize=normalize,
            max_iter=max_iter, tol=tol, init=init, random_state=random_state,
            block_rows=block_rows, verbose=verbose, callback=callback,
            warm_start=_warm_start, extra_params=extra_params)

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
        detalle = ", ".join(f"{len(idx)} of type {tipo!r}" for tipo, idx in vacias.items())
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

    dtype = (np.float32
             if all(r.matrix.dtype == np.float32 for r in relations.values())
             else np.float64)
    if dtype == np.float32 and tol is not None and tol < 1e-4:
        warnings.warn(
            "with float32 data the loss carries relative noise around 4e-5, "
            "so a tol below 1e-4 sits under that floor. Pass tol>=1e-4, or "
            "None to disable the convergence check.",
            stacklevel=2)

    if _warm_start is not None:
        G = {t: np.array(f, dtype=dtype) for t, f in _warm_start.G.items()}
        historia_previa = list(np.asarray(_warm_start.history).ravel())
        iteraciones_previas = _warm_start.n_iter
    else:
        G = _initialize(relations, sizes, ranks, init, random_state, scale)
        G = {t: np.asarray(f, dtype=dtype) for t, f in G.items()}
        historia_previa, iteraciones_previas = [], 0

    supervision = _validate_supervision(supervision, sizes, ranks)
    supervision = {t: m.astype(dtype, copy=False) for t, m in supervision.items()}
    for tipo, permitido in supervision.items():
        G[tipo] = G[tipo] * permitido

    laplacianos = {t: _graph_laplacian(W) for t, W in graphs.items()}
    theta_split = _split_theta(theta or {}, sizes)

    norms_sq = {n: _observed_norm_sq(r, scale[n]) for n, r in relations.items()}

    # Con device="gpu" el loop trabaja sobre vistas residentes en la GPU;
    # todo lo demas (escalas, normas, etiquetas, params) usa las relaciones
    # originales en CPU, y el modelo final vuelve a numpy.
    relations_fit = relations
    if device == "gpu":
        from . import gpu as _gpu
        relations_fit = _gpu.convert_relations(relations)
        G = {t: _gpu.to_device(f) for t, f in G.items()}
        supervision = {t: _gpu.to_device(m) for t, m in supervision.items()}
        laplacianos = {t: (_gpu.to_device(W), _gpu.to_device(g))
                       for t, (W, g) in laplacianos.items()}
        theta_split = {t: (_gpu.to_device(Tp), _gpu.to_device(Tn))
                       for t, (Tp, Tn) in theta_split.items()}

    estado = _FitState(relations_fit, sizes, ranks, scale, weight, block_rows,
                       device=device)

    S = {}
    history = list(historia_previa)
    energia_previa = {t: None for t in G}
    razon = "max_iter"
    convergio = False
    iteracion = 0

    for iteracion in range(1, max_iter + 1):
        for nombre in relations_fit:
            S[nombre] = estado.solve_backbone(nombre, G, lambda_S,
                                              S_prev=S.get(nombre))

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
        perdida = _loss_now(relations_fit, G, S, scale, weight, norms_sq,
                            estado)
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
    for nombre in relations_fit:
        S[nombre] = estado.solve_backbone(nombre, G, lambda_S,
                                          S_prev=S.get(nombre))

    rel_error = {}
    for nombre, relacion in relations_fit.items():
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        err_sq, norm_sq = _error_de(relacion, M, G_src, S[nombre],
                                    G[relacion.dst], scale[nombre],
                                    estado.productos.get(nombre),
                                    norms_sq.get(nombre))
        rel_error[nombre] = float(np.sqrt(err_sq / max(norm_sq, EPS)))

    if device == "gpu":
        G = {t: _gpu.to_host(f) for t, f in G.items()}
        S = {n: _gpu.to_host(s) for n, s in S.items()}
        muertas = {t: _gpu.to_host(idx) for t, idx in muertas.items()}

    params = dict(
        weights=weights, normalize=normalize, eta=eta, gauge=gauge,
        supervision={t: m.astype(bool) for t, m in supervision.items()} or None,
        masks={n: np.asarray(f) for n, f in masks.items()} if masks else None,
        graphs=dict(graphs) or None, theta=dict(theta) if theta else None,
        lambda_S=lambda_S, lambda_G=lambda_G, alpha_graph=alpha_graph,
        max_iter=max_iter, tol=tol, init=init, random_state=random_state,
        block_rows=block_rows, device=device, **extra_params,
    )
    return FusionModel(
        G=G, S=S, rel={n: (r.src, r.dst) for n, r in relations.items()},
        ranks=dict(ranks), sizes=sizes, scale=scale, weight=weight,
        index=_collect_labels(relations),
        history=np.asarray(history), rel_error=rel_error,
        n_iter=iteraciones_previas + iteracion, converged=convergio,
        stop_reason=razon, dead_columns=muertas, empty_rows=vacias, params=params,
    )


# El nombre historico se mantiene como alias.
fit = fuse


# --------------------------------------------------------------------------
# maquinaria interna


class _FitState:
    """Holds the per-iteration work buffers and the Gram cache.

    The Gram of a type is cached per (type, mask) rather than per type:
    with a row mask the destination side needs the MASKED Gram, and using
    the unmasked one changes the resulting factor by 70% in relative norm.

    `productos` caches P = M @ G_dst per unweighted relation. The loss at
    the end of an iteration computes P with the factors it just updated
    and stores it; the backbone solve and the accumulation of the NEXT
    iteration see the same factors, so they reuse it instead of running
    their own sparse pass. This takes the sparse passes per iteration
    from four to two (the loss pass and the transposed pass in
    `accumulate`), at the cost of holding one (n_obs, c_dst) buffer per
    relation across the iteration boundary.
    """

    def __init__(self, relations, sizes, ranks, scale, weight, block_rows,
                 device="cpu"):
        self.relations = relations
        self.sizes = sizes
        self.ranks = ranks
        self.scale = scale
        self.weight = weight
        self.block_rows = max(1, int(block_rows))
        self.device = device
        # Pesos por entrada, normalizados para que el maximo sea 1: la
        # magnitud absoluta se pliega en beta, y con pesos acotados por 1
        # el punto fijo de S queda contraido. Una relacion cuyo peso
        # normalizado es uniforme e igual al fondo ES la perdida clasica
        # con beta * tope, y se enruta a la rama sin pesos.
        self.beta = dict(weight)
        self.productos = {}
        self.pesos = {}
        for nombre, relacion in relations.items():
            if not relacion.weighted:
                continue
            w = relacion.entry_weights
            w0 = relacion.background
            tope = max(float(w.max()) if w.size else 0.0, w0)
            if tope <= 0.0:
                raise ValueError(
                    f"relation {nombre!r} has every entry weight and the "
                    "background at zero; nothing would enter the loss")
            w_norm = w / tope
            w0_norm = w0 / tope
            self.beta[nombre] = weight[nombre] * tope
            delta = w_norm - w0_norm
            if w0_norm == 1.0 and not np.any(delta):
                continue
            M = relacion.matrix
            objetivo = sp.csr_matrix(
                (w_norm * (scale[nombre] * M.data), M.indices, M.indptr),
                shape=M.shape)
            # La reconstruccion solo se necesita donde delta != 0 (en el
            # regimen de entradas retenidas eso es una fraccion chica del
            # patron), asi que el SDDMM corre sobre ese subpatron.
            mascara = delta != 0.0
            filas_nnz = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
            filas_sub = filas_nnz[mascara]
            indptr_sub = np.concatenate(
                ([0], np.cumsum(np.bincount(filas_sub, minlength=M.shape[0]))))
            self.pesos[nombre] = dict(
                w0=w0_norm, tiene_delta=bool(mascara.any()), objetivo=objetivo,
                delta_sub=delta[mascara], filas_sub=filas_sub,
                columnas_sub=M.indices[mascara].copy(),
                indptr_sub=indptr_sub.astype(M.indptr.dtype))

    def _delta_por_reconstruccion(self, info, G_src, S_r, G_dst, shape):
        """The sparse matrix D.R, evaluated only on the support of delta."""
        R_sub = product_at(G_src @ S_r, G_dst, info["filas_sub"],
                           info["columnas_sub"])
        return sp.csr_matrix((info["delta_sub"] * R_sub, info["columnas_sub"],
                              info["indptr_sub"]), shape=shape)

    def _producto(self, nombre, M, G_dst):
        """P = M @ G_dst, reused from the loss of the previous iteration."""
        P = self.productos.get(nombre)
        if P is None:
            P = M @ G_dst
            self.productos[nombre] = P
        return P

    def _gram(self, cache, tipo, G, filas):
        if filas is None:
            clave = (tipo, None)
        elif isinstance(filas, np.ndarray):
            clave = (tipo, filas.tobytes())
        else:
            # En GPU la mascara vive en el dispositivo y es el mismo objeto
            # durante todo el ajuste; bajarla a bytes costaria una copia
            # host-device por consulta.
            clave = (tipo, id(filas))
        if clave not in cache:
            factor = G[tipo] if filas is None else G[tipo][filas]
            cache[clave] = factor.T @ factor
        return cache[clave]

    def solve_backbone(self, nombre, G, lambda_S, S_prev=None):
        """Closed-form S with Tikhonov, on the observed rows only.

        With entry weights there is no closed form: the update is one
        damped preconditioned step,

            S <- (1 - w0) S + (Gam_i + lam I)^{-1} G_i^T (WT - D.R) G_j (Gam_j + lam I)^{-1}

        whose fixed point satisfies the weighted normal equations, and
        which reduces EXACTLY to the closed form when the weights are
        uniform and the background is 1 (then D = 0 and w0 = 1). The
        weight normalization in __init__ bounds the weights by 1, which
        keeps this iteration contractive.
        """
        relacion = self.relations[nombre]
        info = self.pesos.get(nombre)
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        G_dst = G[relacion.dst]
        c_i, c_j = G_src.shape[1], G_dst.shape[1]
        xp = _xp(G_src)
        A = G_src.T @ G_src + lambda_S * xp.eye(c_i)
        B = G_dst.T @ G_dst + lambda_S * xp.eye(c_j)
        if info is None:
            middle = G_src.T @ self._producto(nombre, M, G_dst) * self.scale[nombre]
        elif S_prev is not None and info["tiene_delta"]:
            DR = self._delta_por_reconstruccion(info, G_src, S_prev, G_dst,
                                                M.shape)
            middle = G_src.T @ ((info["objetivo"] - DR) @ G_dst)
        else:
            middle = G_src.T @ (info["objetivo"] @ G_dst)
        X = np.linalg.solve(A, middle.astype(A.dtype, copy=False))
        # El solve corre en float64 (la identidad promueve A y B); el
        # resultado vuelve a la precision de trabajo de los factores.
        paso = np.linalg.solve(B.T, X.T).T.astype(G_dst.dtype, copy=False)
        if info is None or S_prev is None:
            return paso
        return (1.0 - info["w0"]) * S_prev + paso

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
            s, beta = self.scale[nombre], self.beta[nombre]
            M, filas = relacion.observed()
            src, dst = relacion.src, relacion.dst
            G_dst = G[dst]

            info = self.pesos.get(nombre)
            if info is not None:
                self._accumulate_weighted(N, D, G, S_r, relacion, info,
                                          beta, cache)
                continue

            gram_dst = self._gram(cache, dst, G, None)
            B_src = beta * (S_r @ gram_dst @ S_r.T)
            B_src_pos, B_src_neg = _split_signs(B_src)

            G_src_obs = G[src] if filas is None else G[src][filas]
            gram_src = self._gram(cache, src, G, filas)
            B_dst = beta * (S_r.T @ gram_src @ S_r)
            B_dst_pos, B_dst_neg = _split_signs(B_dst)

            n_obs = M.shape[0]
            P = self._producto(nombre, M, G_dst)
            if self.device == "gpu":
                # Sin bloques: el pico de memoria que el bloqueo acota en
                # CPU no gobierna en la GPU, y la pasada transpuesta va por
                # cusparse sin materializar la transpuesta.
                A_full = (beta * s) * (P @ S_r.T)
                A_pos, A_neg = _split_signs(A_full)
                aporte_N = A_pos + G_src_obs @ B_src_neg
                aporte_D = A_neg + G_src_obs @ B_src_pos
                if filas is None:
                    N[src] += aporte_N
                    D[src] += aporte_D
                else:
                    N[src][filas] += aporte_N
                    D[src][filas] += aporte_D
                acc_dst_pos = _gpu_spmm_transpose(M, G_src_obs)
            else:
                acc_dst_pos = np.zeros((self.sizes[dst], self.ranks[src]),
                                       dtype=G_dst.dtype)
                for inicio in range(0, n_obs, self.block_rows):
                    bloque = slice(inicio, min(inicio + self.block_rows, n_obs))
                    M_b = M[bloque]
                    G_src_b = G_src_obs[bloque]

                    A_b = (beta * s) * (P[bloque] @ S_r.T)
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

        energia = {t: float(np.sum(G[t] * D[t], dtype=np.float64)) for t in G}
        return N, D, energia

    def _accumulate_weighted(self, N, D, G, S_r, relacion, info, beta, cache):
        """Contribution of a relation with entry weights.

        Gradient over 2*beta: w0 G_i B + C - A, with A from the weighted
        target, C from the delta weights on the reconstruction (the SDDMM)
        and B the quadratic backbone term. The sign split keeps G_i [B]_+
        in the denominator at FULL weight, not scaled by w0: that is the
        positive floor that the naive split loses, and losing it diverges
        to NaN when the background is zero. With v = (1 - w0) G_i B - C,

            N_i += [A]_+ + G_i [B]_- + [v]_+
            D_i += [A]_- + G_i [B]_+ + [v]_-

        and symmetrically for the dst side.
        """
        src, dst = relacion.src, relacion.dst
        G_src, G_dst = G[src], G[dst]
        M = relacion.matrix
        objetivo = info["objetivo"]
        w0 = info["w0"]

        gram_src = self._gram(cache, src, G, None)
        gram_dst = self._gram(cache, dst, G, None)
        B_i = beta * (S_r @ gram_dst @ S_r.T)
        B_j = beta * (S_r.T @ gram_src @ S_r)
        A_i = beta * ((objetivo @ G_dst) @ S_r.T)
        A_j = beta * ((objetivo.T @ G_src) @ S_r)
        if info["tiene_delta"]:
            DR = self._delta_por_reconstruccion(info, G_src, S_r, G_dst,
                                                M.shape)
            C_i = beta * ((DR @ G_dst) @ S_r.T)
            C_j = beta * ((DR.T @ G_src) @ S_r)
        else:
            C_i = C_j = 0.0
        v_i = (1.0 - w0) * (G_src @ B_i) - C_i
        v_j = (1.0 - w0) * (G_dst @ B_j) - C_j

        for tipo, factor, A_t, B_t, v_t in ((src, G_src, A_i, B_i, v_i),
                                            (dst, G_dst, A_j, B_j, v_j)):
            A_pos, A_neg = _split_signs(A_t)
            B_pos, B_neg = _split_signs(B_t)
            v_pos, v_neg = _split_signs(v_t)
            N[tipo] += A_pos + factor @ B_neg + v_pos
            D[tipo] += A_neg + factor @ B_pos + v_neg


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


def _xp(a):
    """numpy or cupy, whichever module holds the array."""
    if isinstance(a, np.ndarray):
        return np
    import cupy
    return cupy


def _gpu_spmm_transpose(M, B):
    from .gpu import spmm_transpose
    return spmm_transpose(M, B)


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
        if relacion.weighted:
            # Las entradas con peso cero (retenidas) no definen la escala:
            # su valor no debe influir en el ajuste por ninguna via.
            datos = relacion.matrix.data[relacion.entry_weights > 0]
            norma = float(np.sqrt(np.square(
                datos.astype(np.float64, copy=False)).sum()))
        else:
            M, _ = relacion.observed()
            norma = float(np.sqrt(_norm_sq(M)))
        escalas[nombre] = 1.0 / norma if norma > 0 else 1.0
    return escalas


def _norm_sq(M):
    # hasattr(indptr) en vez de sp.issparse para cubrir tambien las
    # matrices de cupyx.scipy.sparse.
    datos = M.data if hasattr(M, "indptr") else M
    return float(np.square(datos.astype(np.float64, copy=False)).sum())


def _observed_norm_sq(relacion, escala):
    M, _ = relacion.observed()
    return _norm_sq(M) * escala * escala


def _error_de(relacion, M, G_src, S_r, G_dst, escala, producto=None,
              norm_sq=None):
    """(squared error, squared norm) of one relation, weighted-aware."""
    if relacion.weighted:
        return _weighted_squared_error(relacion, G_src, S_r, G_dst, escala)
    return _squared_error_scaled(M, G_src, S_r, G_dst, escala, producto,
                                 norm_sq)


def _weighted_squared_error(relacion, G_src, S_r, G_dst, escala):
    """(weighted squared error, weighted squared norm of the target).

    Expanding sum_stored w (t - R)^2 + w0 sum_rest R^2 leaves ||R||^2 by
    the trace identity, a cross term through the sparse weighted target,
    and a quadratic correction that needs R only where w differs from
    w0. In the held-out regime that support is a small fraction of the
    pattern, so the expensive gather stays small. Entries with weight
    zero contribute nothing, so their values cannot move the error.
    """
    M = relacion.matrix
    w = relacion.entry_weights
    w0 = relacion.background
    G_src = np.asarray(G_src, dtype=np.float64)
    G_dst = np.asarray(G_dst, dtype=np.float64)
    objetivo = escala * M.data
    ponderado = sp.csr_matrix((w * objetivo, M.indices, M.indptr), shape=M.shape)
    cross = float(np.sum((G_src.T @ (ponderado @ G_dst)) * S_r))
    gram_i = G_src.T @ G_src
    gram_j = G_dst.T @ G_dst
    total_R2 = float(np.sum((gram_i @ S_r @ gram_j) * S_r))
    delta = w - w0
    mascara = delta != 0.0
    if mascara.any():
        filas = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))[mascara]
        R_sub = product_at(G_src @ S_r, G_dst, filas, M.indices[mascara])
        correccion = float(np.sum(delta[mascara] * np.square(R_sub)))
    else:
        correccion = 0.0
    err = (w0 * total_R2 + float(np.sum(w * np.square(objetivo)))
           - 2.0 * cross + correccion)
    norm = float(np.sum(w * np.square(objetivo)))
    return max(err, 0.0), max(norm, EPS)


def _squared_error_scaled(M, G_src, S_r, G_dst, escala, producto=None,
                          norm_sq=None):
    """(||s M - G_i S G_j^T||^2, ||s M||^2), by the trace identity."""
    G_src = G_src.astype(np.float64, copy=False)
    G_dst = G_dst.astype(np.float64, copy=False)
    if norm_sq is None:
        norm_sq = _norm_sq(M) * escala * escala
    if producto is None:
        producto = M @ G_dst
    middle = (G_src.T @ producto) * escala
    gram_i = G_src.T @ G_src
    gram_j = G_dst.T @ G_dst
    cross = float(np.sum(middle * S_r))
    quad = float(np.sum((gram_i @ S_r @ gram_j) * S_r))
    return max(norm_sq - 2.0 * cross + quad, 0.0), norm_sq


def _loss_now(relations, G, S, scale, weight, norms_sq, estado):
    """Mean relative squared error over relations, by the trace identity.

    For unweighted relations the sparse product M @ G_dst is computed
    here with the factors of the CURRENT iterate and left in the fit
    state, where the next iteration reuses it.
    """
    total = 0.0
    for nombre, relacion in relations.items():
        M, filas = relacion.observed()
        G_src = G[relacion.src] if filas is None else G[relacion.src][filas]
        producto = None
        if not relacion.weighted:
            producto = M @ G[relacion.dst]
            estado.productos[nombre] = producto
        else:
            # La rama con pesos no pasa por aca, pero una relacion de pesos
            # uniformes se enruta a la rama clasica y su producto cacheado
            # quedo calculado con los factores anteriores: se invalida para
            # que el solve siguiente lo recalcule.
            estado.productos.pop(nombre, None)
        err_sq, norm_sq = _error_de(relacion, M, G_src, S[nombre],
                                    G[relacion.dst], scale[nombre], producto,
                                    norms_sq.get(nombre))
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
        elif relacion.weighted and np.any(relacion.entry_weights == 0.0):
            # Las entradas con peso cero se ocultan tambien de la
            # inicializacion, o su valor se filtraria al punto de partida.
            M = sp.csr_matrix(
                (M.data * (relacion.entry_weights != 0.0), M.indices, M.indptr),
                shape=M.shape)
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
    """Whether json.dumps can serialize v. Arrays and sparse matrices cannot."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return True
    if isinstance(v, dict):
        return all(isinstance(k, str) and _jsonable(x) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return all(_jsonable(x) for x in v)
    return False


def _native(v):
    """Numpy scalars as native Python scalars, recursively.

    np.float32 and np.int64 are not subclasses of float and int, so
    without this a params entry like weights={'r': np.float32(30)} would
    be dropped from meta.json in silence and resume after load would
    revert the hyperparameter to its default.
    """
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    return v


def _params_json(params):
    """Params as meta.json stores them: numpy scalars converted, the rest
    filtered to what json accepts (arrays and sparse go to subdirectories)."""
    salida = {}
    for clave, valor in params.items():
        valor = _native(valor)
        if _jsonable(valor):
            salida[clave] = valor
    return salida
