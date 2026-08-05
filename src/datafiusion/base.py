"""DataFrame-friendly wrapper around dfmf_sparse.

The wrapper resolves index coherence across relations: when a type appears
in several relations with different indices, it takes the union (preserving
first appearance) and reindexes each DataFrame to that order, filling
missing cells with `fill_value` (default 0). Aligned matrices are then
converted to scipy.sparse.csr before factorization.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import dfmf_sparse, reconstruction_error


def _as_dataframe(matrix):
    if isinstance(matrix, pd.DataFrame):
        return matrix
    if sp.issparse(matrix):
        return pd.DataFrame(matrix.toarray())
    return pd.DataFrame(np.asarray(matrix))


def _union_preserving_order(indices):
    combined = pd.Index(indices[0])
    for nxt in indices[1:]:
        nxt = pd.Index(nxt)
        combined = combined.append(nxt.difference(combined))
    return combined


class DataFusionModel:
    """Non-negative matrix tri-factorization across heterogeneous DataFrames.

    Parameters
    ----------
    nodes : dict[str, int]
        Rank c_i per object type.
    relations : dict[(str, str), list[pd.DataFrame or scipy.sparse]]
        Relation matrices keyed by (src_type, dst_type). DataFrame index
        labels the rows (src entities), columns label dst entities. Sparse
        inputs must already be aligned with the canonical index order of
        their types.
    laplacians : dict[str, pd.DataFrame or scipy.sparse], optional
        Graph-Laplacian constraints per type (typically D - W).
    lambda_G, lambda_S : float
        L2 regularization weights.
    fill_value : float
        Value used to fill missing entries when reindexing to the union of
        indices. Default 0.
    init : {"random"}
        Initialization strategy.
    max_iter : int
        Number of factorization iterations.
    random_state : int or None
    verbose : int
        Print reconstruction error every `verbose` iterations (0 disables).
    """

    def __init__(
        self,
        nodes,
        relations,
        laplacians=None,
        lambda_G=0.0,
        lambda_S=0.0,
        fill_value=0,
        init="random",
        max_iter=100,
        random_state=666,
        verbose=0,
    ):
        self.nodes = dict(nodes)
        self.relation_definitions = dict(relations)
        self.laplacian_definitions = dict(laplacians) if laplacians else {}
        self.lambda_G = lambda_G
        self.lambda_S = lambda_S
        self.fill_value = fill_value
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self._validate_inputs()

    def _validate_inputs(self):
        for pair in self.relation_definitions:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Relation key {pair!r} must be a (src, dst) tuple")
            src, dst = pair
            if src == dst:
                raise ValueError(
                    f"Self-relation ({src!r}, {dst!r}) not supported. "
                    "Use the `laplacians` argument for type-level constraints."
                )
            for t in (src, dst):
                if t not in self.nodes:
                    raise ValueError(
                        f"Type {t!r} used in relations but missing from nodes"
                    )
        for t in self.laplacian_definitions:
            if t not in self.nodes:
                raise ValueError(f"Type {t!r} used in laplacians but missing from nodes")

    def _resolve_indices(self):
        per_type = {t: [] for t in self.nodes}
        for (src, dst), entries in self.relation_definitions.items():
            for entry in entries:
                if isinstance(entry, pd.DataFrame):
                    per_type[src].append(entry.index)
                    per_type[dst].append(entry.columns)
        for t, L in self.laplacian_definitions.items():
            if isinstance(L, pd.DataFrame):
                per_type[t].append(L.index)
                per_type[t].append(L.columns)

        indices = {}
        for t, idx_list in per_type.items():
            if not idx_list:
                raise ValueError(
                    f"Type {t!r} declared in nodes but absent from every "
                    "DataFrame relation. Pass at least one DataFrame "
                    "referencing it so the wrapper can resolve its index."
                )
            indices[t] = _union_preserving_order(idx_list)
        return indices

    def _build_sparse(self, indices):
        R_sparse = {}
        for (src, dst), entries in self.relation_definitions.items():
            mats = []
            for entry in entries:
                df = _as_dataframe(entry)
                aligned = (
                    df.reindex(index=indices[src], columns=indices[dst])
                    .fillna(self.fill_value)
                    .astype(np.float64)
                )
                mats.append(sp.csr_matrix(aligned.values))
            R_sparse[(src, dst)] = mats

        theta_sparse = {}
        for t, L in self.laplacian_definitions.items():
            n = len(indices[t])
            if isinstance(L, pd.DataFrame):
                aligned = (
                    L.reindex(index=indices[t], columns=indices[t])
                    .fillna(0.0)
                    .astype(np.float64)
                )
                theta_sparse[t] = sp.csr_matrix(aligned.values)
            elif sp.issparse(L):
                if L.shape != (n, n):
                    raise ValueError(
                        f"laplacians[{t!r}] has shape {L.shape}, expected ({n}, {n})"
                    )
                theta_sparse[t] = sp.csr_matrix(L)
            else:
                arr = np.asarray(L, dtype=np.float64)
                if arr.shape != (n, n):
                    raise ValueError(
                        f"laplacians[{t!r}] has shape {arr.shape}, expected ({n}, {n})"
                    )
                theta_sparse[t] = sp.csr_matrix(arr)
        return R_sparse, theta_sparse

    def fit(self):
        self.indices = self._resolve_indices()
        self.R_, self.theta_ = self._build_sparse(self.indices)
        self.G_, self.S_ = dfmf_sparse(
            R=self.R_,
            ranks=self.nodes,
            theta=self.theta_,
            lambda_G=self.lambda_G,
            lambda_S=self.lambda_S,
            max_iter=self.max_iter,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        return self

    def factor(self, type_name, return_dataframe=True):
        G = self.G_[type_name]
        if not return_dataframe:
            return G
        columns = [f"C{i:02}" for i in range(G.shape[1])]
        return pd.DataFrame(G, index=self.indices[type_name], columns=columns)

    def backbone(self, src, dst, idx=0, return_dataframe=True):
        S_ij = self.S_[(src, dst)][idx]
        if not return_dataframe:
            return S_ij
        return pd.DataFrame(
            S_ij,
            index=[f"{src}_C{i:02}" for i in range(S_ij.shape[0])],
            columns=[f"{dst}_C{j:02}" for j in range(S_ij.shape[1])],
        )

    def reconstruct(self, src, dst, idx=0, return_dataframe=True):
        values = self.G_[src] @ self.S_[(src, dst)][idx] @ self.G_[dst].T
        if not return_dataframe:
            return values
        return pd.DataFrame(
            values, index=self.indices[src], columns=self.indices[dst]
        )

    def chain(self, src, dst):
        if src == dst:
            yield [src]
            return
        paths = [[src]]
        while paths:
            new_paths = []
            for path in paths:
                last = path[-1]
                for (a, b) in self.relation_definitions:
                    if a == last and b not in path:
                        extended = path + [b]
                        if b == dst:
                            yield extended
                        else:
                            new_paths.append(extended)
            paths = new_paths

    def relation_profiles(self, src, dst, updated_factors=None, index=None):
        if updated_factors is None:
            updated_factors = {}
        if index is None:
            index = self.indices[src]

        profiles = []
        for path in self.chain(src, dst):
            result = updated_factors.get(path[0], self.G_[path[0]])
            for a, b in zip(path, path[1:]):
                result = result @ self.S_[(a, b)][0]
            end_factor = updated_factors.get(path[-1], self.G_[path[-1]])
            result = result @ end_factor.T
            profiles.append(
                (path, pd.DataFrame(result, index=index, columns=self.indices[dst]))
            )
        return profiles

    def reconstruction_error(self):
        return reconstruction_error(self.R_, self.G_, self.S_)
