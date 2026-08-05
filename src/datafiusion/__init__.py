"""Sparse data fusion by non-negative matrix tri-factorization.

Two entry points coexist.

`fuse` is the current one (`fit` remains as an alias). It takes named
relations, returns a
`FusionModel` that carries the fit-time scaling along with the factors,
and offers row observation masks, a column gauge, graph smoothing
calibrated against the data, a loss trace and a stopping tolerance. Its
loss never materializes a matrix of relation size, so it is the one to
use on relations that do not fit densely.

`dfmf_sparse` is the legacy one, frozen. Its numerical behaviour is
pinned by tests/test_golden.py and does not change, so results produced
with it stay reproducible. New work should use `fuse`.
"""

from .base import DataFusionModel
from .core import dfmf_sparse, reconstruction_error, fold_in_entities, predict_attribute
from .model import FusionModel, Relation, fit, fuse
from .init import init_random, init_nndsvd
from .ops import product_at, sddmm
from .utils import (
    ensure_columns, ensure_index, holdout_entries, labels_to_relation,
    merge_relations, normalize_relations,
)

__all__ = [
    "fuse", "fit", "FusionModel", "Relation",
    "dfmf_sparse", "reconstruction_error", "fold_in_entities", "predict_attribute",
    "DataFusionModel",
    "init_random", "init_nndsvd",
    "sddmm", "product_at", "holdout_entries",
    "ensure_columns", "ensure_index", "labels_to_relation",
    "merge_relations", "normalize_relations",
]
