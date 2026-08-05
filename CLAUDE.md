# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`data-fiusion` is a sparse rewrite of the core algorithm of `scikit-fusion` (Marinka Zitnik's data fusion library). It performs non-negative matrix tri-factorization across heterogeneous relations, keeps all relation matrices as `scipy.sparse`, and uses dense low-rank factors. The upstream library at `~/repositories/scikit-fusion` densifies relations everywhere, which makes it unusable for the project's typical inputs (relations with millions of rows but a few percent density).

The package name on disk is `datafiusion` (no hyphen). The distribution name is `data-fiusion`. The API is organized around four patterns: joint factorization of several relations, per-relation weighting, fold-in of new entities without refitting, and attribute prediction from reconstructed views.

## Commands

The project uses `uv` and ships a `pyproject.toml` plus a populated `.venv` with `data-fiusion` already installed in editable mode and `scikit-fusion` available as a dev dependency for parity tests. Run everything through `uv run` from this directory; it picks up the project's `.venv` automatically.

- Parity test against scikit-fusion: `uv run python tests/test_compare_skfusion.py`. Runs `dfmf_sparse` with both `random` and `nndsvd` initialization, runs scikit-fusion's `Dfmf` on the same synthetic data, prints convergence traces.
- Toy DataFrame wrapper: `uv run python examples/toy_sparse.py`. Exercises `DataFusionModel` end to end with DataFrames and a Laplacian.
- Semi-supervised labels: `uv run python examples/semi_supervised.py`. Documents the labels-as-relation pattern and its limitations.
- Newsgroups counts baseline: `uv run python examples/newsgroups/conteos.py`. Raw counts, sqrt, log1p, shifted Anscombe and TF-IDF under the quadratic loss; the reference any count likelihood has to beat. About 15 minutes.
- Poisson against that baseline: `uv run python examples/newsgroups/poisson_vs_cuadratica.py`. KL on raw counts, on idf-scaled columns and on TF-IDF values, with the 2-SE verdict against the quadratic TF-IDF reference printed at the end.
- Mixed families on Last.fm: `uv run python examples/lastfm/comparacion.py`. Downloads HetRec 2011 Last.fm on first run (2.5 MB, non-commercial license). Poisson listening counts plus masked gaussian tags against the all-gaussian arm, paired over 4 seeds with the 2-SE verdict printed.
- MovieLens benchmark, held-out genre prediction: `uv run python examples/movielens/prediccion.py`. Baselines, hyperparameter sweep on validation, final test evaluation. About three minutes.
- MovieLens benchmark, cost against scikit-fusion: `uv run python examples/movielens/escala.py`. Sweeps rows and columns, runs each configuration in its own subprocess under a memory guard.
- Install or reinstall scikit-fusion into the project venv: `uv pip install -e <path to a scikit-fusion checkout> --python .venv/bin/python`.

Avoid `uv run --with .` from outside this repo: it builds a wheel and caches it under `~/.cache/uv/archive-v0/<hash>/`. Edits to the source then look like they had no effect because the cached wheel is served until `--reinstall-package data-fiusion` is passed. From inside the repo, plain `uv run` resolves through the project `.venv` (editable) and avoids the cache entirely.

## Architecture

Four layers, with the API consolidated around patterns from real use.

1. **Numerical core (`core.py`).** `dfmf_sparse(R, ranks, theta, lambda_G, lambda_S, max_iter, init, random_state, verbose)`. Takes `R: dict[(src, dst), list[sparse]]` and `ranks: dict[str, int]`, returns `(G, S)` with `G[t]: (n_t, c_t)` dense ndarrays and `S[(src, dst)]` a list of `(c_src, c_dst)` backbones (one per matrix in `R[(src, dst)]`). Self-relations (`src == dst`) are rejected; type-level structure goes through `theta`.
   - `_solve_S` uses closed-form Tikhonov: `S = (G_i^T G_i + lambda I)^{-1} G_i^T R G_j (G_j^T G_j + lambda I)^{-1}`. Equivalent to scikit-fusion's `pinv(G^T G)` at `lambda=0`.
   - `reconstruction_error(R, G, S)` is the sum of per-relation relative Frobenius errors.
   - `fold_in_entities(R_new, G, S, target_type, lambda_reg)` projects new rows into `G[target_type]` space via ridge least squares without refitting. Vectorized.
   - `predict_attribute(known, G, S, target_type, view_keys, combine, eps)` predicts a soft distribution over a target type by combining several reconstructed views (geometric mean is the default; product is hard Naive Bayes; sum is additive evidence).

2. **Initialization (`init.py`).** Two strategies.
   - `init_random`: uniform `[0, 1)` per entry, seeded by `np.random.default_rng(random_state)`.
   - `init_nndsvd`: NNDSVDa adapted to multi-relation factorization (Boutsidis and Gallopoulos 2008). For each type, concatenates every relation where the type participates, normalizes each block by Frobenius, takes a truncated SVD, applies NNDSVDa to the left singular vectors. Below-eps entries are replaced by the absolute mean of the concatenated matrix.
   - `_truncated_svd_left` passes a deterministic `v0` to `scipy.sparse.linalg.svds` and runs `_svd_flip_u` on the returned `U` to pin the sign of each column. Both are necessary: ARPACK seeds from global numpy state and singular vectors are defined up to sign, so without these pins NNDSVD output differs depending on whether scikit-fusion was imported earlier in the process.

3. **Data prep utilities (`utils.py`).** Two helpers extracted from multi-source pipelines.
   - `merge_relations(*dicts)`: concatenates lists of matrices per relation key across multiple builder dicts. Used to combine per-source builders into one final `R`.
   - `normalize_relations(R, weights=None)`: Frobenius-normalizes every matrix; optional per-relation scalar multiplier applied after. Returns `(R_normalized, scales)`. The scales are kept so reconstructions can be rescaled to absolute counts if needed.
   - `labels_to_relation(labels, classes, fill)`: one-hot encodes a `Series` of labels into a relation DataFrame. Handles NaN as `fill`.
   - `ensure_columns`, `ensure_index`: smaller pandas helpers.

4. **DataFrame wrapper (`base.py`).** `DataFusionModel` resolves index coherence across DataFrames per type (union, fill with `fill_value`), exposes `factor`, `backbone`, `reconstruct`, `chain`, `relation_profiles` returning DataFrames. Useful for pandas-first workflows, but it densifies each relation to align indices, so it must not be used on large data; operate on `Relation` objects or on `dict[(src, dst), list[sparse]]` directly instead.

5. **Diagram (`diagram.py`).** `fusion_diagram(source, ranks=None, positions=None)` draws the publication-oriented schema of a fusion setup: entity types as boxes (size and rank), relations as arrows annotated with dimensions, density, preprocess and masks, colored by family (Okabe-Ito). Accepts named relations, the legacy keyed dict or a `FusionModel`. Optional, depends only on `matplotlib`. Example figure: `examples/lastfm/diagrama.py`.

6. **Sparse primitives (`ops.py`).** `sddmm(pattern, A, B)` evaluates A @ B^T at the stored entries of a sparse pattern in O(nnz * c), blocked; `product_at(A, B, rows, cols)` is the same kernel on coordinate lists. They are what makes entry weights and count likelihoods expressible without materializing anything of relation size.

7. **Named value transforms (`preprocess.py`).** `Relation(preprocess=...)` names transforms from a closed registry ("log1p", "sqrt", shifted "anscombe", "idf"); `fuse` applies them once (learning the idf vector from the training data), stores names in `params["preprocess"]` and state in `params["idf"]` (persisted by `save` like supervision and masks), and `FusionModel.transform` and `loss` reapply the SAME chain with the SAME state to incoming raw data. Closed registry on purpose: callables cannot persist. The caller's matrix is never modified (only the data array is copied). An incoming relation declaring a different chain than the fit raises; idf cannot apply when the new entities are on the column side; `reconstruct_entries(original=True)` inverts the chain. All transforms map 0 to 0, so patterns and `entry_weights` alignment survive.

8. **Count and mixed families (`poisson.py`).** `fuse` dispatches to `fit_families` when any relation carries `family="poisson"`. Poisson relations contribute generalized KL (S non-negative, multiplicative; the column gauge is compensated in S); gaussian relations contribute through the same `_FitState` machinery as the classic path, including row masks and entry weights. The shared factors receive ONE joint update: the positive root of the sum of both auxiliary majorizers, which reduces exactly to the classic ratio without poisson terms and to the KL rule without gaussian terms (derivation in the module docstring). The loss is the mean of normalized per-relation terms (relative squared error or deviance ratio). The balance BETWEEN families has no natural unit: relative `weights` must be validated. Not supported with poisson relations yet: `transform`, row masks and entry weights on the poisson relation itself, graphs; column or row weighting is expressed by scaling the data. `resume` works; `params["family"]` ("poisson" or "mixed") is informational and popped by resume.

## Use patterns documented in `README.md`

The README covers these end-to-end. When extending or debugging, mirror the structure:

1. **Single-source factorization**: just `dfmf_sparse(R, ranks, ...)` with a dict of sparse matrices.
2. **Multi-source fusion**: one builder per source returning a relation dict, `merge_relations(*dicts)` to combine, `normalize_relations(R, weights={...})` to balance.
3. **Fold-in of new entities**: `fold_in_entities(R_new, G, S, target_type)` after a fit on a smaller sample. Used to scale to millions of users.
4. **Attribute prediction**: `predict_attribute(known, G, S, target_type, view_keys)` for entities whose attribute is missing in the source.
5. **Importance sampling weights** (cookbook only, no API): compute `w(hora, dia) = p_target(hora, dia) / p_observed(hora, dia)` per trip, optionally combine with a model-derived quality weight (`exp(-distance_to_centroid)`).

## Patterns to follow when extending

- New algorithms: pair an internal core (plain dict in / dict out) with a public wrapper. Reuse `init_random` and `init_nndsvd` for initialization if applicable.
- New initialization strategies live in `init.py`. Dispatch via the `init` argument in `dfmf_sparse`.
- New post-fit utilities (anything operating on `(G, S)`) belong in `core.py`. Keep them stateless and vectorized.
- Data prep helpers (operating on raw relations) live in `utils.py`.
- If you touch the SVD path, keep `_svd_flip_u` applied to the returned `U`. The reproducibility test in `tests/test_compare_skfusion.py` will silently give slow convergence if signs flip across processes.
- The closed-form `_solve_S` is the unconstrained least-squares solution; if you add non-negativity to `S` later, swap the solve for a multiplicative update and keep the outer loop.

## Comparison with scikit-fusion (`~/repositories/scikit-fusion`)

- scikit-fusion stores relations as dense `numpy.ndarray` (or `numpy.ma.MaskedArray` for missing data). `data-fiusion` stores them as `scipy.sparse.csr_matrix` and uses dense factors only.
- scikit-fusion uses `FusionGraph` / `ObjectType` / `Relation` as user-facing primitives. `data-fiusion` uses plain dicts keyed by `(src_type, dst_type)`.
- scikit-fusion supports `n_run > 1` parallel restarts through `joblib`. `data-fiusion` offers sequential restarts through `fuse(n_runs=...)`; the legacy `dfmf_sparse` runs a single factorization per call.
- scikit-fusion uses `np.random.RandomState`; `data-fiusion` uses `np.random.default_rng`. Direct numerical comparison is not meaningful.
- scikit-fusion uses `pinv(G^T G)`; `data-fiusion` uses `np.linalg.solve(G^T G + lambda I, ...)`. Equivalent at `lambda=0` in exact arithmetic.

## Numpy compatibility caveats

The codebase targets Python `>=3.10` and `numpy >=1.22`. There are no deprecated `np.float` or `from collections import Iterable` usages here. The parity test monkey-patches `np.float = float` and `collections.Iterable = collections.abc.Iterable` before importing scikit-fusion, because scikit-fusion still relies on those names. When upgrading scikit-fusion, those shims can go away.

## Behaviour change: `fold_in_entities` is non-negative by default

It used to return the raw ridge solution, with 24% to 38% negative entries, while the
fit enforces `G >= 0`. It now refines to a non-negative solution (within 0.5% of
`scipy.optimize.nnls`). Any code that folds in will produce different numbers than before. The direction is measured: the hard clamp improved every comparison, and the
refinement improves on the clamp. Pass `nonneg=False` for the old behaviour.

The fit itself is unchanged: `dfmf_sparse` stays frozen and pinned by the golden test.

## Sparse gives the same answer, cheaper (measured)

`examples/movielens/equivalencia.py`. Both engines fit the same MovieLens
relations and are scored by the same downstream code, so only the fit differs.
With the SAME initialization the two agree; the quality gap that shows up
otherwise is the initializer, not the algorithm:

| motor | init | AP | segundos | memoria |
|---|---|---|---|---|
| dfmf_sparse | random | 0.476 | 3.2 | 35 MB |
| scikit-fusion | random | 0.442 | 7.8 | 249 MB |
| scikit-fusion | random_vcol | 0.519 | 8.9 | |
| dfmf_sparse | nndsvd | 0.730 | 3.9 | 35 MB |

So: the rewrite preserves the model (0.476 against 0.442 is within what two
different RNG streams give), costs 2.1x less time and 7.2x less memory, and
`nndsvd` is worth +0.25 AP on top. scikit-fusion has no nndsvd; its
`random_c` mode crashes on `itemgetter` before producing anything.

**Do not compare engines across different `init` values.** The initializer
dominates everything else here by an order of magnitude more than the engine
does.

## Careful with peak-memory measurements

`resource.getrusage(...).ru_maxrss` is a historical maximum for the whole
process, so subtracting the current RSS from it attributes to the stage under
test everything the process allocated earlier. Doing that made the
initialization look like it used 10 GB when it used 2 GB, and made two
different code paths report byte-identical figures. Measure a stage as
`peak_after - peak_before`, generate test data in a different process, and be
suspicious when two implementations report the same number to the megabyte.

## Two fitting paths

`fuse` (in `model.py`) is the current API, with `fit` kept as an alias;
`dfmf_sparse` (in `core.py`) is frozen.
The design and the reasoning behind it are in `docs/diseno-regularizacion.md`.
The task-oriented user documentation is `docs/guia-de-uso.md`; its code snippets
are meant to run as written, so keep it in sync when the API changes.

The legacy path is frozen **literally**, not as a wrapper over the new loop:
`tests/test_golden.py` pins its output, so results produced with it stay reproducible. The cost is roughly 50
duplicated lines of update loop. Do not "unify" them without a reason.

What `fuse` adds: relations by name (so two matrices between the same pair of
types stop being positional), row observation masks, a column gauge, graph
smoothing calibrated against the data energy, a loss trace with a stopping
tolerance, `transform` with a non-negative fold-in that reapplies the fit-time
scaling itself, `predict_proba` in row batches, random restarts (`n_runs`,
sequential, keeps the run with the lowest final loss), and save/load/resume.
Relations can carry per-entry weights (`Relation.entry_weights` plus
`background`, the implicit-feedback regime of Hu, Koren and Volinsky):
weight-zero entries are hidden from the loss, the scaling and the
initialization, which is what `holdout_entries` plus
`FusionModel.reconstruct_entries` use for held-out entry validation. The
weighted path needs the SDDMM primitive in `ops.py`; uniform weights route to
the classic path bit-identically, and the update formulas are documented in
`_accumulate_weighted` and `solve_backbone`.
`save` persists supervision, masks, graphs and theta in optional
subdirectories, so `resume` after `load` reproduces the fit configuration;
it cleans a reused target directory first, because stale files would
override the null in meta.json. `transform` honours `Relation.rows` (each
new entity is solved from the relations it is observed in) and reports new
entities without observations in the derived model's `empty_rows`; a
transform-derived model cannot be resumed. Relation and type names are used
as file names by `save`, so `fuse` rejects names with path separators.
Row masks are sets: `Relation` deduplicates them and rejects float dtypes.

**No L2 penalty on G is offered, and this is deliberate.** After the closed-form
solve of S the identity `<G_t, N_t> = <G_t, D_t>` holds, so a penalty feeding
only the denominator has no fixed point and collapses G; under the column gauge
`||G_t||_F^2 = c_t` is constant, so the same penalty is inert. `lambda_G`
survives only for the legacy path and defaults to zero. The levers that do
regularize are the rank, `weights`, `masks` and `alpha_graph`.

**`weights` does not mean the same thing in the two paths.** `normalize_relations`
multiplies the matrix (the weight lands inside the norm, so its effect is
quadratic and the reconstruction target moves); `fuse` weights the loss term.
The second is the correct formulation, but a number carried over from old code
will not reproduce the old fit.

### Measured: the new path clusters MUCH better

The tri-factorization is a co-clustering method and that side is evaluated separately
in `examples/movielens/clustering.py`. Grouping movies into 19 clusters with random
init, four seeds, against genres:

| method | NMI | ARI |
|---|---|---|
| k-means on the same matrices | 0.083 | 0.015 |
| `dfmf_sparse` | 0.579 | 0.464 |
| `fuse` without gauge | 0.620 | 0.509 |
| `fuse` with column gauge | **0.756** | **0.741** |

Paired by seed against `dfmf_sparse`: +0.177 NMI (15.7 SE) and +0.277 ARI (14.7 SE).
Without the gauge the advantage drops to +0.041, so the gauge is what is doing the
work. The reason is direct: a row's cluster is the argmax over the columns of `G`, and
without a gauge that argmax is decided by whichever column grew most. With `nndsvd`,
which already starts well scaled, the gap narrows to 0.764 vs 0.745 — the gauge mostly
buys independence from the initializer.

Do not read the absolute numbers as unsupervised clustering quality: the genre relation
is IN the fit, so recovering genres is partly tautological. The unsupervised row is
`fuse` without the genre relation, at NMI 0.140, still above k-means at 0.083.

### Measured: the new path does NOT generalize better at prediction

Paired comparison on MovieLens, three protocols, four seeds, one shared grid
(`examples/movielens/comparacion.py`):

| protocolo | AP media | contra legacy (pareado) |
|---|---|---|
| legacy | 0.732 +- 0.003 | |
| fold-in nuevo | 0.722 +- 0.006 | -0.010 +- 0.004 (2.7 SE) |
| enmascarado | 0.716 +- 0.005 | -0.017 +- 0.002 (7.3 SE) |

The success criterion declared before running (beat legacy by 2 standard
errors) was not met, so no claim of better generalization is made. Seven
candidate causes were ruled out by ablation: the column gauge, `eta`,
`lambda_S`, the non-negative fold-in, the weight grid range (swept from 0.01 to
1000, optimum at 3.0 either way), the final re-solve of S, and the weight
semantics. The remaining suspect is the block ordering inside the NNDSVD
initialization, which is a property of how the data is arranged and not of the
method. Anyone picking this up should test that before assuming the loop is at
fault.

So the two sides disagree, and which path to use depends on the task. For
**clustering**, `fuse` with the column gauge, by a wide margin. For **predicting** a
held-out attribute the two are close and the old one is slightly ahead. The rest of
the new path's justification is cost and correctness: constant memory in the loss, a
fold-in verified against `scipy.optimize.nnls`, label validation that turns a silent
catastrophic failure into a `ValueError`, and masks that make a semi-supervised regime
expressible at all.

## MovieLens benchmark (`examples/movielens/`)

Small end-to-end test of the implementation with complete ground truth, meant to
diagnose the library without the complexity of a real application domain. Uses the MovieLens
copy shipped with scikit-fusion (6.5 MB on disk, no download; `MOVIELENS_DIR`
overrides the path). Three relations, all with `pelicula` as source type so
`fold_in_entities` applies: `(pelicula, usuario)` ratings, `(pelicula, genero)`
labels, `(pelicula, actor)` cast. Movies are split 60/20/20; validation and test
movies stay out of the fit entirely, so the label cannot leak.

`examples/movielens/README.md` has the numbers. Three findings worth carrying
back to the library:

- `fold_in_entities` returns 24% to 38% negative entries in `G_new`, because the
  ridge solve has no sign constraint while the fit enforces `G >= 0`. Clamping
  with `np.maximum(G_new, 0.0)` improves accuracy consistently. The example
  clamps outside the library; folding this into `core.py` (or switching to
  non-negative least squares) is a candidate change.
- In-sample `predict_attribute` on a relation that entered the fit reaches AP
  0.98, and exactly 1.00 when the target type's rank equals the class count.
  Validation has to use entities held out of the fit.
- `lambda_G > 0` degrades generalization here (AP 0.722 at 0.0, 0.541 at 1.0).
  The source type's rank is the lever that controls overfitting instead.

New relations passed to `fold_in_entities` must be scaled by the *training*
Frobenius norm and weight from `normalize_relations`, not their own. `fuse` plus
`FusionModel.transform` does this for you; the legacy path does not.
