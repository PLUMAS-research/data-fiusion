# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`data-fiusion` is a sparse rewrite of the core algorithm of `scikit-fusion` (Marinka Zitnik's data fusion library). It performs non-negative matrix tri-factorization across heterogeneous relations, keeps all relation matrices as `scipy.sparse`, and uses dense low-rank factors. The `scikit-fusion` library densifies relations everywhere, which makes it unusable for the typical large inputs (relations with millions of rows but a few percent density).

The package name on disk is `datafiusion` (no hyphen). The distribution name is `data-fiusion`. The API is organized around four patterns: joint factorization of several relations, per-relation weighting, fold-in of new entities without refitting, and attribute prediction from reconstructed views.

## Commands

The project uses `uv` and ships a `pyproject.toml` plus a populated `.venv` with `data-fiusion` already installed in editable mode. Every test dependency is declared (`dev` group: pytest, scikit-learn), so `uv sync` leaves a working environment. Run everything through `uv run` from this directory; it picks up the project's `.venv` automatically.

- Parity report against scikit-fusion: `uv run python tests/test_compare_skfusion.py`. Runs `dfmf_sparse` with both `random` and `nndsvd` initialization and compares against the stored `Dfmf` traces in `tests/data/referencia_skfusion.json`; scikit-fusion itself is NOT needed. The traces regenerate with `tests/data/generar_referencia_skfusion.py`, the only file that imports scikit-fusion (install a checkout first: `uv pip install -e <path> --python .venv/bin/python`; note that any later `uv sync` removes it again, which is fine).
- Semi-supervised labels: `uv run python examples/semi_supervised.py`. Documents the labels-as-relation pattern and its limitations.
- Newsgroups counts baseline: `uv run python examples/newsgroups/conteos.py`. Raw counts, sqrt, log1p, shifted Anscombe and TF-IDF under the quadratic loss; the reference any count likelihood has to beat. About 15 minutes.
- Poisson against that baseline: `uv run python examples/newsgroups/poisson_vs_cuadratica.py`. KL on raw counts, on idf-scaled columns and on TF-IDF values, with the 2-SE verdict against the quadratic TF-IDF reference printed at the end.
- Mixed families on Last.fm: `uv run python examples/lastfm/comparacion.py`. Downloads HetRec 2011 Last.fm on first run (2.5 MB, non-commercial license). Poisson listening counts plus masked gaussian tags against the all-gaussian arm, paired over 4 seeds with the 2-SE verdict printed.
- MovieLens benchmark, held-out genre prediction: `uv run python examples/movielens/prediccion.py`. Baselines, hyperparameter sweep on validation, final test evaluation. About three minutes.
- MovieLens benchmark, cost against scikit-fusion: `uv run python examples/movielens/escala.py`. Sweeps rows and columns, runs each configuration in its own subprocess under a memory guard.
- GPU against CPU, end to end: `uv run python examples/gpu/comparacion.py`. Needs the gpu extra (`uv sync --extra gpu`) and a CUDA GPU; measured numbers in `examples/gpu/README.md`.

Avoid `uv run --with .` from outside this repo: it builds a wheel and caches it under `~/.cache/uv/archive-v0/<hash>/`. Edits to the source then look like they had no effect because the cached wheel is served until `--reinstall-package data-fiusion` is passed. From inside the repo, plain `uv run` resolves through the project `.venv` (editable) and avoids the cache entirely.

## Architecture

One module per concern, with the API consolidated around patterns from real use.

1. **Numerical core (`core.py`).** `dfmf_sparse(R, ranks, theta, lambda_G, lambda_S, max_iter, init, random_state, verbose)`. Takes `R: dict[(src, dst), list[sparse]]` and `ranks: dict[str, int]`, returns `(G, S)` with `G[t]: (n_t, c_t)` dense ndarrays and `S[(src, dst)]` a list of `(c_src, c_dst)` backbones (one per matrix in `R[(src, dst)]`). Self-relations (`src == dst`) are rejected; type-level structure goes through `theta`.
   - `_solve_S` uses closed-form Tikhonov: `S = (G_i^T G_i + lambda I)^{-1} G_i^T R G_j (G_j^T G_j + lambda I)^{-1}`. Equivalent to scikit-fusion's `pinv(G^T G)` at `lambda=0`.
   - `reconstruction_error(R, G, S)` is the sum of per-relation relative Frobenius errors.
   - `fold_in_entities(R_new, G, S, target_type, lambda_reg)` projects new rows into `G[target_type]` space via ridge least squares without refitting. Vectorized.
   - `predict_attribute(known, G, S, target_type, view_keys, combine, eps)` predicts a soft distribution over a target type by combining several reconstructed views (geometric mean is the default; product is hard Naive Bayes; sum is additive evidence).

2. **Initialization (`init.py`).** Two strategies.
   - `init_random`: uniform `[0, 1)` per entry, seeded by `np.random.default_rng(random_state)`.
   - `init_nndsvd`: NNDSVDa adapted to multi-relation factorization (Boutsidis and Gallopoulos 2008). For each type, concatenates every relation where the type participates, normalizes each block by Frobenius, takes a truncated SVD, applies NNDSVDa to the left singular vectors. Below-eps entries are replaced by the absolute mean of the concatenated matrix.
   - `_truncated_svd_left` passes a deterministic `v0` to `scipy.sparse.linalg.svds` and runs `_svd_flip_u` on the returned `U` to pin the sign of each column. Both are necessary: ARPACK seeds from global numpy state and singular vectors are defined up to sign, so without these pins NNDSVD output differs depending on whether scikit-fusion was imported earlier in the process.

3. **Data prep utilities (`utils.py`).** Three helpers extracted from multi-source pipelines.
   - `merge_relations(*dicts)`: concatenates lists of matrices per relation key across multiple builder dicts. Used to combine per-source builders into one final `R`.
   - `normalize_relations(R, weights=None)`: Frobenius-normalizes every matrix; optional per-relation scalar multiplier applied after. Returns `(R_normalized, scales)`. The scales are kept so reconstructions can be rescaled to absolute counts if needed.
   - `holdout_entries(matrix, fraction)`: picks stored entries to hold out of a fit, returning weights aligned with `matrix.data` plus the coordinates to score afterwards.

4. **Diagram (`diagram.py`).** `fusion_diagram(source, ranks=None, positions=None)` draws the publication-oriented schema of a fusion setup: entity types as boxes (size and rank), relations as arrows annotated with dimensions, density, preprocess and masks, colored by family (Okabe-Ito). Accepts named relations, the legacy keyed dict or a `FusionModel`. Optional, depends only on `matplotlib`. Example figure: `examples/lastfm/diagrama.py`.

5. **GPU backend (`gpu.py`).** `fuse(..., device="gpu")` runs the fit loop on a CUDA GPU through CuPy and returns the model in numpy. `gpu.py` wraps everything CuPy-specific: availability checks with install instructions, host-device conversion, `_DeviceRelation` views (the row mask is applied on CPU and the sliced matrix uploaded once), and `spmm_transpose` (cusparse `transa=True`, which avoids materializing the transpose and is 12x faster than multiplying by one). The default `device="cpu"` never imports cupy, so the dependency stays optional (`pyproject` extra `gpu` resolving to `cupy-cuda13x[ctk]`; `CUDA_PATH` is pointed at the NVIDIA wheels automatically). Model code touches cupy only through `_xp` and the lazy import in `fuse`; most of the loop relies on numpy's `__array_function__` dispatch. Not supported on GPU: poisson relations and entry weights (rejected loudly). GPU tests in `tests/test_gpu.py` skip themselves without a GPU.

6. **Sparse primitives (`ops.py`).** `sddmm(pattern, A, B)` evaluates A @ B^T at the stored entries of a sparse pattern in O(nnz * c), blocked; `product_at(A, B, rows, cols)` is the same kernel on coordinate lists. They are what makes entry weights and count likelihoods expressible without materializing anything of relation size.

7. **Named value transforms (`preprocess.py`).** `Relation(preprocess=...)` names transforms from a closed registry ("log1p", "sqrt", shifted "anscombe", "idf"); `fuse` applies them once (learning the idf vector from the training data), stores names in `params["preprocess"]` and state in `params["idf"]` (persisted by `save` like supervision and masks), and `FusionModel.transform` and `loss` reapply the SAME chain with the SAME state to incoming raw data. Closed registry on purpose: callables cannot persist. The caller's matrix is never modified (only the data array is copied). An incoming relation declaring a different chain than the fit raises; idf cannot apply when the new entities are on the column side; `reconstruct_entries(original=True)` inverts the chain. All transforms map 0 to 0, so patterns and `entry_weights` alignment survive.

8. **Frame ingestion (`frames.py`).** `relations_from_frames(specs, vocabularies, on_unknown, on_missing)` builds every relation from long coordinate DataFrames and resolves each type's vocabulary once (sorted union of what appears, or a fixed list), so shared types come out aligned with labels attached and without dense reindexing. Duplicate pairs are summed (value=None counts occurrences). Mismatch policies are loud: `on_unknown` for categories outside a fixed vocabulary ("error" naming them, "add" appends at the end consistently across relations, "drop" discards those rows with a warning), `on_missing` for fixed categories without observations ("zero" or "error"). `rows` in a spec takes entity LABELS and is mapped to indices. `relation_from_frame` is the single-relation case. The projection pattern fixes the fitted side with `vocabularies={"tipo": modelo.index["tipo"]}`.

9. **Count and mixed families (`poisson.py`).** `fuse` dispatches to `fit_families` when any relation carries `family="poisson"`. Poisson relations contribute generalized KL (S non-negative, multiplicative; the column gauge is compensated in S); gaussian relations contribute through the same `_FitState` machinery as the classic path, including row masks and entry weights. The shared factors receive ONE joint update: the positive root of the sum of both auxiliary majorizers, which reduces exactly to the classic ratio without poisson terms and to the KL rule without gaussian terms (derivation in the module docstring). The loss is the mean of normalized per-relation terms (relative squared error or deviance ratio). The balance BETWEEN families has no natural unit: relative `weights` must be validated. Not supported with poisson relations yet: `transform`, row masks and entry weights on the poisson relation itself, graphs; column or row weighting is expressed by scaling the data. `resume` works; `params["family"]` ("poisson" or "mixed") is informational and popped by resume.

## Patterns to follow when extending

- New algorithms: pair an internal core (plain dict in / dict out) with a public wrapper. Reuse `init_random` and `init_nndsvd` for initialization if applicable.
- New initialization strategies live in `init.py`. Dispatch via the `init` argument in `dfmf_sparse`.
- New post-fit utilities (anything operating on `(G, S)`) belong in `core.py`. Keep them stateless and vectorized.
- Data prep helpers (operating on raw relations) live in `utils.py`.
- If you touch the SVD path, keep `_svd_flip_u` applied to the returned `U`. The reproducibility test in `tests/test_compare_skfusion.py` will silently give slow convergence if signs flip across processes.
- The closed-form `_solve_S` is the unconstrained least-squares solution; if you add non-negativity to `S` later, swap the solve for a multiplicative update and keep the outer loop.

## Comparison with scikit-fusion

scikit-fusion is no longer installed in the venv: the live parity validation is done and the test compares against its stored traces (see Commands). A local scikit-fusion install is still the data source for the MovieLens examples (`MOVIELENS_DIR`) and what `examples/movielens/escala.py` and `equivalencia.py` benchmark against when it is installed. The design differences that motivated this library:

- scikit-fusion stores relations as dense `numpy.ndarray` (or `numpy.ma.MaskedArray` for missing data). `data-fiusion` stores them as `scipy.sparse.csr_matrix` and uses dense factors only.
- scikit-fusion uses `FusionGraph` / `ObjectType` / `Relation` as user-facing primitives. `data-fiusion` uses plain dicts keyed by `(src_type, dst_type)`.
- scikit-fusion supports `n_run > 1` parallel restarts through `joblib`. `data-fiusion` offers sequential restarts through `fuse(n_runs=...)`; the legacy `dfmf_sparse` runs a single factorization per call.
- scikit-fusion uses `np.random.RandomState`; `data-fiusion` uses `np.random.default_rng`. Direct numerical comparison is not meaningful.
- scikit-fusion uses `pinv(G^T G)`; `data-fiusion` uses `np.linalg.solve(G^T G + lambda I, ...)`. Equivalent at `lambda=0` in exact arithmetic.

## Numpy compatibility caveats

The codebase targets Python `>=3.10` and `numpy >=1.22`. There are no deprecated `np.float` or `from collections import Iterable` usages here. The reference generator (`tests/data/generar_referencia_skfusion.py`) monkey-patches `np.float = float` and `collections.Iterable = collections.abc.Iterable` before importing scikit-fusion, because scikit-fusion still relies on those names; the same shims live in `examples/movielens/escala.py` for its optional comparison.

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
The reasoning behind each lever lives in the `fuse` docstring (what `lambda_S`
actually penalizes, why the graph term uses `diag(W_sym 1) - W_sym`, why the
column gauge exists); `README.md` records why no L2 on G is offered.
The task-oriented user documentation is `docs/guia-de-uso.md`; its snippets are
templates over a fictional domain, not runnable scripts, so keep them in sync
when the API changes.

The legacy path is frozen **literally**, not as a wrapper over the new loop:
`tests/test_golden.py` pins its output, so results produced with it stay reproducible. The cost is roughly 50
duplicated lines of update loop. Do not "unify" them without a reason.

What `fuse` adds over the legacy path is user documentation, not agent lore:
the feature list lives in `README.md` and the workflows in
`docs/guia-de-uso.md`. Two pointers that only exist in code: the update
formulas of the entry-weighted path are derived in `_accumulate_weighted` and
`solve_backbone`, and the joint mixed-family update in the `poisson.py` module
docstring.

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

Which path for what, per the measurements: for **clustering**, `fuse` with the
column gauge, by a wide margin; for **predicting** a held-out attribute the two
are close and the legacy path is slightly ahead (the open 1.6% gap and its
remaining suspect are in `docs/oportunidades.md`). Never compare two fits with
different `init` values: the initializer dominates everything else by an order
of magnitude. The measured tables live in the bitacora (`slides/`), `README.md`
and the READMEs under `examples/`.

## MovieLens benchmark (`examples/movielens/`)

Small end-to-end test with complete ground truth, meant to diagnose the library
without the complexity of a real domain. Uses the MovieLens copy that ships with
scikit-fusion (6.5 MB, no download). With the package no longer installed, point
`MOVIELENS_DIR` at a scikit-fusion checkout's `skfusion/datasets/data/movielens`. Three
relations with `pelicula` as source type: ratings, genre labels, cast. Movies
split 60/20/20; validation and test movies stay out of the fit entirely, so the
label cannot leak. `examples/movielens/README.md` has the numbers.

One trap that only bites the legacy path: new relations passed to
`fold_in_entities` must be scaled by the *training* Frobenius norm and weight
from `normalize_relations`, not their own. `fuse` plus `FusionModel.transform`
does this by itself.
