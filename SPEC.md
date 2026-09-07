# SPEC.md — benchprobe

Scope is fixed by `HANDOFF_BENCHPROBE.md` (6 September 2026) §2, reproduced in §1 below. Function
signatures are agreed per ticket and appended in §2; anything not listed there is out of scope until a
ticket adds it (rule 4).

## 0. What benchprobe is, and is not

It is the analysis layer behind the evaluation papers and the MSc thesis pipeline, extracted from code
that already produced published numbers, packaged so the same computation runs by one command on a
fresh machine. Three families of function, one per programme layer: **trust** (reliability and grader
agreement), **measure** (factor structure and incremental validity), **predict** (held-out designs and
uncertainty for criterion studies). An IRT estimator sits under measure.

It is not a general-purpose benchmark-auditing product. It carries no claim the papers do not carry.
It is "in development" on every surface until the day it is the analysis code behind a real study.

## 1. Scope (handoff §2, verbatim)

| Module | Layer | Functions | Extracted from |
|---|---|---|---|
| `benchprobe.trust` | Trust | KR-20, dependability index, Livingston–Lewis classification accuracy, Krippendorff's α, Cohen's κ, generalisability-theory variance components, human-vs-grader agreement with bootstrap intervals | `llm-judge-reliability` |
| `benchprobe.measure` | Measure | KMO, Horn's parallel analysis, maximum-likelihood EFA with oblimin rotation, factor scores, Thurstone weights W = R⁻¹Λ, date-residualisation of a factor | `frontier-ai-economic-validity` |
| `benchprobe.predict` | Predict | Leave-one-benchmark-out with factor re-estimation inside every fold, pooled ΔMSE with bootstrap intervals, general-index baseline, hold-out by task, model, family or context, effective-sample-size reporting | `frontier-ai-economic-validity`; thesis pipeline needs |
| `benchprobe.irt` | Measure | Two-parameter and graded-response estimation, ability scores with standard errors; torch optional | Price of Intelligence archive |
| `benchprobe.io` | all | Hash-pinned snapshot loading, provenance record, coverage rules (models ≥ n, benchmarks ≥ k), config-driven runs | `frontier-ai-economic-validity` |
| `benchprobe.report` | all | Tables and figures regenerated from run outputs; the archived-vs-recomputed boundary written into every table caption | both |

**Out of scope, by decision:** benchmark construction, scraping, model inference, leaderboards,
dashboards, a web UI, "auditing" wrappers that hide the statistics. The thesis pipeline's inference and
grading layers live in the thesis repository, and call benchprobe for analysis only.

## 2. Signatures agreed per ticket

Conventions. "Indicator matrix" means models × benchmarks, one column per benchmark, z-scored with
population standard deviation (`ddof=0`) unless stated. Every function that produces a reported number
returns plain floats, NumPy arrays or pandas objects; nothing prints. Random draws take an explicit
`seed`. Names of benchmarks, grids and rungs are the archive's keys (§3), so tables recomputed here
join to the archived tables without renaming.

### T1 (6 September 2026) — provisional signatures the golden tests call

Agreed by the coordinating session so that `tests/golden/test_one_capability.py` could be written.
Each is confirmed or amended by the ticket that implements it (T2 `io`, T3 `measure`, T4 `predict`);
an amendment changes the test's call, never its number.

**`benchprobe.io`** (T2 — confirmed 6 September 2026, as implemented)

- `load_snapshot(name: str = "one_capability_2026-07-06", *, data_dir: str | os.PathLike | None = None) -> Snapshot`
  Loads the table the snapshot's `MANIFEST.json` names in its `table` field, after verifying the
  SHA-256 and size of every file the manifest lists; a mismatch, a missing file or a manifest whose
  `name` differs from its folder raises `SnapshotIntegrityError` (never a warning). `Snapshot` carries
  `.name`, `.table` (DataFrame; `releaseDate` parsed to datetime), `.sha256` (of the table file),
  `.manifest`, `.origin` (`"vendored"`, `"env:BENCHPROBE_DATA_DIR"` or `"data_dir"`), `.loaded_at`
  (UTC) and `.provenance()` (a JSON-serialisable record: snapshot, table, hash, size, shape, origin,
  source, upstream hashes, load time, benchprobe version). `data_dir` defaults to
  `$BENCHPROBE_DATA_DIR`, then to the copies vendored under `benchprobe/data/`. `list_snapshots()`
  names the folders that would be searched.
- `build_grid(snapshot: Snapshot, grid: str) -> Grid`
  Builds one of the named grids in §3. `Grid` carries `.name`, `.frame` (the selected rows, index
  reset), `.benchmarks` (column keys, archive order), `.labels`, `.blocks`, `.z` (indicator matrix,
  population SD over the grid's own rows, as the archive's `zmat`), `.days` (days since the earliest
  release date in the *full* snapshot, integer; NaN only on the `economic_dense` coverage grid) and
  `.n`. An unknown grid name raises `ValueError` listing the known names.
- Helpers, public because the grids are built from them and the thesis pipeline will reuse them:
  `zscore(frame, columns, *, ddof=0)`, `days_since_earliest_release(frame, reference, *, strict=True)`,
  `base_model_key(name)` (the archive's regex, verbatim), `deduplicate_by_base_model(table)` (the
  archive's `sort_values("intelligenceIndex", descending).groupby("base").first()`, verbatim — see
  `docs/decisions.md` T2 for what `first()` does), `benchmark_coverage(table, benchmarks, *,
  min_models=60)` and `model_coverage(table, benchmarks, *, min_benchmarks=8)` (the archive's Phase 1
  inclusion rules), `sha256_of(path)`. Constants: `PRIMARY_BENCHMARKS`, `DENSE9_BENCHMARKS`,
  `ECONOMIC_BENCHMARKS`, `LABELS`, `BLOCKS`, `GRIDS`, `ENV_DATA_DIR`.
- Config-driven runs (§1) are deferred to T7.

**`benchprobe.measure`** (T3)

- `kmo(z) -> float` — overall Kaiser–Meyer–Olkin measure of the indicator matrix.
- `parallel_analysis(z, *, n_iter: int = 1000, seed: int = 42, percentile: float = 95) -> ParallelAnalysis`
  Horn's parallel analysis on the correlation matrix against `n_iter` standard-normal draws of the same
  shape; `.observed`, `.threshold` (per-eigenvalue percentile of the random draws) and `.k_retained`.
- `efa(z, *, k: int = 3, rotation: str | None = "oblimin", method: str = "ml", align_to_mean_score: bool = True) -> Efa`
  Maximum-likelihood exploratory factor analysis. `.loadings` (DataFrame of pattern loadings,
  benchmarks × `F1..Fk`), `.phi` (factor correlations; identity when unrotated), `.uniquenesses`,
  `.ssl` (sum of squared loadings per factor), `.weights` (Thurstone regression weights on the pattern
  loadings, `W = R⁻¹Λ` by pseudo-inverse) and `.scores` (regression factor scores `z @ R⁻¹S` with
  `S = ΛΦ` the structure matrix, which is what `factor_analyzer.transform` computes and what the
  archive's Task-1 date-R² uses; `S = Λ` when unrotated). The two weightings coincide when unrotated
  and differ under oblimin; the archive's LOBO path projects held-out rows with `W = R⁻¹Λ`, and each
  ticket keeps its archive's choice. With `align_to_mean_score`, each factor's sign is chosen so that
  its scores correlate positively with the row-mean of `z`, as the archive does.
- `thurstone_weights(R, loadings) -> ndarray` — `W = R⁻¹Λ` by pseudo-inverse; used by `efa` and by
  `predict.lobo` to project held-out rows.
- `first_factor_share(z, *, k: int = 3) -> float`
  Share of common variance carried by the first factor of the *unrotated* `k`-factor ML solution:
  `ssl[0] / ssl.sum()`, as a fraction in `[0, 1]`.
- `residualise_on_date(z, days) -> DataFrame`
  Ordinary-least-squares residual of every column on `days` (with intercept). The archive residualises
  the indicators and re-estimates the factor solution; handoff §2 says "date-residualisation of a
  factor", and this signature follows the archived computation (see `docs/decisions.md`).
- `residualisation_drop_bootstrap(z, days, *, k: int = 3, B: int = 2000, seed: int = 42) -> DropBootstrap`
  Resamples rows with replacement `B` times; on each draw recomputes `first_factor_share` before and
  after `residualise_on_date` and records the drop (as a fraction). `.point` (the full-sample drop),
  `.ci` (2.5th and 97.5th percentiles of the draws), `.draws`. The archive's H2(ii) check.
- `date_r2(x, days, *, form: str = "ols") -> float`
  R² of a one-dimensional score vector on `days`. `form="ols"` is the archived computation;
  `form="logistic"` is the pre-registered functional form behind the paper's 0.505, whose exact
  specification is not in the archived notebook and must be recovered before T3 can implement it
  (`docs/reproducibility.md`, open item 2).

**`benchprobe.predict`** (T4)

- `lobo(grid: Grid, *, targets: Sequence[str], k: int = 3, learners: str | Mapping = "registered", outer_folds: int = 5, inner_folds: int = 5, seed: int = 0, covariates: bool = True) -> LoboResult`
  Leave-one-benchmark-out: each target is predicted from the remaining benchmarks of the grid. Inside
  every outer fold the predictors are standardised on the training rows, the `k`-factor oblimin ML
  solution is fitted on the training rows only and projected to the held-out rows by Thurstone weights,
  and each rung (§3) is fitted with inner `GridSearchCV` over the registered learner grids.
  `learners="registered"` is the archive's four (ridge, elastic net, random forest, gradient boosting)
  with their grids; `learners="ridge"` is the fast path. `.metrics` is a DataFrame with the archive's
  columns (`target, block, rung, learner, train_rmse, test_rmse, train_mae, test_mae, train_r2,
  test_r2, test_mse`); `.oof` maps `(target, rung, learner)` to out-of-fold `(y_true, y_pred)`.
- `ladder(result: LoboResult, *, targets: Sequence[str]) -> DataFrame`
  Indexed by rung (`i_date` … `v_kfac_cov`), columns `learner`, `train_rmse`, `test_rmse`, `test_r2`,
  `runner_up_gap`, as the archive's `_ladder` emits: the best learner by pooled test RMSE over
  `targets` (square root of the mean test MSE), that learner's pooled train RMSE (the same pooling),
  its pooled test RMSE, its mean test R² across targets, and the pooled-RMSE gap to the runner-up.
- `pooled_delta_mse(result: LoboResult, *, baseline: str = "ii_meanidx", model: str = "iv_kfac", targets: Sequence[str], learner: str = "ridge", B: int = 2000, seed: int = 42) -> DeltaMse`
  Pooled ΔMSE = MSE(baseline) − MSE(model) over the concatenated out-of-fold predictions of `targets`,
  with a percentile bootstrap over models (`B` resamples). `.point`, `.ci` (2.5th, 97.5th percentiles),
  `.per_target` (DataFrame of the same per target), `.n` (number of pooled residual differences).

**`benchprobe.trust`**, **`benchprobe.irt`**, **`benchprobe.report`** — signatures are agreed at T5,
T6 and T7. Note for T5: the JUDGe archive (`estimators.py`) carries KR-20, KR-21, the intra-item
phi, beta-binomial fit and goodness of fit, Livingston–Lewis, Φ(λ), classification accuracy and a
cluster bootstrap; Krippendorff's α and Cohen's κ are in scope by §1 but are not in that archive and
will be new code under the T5 ticket.

## 3. Names used by the archive and kept here

**Benchmarks** (key → label, block), in the archive's order:

| key | label | block |
|---|---|---|
| `gpqa` | GPQA Diamond | Academic |
| `hle` | HLE | Academic |
| `omniscience` | AA-Omniscience | Academic |
| `tau2` | τ²-Bench | Economic |
| `ifbench` | IFBench | Instruction-following |
| `lcr` | AA-LCR | Long-context |
| `scicode` | SciCode | Scientific-coding |
| `critpt` | CritPt | Scientific-coding |
| `terminalbenchHard` | Terminal-Bench Hard | Scientific-coding |
| `gdpval_elo` | GDPval (Elo) | Economic |
| `terminalbenchV21` | Terminal-Bench v2.1 | Economic |
| `tauBanking` | τ³-Banking | Economic |

`mmmuPro` (MMMU-Pro) is held for sensitivity only; APEX-Agents was dropped by the coverage rule
(fewer than 60 models).

**Grids** on the One Capability snapshot (`aa_analysis_models.csv`, 421 configurations):

| grid | definition (archive) | n |
|---|---|---|
| `complete_case` | rows complete on all twelve primary benchmarks ("G1") | 96 |
| `dense9` | rows complete on the nine near-universal benchmarks ("G2"; the twelve minus `gdpval_elo`, `terminalbenchV21`, `tauBanking`) | 409 |
| `economic_dense` | rows carrying `gdpval_elo`, `terminalbenchV21` and `tauBanking` | 103 |
| `deduplicated` | one row per base model (highest `intelligenceIndex` kept; base key strips reasoning/effort suffixes), then complete on all twelve | 89 |
| `compute_known` | `complete_case` rows with non-null `totalParameters` (the compute proxy; the archived notebook does not build this subsample, T3 confirms by reproducing the archived shares) | 58 |

**Rungs** of the LOBO ladder: `i_date` (standardised release date), `ii_meanidx` (mean of the
standardised predictors, the general-index baseline), `iii_f1` (first factor score alone), `iv_kfac`
(all `k` factor scores), `v_kfac_cov` (factor scores plus reasoning flag, open-weights flag,
log₁₀ parameters, and a missing-parameters indicator).

**Learners** (registered): `ridge` (alpha ∈ {0.03, 0.1, 0.3, 1, 3, 10, 30}); `elasticnet`
(alpha ∈ {0.03, 0.1, 0.3, 1}, l1_ratio ∈ {0.2, 0.5, 0.8}, max_iter 5000); `rf` (300 trees,
max_depth ∈ {None, 4}, min_samples_leaf ∈ {1, 3}, random_state 0); `gbm` (200 estimators,
max_depth ∈ {2, 3}, learning_rate ∈ {0.05, 0.1}, random_state 0). Outer `KFold(5, shuffle=True,
random_state=0)`; inner `GridSearchCV(cv=5, scoring="neg_mean_squared_error")`; the target is
standardised on training-fold statistics.
