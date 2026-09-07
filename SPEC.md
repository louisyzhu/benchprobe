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
  (T5 addition: `.folder`, the verified snapshot folder, and `.file(name)`, the path of a
  manifest-listed file — a `KeyError` for anything the manifest does not list.)
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
  `zscore(frame, columns, *, ddof=0)` (raises on a zero-variance column),
  `days_since_earliest_release(frame, reference, *, strict=True)`,
  `base_model_key(name)` (the archive's regex, verbatim), `deduplicate_by_base_model(table)` (the
  archive's `sort_values("intelligenceIndex", descending).groupby("base").first()`, verbatim — see
  `docs/decisions.md` T2 for what `first()` does; the result carries the `base` key as its last
  column), `benchmark_coverage(table, benchmarks, *,
  min_models=60)` and `model_coverage(table, benchmarks, *, min_benchmarks=8)` (the archive's Phase 1
  inclusion rules), `sha256_of(path)`. Constants: `PRIMARY_BENCHMARKS`, `DENSE9_BENCHMARKS`,
  `ECONOMIC_BENCHMARKS`, `LABELS`, `BLOCKS`, `GRIDS`, `ENV_DATA_DIR`.
- Config-driven runs (§1) are deferred to T7.

**`benchprobe.measure`** (T3 — confirmed 6 September 2026, as implemented)

- `kmo(z) -> float` — overall Kaiser–Meyer–Olkin measure of the indicator matrix.
- `parallel_analysis(z, *, n_iter: int = 1000, seed: int = 42, percentile: float = 95) -> ParallelAnalysis`
  Horn's parallel analysis on the correlation matrix against `n_iter` standard-normal draws of the same
  shape; `.observed`, `.threshold` (per-eigenvalue percentile of the random draws) and `.k_retained`.
- `efa(z, *, k: int = 3, rotation: str | None = "oblimin", method: str = "ml", align_to_mean_score: bool = True) -> Efa`
  Maximum-likelihood exploratory factor analysis. `.loadings` (DataFrame of pattern loadings,
  benchmarks × `F1..Fk`), `.structure` (`ΛΦ`), `.phi` (factor correlations; identity when
  unrotated), `.uniquenesses`, `.ssl` (sum of squared pattern loadings per factor), `.weights`
  (Thurstone regression weights on the pattern loadings, `W = R⁻¹Λ` by pseudo-inverse), `.scores`
  (regression factor scores `z @ R⁻¹S`, computed explicitly as `factor_analyzer.transform` does but
  without its silent fallback to `Λ` when `R` is singular — that case raises; `S = Λ` when
  unrotated), `.signs`, `.k`, `.rotation`, `.method`. Result objects are frozen dataclasses with
  identity equality (`eq=False`), since they hold DataFrames.
  The two weightings coincide when unrotated and differ under oblimin; the archive's LOBO path
  projects held-out rows with `W = R⁻¹Λ`, and each ticket keeps its archive's choice. With
  `align_to_mean_score`, each factor's sign is chosen so that its scores correlate positively with
  the row-mean of `z`, as the archive does; the sign is applied to scores, loadings, structure,
  weights and `phi` together. NaN anywhere in the input raises.
- `thurstone_weights(R, loadings) -> ndarray` — `W = R⁻¹Λ` by pseudo-inverse; used by `efa` and by
  `predict.lobo` to project held-out rows.
- `first_factor_share(z, *, k: int = 3) -> float`
  Share of common variance carried by the first factor of the *unrotated* `k`-factor ML solution:
  `ssl[0] / ssl.sum()`, as a fraction in `[0, 1]`, column 0 as the archive takes it; a
  `RuntimeWarning` is raised if column 0 is not the largest factor (it is on every archive grid,
  with the thinnest margin on the residualised `compute_known` grid, 4.78 vs 4.27).
- `residualise_on_date(z, days) -> DataFrame`
  Ordinary-least-squares residual of every column on `days` (with intercept). The archive residualises
  the indicators and re-estimates the factor solution; handoff §2 says "date-residualisation of a
  factor", and this signature follows the archived computation (see `docs/decisions.md`).
- `residualisation_drop_bootstrap(z, days, *, k: int = 3, B: int = 2000, seed: int = 42) -> DropBootstrap`
  Resamples rows with replacement `B` times (`default_rng(seed)`, `rng.integers(0, n, n)` per
  replicate, the archive's draw order); on each draw recomputes `first_factor_share` before and after
  `residualise_on_date` and records the drop (as a fraction). `.point` (the full-sample drop), `.ci`
  (2.5th and 97.5th percentiles of the draws), `.draws`, `.B`, `.seed`, `.k`. The archive's H2(ii)
  check; reproduces its interval exactly.
- `date_r2(x, days, *, form: str = "ols") -> float`
  R² (`1 − SS_res/SS_tot`) of a one-dimensional score vector on `days`. `form="ols"` is a straight
  line (the archived Task-1 computation, 0.477). `form="logistic"` is the four-parameter logistic
  `floor + upper / (1 + exp(−rate (t − midpoint)))` fitted by least squares (`scipy.optimize.curve_fit`)
  from a 4 × 3 grid of starting values on standardised time, best fit kept; recovered at T3 as the
  pre-registered form because it reproduces all three archived factor values, 0.505 / 0.364 / 0.286
  (`docs/decisions.md`, T3).

**`benchprobe.predict`** (T4 — confirmed 6 September 2026, as implemented)

- `lobo(grid: Grid, *, targets: Sequence[str], k: int = 3, learners: str | Sequence[str] | Mapping = "registered", outer_folds: int = 5, inner_folds: int = 5, seed: int = 0, covariates: bool = True, rungs: Sequence[str] = RUNGS) -> LoboResult`
  Leave-one-benchmark-out: each target is predicted from the remaining benchmarks of the grid. Inside
  every outer fold the predictors are standardised on the training rows, the `k`-factor oblimin ML
  solution is fitted on the training rows only and projected to the held-out rows by Thurstone weights,
  and each rung (§3) is fitted with inner `GridSearchCV` over the registered learner grids.
  `learners="registered"` is the archive's four (ridge, elastic net, random forest, gradient boosting)
  with their grids (`registered_learners()`); a learner name (`"ridge"`, the fast path), a sequence
  of names, or a mapping `name → (estimator, grid)` are also accepted. `rungs` selects a subset of
  `RUNGS`; `v_kfac_cov` is dropped when `covariates=False` or the grid lacks `isReasoning`,
  `isOpenWeights`, `totalParameters`. `.metrics` is a DataFrame with the archive's columns
  (`METRIC_COLUMNS`: `target, block, rung, learner, train_rmse, test_rmse, train_mae, test_mae,
  train_r2, test_r2, test_mse`; the train columns pool the training-fold predictions across outer
  folds, as the archive does); `.oof` maps `(target, rung, learner)` to out-of-fold `(y_true,
  y_pred)` on the training-fold-standardised target; `.grid`, `.n`, `.targets`, `.rungs`,
  `.learners`, `.k`, `.outer_folds`, `.inner_folds`, `.seed` record the run. NaN in any input raises.
- `ladder(result: LoboResult, *, targets: Sequence[str], pooling: str = "mean_rmse") -> DataFrame`
  Indexed by rung (`i_date` … `v_kfac_cov`), columns `learner`, `train_rmse`, `test_rmse`, `test_r2`,
  `runner_up_gap`: the best learner by pooled test RMSE over `targets`, that learner's pooled train
  RMSE (the same pooling), its pooled test RMSE, its mean test R² across targets, and the
  pooled-RMSE gap to the runner-up. `pooling="mean_rmse"` is the arithmetic mean of per-target
  RMSEs — the pooling that reproduces every row of the archived `lobo_rung_summary.csv` and the
  paper's ladder table; `pooling="rms"` (square root of the mean per-target MSE) is what the
  archive notebook's `_ladder` helper computes and does not reproduce that table (T4 finding,
  `docs/decisions.md`).
- `pooled_delta_mse(result: LoboResult, *, baseline: str = "ii_meanidx", model: str = "iv_kfac", targets: Sequence[str], learner: str = "ridge", B: int = 2000, seed: int = 42) -> DeltaMse`
  Pooled ΔMSE = MSE(baseline) − MSE(model) over the concatenated out-of-fold rows of `targets`, with
  a percentile bootstrap over those rows (`B` resamples of `rng.integers(0, n, n)`, the archive's
  draw). `.point`, `.ci` (2.5th, 97.5th percentiles), `.per_target` (DataFrame: `dMSE`, `n` per
  target), `.draws`, `.n`, `.B`, `.seed`, `.baseline`, `.model`, `.learner`. An optional `stream`
  (`numpy.random.Generator`) replaces `seed` so that a sequence of comparisons can be drawn from one
  stream in the archive's order.
- `h4_table(result, *, baselines=("i_date", "ii_meanidx", "iii_f1"), model="iv_kfac", targets, learner="ridge", B=2000, seed=42, per_target_baselines=("ii_meanidx",)) -> DataFrame`
  The archive's H4 table (`h4_bootstrap_dmse.csv` layout: `scope, baseline, kmodel, dMSE, ci_lo,
  ci_hi, excludes_zero`), all intervals drawn from one `default_rng(seed)` in the order of the
  archive's H4 cell: pooled per baseline, then per target per `per_target_baselines`.
- Hold-out by task, model, family or context, and effective-sample-size reporting (§1, "thesis
  pipeline needs"): deferred to a ticket that can name the thesis design (handoff §8).

**`benchprobe.trust`** (T5 — agreed and implemented 7 September 2026)

Extracted verbatim from the JUDGe archive (`estimators.py`, `sweeps.py`, commit `126d1bce`), archive
names kept. Verdict matrices are `(n_items, K)` 0/1 arrays; totals are `(n_items,)`.

- `load_bank(snapshot=None, *, K=10) -> Bank` — the archive's `load_bank`: rows carrying judge
  verdicts (180 of 210) as `.gold` and `.judge` element matrices, `.gold_total`, `.judge_total`,
  `.question_id` (the cluster for bootstraps), `.item_id`, `.K`, `.n`, `.n_items_total`,
  `.judge_model`. Default snapshot `judge_2026-08-31` (vendored: `judge_item_bank.csv`,
  `sweep_grid.json`). `published_grid(snapshot=None) -> dict` returns the published simulation values.
- Reliability: `kr20(P)`, `kr21(tot, K)`, `intra_item_rho(P)`.
- Beta-binomial: `bb_mle(tot, K) -> (alpha, beta)`, `bb_pmf(alpha, beta, K)`,
  `bb_gof(tot, K, min_expected=5) -> BetaBinomialFit(chi2, df, p, alpha, beta)`.
- Classification: `livingston_lewis(tot, K, cut, n_grid=None) -> LivingstonLewis(decision_consistency,
  classification_accuracy)` (accuracy indexed to the instrument's own true score, as the archive
  warns); `phi_lambda(P, cut)` — the G-theory dependability index Φ(λ), a ratio of variance
  components; `classification_accuracy(judge_tot, ref_tot, cut)`;
  `cluster_bootstrap(stat_fn, cluster_ids, B=2000, seed=0)` (archive draw order; non-finite draws
  dropped); `agreement_with_interval(judge_tot, ref_tot, cut, *, cluster_ids, B=2000, seed=0) ->
  Agreement(point, ci, draws, cut, n_clusters, B, seed)` — human-vs-grader agreement with a
  cluster-bootstrap 95 % interval.
- Agreement coefficients (new code under T5, not in the archive): `cohens_kappa(a, b, *,
  weights=None)` (scikit-learn's `cohen_kappa_score`); `krippendorffs_alpha(data, *,
  level="nominal")` for a raters × units matrix with NaN for missing ratings, levels nominal,
  ordinal, interval, ratio (Krippendorff 2011; reproduces its worked example, nominal 0.743 and
  interval 0.849, and the `krippendorff` package to 1e-15 on random fixtures).
- G-theory: `gstudy_two_way(X) -> GStudy(v_items, v_facet, v_resid, Phi)` — variance components of a
  fully crossed items × facet design by two-way random-effects ANOVA (negative estimates truncated at
  zero), Φ for one condition of the facet; the archive's phrasing G-study inline computation.
- Real bank: `bank_summary(bank) -> dict` — KR-20 on judge and gold, per-element error, judge–gold
  correlation, leniency, beta-binomial fit on judge totals (the archive README's bit-for-bit set).
- Simulations (archive `sweeps.py`, seeds and draw order preserved): `simulate_bank(spread, err,
  n=210, K=10, rng=None)`, `two_way_sweep(reps=60, seed=11, *, spreads, errors, n, K) -> (mean, sd)`,
  `phi_control(seed=101, n=4000)`, `ll_estimand_control(seed=303, n=4000)`, `phrasing_gstudy(v_phrasing,
  v_resid, n=210, R=3, sims=200, seed=7)`, `sweeps()` (the archive's main block, in its order).
  Constants `K_DEFAULT`, `N_ITEMS`, `MEASURED_ERROR`, `BANK_SPREADS`, `JUDGE_ERRORS`.

**`benchprobe.irt`**, **`benchprobe.report`** — signatures are agreed at T6 and T7.

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
