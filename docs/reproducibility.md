# Reproducibility

Every output benchprobe reports is stated as one of three tiers (rule 7):

- **recomputed** — produced again from the hash-pinned inputs by this package and equal to the
  archived value within the stated tolerance; the only tier that may be called "fully reproducible";
- **statistically reproduced** — produced again from the hash-pinned inputs with a different random
  stream (seed or draw order not recoverable), and equal to the archived value within a tolerance set
  for that Monte-Carlo variation;
- **regenerated** — read from an archived table and re-rendered; not recomputed by this package.

Every table and figure the package emits names its tier in the caption (rule 8). This file is the
ledger: one row per output, kept current by the ticket that produces it.

## Outputs

The One Capability acceptance rows (handoff §6) are encoded in `tests/golden/test_one_capability.py`
(T1) with the tolerances below; each row carries the tier and the recomputed value from the ticket
that computed it (T2 grids, T3 measure, T4 predict). "Archive file" is where the value was verified on
6 September 2026 (`frontier-ai-economic-validity`, commit `946ce845`). Last full run:
`uv run pytest -m golden` on `t6-irt` after T6, **39 passed in 3008 s (50 min)**, covering all three
studies' rows; the KMO note below is the one warning. (Previous: 31 passed in 2894 s on `main` after
T7; T6 adds the eight Price of Intelligence rows, which take 4 s of the total.)

| Output | Grid | Archive value | Archive file | Tolerance | Tier | Recomputed | Ticket |
|---|---|---|---|---|---|---|---|
| KMO | complete_case (96) | 0.933 | task1_structure_results.json (0.9326) | 0.005 | recomputed (factor_analyzer warns that it used the Moore–Penrose inverse: the correlation matrix is near-singular on this grid, Terminal-Bench v2.1's uniqueness being ≈ 0; the archive's `calculate_kmo` does the same) | 0.9326 | T3 |
| First-factor share of common variance | complete_case | 74.5 % | task1_structure_results.json (0.74544) | 0.5 pp | recomputed | 74.544 % | T3 |
| Logistic-fit date-R² of the dominant factor | complete_case | 0.505 | hypothesis_adjudication.csv (OLS 0.477 in task1_structure_results.json) | 0.005 | recomputed (four-parameter logistic recovered at T3; other factors 0.3641 / 0.2859 vs archived 0.364 / 0.286; OLS 0.4767) | 0.5048 | T3 |
| Residualised first-factor share | complete_case | 59.6 % | task1_structure_results.json (0.59635) | 0.5 pp | recomputed | 59.635 % | T3 |
| Residualisation drop | complete_case | 14.9 pp | task1_structure_results.json (14.909) | 1.0 pp | recomputed | 14.909 pp | T3 |
| Residualisation drop, bootstrap 95 % interval | complete_case | [−5.3, +32.7] pp | bootstrap_h2_drop.csv (−5.337, 32.731) | 2.0 pp per endpoint | recomputed (seed 42 and draw order replicated; P(drop ≥ 15 pp) = 0.346 as archived) | [−5.336, 32.731] pp | T3 |
| Residualisation drop | deduplicated (89) | 24.1 pp | dedup_r1_comparison.csv | 1.0 pp | recomputed (90.23 → 66.13) | 24.102 pp | T3 |
| Residualisation drop, date only | compute_known (58) | 16.5 pp | subsample_date_compute.csv (16.48) | 1.0 pp | recomputed (66.32 → 49.84; subsample definition confirmed, open item 2) | 16.479 pp | T3 |
| Residualised oblimin loadings, economic benchmarks on the economic factor | complete_case | 1.01 / 0.84 / 0.54 / 0.50 | loadings_residualised.csv; paper Table tab:loadings | 0.02 | recomputed (economic factor = F1; φ 0.3643 / 0.6718 / 0.7205 and uniquenesses also match the archive) | 1.0063 / 0.8418 / 0.5420 / 0.4991 | T3 |
| Largest economic cross-loading | complete_case | 0.38 | loadings_residualised.csv (0.3826) | 0.02 | recomputed | 0.3826 | T3 |
| Pooled economic ΔMSE, mean index vs k-factor, ridge | complete_case | +0.037 | h4_bootstrap_dmse.csv; task2_prediction_results.json (0.037342) | 0.002 | recomputed (per target 0.0263 / 0.0284 / 0.0564 / 0.0383 as archived) | +0.0373 | T4 |
| Its bootstrap 95 % interval | complete_case | [+0.019, +0.055] | h4_bootstrap_dmse.csv | 0.005 per endpoint | statistically reproduced (the archived table is read, not derived, by the notebook; its draw order is unrecoverable — decisions T4) | [+0.0180, +0.0558] fresh stream; [+0.0189, +0.0556] in H4-cell order | T4 |
| Pooled economic ΔMSE | deduplicated (89) | +0.038 | h4_bootstrap_dedup.csv (0.0378) | 0.002 | recomputed, interval too ([0.0198, 0.0561] exactly; per target 0.0039 / 0.0370 / 0.0721 / 0.0381) | +0.0378 | T4 |
| Ladder, single index (ii), best learner | complete_case, economic block | RMSE 0.474, R² 0.771 | lobo_rung_summary.csv | 0.005 each | recomputed (ridge; mean-of-RMSE pooling, decisions T4; four learners, about 10 min on two cores) | RMSE 0.4745, R² 0.7710 (train RMSE 0.4631 vs 0.463) | T4 |
| Ladder, k-factor (iv), best learner | same | RMSE 0.433, R² 0.808 | lobo_rung_summary.csv | 0.005 each | recomputed (ridge) | RMSE 0.4333, R² 0.8077 (train 0.4098 vs 0.410) | T4 |
| Ladder, first factor alone (iii), best learner | same | RMSE 0.950 | lobo_rung_summary.csv | 0.005 | recomputed (elastic net; R² 0.1096 vs 0.110; the other two rungs the paper prints, i and v, give rf 0.7406 / R² 0.4587 and ridge 0.4382 / R² 0.8005 against 0.741 / 0.459 and 0.438 / 0.800) | RMSE 0.9503 | T4 |
| Grid sizes | complete_case, deduplicated, compute_known, economic_dense | 96, 89, 58, 103 | task1_structure_results.json, dedup_r1_comparison.csv, subsample_date_compute.csv, archive README | exact | recomputed | 96, 89, 58, 103 (T2, `pytest -m golden -k grid_size`: 4 passed) | T2 |

### One Capability, the twelve archived tables (T7)

`python -m benchprobe.report one-capability --out DIR` writes all twelve with captions and a
`comparison.csv`. The table below is the run of 7 September 2026; the fast path (no ladder refit)
takes 25–50 s on two cores depending on load, the `--full-ladder` run took 33 minutes and
reproduced `lobo_rung_summary.csv` byte for byte (both scopes, all five rungs, best learners and
every value at the archive's three decimals).

| Table | Tier | Deviation from archive | Tolerance |
|---|---|---|---|
| loadings_raw.csv | recomputed | 1.4e-6 | 5e-4 |
| loadings_residualised.csv | recomputed | 1.6e-7 | 5e-4 |
| task1_structure_results.json | recomputed (logit share and Kearns constants regenerated) | 1.5e-4 | 5e-4 |
| cluster_validation.csv | regenerated (clustering outside SPEC §1) | — | — |
| lobo_rung_summary.csv | regenerated on the fast path; recomputed with `--full-ladder` (four learners, twelve targets) | 0 (identical) | 1e-3 |
| h4_bootstrap_dmse.csv | statistically reproduced (points exact at 3 dp; excludes_zero agrees on every row) | 1.97 MC SD, worst endpoint | 3 MC SD |
| task2_error_analysis_gdpval.csv | recomputed | 7.0e-7 | 5e-4 |
| k_selection_evidence.csv | recomputed (BIC row regenerated) | 0 | exact on selected_k |
| ksweep_rung_iv.csv | statistically reproduced (points within 2e-4; excludes_zero agrees on every row) | 1.34 MC SD (compared before rounding, T7.1) | 3 MC SD |
| efa_factor_correlations.csv | recomputed | 0 | 5e-4 |
| efa_uniquenesses.csv | recomputed | 0 | 5e-4 |
| eda_distribution_stats.csv | recomputed | 4e-16 | 5e-4 |

### JUDGe (T5)

| Output | Archive value | Archive file | Tolerance | Tier | Recomputed | Ticket |
|---|---|---|---|---|---|---|
| Judged items | 180 of 210 | make_figures.load_bank | exact | recomputed | 180 | T5 |
| KR-20, judge verdicts / gold verdicts | 0.5223 / 0.5231 | README | 0.00005 | recomputed | 0.5223 / 0.5231 | T5 |
| Per-element judge error | 4.72 % | README | 0.005 pp | recomputed | 4.72 % | T5 |
| Judge–gold correlation; leniency | 0.921; +0.46 elements | README | 0.0005; 0.005 | recomputed | 0.9206; +0.4611 | T5 |
| Beta-binomial fit on judge totals | χ² 5.31, 6 df, p 0.504 | README | 0.005; exact; 0.0005 | recomputed (α 5.4521, β 3.6632) | χ² 5.3122, 6 df, p 0.5044 | T5 |
| Simulated quantities (two-way grid, controls, G-study) | the archive's `sweeps.py` output | `tests/golden/judge_sweeps_reference.json` | 1e-9 | recomputed (seeds 11 / 101 / 303 / 7, draw order preserved) | identical (0.0) | T5 |
| Published two-way KR-20 grid | `sweep_grid.json` | vendored | 3 SE of the difference of two 60-replicate means | statistically reproduced (different stream, README) | 1.72 SE worst cell, 0.88 mean (re-expressed at T5.1; the 0.315 SD recorded before was the same gap on the wrong scale) | T5, T5.1 |
| Published 60-replicate run at 4.72 % error | `sweep_grid.json` | vendored | 3 SE of the difference of two 60-replicate means | statistically reproduced (seed unknown) | within tolerance | T5, T5.1 |

### Price of Intelligence — item-response fit (T6)

Snapshot `price_of_intelligence_2026-08-09` (archive v1.0.0, DOI 10.5281/zenodo.22177190; all four
vendored files hash-identical to the archive's own `MANIFEST.sha256`). **The archive ships no
estimation code**, so `benchprobe.irt` is a re-implementation of the model the archive states it
fitted — Samejima's continuous response model on logit scores, homoscedastic — validated against
its published outputs. The *recomputed* tier below is claimed for these quantities and no others.
Every row requires the scale repair (`docs/decisions.md`, 2026-09-10); without it none reproduces.
Locked in `tests/golden/test_price_of_intelligence.py`; whole file runs in 4 s.

| Output | Archive value | Archive file | Tolerance | Tier | Recomputed | Ticket |
|---|---|---|---|---|---|---|
| Scored cells / models / benchmarks | 4605 / 782 / 64 | theta_estimation.json | exact | recomputed | 4605 / 782 / 64 | T6 |
| Cells squeezed off the boundary | 89 | theta_estimation.json | exact | recomputed | 89 | T6 |
| Models with an anchor cell | 480 | theta_estimation.json | exact | recomputed | 480 | T6 |
| Free parameters, stage 1 / stage 2 | 607 / 891 | theta_estimation.json | exact | recomputed | 607 / 891 | T6 |
| Stage-1 objective (anchors only) | −1044.5802 | theta_estimation.json | 0.00005 | recomputed | −1044.58017655 | T6 |
| Stage-1 residual scale | 0.30762771382713755 | theta_estimation.json | 1e-6 | recomputed | 0.30762771557809 | T6 |
| Stage-2 objective (full panel) | −1328.6389 | theta_estimation.json | 0.00005 | recomputed (priors on held-fixed coordinates excluded; see decisions) | −1328.63888738 | T6 |
| Stage-2 residual scale | 0.44927472556854087 | theta_estimation.json | 1e-6 | recomputed | 0.44927473150302 | T6 |
| Item difficulty, all 64 benchmarks | item_parameters.csv | item_parameters.csv | 5e-4 | recomputed | worst 7.6e-5 | T6 |
| Item discrimination, all 64 benchmarks | item_parameters.csv | item_parameters.csv | 5e-4 | recomputed | worst 1.3e-5 | T6 |
| Ability θ, all 782 models | se_theta_structure.csv | se_theta_structure.csv | 1e-3 | recomputed | worst 3.3e-4 | T6 |
| se(θ), all 782 models | se_theta_structure.csv | se_theta_structure.csv | 5e-4 | recomputed (cell information + ability prior) | worst 2.7e-5 | T6 |
| Test information; se predicted from it | se_theta_structure.csv | se_theta_structure.csv | 5e-3; 5e-4 | recomputed (distinct-benchmark information — a different set from se(θ); finding, see decisions) | worst 8.7e-4; 2.9e-5 | T6 |
| Mean discrimination per model | se_theta_structure.csv | se_theta_structure.csv | 5e-4 | recomputed (over distinct benchmarks) | worst 1.1e-5 | T6 |

Not reproduced, and not claimed: the archive's recorded `initial_objective` and
`after_adam_objective` for stage 2 (10777.8325 / −1328.2421). Its stage-2 *starting point* is not
recorded, and benchprobe's differs; stage 1 matches on both (1875.7572 initial). These are
optimiser-path diagnostics, not results, and the two stages land on the same optimum either way —
recorded here so the gap is not mistaken for agreement. Also out of scope at T6:
`information_curves.csv`, `dif_tests.csv`, `agreement_2pl.json`, `saturation*`, `kane_audit.json`
and the sensitivity fits — none is vendored and none is claimed.

## Seeds and random streams recovered from the archives

One Capability (`frontier-ai-economic-validity`, `notebook/analysis.ipynb`, commit `946ce845`):

| Draw | Stream | Where |
|---|---|---|
| Horn's parallel analysis | `numpy.random.default_rng(42)`, 1000 draws | Task-1 cell "PCA and parallel analysis" |
| LOBO outer folds | `KFold(5, shuffle=True, random_state=0)` | Task-2 cell "LOBO nested-CV loop" |
| Random forest, gradient boosting | `random_state=0` | same |
| H4 pooled ΔMSE bootstrap | `numpy.random.default_rng(42)`, B = 2000, one stream shared in the order `i_date`, `ii_meanidx`, `iii_f1`, then the four per-target bootstraps | Task-2 cell "H4 — bootstrap ΔMSE" |
| H2(ii) share-drop bootstrap | `numpy.random.default_rng(42)`, B = 2000 | cell "Registration-honest robustness statistics" |
| Target-clustered ΔMSE bootstrap | `numpy.random.default_rng(42)`, 5000 draws | same |
| Deduplicated-grid H4 re-run | `KFold(5, shuffle=True, random_state=0)`; bootstrap `default_rng(42)` | cell "Registered R1 deduplication" |
| Model clustering (not in scope) | `KMeans(n_init=10, random_state=0)` | Task-1 cell "Clustering" |

JUDGe (`llm-judge-reliability`, commit `126d1bce`): real-bank quantities reproduce bit-for-bit
(README, confirmed at T5); simulated sweeps carry fixed seeds (`two_way_sweep` 11, `phi_control` 101,
`ll_estimand_control` 303, `phrasing_gstudy` 7; `cluster_bootstrap` 0) and one stream per sweep in
the archive's order, which `trust.sweeps()` reproduces exactly; the published two-way grid came from
a different RNG stream, so re-simulation lands within a third of a per-cell standard deviation — the
statistically-reproduced tier by construction (0.315 SD measured at T5).

## Expected runtime and cost

From the One Capability archive README, laptop-class 12-core machine: the whole notebook about
26 minutes end to end with all four learners, nearly all of it in the LOBO ladder and the
hyperparameter-table sweep; about 75 seconds on the ridge-only path; everything else seconds.
Measured here (two cores, T3–T4, unloaded to lightly loaded): the four-learner LOBO on the four
economic targets 572 s; the ridge-only LOBO on those targets 10–20 s; the H2(ii) bootstrap (4000 ML
factor fits) 90–125 s; `pytest -m golden` 11–12 minutes in all, `pytest -m "golden and not slow"`
about 2 minutes, `pytest -m smoke` 10–25 s. No output needs inference credits, an API key or network access: the
package analyses tables (handoff §8).

## Environment

`uv.lock` pins the environment. At T0 it resolved to numpy 2.5.3, pandas 3.0.5, scipy 1.18.1,
scikit-learn 1.9.0, factor_analyzer 0.5.1, statsmodels 0.15.0 on CPython 3.13; the archive ran
numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0, factor_analyzer 0.5.1,
statsmodels 0.14.6 on Python 3.13. Any golden difference attributable to this drift is reported with
both values (rule 3) and decided at T3.

## Open items (report, do not guess)

0. **The Price of Intelligence panel's `score_scale`/`score_divisor` metadata is wrong for four
   benchmarks** (221 of 4605 scored cells), and the archive's `se_theta_structure.csv` reports a
   `test_information` computed over a different set from the `se_theta` in the same row. Both are
   evidenced, reproduced and locked at T6 (`docs/decisions.md`, 2026-09-10). Both need fixing at
   source; neither changes a published number.

1. **Are the 103-model economic-dense subset and the 96-model complete-case grid the same object?**
   Evidence at T0, computed from the vendored table: the 96 rows complete on all twelve benchmarks are a
   strict subset of the 103 rows carrying GDPval, Terminal-Bench v2.1 and τ³-Banking; the seven extra
   rows lack at least one of the other nine. The archive's results files use n = 96 for both KMO
   (`n_G1`) and LOBO (`n_LOBO`), and the paper's text says "the $n=96$ complete-case grid"; the archive
   README's line "Economic-dense subset: 103 models … (Task-2 grid)" is the one surface that disagrees.
   Settled at T2: `build_grid` produces both (96 and 103), the golden grid rows pass, and
   `tests/io/test_io.py::test_economic_dense_contains_the_complete_case_grid` pins the strict nesting.
   The README wording is a finding for the coordinating thread.
2. **Which of the twelve archived tables can be recomputed at all from the archived inputs?** The
   archive README lists twelve tables the notebook reads rather than derives. Two acceptance rows
   depended on values that exist only in those tables: the logistic-fit date-R² of 0.505 (the notebook
   derives the OLS 0.477 only) and the compute-known subsample (n = 58; the notebook does not build it).
   Both recomputed at T3: the four-parameter logistic reproduces 0.505 / 0.364 / 0.286, and
   `complete_case ∩ non-null totalParameters` reproduces 66.32 / 49.84. Closed at T7 (table above):
   ten of the twelve are recomputed or statistically reproduced; `cluster_validation.csv` is out of
   scope; within the recomputed tables the logit first-factor share, the Kearns constants and the
   BIC row are regenerated because their computations are not in the archive.
