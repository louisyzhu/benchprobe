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

None computed yet. The One Capability acceptance rows (handoff §6) are encoded in
`tests/golden/test_one_capability.py` (T1) with the tolerances below; each row gets its tier and its
recomputed value from the ticket that computes it. "Archive file" is where the value was verified on
6 September 2026 (`frontier-ai-economic-validity`, commit `946ce845`).

| Output | Grid | Archive value | Archive file | Tolerance | Tier | Recomputed | Ticket |
|---|---|---|---|---|---|---|---|
| KMO | complete_case (96) | 0.933 | task1_structure_results.json (0.9326) | 0.005 | not yet computed | — | T3 |
| First-factor share of common variance | complete_case | 74.5 % | task1_structure_results.json (0.74544) | 0.5 pp | not yet computed | — | T3 |
| Logistic-fit date-R² of the dominant factor | complete_case | 0.505 | hypothesis_adjudication.csv (OLS 0.477 in task1_structure_results.json) | 0.005 | not yet computed; functional form to recover (open item 2) | — | T3 |
| Residualised first-factor share | complete_case | 59.6 % | task1_structure_results.json (0.59635) | 0.5 pp | not yet computed | — | T3 |
| Residualisation drop | complete_case | 14.9 pp | task1_structure_results.json (14.909) | 1.0 pp | not yet computed | — | T3 |
| Residualisation drop, bootstrap 95 % interval | complete_case | [−5.3, +32.7] pp | bootstrap_h2_drop.csv (−5.337, 32.731) | 2.0 pp per endpoint | not yet computed; statistically reproduced unless draw order matches | — | T3 |
| Residualisation drop | deduplicated (89) | 24.1 pp | dedup_r1_comparison.csv | 1.0 pp | not yet computed | — | T3 |
| Residualisation drop, date only | compute_known (58) | 16.5 pp | subsample_date_compute.csv (16.48) | 1.0 pp | not yet computed; subsample not derived in the archived notebook (open item 2) | — | T3 |
| Residualised oblimin loadings, economic benchmarks on the economic factor | complete_case | 1.01 / 0.84 / 0.54 / 0.50 | loadings_residualised.csv; paper Table tab:loadings | 0.02 | not yet computed | — | T3 |
| Largest economic cross-loading | complete_case | 0.38 | loadings_residualised.csv (0.3826) | 0.02 | not yet computed | — | T3 |
| Pooled economic ΔMSE, mean index vs k-factor, ridge | complete_case | +0.037 | h4_bootstrap_dmse.csv; task2_prediction_results.json (0.037342) | 0.002 | not yet computed | — | T4 |
| Its bootstrap 95 % interval | complete_case | [+0.019, +0.055] | h4_bootstrap_dmse.csv | 0.005 per endpoint | not yet computed; statistically reproduced unless draw order matches | — | T4 |
| Pooled economic ΔMSE | deduplicated (89) | +0.038 | h4_bootstrap_dedup.csv (0.0378) | 0.002 | not yet computed | — | T4 |
| Ladder, single index (ii), best learner | complete_case, economic block | RMSE 0.474, R² 0.771 | lobo_rung_summary.csv | 0.005 each | not yet computed (slow: four learners) | — | T4 |
| Ladder, k-factor (iv), best learner | same | RMSE 0.433, R² 0.808 | lobo_rung_summary.csv | 0.005 each | not yet computed (slow) | — | T4 |
| Ladder, first factor alone (iii), best learner | same | RMSE 0.950 | lobo_rung_summary.csv | 0.005 | not yet computed (slow; platform-sensitive per archive README) | — | T4 |
| Grid sizes | complete_case, deduplicated, compute_known, economic_dense | 96, 89, 58, 103 | task1_structure_results.json, dedup_r1_comparison.csv, subsample_date_compute.csv, archive README | exact | recomputed | 96, 89, 58, 103 (T2, `pytest -m golden -k grid_size`: 4 passed) | T2 |

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
(README); simulated sweeps carry fixed seeds (`two_way_sweep` 11, `phi_control` 101,
`ll_estimand_control` 303, `phrasing_gstudy` 7) but the published two-way grid came from a different
RNG stream, so re-simulation moves cells by about a third of a per-cell standard deviation — the
statistically-reproduced tier by construction. Details at T5.

## Expected runtime and cost

From the One Capability archive README, laptop-class 12-core machine: the whole notebook about
26 minutes end to end with all four learners, nearly all of it in the LOBO ladder and the
hyperparameter-table sweep; about 75 seconds on the ridge-only path; everything else seconds. No output
needs inference credits, an API key or network access: the package analyses tables (handoff §8).

## Environment

`uv.lock` pins the environment. At T0 it resolved to numpy 2.5.3, pandas 3.0.5, scipy 1.18.1,
scikit-learn 1.9.0, factor_analyzer 0.5.1, statsmodels 0.15.0 on CPython 3.13; the archive ran
numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0, factor_analyzer 0.5.1,
statsmodels 0.14.6 on Python 3.13. Any golden difference attributable to this drift is reported with
both values (rule 3) and decided at T3.

## Open items (report, do not guess)

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
   depend on values that exist only in those tables: the logistic-fit date-R² of 0.505 (the notebook
   derives the OLS 0.477 only) and the compute-known subsample (n = 58; the notebook does not build it).
   T3 reports which rows are recomputable, which are regenerated, and why.
