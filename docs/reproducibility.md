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

None computed yet (T0). The One Capability acceptance rows are listed with their tolerances in
`tests/golden/test_one_capability.py` from T1; each moves into the table below, with its tier, when
the ticket that computes it lands.

| Output | Study | Tier | Archive value | Recomputed value | Tolerance | Ticket |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

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
   T2's grid tests encode both counts; the README wording is a finding for the coordinating thread.
2. **Which of the twelve archived tables can be recomputed at all from the archived inputs?** The
   archive README lists twelve tables the notebook reads rather than derives. Two acceptance rows
   depend on values that exist only in those tables: the logistic-fit date-R² of 0.505 (the notebook
   derives the OLS 0.477 only) and the compute-known subsample (n = 58; the notebook does not build it).
   T3 reports which rows are recomputable, which are regenerated, and why.
