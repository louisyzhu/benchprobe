# Decisions

One dated line per design decision: what, who, why. "Coordinating session" is the Claude session that
wrote the ticket; "Louis" is the maintainer. Nothing here changes a golden number, a tolerance or a seed
without a ticket that says so.

## 2026-09-06 — T0 skeleton

- Src layout with one flat module per SPEC family (`trust`, `measure`, `predict`, `irt`, `io`, `report`), no subpackages — coordinating session; the handoff names modules, and flat modules keep the import surface identical to SPEC.md §1. `benchprobe.io` keeps the handoff's name; absolute imports mean it cannot shadow the standard-library `io` inside the package.
- `requires-python >= 3.12`, `.python-version` 3.13 — coordinating session; the One Capability archive ran on 3.13 and its scipy lower bound (1.18.0) needs ≥ 3.12 (uv resolution error under ≥ 3.11, seen at T0).
- Runtime dependencies declared at T0 with lower bounds equal to the archive's pinned versions (numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0, factor_analyzer 0.5.1, statsmodels 0.14.6); exact versions pinned by `uv.lock` — coordinating session; rule 9 needs a lockfile from the first commit and the extraction tickets should not churn it. The lock resolved to numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0 (newer than the archive), scikit-learn 1.9.0 and factor_analyzer 0.5.1 (identical). Whether to pin the archive's exact versions instead is T3's call, taken on the golden-test result.
- PyTorch as the optional extra `irt`, nothing else optional — coordinating session; handoff §1 ("Python package", torch optional for IRT).
- Build backend hatchling; the vendored data under `src/benchprobe/data/` ships in the wheel — coordinating session; `load_snapshot()` must work after `pip install` with no download step.
- The One Capability analysis-ready table (`aa_analysis_models.csv`, 137 047 bytes, SHA-256 `0c78f8c6…d569`, CC BY 4.0) vendored at T0 with a manifest recording its source commit and the upstream raw-snapshot hashes; raw snapshots not vendored — coordinating session; the golden tests need a hash-pinned input on a fresh clone without network, and the raw HTML (4.5 MB, site terms) is not needed for any §6 row. Whether M3's twelve-table recomputation needs the raw parse is decided at T7.
- Markers `smoke`, `golden`, `slow` with `--strict-markers`; CI runs `smoke` only and collects `golden` without running it — coordinating session; rule 6 (sub-minute smoke test) and handoff §4 ("run and paste" is the only evidence for golden results).
- CI pins actions by major tag (`actions/checkout@v4`, `astral-sh/setup-uv@v6`) — coordinating session; could not be verified against GitHub from the T0 environment, see the T0 report.
- Licence file deferred to v0.1, README says "all rights reserved until then" — coordinating session; handoff §3 lists licence and CITATION.cff as v0.1 items and the choice is Louis's.
- Commits authored as `Louis Yiven Zhu <louisyzhu@users.noreply.github.com>` with the agent as co-author — coordinating session; the maintainer pushes and may reset the author before doing so.

## 2026-09-06 — T1 golden tests

- Golden tests import each SPEC function inside the test body, so an unimplemented function fails that test with a named reason instead of breaking collection — coordinating session; "all failing" must be visible per row, and the smoke test and `--collect-only` must keep working.
- Provisional signatures for `io`, `measure` and `predict` written into SPEC.md §2 so the tests can call them; the implementing ticket may amend a signature and then changes the call, never the number — coordinating session; handoff §4 ("function signatures agreed per ticket").
- Row "Three-factor loadings" encoded against the *residualised* (date-partialled) oblimin solution — coordinating session; the archive's `loadings_residualised.csv` and the paper's Table (tab:loadings) carry 1.01 / 0.84 / 0.54 / 0.50 and max cross-loading 0.38; the raw solution (`loadings_raw.csv`) has 1.05 / 0.87 / 0.57 / 0.71. The handoff row does not say which; the numbers decide.
- Row "Residualised first-factor share": "deduplicated 24.1" and "compute-known 16.5" encoded as *drops in percentage points* on the `deduplicated` (n = 89) and `compute_known` (n = 58, date-only) grids — coordinating session; `dedup_r1_comparison.csv` (drop 24.1) and `subsample_date_compute.csv` (date_only drop 16.48) fix the meaning.
- Economic factor identified as the factor carrying the largest sum of squared loadings across the four economic benchmarks, not assumed to be column 0 — coordinating session; a rotation may permute columns without changing any number, and the paper's own reading (F1 = agentic/work-realistic) is what the test checks.
- Tolerances the handoff did not set, fixed at T1 and adjustable only by ticket: loadings 0.02; RMSE 0.005; a drop in share 1.0 pp (it is the difference of two shares each carrying 0.5 pp); bootstrap-interval endpoints 0.005 (ΔMSE) and 2.0 pp (share drop), the statistically-reproduced tier — coordinating session; reported precision is two or three decimals and the archive's own re-run check asserts at 5e-4, so these are loose by design and tighten only by ticket.
- Ladder rows (best learner per rung) marked `slow`: they need the four-learner refit (most of the archive's 26-minute notebook run); ΔMSE rows need ridge only — coordinating session; keeps `pytest -m "golden and not slow"` usable in an evening.
- `residualise_on_date` operates on the indicator matrix, then the factor solution is re-estimated — coordinating session; handoff §2 says "date-residualisation of a factor", the archived computation residualises the twelve indicators (notebook cell "Residualise on release date — H2"), and the code is what gets extracted. Wording in the handoff to be reconciled by the coordinating thread.
- `compute_known` grid defined provisionally as `complete_case` rows with non-null `totalParameters` — coordinating session; that selection gives n = 58, the archived count, but the archived notebook does not derive the subsample; T3 confirms by reproducing 66.32 / 49.84 (raw / date-only share).
- `measure.residualisation_drop_bootstrap` added to SPEC §2 as the H2(ii) bootstrap (model resampling, B = 2000, seed 42) so the [−5.3, +32.7] row has a callable — coordinating session; the archive computes it inline in the "Registration-honest robustness statistics" cell.
- `economic_dense` grid is counted only; its `.benchmarks` are not asserted — coordinating session; the grid is incomplete on the other nine benchmarks by construction and no §6 row computes on it.

## 2026-09-06 — T2 io

- `MANIFEST.json` gains a `table` field naming the file to load; every file under `files` is hashed and size-checked on load — coordinating session; one manifest schema for every future snapshot (thesis tables included) without hard-coding file names in `io`.
- Grids reproduce the archive's row selections verbatim, including `deduplicate_by_base_model`'s `groupby("base").first()` — coordinating session. Finding: pandas' `first()` takes the first *non-null* value per column, so for 4 of the 320 base models the deduplicated row carries a benchmark score from a lower-ranked configuration; a strict one-row-per-base-model selection (`head(1)`) gives 86 complete rows, the archived computation 89. benchprobe reproduces the archived 89 (and its 24.1 pp / +0.038 rows); the paper's R1 wording ("one row per base model") is a finding for the coordinating thread, not something this repository changes. `tests/io/test_io.py::test_deduplication_is_first_non_null_per_column_as_archived` pins both counts.
- `economic_dense` is a coverage grid: `.z` keeps NaN and `.days` may be NaN (one of its 103 rows has no release date); every other grid is strict — coordinating session; no §6 row computes on it, and the strict grids must never carry NaN into the estimators.
- `io` unit tests carry the `smoke` marker — coordinating session; they run in well under a second and rule 6 wants every module's smoke test in CI.
- Config-driven runs deferred to T7 — coordinating session; nothing before the one-command reproduction needs a config file, and rule 4 forbids speculative scope.

## 2026-09-06 — T3 measure

- The logistic form behind the paper's date-R² (0.505) recovered as the four-parameter logistic `floor + upper/(1 + exp(−rate(t − midpoint)))`, least squares, best of a 4 × 3 grid of starts — coordinating session; it was not in the archived notebook (OLS 0.477 was). Evidence: it reproduces all three archived per-factor values at once (0.5048 / 0.3641 / 0.2859 against 0.505 / 0.364 / 0.286, `PHASE2_REPORT.md`), which a form tuned to one number would not; the three-parameter logistic (no floor) gives 0.235 and is ruled out. Louis to confirm against the pre-registration text; the code is what reproduces the paper either way.
- `compute_known` confirmed as `complete_case ∩ non-null totalParameters`: raw share 66.32 → date-only 49.84, drop 16.48 pp, the archived `subsample_date_compute.csv` values to the second decimal — coordinating session; open item 2's second half closes.
- `first_factor_share` takes column 0 of the unrotated ML solution, as the archive's `factor1_share` does, without re-ordering by size — coordinating session; on every grid tested column 0 is also the largest, and the archive's number is what the test locks.
- Sign alignment applied to loadings, structure, weights and `phi` together with the scores — coordinating session; the archive flips scores (Task-1 date cell) and loadings (residualised-loadings cell) by the same score-based sign, and a consistent solution needs `phi` flipped too. All signs were +1 on the archive's grids, so no reported number moves.
- The factor_analyzer/scikit-learn shim is applied inside `measure` at first use, never at import — coordinating session; verified needed under the locked scikit-learn 1.9.0 (`transform` passes `force_all_finite`), and a shim at import time would patch scikit-learn for every user of the process.
- Environment drift (numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0 vs the archive's) moved no measure-layer number beyond the fourth decimal; the archive's exact versions are not pinned — coordinating session; `docs/reproducibility.md` records both values per row.
- Measure unit tests carry the `smoke` marker (synthetic data, under two seconds) — coordinating session; rule 6.

## 2026-09-06 — T4 predict

- `lobo` reproduces the archive's Task-2 loop verbatim (train-fold standardisation, in-fold oblimin ML solution projected by `pinv(R) Λ`, sign alignment to the mean index, inner `GridSearchCV(cv=5)` on the train-fold-standardised target, metrics pooled over outer folds) and accepts `rungs` and `learners` subsets — coordinating session; the golden ΔMSE rows need ridge and two rungs, the ladder rows need all four learners and all five rungs, and the thesis pipeline will need neither in full.
- The pooled-ΔMSE bootstrap draws `rng.integers(0, n, n)` per replicate from a fresh `default_rng(seed)`; `stream=` lets `h4_table` draw a whole table from one stream in the archive's H4-cell order — coordinating session. Finding: the archived `h4_bootstrap_dmse.csv` is one of the twelve tables the notebook reads rather than derives, and it carries per-target `i_date` rows the current H4 cell does not compute, so the draw order that produced its intervals is not in the archive; its intervals are the statistically-reproduced tier ([0.0180, 0.0558] fresh stream and [0.0189, 0.0556] in H4-cell order, against the archived [0.019, 0.055]). The deduplicated-grid table *is* notebook-derived and reproduces exactly ([0.0198, 0.0561]).
- `ladder()` pools per-target RMSEs by their arithmetic mean (`pooling="mean_rmse"`), with the notebook helper's root-mean-square available as `pooling="rms"` — coordinating session. Finding: the archived `lobo_rung_summary.csv` (one of the twelve read-not-derived tables) and the paper's ladder table reproduce from `lobo_metrics_full.csv` only under mean-of-RMSE pooling (all ten rows, both scopes, to three decimals), while the notebook's `_ladder` helper — the one its self-verification uses — pools by root-mean-square and gives 0.483 / 0.442 where the paper prints 0.474 / 0.433. The paper's numbers are the golden numbers; the helper-vs-table inconsistency and the method text's silence on the pooling are findings for the coordinating thread.
- Finding, platform sensitivity: rerunning the four learners here reproduces the archived `lobo_metrics_full.csv` cell for cell to 1e-6 for ridge, elastic net and gradient boosting, and to 1e-16 on the date and mean-index rungs for every learner; random forest differs on the factor rungs (max 6.6e-4 on rung iv, 1.7e-2 on rung v), the pattern the archive README describes (a split threshold flips on a factor score that differs in the last digits). No ladder row the paper prints depends on a random-forest cell except rung i (rf, 0.7406 here against 0.741), which is within tolerance. Recorded, not corrected.
- Golden ladder rows run on the four economic targets only — coordinating session; each target's LOBO fit is independent of the others, so the economic-block ladder and ΔMSE are identical to a twelve-target run, at a third of the cost.
- Hold-out by task, model, family or context, and effective-sample-size reporting (SPEC §1, "thesis pipeline needs") not implemented at T4 — coordinating session; they need the thesis design, which handoff §8 keeps out of this repository until the OSF deposit. Ticketed then.
- Predict unit tests carry the `smoke` marker (ridge only; about 20 s) — coordinating session; rule 6's one-minute bound holds for the module and the full smoke set.

## 2026-09-06 — T4.1 review fixes (after the second same-family review)

- Factor scores computed explicitly (`(z − mean)/std @ solve(R, S)`), reproducing `factor_analyzer.transform` without its silent fallback to `Λ` on a singular `R`; blanket `warnings` suppression removed from `measure` (factor_analyzer emits none on the archive grids or under resampling, checked over 300 bootstrap fits) and narrowed to scikit-learn's `ConvergenceWarning` around the inner grid search in `predict` (the archive silences all warnings in its Task-2 setup) — coordinating session; a swallowed substitution would change a reported number without a trace.
- `io.zscore` raises on a zero-variance column instead of returning NaN; `date_r2` raises on a constant score vector; `first_factor_share` warns if column 0 is not the largest factor; unknown learner names raise `ValueError` on every path; `ladder` and `h4_table` name the missing rung instead of failing with `KeyError`/`IndexError`; `lobo` checks release dates only when rung `i_date` is requested; result dataclasses use identity equality (`eq=False`) — coordinating session; none of these changes a number on the archive grids (smoke and golden sets re-run).
- `statsmodels` removed from the runtime dependencies (declared at T0 from the archive's requirements, imported nowhere) — coordinating session; lockfile regenerated.
- Deduplicated frame keeps the `base` key as its last column — coordinating session; the other grids' column order is the snapshot's.
- Runtime figures in `docs/reproducibility.md` given as ranges measured under light load — coordinating session; the reviewer measured 123 s and 19 s where the first run measured 90 s and 10 s.

## 2026-09-07 — T5 trust

- JUDGe item bank and published grid vendored as snapshot `judge_2026-08-31` (CC BY 4.0, commit `126d1bce`), loaded through `io.load_snapshot` like the One Capability table; `Snapshot` gains `.folder` and `.file(name)` so a multi-file snapshot's extra files are reachable only through the manifest — coordinating session; one loader, one integrity rule.
- Archive function names kept (`kr20`, `bb_gof`, `livingston_lewis`, `phi_lambda`, …), with `bb_gof` and `livingston_lewis` returning named tuples instead of bare tuples — coordinating session; names join to the archive. (Corrected at T8: the archive's `bb_gof` returned `(chi2, df, p, (a, b))`, four values with a nested pair; benchprobe's `BetaBinomialFit` is five flat fields, so four-value unpacking written against the archive does *not* work. The earlier line said it did; that was wrong.)
- JUDGe golden rows added in `tests/golden/test_judge.py`: the six real-bank quantities the archive README states reproduce bit-for-bit (tolerance half a unit of the last printed digit), the archive's own `sweeps.py` output (tolerance 1e-9; `tests/golden/judge_sweeps_reference.json` is that script's output under the locked environment, 7 Sept 2026), and the published grid and measured-error run at 0.5 SD per cell — coordinating session; handoff §6 fixes only the One Capability rows, so these are the T5 acceptance test the handoff's T5 line ("document the statistically-reproduced tier and seeds") implies. Result: real-bank rows recomputed exactly (0.5223 / 0.5231 / 4.72 % / 0.9206 / +0.4611 / χ² 5.3122, df 6, p 0.5044); `sweeps()` reproduces the archive script to 0.0; the published grid to 0.315 SD (README: 0.31).
- Krippendorff's α implemented from the coincidence-matrix definition (nominal, ordinal, interval, ratio) and Cohen's κ delegated to scikit-learn — coordinating session (new code, in scope by SPEC §1). Reference values: Krippendorff (2011)'s worked example (0.7434 / 0.8154 / 0.8491 / 0.7974, matching the published 0.743 and 0.849) and six random fixtures computed with the `krippendorff` package (`tests/trust/krippendorff_reference.json`, agreement 2e-16); κ against the classic 2 × 2 example (0.4).
- `gstudy_two_way` factored out of the archive's `phrasing_gstudy` loop so the variance components are callable on real data — coordinating session; `phrasing_gstudy` calls it and still reproduces the archive to 0.0.
- Trust unit tests carry the `smoke` marker (about a second) — coordinating session; rule 6.

## 2026-09-07 — T7 report

- The twelve archived tables are vendored under the One Capability snapshot (`archived/`, hashed in the manifest) so the one-command reproduction can state each table's deviation from the archive on a fresh clone — coordinating session; the M3 gate is "diffs against archive within tolerance", and a diff needs the archive present.
- Tiers per table fixed as in SPEC §2 (eight recomputed, two statistically reproduced, `cluster_validation.csv` regenerated as out of scope, `lobo_rung_summary.csv` recomputed only with `--full-ladder`) — coordinating session. Fields that could not be recomputed are regenerated and named in the caption rather than approximated: the logit first-factor share (91.81 %; the archive's transform is not recorded — a logit of the bounded benchmarks clipped at 1e-3 gives 91.9 %, at 5e-3 91.36 %, so the form is not recoverable from the number), the two Kearns literature constants, and the BIC row of `k_selection_evidence.csv` (no BIC computation in the archived notebook).
- Bootstrap intervals in the two statistically-reproduced tables are judged in Monte-Carlo standard deviations: the same bootstrap is re-run under 20 other seeds, and an archived endpoint counts as reproduced when it lies within 3 SD of benchprobe's after subtracting the archive's rounding — coordinating session; a fixed absolute tolerance would have been a guess (the per-target `i_date` intervals move by ±0.004–0.007 between seeds, the pooled `ii_meanidx` interval by ±0.0005), and this criterion calibrates itself. Result on the fast path: worst cell 1.97 SD (`h4_bootstrap_dmse.csv`, τ³-Banking vs `i_date` upper endpoint) and 1.24 SD (`ksweep_rung_iv.csv`); every point estimate matches the archive at its printed precision.
- Findings on the archived tables, recorded not corrected: `ksweep_rung_iv.csv` pools RMSE as the root of the mean per-target MSE (0.4424 at k = 3) while `lobo_rung_summary.csv` pools by the arithmetic mean of per-target RMSEs (0.433 for the same fit) — the two archived tables use different poolings for the same quantity; `mean_offdiag_spearman` in `task1_structure_results.json` is the pairwise-complete Spearman correlation over all 421 configurations (0.7905), not over the 96-model grid (0.8188); `subsample_res_share` (57.05 %) residualises on release date and log₁₀ parameters jointly, confirmed by reproduction.
- `k_selection_evidence.csv` detail strings are regenerated in the archive's own format so the table diffs cleanly; only `selected_k` is compared — coordinating session; text cells are not numbers.
- `predict.LoboResult.oof_index` added so `task2_error_analysis_gdpval.csv` can be rebuilt by model name — coordinating session; it reproduces the archive to 7e-7.
- Report unit tests run the quick path (B = 50, 3 re-runs, 100 parallel-analysis draws) in the smoke set; the golden set runs the fast path and, marked `slow`, the full ladder (33 min measured) — coordinating session; the quick and fast paths both take 25–50 s on two cores depending on load (the quick path saves little because the ridge LOBO fits dominate), within rule 6's bound for the module.

## 2026-09-07 — T7.1 review fixes (after the third same-family review)

- The twelve-table golden test now locks its own tolerances and the SHA-256 of every vendored archived table — coordinating session; the reviewer showed the previous version took both deviation and tolerance from `src/`, so a tolerance change inside the package would have passed the M3 gate without touching `tests/golden/`.
- `report.interval_agreement` and `report.compare` treat a NaN on either side as a failure (`inf` / `NaN` deviation), never as agreement — coordinating session; the reviewer demonstrated an all-NaN table reporting perfect agreement.
- A table outside its tolerance now says "NOT reproduced within tolerance" in its caption instead of asserting the tier; `excludes_zero` is compared on both statistical tables and any flip fails the table; point estimates off at the printed precision fail the table even when the interval z is small — coordinating session; rules 7 and 8 need the caption to state what the run found.
- The quick path (`--quick`, tests and demos) no longer judges the two statistical tables: with three re-runs the Monte-Carlo SD is a two-degree-of-freedom estimate and produced a false "outside tolerance" alarm; the tables are written, their point values are the recomputed ones, and the caption says the criterion was not applied — coordinating session.
- k-sweep intervals compared before rounding (as the H4 table already was), with rerun alignment by key rather than row order; text columns copied from the archive (`ksweep_rung_iv.csv` `note`, `k_selection_evidence.csv` `detail`) declared in the captions — coordinating session.
- Stable sorts in `io.deduplicate_by_base_model` and `predict.ladder`; a constant factor score on a training fold raises in `predict._build_rungs` instead of skipping the sign flip; `trust.agreement_with_interval` raises on an empty bootstrap; `report` removes its own stale output files before writing — coordinating session; none changes a number on the archive grids (the deduplicated grid has no ties, checked by the reviewer).

## 2026-09-10 — repository published, CI first run

- `main` and the ten ticket branches pushed to `github.com/louisyzhu/benchprobe`, private (21 commits on `main`, tip `02f1dd7`) — Louis, 10 September 2026; the "in development" surface rule keeps it private until the v0.1 tag. This closes the M0-report item "the GitHub Actions workflow has not run: no repository on GitHub yet".
- Action pins verified against GitHub's actual tag lists on the day of the push: `actions/checkout` publishes `v1`–`v7` and `astral-sh/setup-uv` publishes `v1`–`v7`, so the workflow's `@v4` and `@v6` both resolve — coordinating session; this was the open "could not be verified from the sandbox" note in `docs/decisions.md` (T0) and it is now closed. Newer majors exist and are not adopted: the pinned ones work and a bump is scope without a ticket (rule 4).
- CI run #2 (commit `5f17347`) succeeded in 51 s on `ubuntu-latest`: lockfile check, locked install, lint, the 103-test smoke set and golden collection — the workflow is now written, run and proven rather than assumed. It warned that `actions/checkout@v4` and `astral-sh/setup-uv@v6` target Node 20, which GitHub deprecated (changelog, 19 September 2025), and forced them onto Node 24. Both bumped to `@v7`, whose `action.yml` at that tag declares `using: node24` — read from a shallow clone of each tag, not inferred. Neither action is passed any input here, so the major bump changes nothing else. This supersedes the note made an hour earlier that a bump was scope without a ticket: that was written before CI had ever run, and the run is the evidence that changes it.
- **Finding, from the first CI run ever (10 September 2026): the workflow was invalid YAML and had been since T0.** GitHub rejected `.github/workflows/ci.yml` at line 21, `- name: Smoke test (rule 6: under a minute)` — an unquoted `: ` inside a value, which YAML reads as a nested mapping, so the step had neither `uses` nor `run`. Written blind at T0 because the sandbox cannot reach GitHub, and invisible for four days because nothing had ever parsed it. Fixed by quoting the two step names that contain punctuation. This is the exact class of defect the M0 report listed under "not verified in this session", and it is the argument for the house rule that a claim about state needs a run behind it.
- Guard added so it cannot recur: `tests/smoke/test_skeleton.py` parses every file under `.github/workflows/` and requires each step to carry `uses` or `run` and each step name to remain a string — coordinating session; verified by reintroducing the original bug, which the test fails on, then restoring. `pyyaml` added to the dev dependency group for it (dev-only; the package's runtime dependencies are unchanged).
- `workflow_dispatch` added to the workflow triggers — coordinating session; GitHub Actions was disabled at the repository level when the code was first pushed, so no run started, and a manual trigger is what lets CI be started (and re-started) without an empty commit.

## 2026-09-10 — T5.1 / T7.2, from a fourth same-family review

A review by Claude Code (a Claude model, so this is a fourth same-family pass and **not** the T8
cross-family sweep, which remains open). Each claim was checked against the archives and the code
before acting; two were not substantiated and are recorded as such.

**Confirmed and fixed.**

- `report.compare()` returned a deviation of 0.0 when it found no numeric columns in common, so a table could be judged "within tolerance" on a comparison of zero cells — demonstrated: two tables disagreeing on every shared column returned `0.0`, `within_tolerance` True. It now raises. Six of the twelve tables let it choose the columns, so this was a live hole, not a hypothetical one.
- `k_selection_evidence.csv` included the `BIC_min` row — which benchprobe copies out of the archive because the BIC computation is not in the notebook — in the comparison *against that same archive*, at tolerance zero: a self-comparison dressed as a passing check. The comparison now covers only the three recomputed rows, and the caption says which.
- The ladder table's caption now states that its pooling rule was **inferred, not extracted**: the archive ships no code producing `lobo_rung_summary.csv`, and mean-of-RMSE was chosen because it reproduces all ten of its rows while the notebook's own `_ladder` helper (root-mean-square) reproduces none. The table is still recomputed, but on a rule benchprobe reverse-engineered, and a reader is entitled to know that.
- **The published-grid test was mis-scaled, and this is the one arithmetic error the review found.** Each grid cell is a mean of 60 replicates; the difference between two such means has standard error `sd·sqrt(2/60)`, not the per-replicate `sd` the test divided by. The old 0.5-SD threshold was 2.74 standard errors under a stricter-sounding name. Re-expressed at three standard errors of the difference (`TOL_PUBLISHED_SE`), which is both the right scale and a tighter test: worst cell 1.72 SE, mean 0.88. Changing a golden tolerance is permitted only by ticket (rule 1); this is that ticket, the change makes the test stricter, and it was made after the row passed, never to rescue it. The archive README's own "0.31 of one per-cell SD" carries the same mis-scaling and is a note for the coordinating thread.

**Not substantiated.**

- *"The 4000 resamples behind the [−5.3, +32.7] pp interval mix shares of different factors."* Measured over the interval's own 2000 draws: column 0 was the largest factor in **2000 of 2000** draws on the raw grid and 2000 of 2000 on the residualised grid (full-sample sums of squared loadings 7.79 / 2.34 / 0.32 and 5.80 / 3.42 / 0.51). The thin margin the review is thinking of (4.78 vs 4.27) is on the residualised `compute_known` grid, which carries no bootstrap; it was transposed onto an interval computed on `complete_case`. The `first_factor_share` warning stands as a guard, but nothing is contaminated.
- *"`predict.ladder`'s default pooling matches nothing in the archive."* It matches the archived `lobo_rung_summary.csv` in all ten rows, verified twice independently. What it matches nothing of is the notebook's `_ladder` *helper* — which is the finding already recorded at T4, stated the other way round.

**Accepted as open, not yet acted on.**

- `MANIFEST.json` records the hash of every file it lists but is not itself hashed, so a tampered manifest would relabel rather than fail. Low severity for a single-author repository; the fix is a hash of the manifest in the golden tests, which is a ticket of its own.
- The review's point that a passing `pytest -m "golden and not slow"` does not establish the ladder rows is correct and is why the slow variant exists; the reproducibility ledger already distinguishes them.

**Finding about the JUDGe archive, carried faithfully by benchprobe.** `sweeps.py` builds
`range_at_measured_error` from `mean[:, 1]`, which is the 5 % column of `JUDGE_ERRORS`, while
labelling it `judge_error: 0.0472`. The range reported there is the range at 5 %, not at the
measured 4.72 % rate. The separate `measured_error_run` block in `sweep_grid.json` *is* at 0.0472
and is unaffected. If the paper quotes that field, the sentence needs checking.

# T6 — the Price of Intelligence item-response estimator (2026-09-10)

The archive arrived by upload (the proxy still refuses zenodo.org). Four decisions.

**1. The model is not the one the skeleton assumed.** `SPEC.md` said "two-parameter and
graded-response estimation". The archive's own estimation record says *Samejima continuous
response model on logit scores, homoscedastic*, and its parameter counts (607 free at stage 1,
891 at stage 2) confirm it. The scope row and the signatures section were corrected. This moves
no number; it corrects a description that was wrong from T0.

**2. This module is a re-implementation, not an extraction.** The archive ships its frozen panel
and its Phase-2 outputs and **no estimation code**. Every other benchprobe module was extracted
from code that produced the published numbers; this one was written from the archive's stated
model, optimiser (Adam 3000 steps at lr 0.05, then L-BFGS) and penalty scales, and then held
against its published outputs. `docs/reproducibility.md` records it at the *recomputed* tier for
the quantities `tests/golden/test_price_of_intelligence.py` locks, and for nothing else. The
recorded `seed` does not enter: Adam starts from zeros and nothing here is stochastic.

**3. The scale repair — a correction to the panel's metadata, carried in the open.** Four
benchmarks (`aider_polyglot_external`, `forecastbench_external`, `live_bench_external`,
`os_world_external`; 221 of 4605 scored cells) carry `score_scale="percentage"` and
`score_divisor=100` over scores that are already proportions. The evidence, recomputed by
`irt.scale_repair_report` on every run rather than asserted: all 221 stored values are
**bit-for-bit** equal to `round(score*100, 2)/100`, the floating-point residue of a percentage
divided by 100 (`0.036000000000000004`, `0.08900000000000001`); across the other 4384 scored
cells, which store native ratios of counts, only 59.8 % are. These four are the only rows in the
panel declared `percentage`; scale is constant per benchmark for all 64.

Applying the recorded divisor a second time reproduces neither the published item parameters
(worst |Δ discrimination| 1.11, worst |Δ difficulty| 24.6, 55 of 64 benchmarks within 0.05 on
both) nor the residual scale (0.444766 against the published 0.44927472556854087). Not applying
it reproduces both (worst |Δ difficulty| 7.6e-5, |Δ discrimination| 1.3e-5, residual scale to
5.9e-9). So the *published* numbers were computed on the corrected reading; it is the shipped
metadata, not the paper, that is wrong, and anyone re-running from the panel as documented would
diverge silently. `build_panel(repair_scale=True)` is therefore the default, the repaired
benchmarks are named rather than detected (the set cannot silently grow), a named benchmark whose
evidence no longer supports the repair raises instead of being repaired, and
`test_the_unrepaired_panel_does_not_reproduce_the_archive` fits the literal metadata and asserts
that it *fails*, so the correction can never be quietly dropped and mistaken for agreement.
Rule 3 holds: this moves no published number — it is what makes the published numbers reproduce.

Provenance of the 221 rows, from the panel itself: they come from `aider.chat` (62),
`livebench.ai` (54) and `os-world.github.io` (20) — hosts that appear nowhere else in the panel —
plus 85 forecastbench rows from `epoch.ai`; all 221 are `item_level_readable = False`. Two
tempting explanations were checked and **do not hold**: `archive_timestamp` is `20260809T212305Z`
for all 5067 rows and so distinguishes nothing, and a percentage-sounding column header does not
predict the label (`Average (%)`, `Overall pass (%)`, `Win Rate (%)` and 36 rows reading
`Overall score` are all labelled `proportion`). Without the ingestion code the entry point can be
located and the double division proved; the intent cannot be read.

**4. Two arithmetic findings about the archive's own outputs, reproduced rather than reconciled.**

- *The objective excludes priors on fixed parameters.* Including them left benchprobe's stage-2
  objective exactly 1.8839 above the archive's — the Gaussian penalty carried by the nine fixed
  non-reference anchors, computed from the archive's own item parameters. A penalty on a
  held-fixed coordinate is an additive constant: it cannot move the optimum, and counting it makes
  the reported objective depend on which parameters happen to be fixed. Excluded, and stage 2 lands
  on −1328.6389 (recomputed −1328.63888738). Stage 1 is unaffected, because there only the
  reference item is fixed and its parameters are zero.
- *`se_theta_structure.csv` reports two different informations in the same row.* Its
  `test_information` and `se_predicted_from_information` sum `a_k²/σ²` over a model's **distinct
  benchmarks**; its `se_theta` sums over the model's **cells** and adds the ability prior's
  precision. 148 of 782 models are scored more than once on some benchmark, so for those
  `se_predicted_from_information` is not the likelihood-only counterpart of `se_theta` its name
  implies (worst gap 0.185). Both were reconstructed from the archive's own item parameters and
  residual scale, to 4e-16 and 1.7e-13, so this is a definitional split in the archive, not
  estimation noise. `ability_table` reproduces both and labels which set each column used; the
  finding is locked by `test_the_archives_two_informations_really_do_differ`. For the coordinating
  thread: if any paper quotes `se_predicted_from_information` as the standard error before the
  prior, the sentence needs checking.

**Dependency.** `pyarrow>=17.0` added: the archive freezes its panel as Parquet and that is the
file its own `MANIFEST.sha256` hashes, so vendoring a CSV conversion would break the provenance
chain. `io.load_snapshot` now reads a `.parquet` table as well as a `.csv`. The optional `irt`
extra (`torch`) is retained but unused — the recorded fit reproduces in NumPy/SciPy with an
analytic gradient — and the pyproject comment now says so instead of claiming torch is needed.

**Tolerances.** Nothing here can be bit-for-bit; a different optimiser implementation never is.
They are declared from the precision at which each quantity is reported and used — objectives at
half a unit of the last printed digit, residual scale 1e-6, item parameters and standard errors
5e-4, abilities 1e-3, information 5e-3 — and the observed worst deviations are recorded in the
test's docstring so drift is visible. They were set before the first golden run, not after.

**Two things T6 does not do, stated so the gap is not mistaken for coverage.**

- *Convergence is claimed on a weaker criterion than the archive's.* `CrmFit.converged` is
  L-BFGS's success flag plus a gradient bound. The archive judges convergence on the Newton
  decrement and the Hessian spectrum, and says why: an earlier specification of theirs reported a
  small gradient at a saddle point, so the eigenvalue check is part of their criterion rather than
  a diagnostic beside it (`docs/phase2-1-estimation.md` §4). An absolute gradient bound is the
  criterion they explicitly rejected. What establishes the T6 fit is not benchprobe's convergence
  flag but the agreement of its parameters with the published ones. **Ticket T6.1:** Hessian
  eigenvalues and the Newton decrement on the free block, to reproduce the archive's four
  convergence rows (stage-1 min eigenvalue 0.040, decrement 1.9e-9; stage-2 0.086, 4.8e-8,
  condition number 217,549).
- *Plausible values are not implemented.* The archive draws L = 20 per model
  (`data/interim/theta_draws.parquet`, 15,640 rows), because the upper end of the posterior SD
  range belongs to models observed on one uninformative benchmark. benchprobe reports the point
  estimate and its standard error only. Not vendored, not claimed, and out of SPEC.md's `irt`
  scope as written.

**Cross-checks against the archive's prose that T6 passes but does not lock.** The archive's
estimation note quotes summary statistics that the vendored tables reproduce to the digits printed:
posterior SD of θ median 0.1354 / min 0.0326 / max 3.3368 against its "0.135 / 0.033 / 3.337", and
discrimination min 0.1003 / max 7.8365 / median 1.0094 against its "0.100 / 7.837 / 1.009". These
are checks on the *archived tables*, not on benchprobe, which is why they are recorded here rather
than added as golden rows.

# T8 — cross-family QA (2026-09-10)

Reviewer: **Codex (OpenAI, "model 5.6 sol high" as Louis reported it)**, reading the T8 packet at
commit `7e253a1`. First review of this repository by a model outside the Claude family. Twelve
defects and a set of section findings; each is recorded below with what was done. Nothing in
`tests/golden/` was loosened; four tolerances were tightened and are recorded under their ticket.

**Defects, in the reviewer's severity order.**

1. *`worst()` in the T6 golden test skips NaN.* **Confirmed** — `Series.max()` and `idxmax()` skip
   NaN, so a partly-NaN recomputation could pass on its finite remainder. **Fixed:** an explicit
   non-finite check on both sides before the comparison. (`compare()` in `report.py` already
   treated NaN as failure; the T6 helper did not.)
2. *`judge_sweeps_reference.json` and `sweep_grid.json` are read from files, not pinned.*
   **Confirmed. Fixed:** `test_judge.py` pins the SHA-256 of both, and of the snapshot's
   `MANIFEST.json`. The same manifest pin was added for the other two snapshots
   (`test_one_capability_tables.py`, `test_price_of_intelligence.py`), which closes the open ticket
   "hash `MANIFEST.json` itself". `io.py`'s docstring, which said the manifest's SHA-256 was
   verified on load, was wrong and now says what is and is not authenticated.
3. *`CrmFit.converged` was `result.success` only, while the docs said "plus a gradient bound".*
   **Confirmed** — the docs overstated what the code did. **Fixed by doing T6.1:** the Hessian on
   the free block (central differences of the analytic gradient), its spectrum, the Newton
   decrement and the largest remaining ability step, judged by the archive's own rule. Reproduces
   the archive's recorded diagnostics: min eigenvalue 0.040 / 0.08614 (relative 4e-12 / 2e-5),
   max eigenvalue 19747.0 / 18740.3, condition number 493675 / 217544 (archive 217549), zero
   negative eigenvalues, both stages. benchprobe's own decrement (1.4e-10 / 1.6e-10) and ability
   step (5e-6 / 1.7e-5) are smaller than the archive's (1.9e-9 / 4.8e-8; 2e-5 / 3.5e-4): it
   stopped closer. Locked as golden rows. Whole file still runs in 5 s.
4. *The package-level "extracted" claim covers reconstructed methods.* **Confirmed.** README,
   SPEC §0 and the package docstring now say: extracted where the archive ships code,
   reconstructed and named where it does not — the logistic form, the ladder pooling rule, the
   `compute_known` grid, and all of `irt`.
5. *T6's vector tolerances can hide small disagreements; the "reporting precision" justification is
   contradicted by the full-float CSVs.* **Confirmed on the justification; partly on the
   tolerances.** T6.1 supplied the real reason, and it is recorded in the test docstring: the
   archive's published solution is up to its own remaining Newton step from the optimum (3.5e-4 on
   abilities, 7.5e-5 on difficulties, 2.5e-5 on log-discriminations, 1.1e-8 on the log residual
   scale — all computed by evaluating benchprobe's objective at the archive's point), and the
   observed deviations per block are those steps to within 10 %. Tolerances are now set from
   them: abilities 1e-3 (the archive's own convergence resolution, unchanged), items and standard
   errors 2e-4 (from 5e-4), residual scale 1e-7 (from 1e-6); information 5e-3 unchanged, with its
   derivation stated as loose. Tightened by this ticket after the rows passed, never to rescue
   them (rule 1, as at T5.1). **Added, as the reviewer asked:** a rank-order test — zero reversals
   among model pairs the archive separates by more than the tolerance, at most five in total,
   Spearman ≥ 0.999999.
6. *The scale repair is operationally justified but the documentation overstates: the bit-for-bit
   test proves grid membership, not transformation history, and the packet cannot distinguish
   "metadata wrong" from "estimator ignored correct metadata".* **Confirmed, and the reviewer's
   phrasing is the right one.** The column is renamed `share_on_two_decimal_percent_grid`; the
   docstring says what the test measures and that the evidence is the joint pattern (221 of 221 on
   the grid against a 60 % background, no score above 1, and joint reproduction of every published
   quantity only without the divisor), not any single value. The claim in T6's own section above
   is left as written and corrected here: the defensible statement is that *the published fit
   treated the stored values as proportions despite the recorded divisor*, so the panel's metadata
   is inconsistent with the fit that was published from it; which of the two is "wrong" is a
   question for the ingestion code, which the archive does not ship.
7. *The JUDGe G-study's stated interpretation contradicts its formula.* **Confirmed — and this is
   a finding about the archive and the paper, not about benchprobe.** `sweeps.py` says Φ "is
   v_p/(v_p + (v_ph + v_e)/R), since a deployment commits to one phrasing rather than averaging
   over R of them". Dividing the phrasing and residual components by R is the dependability of a
   *mean over R phrasings* (Brennan's Φ with n′ = R); a deployment that commits to one phrasing has
   Φ = v_p/(v_p + v_ph + v_e). benchprobe reproduces the archive's number to 1e-9 and its docstring
   carried the same contradiction; the docstring now states the formula and the discrepancy. The
   difference is not small. From the reproduced components (R = 3): *irrelevant* 0.970 as computed
   vs 0.917 for one phrasing; *mild* 0.958 vs 0.885; *large* 0.876 vs 0.686;
   *deterministic_given_phrasing* 0.947 vs 0.849. **For the coordinating thread, with priority:**
   if the paper reports these Φ values under the single-phrasing reading, they are overstated by
   0.05–0.19. No number in benchprobe changes (rule 3); a `single_condition=True` option is a
   ticket, not a silent addition (rule 4).
8. *"Recomputed" tables with copied fields; quick-mode captions assert the tier before
   disclaiming.* **Confirmed. Fixed:** `TableResult` gains `regenerated_columns` and `judged`; a
   partly-copied table's caption opens "recomputed EXCEPT <columns>, which are regenerated"; a
   table not judged opens "NOT JUDGED in this run" before naming its intended tier. The tier
   *label* on `task1_structure_results.json` and `k_selection_evidence.csv` stays `recomputed`,
   because the golden tables test locks tiers and the boundary is now in the caption where rule 8
   wants it; the reviewer's point that the label alone is misleading is accepted, and the
   caption is the fix.
9. *`ability_table` took its prior from the global archive record, not from the fit.*
   **Confirmed. Fixed:** `CrmFit.priors` records the penalty scales the fit used and
   `ability_table` reads them from `fit.stage2`. Same numbers on the archive's priors (worst
   se(θ) deviation 2.66e-5, as before).
10. *`compare()` does not enforce unique keys.* **Confirmed. Fixed:** raises on a duplicated key in
    either table; smoke test added.
11. *`bb_gof`'s return shape changed incompatibly, and the decision record said unpacking still
    works.* **Confirmed on the record; the API stays.** The line at T5 was wrong and is corrected
    in place above. benchprobe does not promise the archive's call signatures.
12. *The manifest is trusted but unauthenticated.* **Confirmed.** Closed together with 2.

**Section findings, and what was done with them.**

- *Part 4 omitted the counterparts for `io.benchmark_coverage`, `io.model_coverage`,
  `trust.load_bank` and most of `report.py`.* Correct; a packet limitation. `load_bank` was
  extracted from `make_figures.load_bank` and the coverage rules from the archive's Phase-1 cells,
  neither of which the packet carried. The next packet includes them. The reviewer's "unverified,
  not clean" is the right status for those four until then.
- *`build_panel` clips rather than failing fast, and counts cells within ε rather than "exactly
  0 or 1" as the archive's note says.* **Confirmed on the first; the second turned into a finding
  about the archive.** Fail-fast added: any proportion outside [0, 1] after the recorded scale
  raises. On the count: the note says 89 cells "sat at exactly 0 or 1"; counting exactly-boundary
  cells in the panel gives **88**. One cell — `glm-5.2_unknown` on `gbaeval_external`, score
  1/5807 = 0.000172 — lies inside ε without being on the boundary, and the archive's recorded 89
  includes it. So the archive counted the cells its clip moved, and its prose is off by one.
  benchprobe keeps the rule that reproduces the recorded count and says why in the code.
- *The gradient had no finite-difference test; coordinate fixation was not independently
  checked.* **Fixed:** both added to the smoke set, with a check that a non-minimum is not called
  converged whatever the optimiser flag says.
- *All T6 comparison tables come from the same unpublished optimum; nothing breaks the
  shared-dependence loop.* **This was the most useful point in the review, and T6.1 answers it.**
  The archive records the gradient, Newton decrement and remaining ability step *at its own
  published solution*. Assembling that solution from the published tables and evaluating
  benchprobe's objective there returns 5.487e-5, 4.787e-8 and 3.480e-4 — the archive's recorded
  5.4868e-5, 4.7873e-8 and 3.4798e-4, to the digits printed. Those depend on the objective and its
  curvature, not on any optimiser: the two implementations are the same function. Locked as
  `test_the_archives_own_diagnostics_reproduce_at_its_published_point`. One observation from the
  same computation is recorded and not explained: the objective at the archive's point is 8.8e-6
  above benchprobe's, more than the 2.4e-8 its own decrement predicts is available; it is within
  the objective tolerance and nothing is locked on it.
- *The three-Monte-Carlo-SD interval criterion is a proximity heuristic, not a fidelity test; the
  implementation's own instability enlarges its tolerance; dividing by one run's SD is stricter
  than three SDs of a difference.* **Agreed on all three, recorded, not changed.** The criterion
  was set by ticket at T7 for the statistically-reproduced tier, which by definition claims
  proximity under a different random stream and nothing more; the ledger says so. A calibrated
  equivalence test is a ticket if a paper ever needs the stronger claim.
- *`trust.sweeps` preserves the 5 % / 4.72 % mislabel.* Already recorded at T5.1; unchanged, by
  design (the archive's output is the oracle).
- *Threshold-20 sensitivity, external-validity checks and plausible values are not implemented.*
  Correct and declared; SPEC.md's `irt` row is the scope and these are outside it. benchprobe is
  not the complete analysis code behind the Price of Intelligence study and does not claim to be.
- *Deduplication, ladder pooling, logistic form, unrecoverable bootstrap streams, fixed-parameter
  penalty arithmetic, the two informations:* the reviewer verified each and **agrees**, with one
  correction accepted — the fixed-parameter penalty is a reporting convention, not an archive
  defect, and T6's section above should be read that way.

**Louis's checklist item "cross-family QA sweep recorded" (handoff §9) is met by this section.**

## Open, assigned

- Louis: confirm the recovered four-parameter logistic (T3) against the pre-registration text.
- Coordinating thread, JUDGe paper: the phrasing G-study Φ values and the single-phrasing sentence (T8, defect 7).
- Ticket, unscheduled: `gstudy_two_way(single_condition=True)` for the one-phrasing dependability.
- Coordinating thread: the panel's `score_scale`/`score_divisor` metadata for the four benchmarks
  above should be corrected at source, and `se_predicted_from_information` renamed or recomputed.
- v0.1: licence choice; CITATION.cff; commit author identity.
