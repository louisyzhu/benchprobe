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

## Open, assigned

- T3: whether the sklearn `check_array` shim the archive applies to factor_analyzer 0.5.1 is still needed under the locked versions; whether to pin the archive's exact numeric-stack versions.
- T3: the logistic functional form behind the paper's date-R² of 0.505 (not in the archived notebook; OLS 0.477 is). Recover from the pre-registration or Phase 2 working code before implementing `date_r2(form="logistic")`.
- T4: the archive draws the three baseline bootstraps (`i_date`, `ii_meanidx`, `iii_f1`) from one `default_rng(42)` stream in that order; exact reproduction of [+0.019, +0.055] needs the same order, otherwise the row is statistically reproduced.
- T5: Krippendorff's α and Cohen's κ are in SPEC §1 but not in the JUDGe archive; they are new code and need their own tests and reference values.
- v0.1: licence choice; CITATION.cff; commit author identity.
