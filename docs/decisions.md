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

## Open, assigned

- T2: source of the grids' base-model deduplication key (regex in the archive's R1 block) — extract verbatim.
- T3: whether the sklearn `check_array` shim the archive applies to factor_analyzer 0.5.1 is still needed under the locked versions; whether to pin the archive's exact numeric-stack versions.
- T3: the logistic functional form behind the paper's date-R² of 0.505 (not in the archived notebook; OLS 0.477 is). Recover from the pre-registration or Phase 2 working code before implementing `date_r2(form="logistic")`.
- T4: the archive draws the three baseline bootstraps (`i_date`, `ii_meanidx`, `iii_f1`) from one `default_rng(42)` stream in that order; exact reproduction of [+0.019, +0.055] needs the same order, otherwise the row is statistically reproduced.
- T5: Krippendorff's α and Cohen's κ are in SPEC §1 but not in the JUDGe archive; they are new code and need their own tests and reference values.
- v0.1: licence choice; CITATION.cff; commit author identity.
