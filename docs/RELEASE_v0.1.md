# v0.1 — release checklist and the tiered statement (draft; Louis tags)

Handoff §9, definition of done. Status on 7 September 2026, `main` after T7.1:

| Item | State |
|---|---|
| A fresh clone; one command installs | `uv sync` — done, checked on a fresh clone of the bundle |
| One command runs the smoke test in under a minute | `uv run pytest -m smoke` — 99 tests, 10–65 s depending on load |
| One command reproduces the twelve One Capability tables with the boundary stated per table | `uv run python -m benchprobe.report one-capability --out DIR [--full-ladder]` — done; ladder byte-identical with `--full-ladder` |
| Golden tests green within tolerance | 30 golden tests (One Capability 20 + tables 3 + JUDGe 8, one of the tables tests `slow`) |
| `docs/reproducibility.md` states the tier for every output | done |
| CITATION.cff and licence present | on this branch (`v0.1-prep`): MIT, as the two archives' code; Louis confirms |
| Cross-family QA sweep recorded in `docs/decisions.md` | **not done** — `docs/qa/T8_cross_family_brief.md` is the brief; Louis runs it with a non-Claude model |
| The paper or thesis it ships with cites it by version | **not done** — the One Capability camera-ready or E&D version cites benchprobe v0.1 (M3 route) |
| T6 `irt` | **blocked**: the Price of Intelligence archive is not reachable from the session environment; not required for the M3 ship route |

## Tiered reproducibility statement (for the README and the paper, at v0.1)

> benchprobe v0.1 is the analysis code behind *One Capability or Many?* (arXiv:2608.29420) and
> *Three Ways Classical Test Theory Misleads for LLM Judges*. Every reported quantity is stated
> in `docs/reproducibility.md` as recomputed, statistically reproduced or regenerated. From the
> hash-pinned snapshots, the package recomputes the One Capability structure results (KMO, factor
> shares, residualisation, loadings, factor correlations, uniquenesses), the leave-one-benchmark-out
> point estimates and the four-learner ladder to the archive's printed precision; bootstrap
> intervals whose archived random stream is unrecorded are reproduced within three Monte-Carlo
> standard deviations; model clustering and three fields whose computation is not in the archive
> are regenerated from it, and say so in their captions. The JUDGe real-bank quantities reproduce
> exactly and the simulations reproduce the archive's script to floating-point identity.

Until Louis tags v0.1, every surface says "in development" (handoff §1).
