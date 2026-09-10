# v0.1 — release checklist (Louis tags; rule 10)

Status on 10 September 2026, branch `v0.1-door`. Handoff §9, definition of done:

| Item | State |
|---|---|
| A fresh clone; one command installs | `uv sync` — done |
| One command runs the smoke test in under a minute | `uv run pytest -m smoke` — 124 tests, ~40 s |
| One command reproduces the twelve One Capability tables with the boundary stated per table | `python -m benchprobe.report one-capability --out DIR [--full-ladder]` — done; ladder byte-identical |
| Golden tests green within tolerance | 44 (One Capability 20 + tables 3 + JUDGe 9 + Price of Intelligence 12); last full run 44 passed in 2777 s |
| `docs/reproducibility.md` states the tier for every output | done |
| CITATION.cff and licence present | MIT; CITATION.cff with the three studies as references |
| Cross-family QA sweep recorded | done — Codex, 10 Sept 2026, `docs/decisions.md` T8 |
| Ship condition: the analysis code behind a real study | met by the One Capability recomputation (arXiv:2608.29420) |
| A door for data that is not one of the studies | `benchprobe.scores.ScoreMatrix`, `irt.panel_from_long` / `fit_single_stage`, `benchprobe.studies`, `examples/quickstart.py` — done |
| Wheel builds and installs in a clean environment | `uv build`; `pip install dist/benchprobe-0.1.0-py3-none-any.whl` in a fresh venv imports every module, verifies the vendored snapshots and reproduces the JUDGe KR-20 and the Price of Intelligence residual scale |
| The paper cites it by version | after the tag: the One Capability camera-ready / E&D version cites `benchprobe v0.1` |

## What Louis does (in this order; nothing else is required)

```sh
cd ~/Documents/benchprobe
git pull <bundle> main && git push origin main
git tag -a v0.1.0 -m "benchprobe v0.1.0"
git push origin v0.1.0
```

Then on GitHub: *Releases → Draft a new release → choose tag v0.1.0 → title "benchprobe v0.1.0" →
paste the tiered statement below → Publish*. If the repository is linked to Zenodo
(zenodo.org → GitHub → toggle `louisyzhu/benchprobe` on, once), publishing the release mints a DOI
automatically; add it to `CITATION.cff` (`doi:` field) in the next commit.

PyPI, optional and separate: `uv build && uv publish` with a PyPI API token in `UV_PUBLISH_TOKEN`.
The name `benchprobe` was free on 6 September 2026.

The repository can stay private until the release; a private repository cannot be installed with
`pip install git+…` by anyone else, so the release is what makes it citable.

## Tiered reproducibility statement (release notes, README, and the paper)

> benchprobe v0.1 is the analysis code behind *One Capability or Many?* (arXiv:2608.29420) and is
> validated against two further studies. Every reported quantity is stated in
> `docs/reproducibility.md` as recomputed, statistically reproduced or regenerated. From the
> hash-pinned snapshots it recomputes the One Capability structure results, leave-one-benchmark-out
> point estimates and the four-learner ladder to the archive's printed precision; two bootstrap
> tables whose archived random stream is unrecorded are reproduced within three Monte-Carlo
> standard deviations; model clustering and three fields whose computation is not in the archive
> are regenerated and say so in their captions. The JUDGe real-bank quantities reproduce exactly
> and its simulations reproduce the archive's script to 1e-9. The Price of Intelligence item
> parameters, abilities, objectives and Hessian convergence diagnostics reproduce to the archive's
> own stopping distance, from a re-implementation of its stated model, since that archive ships no
> estimation code. Three corrections to the archives found in the course of this work are recorded
> in `docs/decisions.md`. A review by a model from a different family (Codex) is recorded there too.
