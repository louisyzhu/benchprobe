# benchprobe

**In development.** benchprobe is the analysis layer behind Louis Zhu's evaluation papers, packaged
so the same computation runs by one command on a fresh machine. Where an archive ships the code that
produced its published numbers, benchprobe is an extraction of it; where it does not, benchprobe is a
reconstruction validated against the published outputs, and every such place is named
(`docs/decisions.md`): the logistic date-R² form and the ladder pooling rule of One Capability, the
`compute_known` grid, and the whole of `benchprobe.irt`, whose archive ships no estimation code. Three families of function, one per programme layer: **trust**
(reliability and grader agreement), **measure** (factor structure and incremental validity, with an
IRT estimator underneath), **predict** (held-out designs and uncertainty for criterion studies).
`SPEC.md` fixes the scope. It is a Python package; PyTorch is one optional dependency, for the IRT
estimator only.

It is not a general-purpose benchmark-auditing tool, it has no public release date, and it carries no
claim the papers do not carry. It ships (v0.1) on the day it is the analysis code behind a real study,
with that study's tiered reproducibility statement; until then nothing here is tagged or published.

## Install and run

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12 (uv fetches 3.13 from `.python-version`
if it is not installed). From a fresh clone:

```sh
uv sync                     # locked environment, dev tools included
uv run pytest -m smoke      # the smoke test: seconds
```

Other commands:

```sh
uv run pytest -m golden                  # acceptance tests against the locked numbers (see below)
uv run pytest -m "golden and not slow"   # the same without the full four-learner ladder refit
uv run ruff check .                      # lint
uv sync --extra irt                      # add PyTorch, for benchprobe.irt
```

One command reproduces the twelve result tables of *One Capability or Many?* that the archive
notebook reads rather than derives, each with a caption stating whether it is recomputed,
statistically reproduced or regenerated, and its deviation from the archived copy:

```sh
uv run python -m benchprobe.report one-capability --out runs/one_capability   # 25–50 s
uv run python -m benchprobe.report one-capability --out runs/one_capability --full-ladder  # + ~35 min
```

`--config file.json` overrides the defaults in `report.default_config()` (seeds, bootstrap size, k).

## Configuration

Everything is configured by environment variable; nothing in the repository refers to a local path.

| Variable | Meaning | Default |
|---|---|---|
| `BENCHPROBE_DATA_DIR` | Directory holding snapshot folders (`<name>/MANIFEST.json` plus the files it lists) to load instead of the copies vendored under `benchprobe/data/` | unset: vendored copies |

## Tests and the acceptance numbers

- `tests/smoke/` — imports every module, checks the vendored snapshots against their manifests, and
  checks the repository rules are in place. CI runs this test set (with the lint, the lockfile check
  and a collection-only pass over the golden tests), never the golden tests themselves.
- `tests/golden/` — the locked numbers of the studies benchprobe reproduces, each with its
  tolerance. `test_one_capability.py` encodes the One Capability results on the 6 July 2026 snapshot
  (KMO 0.933, first-factor share 74.5 %, pooled LOBO ΔMSE +0.037, and the rest); `test_judge.py` the
  JUDGe real-bank quantities and simulations. These tests are the
  ship condition: they were written before the code and are never edited to pass. A failing golden
  test is a finding and is reported as such. As of T7 all thirty pass (`docs/reproducibility.md`
  states each row's tier and recomputed value); the three `slow` rows refit four learners and take
  about ten minutes.
- `tests/io/`, `tests/measure/`, `tests/predict/`, `tests/trust/`, `tests/report/` — unit tests on
  synthetic data and the vendored snapshots, in the smoke set. `tests/golden/test_one_capability_tables.py`
  runs the twelve-table reproduction and requires every table within its tolerance.

`docs/reproducibility.md` states, for every output, whether it is *recomputed*, *statistically
reproduced* or *regenerated*, with tolerances, seeds, expected runtime and cost. `docs/decisions.md`
is one dated line per design decision.

## Data

`benchprobe/data/one_capability_2026-07-06/` vendors the analysis-ready table of
[frontier-ai-economic-validity](https://github.com/louisyzhu/frontier-ai-economic-validity) (CC BY 4.0;
figures originate with Artificial Analysis and Epoch AI, see the folder's README), and
`benchprobe/data/judge_2026-08-31/` the item bank and published simulation grid of
[llm-judge-reliability](https://github.com/louisyzhu/llm-judge-reliability) (CC BY 4.0). Each folder
carries a manifest recording its files' SHA-256, which `benchprobe.io.load_snapshot` verifies on every
load (and the smoke test checks independently). Raw snapshots are not vendored. The package analyses tables; it
performs no inference and needs no credentials.

## Licence

To be set by the author at v0.1. Until then, all rights reserved; the vendored data keeps its own
CC BY 4.0 terms.

## Citation

Cite the study the code ships with, by version, once v0.1 exists. `CITATION.cff` is added at v0.1.
