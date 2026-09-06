# benchprobe

**In development.** benchprobe is the analysis layer behind Louis Zhu's evaluation papers, extracted
from code that already produced published numbers and packaged so the same computation runs by one
command on a fresh machine. Three families of function, one per programme layer: **trust**
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

## Configuration

Everything is configured by environment variable; nothing in the repository refers to a local path.

| Variable | Meaning | Default |
|---|---|---|
| `BENCHPROBE_DATA_DIR` | Directory holding snapshot folders (`<name>/MANIFEST.json` plus the files it lists) to load instead of the copies vendored under `benchprobe/data/` | unset: vendored copies |

## Tests and the acceptance numbers

- `tests/smoke/` — imports every module, checks the vendored snapshots against their manifests, and
  checks the repository rules are in place. CI runs this test set (with the lint, the lockfile check
  and a collection-only pass over the golden tests), never the golden tests themselves.
- `tests/golden/` — the locked numbers of the studies benchprobe is extracted from, each with its
  tolerance. `test_one_capability.py` encodes the One Capability results on the 6 July 2026 snapshot
  (KMO 0.933, first-factor share 74.5 %, pooled LOBO ΔMSE +0.037, and the rest). These tests are the
  ship condition: they are written before the code, fail until the extraction tickets land, and are
  never edited to pass. A failing golden test is a finding and is reported as such.

`docs/reproducibility.md` states, for every output, whether it is *recomputed*, *statistically
reproduced* or *regenerated*, with tolerances, seeds, expected runtime and cost. `docs/decisions.md`
is one dated line per design decision.

## Data

`benchprobe/data/one_capability_2026-07-06/` vendors the analysis-ready table of
[frontier-ai-economic-validity](https://github.com/louisyzhu/frontier-ai-economic-validity) (CC BY 4.0;
figures originate with Artificial Analysis and Epoch AI, see the folder's README) with a manifest
recording its SHA-256; the smoke test verifies the hash today, and `benchprobe.io` will verify it on
every load once ticket T2 lands. Raw snapshots are not vendored. The package analyses tables; it
performs no inference and needs no credentials.

## Licence

To be set by the author at v0.1. Until then, all rights reserved; the vendored data keeps its own
CC BY 4.0 terms.

## Citation

Cite the study the code ships with, by version, once v0.1 exists. `CITATION.cff` is added at v0.1.
