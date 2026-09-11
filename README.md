# benchprobe

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22705351.svg)](https://doi.org/10.5281/zenodo.22705351)

**A Python psychometrics library for AI benchmark scores.** Give it a model × benchmark matrix and
it answers four questions the evaluation literature keeps asking by hand:

| Layer | Question | Functions |
|---|---|---|
| `trust` | Are these scores reliable, and does the grader agree with gold? | KR-20/KR-21, beta-binomial fit, Livingston–Lewis classification accuracy, Φ(λ), Krippendorff's α, Cohen's κ, two-way G-study |
| `measure` | Do the benchmarks measure one thing or several — and does that survive controlling for release date? | KMO, Horn's parallel analysis, ML factor analysis with oblimin rotation, Thurstone factor scores, date residualisation and date-R² |
| `predict` | Does the battery predict a held-out criterion better than a single index? | Leave-one-benchmark-out with in-fold factor re-estimation, a rung ladder from date-only to k-factor + covariates, pooled ΔMSE with bootstrap intervals |
| `irt` | Where does each model sit on a common scale, with a standard error, and which benchmarks discriminate? | Samejima continuous response model on logit scores, two-stage fixed-parameter anchor linking, Hessian-based convergence diagnostics |

What makes it different from writing these yourself: **every estimator is validated by reproducing
the published numbers of three studies from hash-pinned data** — 44 acceptance tests, tolerances
declared before the code was written and never loosened, a reproducibility ledger stating for each
number whether it is *recomputed*, *statistically reproduced* or *regenerated*, and a review by a
model from a different family on record. The studies are the library's validation suite, not its
purpose; your data is.

NumPy, SciPy, pandas, scikit-learn, factor_analyzer. No GPU, no service, no UI. Python ≥ 3.12.

## Install

```sh
pip install benchprobe            # from PyPI, once v0.1 is tagged
pip install git+https://github.com/louisyzhu/benchprobe   # or from the repository
```

For development: `uv sync && uv run pytest -m smoke` (locked environment; the smoke set runs in
about 30 s).

## Ten lines on your own data

```python
import pandas as pd
import benchprobe.irt as irt
import benchprobe.measure as measure
import benchprobe.predict as predict
from benchprobe.scores import ScoreMatrix

long = pd.read_csv("scores.csv")  # columns: model, item, score (0–1), release_date
sm = ScoreMatrix.from_long(long, release_date="release_date")

z = sm.complete()  # rows scored on every benchmark
z = (z - z.mean()) / z.std(ddof=0)
print(measure.kmo(z), measure.parallel_analysis(z).k_retained)
print(measure.first_factor_share(z, k=3))  # share of common variance on the first of k factors
print(measure.date_r2(measure.efa(z, k=1).scores, sm.grid().days, form="logistic"))

res = predict.lobo(sm.grid(), targets=["your_criterion"], k=3, learners=["ridge"])
print(res.metrics[["target", "rung", "learner", "test_rmse", "test_r2"]])

fit = irt.fit_single_stage(irt.panel_from_long(sm.long()))
print(
    irt.ability_table(fit, panel=irt.panel_from_long(sm.long()))[["model_id", "theta", "se_theta"]]
)
```

`examples/quickstart.py` is that script on simulated data, end to end, in about a second:
`uv run python examples/quickstart.py`.

Conventions: scores are proportions in [0, 1] (`scale="percent"` divides by 100); a missing cell is
`NaN` wide and an absent row long; a repeated (model, item) cell is an error unless you say what it
means (`aggregate="mean"`). `ScoreMatrix` does bookkeeping only — every number comes from the four
layers.

## The three studies it reproduces

`benchprobe.studies` holds one function per study; nothing outside that package knows a benchmark
by name.

| Study | Entry point | What reproduces | How well |
|---|---|---|---|
| *One Capability or Many?* (arXiv:2608.29420) | `studies.one_capability(out_dir)` — or `python -m benchprobe.report one-capability --out DIR` | the twelve archived result tables, each with a reproducibility tier in its caption; KMO 0.933, first-factor share 74.5 %, logistic date-R² 0.505, pooled LOBO ΔMSE +0.037 [+0.019, +0.055], the four-learner ladder | recomputed to the archive's printed precision; two bootstrap tables statistically reproduced (their random stream is unrecorded); `lobo_rung_summary.csv` byte-identical with `--full-ladder` |
| *Three Ways Classical Test Theory Misleads for LLM Judges* | `studies.judge()` | KR-20 0.5223/0.5231, per-element error 4.72 %, judge–gold r 0.921, beta-binomial χ² 5.31 (p 0.504); the simulation sweeps | real bank exact; sweeps reproduce the archive's script to 1e-9; the published grid within 1.72 standard errors |
| *The Price of Intelligence* (Zenodo 10.5281/zenodo.22177190) | `studies.price_of_intelligence()` | 64 item parameters, 782 abilities with standard errors, both stage objectives and residual scales, the Hessian convergence diagnostics | item parameters to 8e-5, abilities to 3e-4 (the archive's own stopping distance), objectives to 1e-5; the archive's recorded gradient and Newton decrement reproduced *at its published point* |

`docs/reproducibility.md` is the ledger: one row per number, with archive value, recomputed value,
tolerance and tier. `docs/decisions.md` records every design decision and every finding about the
archives — including the ones that are corrections to the archives themselves. Full acceptance run:
`uv run pytest -m golden` (46 min; the One Capability ladder is most of it).

## What it is not

Not a leaderboard, a scraper, a model runner, a dashboard or an "audit in an afternoon". It
computes statistics on tables you already have, and it carries no claim its validation studies do
not carry. If you need a general benchmark-auditing product, see tinyBenchmarks, MetaBench or
PSN-IRT; if you need the specific estimators those papers use, validated against their numbers,
this is it.

## Rules the repository runs by

`CLAUDE.md` — ten rules, checked by a smoke test. The ones a user should know: a golden number,
tolerance or seed is never changed to make a test pass (a failing golden test is a finding and is
reported); every reported number has a test; every output states its reproducibility tier; the
environment is pinned; nothing is published by an agent.

## Data

Three snapshots are vendored under `benchprobe/data/`, each with a `MANIFEST.json` whose entries
`load_snapshot` verifies on every load and whose own hash the golden tests pin: the One Capability
analysis table ([frontier-ai-economic-validity](https://github.com/louisyzhu/frontier-ai-economic-validity),
CC BY 4.0; figures originate with Artificial Analysis and Epoch AI), the JUDGe item bank
([llm-judge-reliability](https://github.com/louisyzhu/llm-judge-reliability), CC BY 4.0), and the
Price of Intelligence panel with three Phase-2 output files (Zenodo, hash-identical to the archive's
own manifest). `BENCHPROBE_DATA_DIR` points at alternative snapshot folders. Nothing here performs
inference or needs credentials.

## Licence and citation

MIT for the code; the vendored data keeps its CC BY 4.0 terms. Cite the release
(`CITATION.cff`) and the study whose numbers you rely on.
