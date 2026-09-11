**benchprobe v0.1.0** — a Python psychometrics library for AI benchmark scores.

Give it a model × benchmark matrix and it answers four questions the measurement literature asks of
any test battery:

- **`trust`** — are the scores reliable, and does the grader agree with gold? KR-20/KR-21,
  beta-binomial fit, Livingston–Lewis classification accuracy, Φ(λ), Krippendorff's α, Cohen's κ,
  two-way generalisability study.
- **`measure`** — do the benchmarks measure one thing or several, and does that survive controlling
  for release date? KMO, Horn's parallel analysis, ML factor analysis with oblimin rotation,
  Thurstone factor scores, date residualisation and date-R².
- **`predict`** — does the battery predict a held-out criterion better than a single index?
  Leave-one-benchmark-out with in-fold factor re-estimation, a rung ladder from date-only to
  k-factor plus covariates, pooled ΔMSE with bootstrap intervals.
- **`irt`** — where does each model sit on a common scale, with a standard error, and which
  benchmarks discriminate? Samejima continuous response model on logit scores, two-stage
  fixed-parameter anchor linking, Hessian-based convergence diagnostics.

### Why trust it

Every estimator is validated by reproducing the published numbers of three studies from hash-pinned
data. 44 acceptance tests, with tolerances declared before the code was written and never loosened;
a per-number reproducibility ledger (`docs/reproducibility.md`) stating whether each figure is
*recomputed*, *statistically reproduced* or *regenerated*; and an independent review by a model from
a different family, adjudicated finding by finding in `docs/decisions.md`.

| Study | Reproduced to |
|---|---|
| *One Capability or Many?* (arXiv:2608.29420) | the twelve archived result tables, at the archive's printed precision; the four-learner LOBO ladder byte-identical |
| *Three Ways Classical Test Theory Misleads for LLM Judges* | real-bank quantities exactly; simulations to 1e-9 against the archive's own script |
| *The Price of Intelligence* (10.5281/zenodo.22177190) | 64 item parameters to 8e-5, 782 abilities to 3e-4 — the archive's own stopping distance — from a re-implementation of its stated model, since that archive ships no estimation code |

Three corrections to those archives were found in the course of this work and are recorded in
`docs/decisions.md`: a scale-metadata inconsistency in a frozen panel (221 of 4,605 cells), a
dependability coefficient reported under the wrong interpretation, and an off-by-one in a stated
count.

### Using it on your own data

```python
from benchprobe.scores import ScoreMatrix
sm = ScoreMatrix.from_long(df)          # columns: model, item, score in [0, 1]
```

`sm.grid()` feeds `predict.lobo`, `sm.long()` feeds `irt.panel_from_long`, `sm.complete()` feeds the
factor functions. `examples/quickstart.py` runs all four layers end to end in about a second.
`benchprobe.studies` holds the three validation studies behind one function each; nothing outside
that package knows a benchmark by name.

### Tiered reproducibility statement

> benchprobe v0.1 is the analysis code behind *One Capability or Many?* (arXiv:2608.29420) and is
> validated against two further studies. Every reported quantity is stated in
> `docs/reproducibility.md` as recomputed, statistically reproduced or regenerated. From the
> hash-pinned snapshots it recomputes the One Capability structure results, leave-one-benchmark-out
> point estimates and the four-learner ladder to the archive's printed precision; two bootstrap
> tables whose archived random stream is unrecorded are reproduced within three Monte-Carlo standard
> deviations; model clustering and three fields whose computation is not in the archive are
> regenerated and say so in their captions. The JUDGe real-bank quantities reproduce exactly and its
> simulations reproduce the archive's script to 1e-9. The Price of Intelligence item parameters,
> abilities, objectives and Hessian convergence diagnostics reproduce to the archive's own stopping
> distance.

### Install

```sh
pip install benchprobe
```

Python ≥ 3.12. NumPy, SciPy, pandas, scikit-learn, factor_analyzer, pyarrow. No GPU, no service, no
UI. MIT licence; the vendored data keeps its CC BY 4.0 terms.
