# Contributing to benchprobe

benchprobe is the analysis layer behind a set of evaluation-validity studies. Its scope is fixed by
[`SPEC.md`](SPEC.md) and its working rules by [`CLAUDE.md`](CLAUDE.md); both are short and both are
binding on contributions.

## Reporting a problem

Open an issue at <https://github.com/louisyzhu/benchprobe/issues>. The most useful report contains:

- the version (`python -c "import benchprobe; print(benchprobe.__version__)"`), your Python version
  and platform;
- a minimal script and the data shape it runs on (a synthetic `ScoreMatrix` is usually enough —
  see `examples/quickstart.py`);
- what you expected and what happened, with the full traceback.

**If a number is wrong, say so explicitly.** A disagreement with a published value is the most
important class of issue this project can receive, and it is treated as a finding rather than a bug
report: give the archive value, the value benchprobe produced, and the tolerance if you know it.

## Getting support

Open an issue with the `question` label. For a methodological question — which estimator applies,
what a tolerance means, whether an output is *recomputed* or *statistically reproduced* —
[`docs/reproducibility.md`](docs/reproducibility.md) states the tier and provenance of every number
the package reports, and [`docs/decisions.md`](docs/decisions.md) records why each design decision
was taken.

## Contributing code

```sh
git clone https://github.com/louisyzhu/benchprobe
cd benchprobe
uv sync
uv run pytest -m smoke      # ~40 s; CI runs exactly this
uv run ruff check . && uv run ruff format --check .
```

Work on a branch, one topic per branch, and open a pull request. Before it can be merged:

- `uv run pytest -m smoke` passes and `ruff` is clean;
- every function that produces a reported number has a test;
- if you touched an estimator, `uv run pytest -m golden` passes (about an hour — paste the output
  in the pull request);
- new behaviour is documented, and any design decision is added to `docs/decisions.md` as one dated
  entry.

### Three rules that are not negotiable

1. **Golden numbers, tolerances and seeds are never changed to make a test pass.** The values in
   `tests/golden/` are the published results of the studies this package reproduces. A failing
   golden test is a finding: report it in the issue or pull request with the archive value, the
   recomputed value and the tolerance. Changing one requires its own issue, an argument, and a
   record in `docs/decisions.md`.
2. **Every reported output states its reproducibility tier** — *recomputed*, *statistically
   reproduced*, or *regenerated* — and "fully reproducible" is used only for the first.
3. **New statistical methods need an issue first.** Scope is `SPEC.md`; the package deliberately
   does not grow into benchmark construction, scraping, model inference, leaderboards or dashboards.

## Adding a study

`benchprobe.studies` holds one entry point per validated study, and nothing outside that package
knows a benchmark by name. A new study needs a hash-pinned snapshot with a `MANIFEST.json`, golden
tests locking its published numbers with tolerances justified in the test's own docstring, and a row
per output in `docs/reproducibility.md`. This is the highest-value contribution the project can
receive: every study added is another independent check on the estimators.

## Conduct

Be straightforward and assume good faith. Disagreements about numbers are settled by computation,
not by seniority.
