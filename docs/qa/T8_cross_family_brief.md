# T8 — cross-family QA sweep (brief for a non-Claude model)

Handoff §4: before any milestone closes, a QA sweep by a model from a different family than wrote
the code (Codex or equivalent), reading the code against `SPEC.md` and the golden tests. Every line
of this repository was written by Claude sessions and reviewed by Claude sessions; this sweep is the
independent check. Louis runs it; the result goes into `docs/decisions.md` as one dated line per
finding, and nothing in `tests/golden/` changes as a consequence except by ticket.

## What to give the reviewer

1. This repository at the commit under review (`git rev-parse HEAD`), with `uv sync` done.
2. The two source archives, read-only: `github.com/louisyzhu/frontier-ai-economic-validity`
   (commit `946ce845`) and `github.com/louisyzhu/llm-judge-reliability` (commit `126d1bce`).
3. `HANDOFF_BENCHPROBE.md`.

## Instructions to paste

> You are reviewing a Python package, benchprobe, whose claim is narrow: it is the analysis code
> behind two published studies, extracted from their archived notebooks and scripts, and it
> reproduces their numbers. Read `CLAUDE.md`, then `SPEC.md`, then `docs/decisions.md` and
> `docs/reproducibility.md`. Then, without modifying any file:
>
> 1. **Extraction fidelity.** For each function in `src/benchprobe/measure.py`,
>    `src/benchprobe/predict.py` and `src/benchprobe/trust.py`, find its counterpart in the archive
>    (`frontier-ai-economic-validity/notebook/analysis.ipynb` cells 17–27, 37–43, 58, 60;
>    `llm-judge-reliability/estimators.py`, `sweeps.py`) and list every difference, however small,
>    with a judgement of whether it can change a reported number.
> 2. **Golden tests.** Check that every number in `tests/golden/*.py` is a literal in the test file
>    or a hash-pinned archive file, never a value the package computes; that no test is skipped,
>    marked expected-to-fail, or able to pass vacuously (NaN, empty input, zero tolerance on a
>    self-comparison); and that `git log -- tests/golden/test_one_capability.py` shows no edit after
>    the T1 commit.
> 3. **Run and paste.** `uv run pytest -m smoke -q`; `uv run pytest -m "golden and not slow" -q`
>    (about three minutes); `uv run python -m benchprobe.report one-capability --out /tmp/t8`
>    (under a minute) and paste `comparison.csv`. If you have forty minutes, also
>    `uv run pytest -m "golden and slow" -q`.
> 4. **Findings in `docs/decisions.md`.** The package records several findings about the archives
>    (ladder pooling, deduplication semantics, the recovered logistic form, unrecorded bootstrap
>    streams). Verify each independently from the archive files and say whether you agree.
> 5. **Anything else** that would make a reader distrust a number this package reports.
>
> Report as a numbered list with file:line evidence, then defects ordered by severity. Do not
> propose changes to golden numbers, tolerances or seeds; if one looks wrong, say why and stop.

## Recording the result

One line per finding in `docs/decisions.md` under a dated "T8 cross-family QA" heading: the finding,
the model that made it, and what was done (fixed by ticket, or recorded as a known limitation).
Then, and only then, the v0.1 checklist item "cross-family QA sweep recorded" is met (handoff §9).
