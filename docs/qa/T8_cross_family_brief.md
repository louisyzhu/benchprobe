# T8 — cross-family QA sweep (brief for a non-Claude model)

Handoff §4: before any milestone closes, a QA sweep by a model from a different family than wrote
the code (Codex or equivalent), reading the code against `SPEC.md` and the golden tests. Every line
of this repository was written by Claude sessions and reviewed by Claude sessions; this sweep is the
independent check. Louis runs it; the result goes into `docs/decisions.md` as one dated line per
finding, and nothing in `tests/golden/` changes as a consequence except by ticket.

## What to give the reviewer

1. This repository at the commit under review (`git rev-parse HEAD`), with `uv sync` done.
2. The three source archives, read-only: `github.com/louisyzhu/frontier-ai-economic-validity`
   (commit `946ce845`), `github.com/louisyzhu/llm-judge-reliability` (commit `126d1bce`), and the
   Price of Intelligence archive v1.0.0 (DOI 10.5281/zenodo.22177190) — which ships **no
   estimation code**, only its frozen panel, its written estimation note
   (`docs/phase2-1-estimation.md`) and its Phase-2 output tables.
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
> 1b. **`src/benchprobe/irt.py` is different in kind and needs a different review.** Its archive
>    ships no estimation code, so it is a re-implementation from a written specification, checked
>    against published output tables. There is nothing to diff. Instead: does the code implement
>    the model that `docs/phase2-1-estimation.md` describes — Samejima CRM on logit scores, one
>    shared residual SD, reference anchor pinned at a = 1, d = 0, two-stage fixed-parameter link?
>    And attack the hardest thing in the repository: `build_panel` **suspends the panel's own
>    recorded `score_divisor` for four benchmarks by default**, and only with that suspension does
>    the fit reproduce the published numbers. Read `irt.scale_repair_report`, the evidence recorded
>    in `docs/decisions.md` (2026-09-10), and `test_the_unrepaired_panel_does_not_reproduce_the_archive`.
>    Is this a justified correction to wrong metadata, or is it a package altering its input until
>    the answer comes out right? Say which, and why. This is the single claim most worth
>    disbelieving.
> 2. **Golden tests.** Check that every number in `tests/golden/*.py` is a literal in the test file
>    or a hash-pinned archive file, never a value the package computes; that no test is skipped,
>    marked expected-to-fail, or able to pass vacuously (NaN, empty input, zero tolerance on a
>    self-comparison); and that `git log -- tests/golden/test_one_capability.py` shows no edit after
>    the T1 commit.
> 2b. **T6's tolerances were set by the model that wrote the code, and nothing there is
>    bit-for-bit.** `tests/golden/test_price_of_intelligence.py` declares six tolerances and
>    justifies each by "the precision at which this quantity is reported and used". Judge that
>    claim. Is any of them loose enough to hide a real disagreement — in particular `TOL_THETA` at
>    1e-3 against an observed worst deviation of 3.3e-4, a margin of only 3x? Would a wrong
>    implementation plausibly land inside these bounds?
> 3. **Run and paste.** `uv run pytest -m smoke -q`; `uv run pytest -m "golden and not slow" -q`
>    (about three minutes, and it includes all eight T6 rows — those take 4 seconds); `uv run python -m benchprobe.report one-capability --out /tmp/t8`
>    (under a minute) and paste `comparison.csv`. If you have forty minutes, also
>    `uv run pytest -m "golden and slow" -q`.
> 4. **Findings in `docs/decisions.md`.** The package records several findings about the archives
>    (ladder pooling, deduplication semantics, the recovered logistic form, unrecorded bootstrap
>    streams, and at T6: the four benchmarks' scale metadata, the objective's treatment of priors
>    on held-fixed parameters, and `se_theta_structure.csv` reporting an information computed over
>    a different set from the standard error beside it). Verify each independently from the archive
>    files and say whether you agree. The T6 findings are claims that a *published archive is
>    internally inconsistent*; hold them to a high bar.
> 5. **The gaps T6 declares.** `docs/decisions.md` states that `CrmFit.converged` is a weaker
>    criterion than the archive's (L-BFGS success plus a gradient bound, where the archive uses the
>    Newton decrement and Hessian spectrum precisely because a gradient bound once passed a
>    saddle point), and that plausible values are not implemented. Are those the right gaps to
>    declare, and are there others the package has not noticed?
> 6. **Anything else** that would make a reader distrust a number this package reports.
>
> Report as a numbered list with file:line evidence, then defects ordered by severity. Do not
> propose changes to golden numbers, tolerances or seeds; if one looks wrong, say why and stop.

## Recording the result

One line per finding in `docs/decisions.md` under a dated "T8 cross-family QA" heading: the finding,
the model that made it, and what was done (fixed by ticket, or recorded as a known limitation).
Then, and only then, the v0.1 checklist item "cross-family QA sweep recorded" is met (handoff §9).
