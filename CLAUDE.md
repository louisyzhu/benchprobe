# CLAUDE.md — rules for every session in this repository

benchprobe is the analysis layer behind Louis Zhu's evaluation papers, extracted from code that
already produced published numbers. `SPEC.md` is the scope; `tests/golden/` holds the locked
numbers; `docs/decisions.md` and `docs/reproducibility.md` are kept current with every ticket.
Work happens on a branch per ticket. Return the diff summary, the full test output, and a list
of anything you could not verify, all verbatim.

Commands: `uv sync` installs; `uv run pytest -m smoke` is the smoke test; `uv run pytest -m golden`
runs the acceptance tests; `uv run ruff check .` lints.

## Rules

1. Never alter a golden number, a tolerance or a seed to make a test pass. A failing golden test is a finding; report it.
2. Never claim a result, test outcome or file state not present in this session's own output.
3. Edits move code, never numbers. Any numerical difference from an archived table is reported with the archive value, the recomputed value and the tolerance.
4. No new statistical methods without a ticket. No scope beyond `SPEC.md`.
5. No secrets, API keys, tokens or local paths in the repository. Config by environment variable, documented in README.
6. Every function that produces a reported number has a test. Every module has a smoke test that runs in under a minute on a laptop.
7. Reproducibility is stated per output as recomputed, statistically reproduced, or regenerated. "Fully reproducible" is used only for the first.
8. The archived-vs-recomputed boundary is visible in every table and figure caption the package emits.
9. The environment is pinned (uv, lockfile). A fresh clone runs the smoke test by one documented command.
10. Nothing is tagged, published to PyPI, or made public by an agent. Louis does that, on the ship date.
