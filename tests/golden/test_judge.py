"""Golden acceptance tests for the JUDGe study (*Three Ways Classical Test Theory Misleads for LLM
Judges*), ticket T5.

Two tiers of locked value, from ``llm-judge-reliability`` at commit ``126d1bce``:

* the real-bank quantities the archive README states reproduce bit-for-bit (KR-20 on judge and
  gold verdicts, per-element error, judge–gold correlation, leniency, beta-binomial goodness of
  fit);
* the simulated quantities. ``judge_sweeps_reference.json`` is the output of the archive's own
  ``python sweeps.py`` (run on 7 September 2026 under this repository's locked environment), which
  benchprobe's ``sweeps()`` must reproduce to floating-point identity because the seeds and draw
  order are the archive's; the *published* grid (``sweep_grid.json``, vendored) came from a
  different random stream and is reproduced statistically, within 0.5 of one per-cell SD (the
  archive README measures 0.31).

Tolerances set at T5 (docs/decisions.md): half a unit of the last printed digit for the bit-for-bit
quantities; 1e-9 for the archive-code simulations; 0.5 SD per cell for the published grid. Adjust
only by ticket (rule 1).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.golden

SNAPSHOT = "judge_2026-08-31"
REFERENCE = Path(__file__).with_name("judge_sweeps_reference.json")

# Real bank (archive README, "Reproduction notes")
N_JUDGED = 180  # of 210 items (archive make_figures.load_bank drops rows without judge verdicts)
KR20_JUDGE = 0.5223
KR20_GOLD = 0.5231
PER_ELEMENT_ERROR_PCT = 4.72
JUDGE_GOLD_CORRELATION = 0.921
LENIENCY_ELEMENTS = 0.46
BB_CHI2 = 5.31
BB_DF = 6
BB_P = 0.504

TOL_KR20 = 0.00005
TOL_ERROR_PP = 0.005
TOL_CORR = 0.0005
TOL_LENIENCY = 0.005
TOL_CHI2 = 0.005
TOL_P = 0.0005
TOL_SIMULATION = 1e-9
TOL_PUBLISHED_SD = 0.5


def api(module: str, name: str):
    try:
        mod = importlib.import_module(f"benchprobe.{module}")
    except ImportError as exc:  # pragma: no cover
        pytest.fail(f"benchprobe.{module} cannot be imported: {exc}")
    fn = getattr(mod, name, None)
    if fn is None:
        pytest.fail(f"benchprobe.{module}.{name} is not implemented (SPEC.md §2)")
    return fn


def close(actual: float, expected: float, tol: float, label: str) -> None:
    assert abs(actual - expected) <= tol, (
        f"{label}: archive {expected}, recomputed {actual:.6g}, tolerance {tol}"
    )


@pytest.fixture(scope="module")
def bank():
    load_snapshot = api("io", "load_snapshot")
    load_bank = api("trust", "load_bank")
    return load_bank(load_snapshot(SNAPSHOT))


@pytest.fixture(scope="module")
def summary(bank):
    return api("trust", "bank_summary")(bank)


def test_bank_size(bank):
    assert bank.n == N_JUDGED and bank.K == 10 and bank.n_items_total == 210


def test_kr20_judge_and_gold(summary):
    close(summary["kr20_judge"], KR20_JUDGE, TOL_KR20, "KR-20, judge verdicts")
    close(summary["kr20_gold"], KR20_GOLD, TOL_KR20, "KR-20, gold verdicts")


def test_per_element_error(summary):
    close(100 * summary["per_element_error"], PER_ELEMENT_ERROR_PCT, TOL_ERROR_PP, "error (%)")


def test_judge_gold_correlation_and_leniency(summary):
    close(summary["judge_gold_correlation"], JUDGE_GOLD_CORRELATION, TOL_CORR, "judge–gold r")
    close(summary["leniency_elements"], LENIENCY_ELEMENTS, TOL_LENIENCY, "leniency (elements)")


def test_beta_binomial_goodness_of_fit(summary):
    close(summary["bb_chi2"], BB_CHI2, TOL_CHI2, "beta-binomial χ²")
    assert summary["bb_df"] == BB_DF, f"df: archive {BB_DF}, recomputed {summary['bb_df']}"
    close(summary["bb_p"], BB_P, TOL_P, "beta-binomial p")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        else:
            out[key] = np.asarray(v, dtype=float)
    return out


def test_simulations_reproduce_the_archive_code_exactly():
    """``sweeps()`` against the archive's own ``sweeps.py`` output (seeds 11, 101, 303, 7)."""
    sweeps = api("trust", "sweeps")
    ours = _flatten(sweeps())
    reference = _flatten(json.loads(REFERENCE.read_text(encoding="utf-8")))
    assert set(ours) == set(reference), sorted(set(ours) ^ set(reference))
    worst = max(float(np.max(np.abs(ours[k] - reference[k]))) for k in reference)
    assert worst <= TOL_SIMULATION, (
        f"largest deviation from the archive's sweeps.py output {worst:.3e}"
    )


def test_two_way_sweep_reproduces_the_published_grid_statistically():
    """The published grid (a different random stream) within 0.5 SD per cell; the archive README
    reports about 0.31."""
    two_way_sweep = api("trust", "two_way_sweep")
    published_grid = api("trust", "published_grid")
    grid = published_grid()
    mean, _ = two_way_sweep(reps=60, seed=11)
    pub_mean = np.asarray(grid["kr20_mean"], dtype=float)
    pub_sd = np.asarray(grid["kr20_sd"], dtype=float)
    worst = float(np.max(np.abs(mean - pub_mean) / pub_sd))
    assert worst <= TOL_PUBLISHED_SD, f"largest cell deviation {worst:.3f} SD"


def test_measured_error_run_reproduces_statistically():
    """The published 60-replicate run at the measured 4.72 % error rate, by bank spread."""
    two_way_sweep = api("trust", "two_way_sweep")
    published_grid = api("trust", "published_grid")
    run = published_grid()["measured_error_run"]
    mean, _ = two_way_sweep(reps=run["reps"], seed=11, errors=[run["judge_error"]])
    pub_mean = np.asarray(run["kr20_mean"], dtype=float)
    pub_sd = np.asarray(run["kr20_sd"], dtype=float)
    worst = float(np.max(np.abs(mean[:, 0] - pub_mean) / pub_sd))
    assert worst <= TOL_PUBLISHED_SD, f"largest deviation {worst:.3f} SD"
