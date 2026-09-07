"""Golden test for the M3 gate: the twelve One Capability tables reproduced by one command, each
recomputed or statistically reproduced within its tolerance, or regenerated with that stated.

Tolerances (T7, docs/decisions.md): 5e-4 on recomputed tables (the archive's own re-run
tolerance) after the archive's rounding; bootstrap intervals within 3 Monte-Carlo standard
deviations (20 re-runs under other seeds) after rounding. The four-learner ladder needs
``--full-ladder`` and is the ``slow`` variant.
"""

import pytest

import benchprobe.report as r

pytestmark = pytest.mark.golden

FAST_PATH_REGENERATED = {"cluster_validation.csv", "lobo_rung_summary.csv"}


def _check(rec, regenerated):
    by_name = {t.name: t for t in rec.tables}
    assert set(by_name) == set(r.ONE_CAPABILITY_TABLES)
    for name, t in by_name.items():
        if name in regenerated:
            assert t.tier == r.REGENERATED, name
        else:
            assert t.tier in (r.RECOMPUTED, r.STATISTICAL), name
            assert t.within_tolerance, (
                f"{name}: {t.tier}, deviation {t.max_abs_deviation}, tolerance {t.tolerance}"
            )


def test_twelve_tables_fast_path(tmp_path):
    rec = r.reproduce_one_capability(tmp_path / "fast")
    _check(rec, FAST_PATH_REGENERATED)


@pytest.mark.slow
def test_twelve_tables_full_ladder(tmp_path):
    rec = r.reproduce_one_capability(tmp_path / "full", full_ladder=True)
    _check(rec, {"cluster_validation.csv"})
    ladder = {t.name: t for t in rec.tables}["lobo_rung_summary.csv"]
    assert "matches the archive" in " ".join(ladder.notes)
