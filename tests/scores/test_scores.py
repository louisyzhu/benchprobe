"""Unit tests for benchprobe.scores and benchprobe.studies (v0.1 door). Smoke set."""

import numpy as np
import pandas as pd
import pytest

import benchprobe.irt as irt
import benchprobe.predict as predict
import benchprobe.studies as studies
from benchprobe.scores import ScoreMatrix

pytestmark = pytest.mark.smoke


@pytest.fixture
def long():
    rows = []
    for i in range(6):
        for j in range(4):
            if (i, j) != (5, 3):
                rows.append((f"m{i}", f"b{j}", 0.1 * (i + 1) + 0.05 * j, f"2024-0{j + 1}-01"))
    return pd.DataFrame(rows, columns=["model", "item", "score", "date"])


def test_from_long_pivots_and_keeps_missing_cells_missing(long):
    sm = ScoreMatrix.from_long(long, release_date="date")
    assert sm.shape == (6, 4)
    assert np.isnan(sm.wide.loc["m5", "b3"])
    assert sm.long().shape[0] == 23
    assert sm.complete().shape == (5, 4)
    assert sm.coverage().loc["b3", "n_models"] == 5
    assert sm.release_dates is not None and sm.release_dates.notna().all()
    assert sm.days_since_earliest_release().min() == 0


def test_from_long_refuses_duplicates_unless_told(long):
    dup = pd.concat([long, long.iloc[[0]]])
    with pytest.raises(ValueError, match="share a \\(model, item\\) pair"):
        ScoreMatrix.from_long(dup)
    sm = ScoreMatrix.from_long(dup, aggregate="mean")
    assert sm.wide.loc["m0", "b0"] == pytest.approx(long.score.iloc[0])


def test_percent_scale_and_range_check(long):
    pct = long.assign(score=long.score * 100)
    with pytest.raises(ValueError, match="scale='percent'"):
        ScoreMatrix.from_long(pct)
    sm = ScoreMatrix.from_long(pct, scale="percent")
    assert sm.wide.max().max() <= 1.0


def test_from_wide_roundtrips(long):
    sm = ScoreMatrix.from_long(long)
    again = ScoreMatrix.from_wide(sm.wide.reset_index())
    pd.testing.assert_frame_equal(sm.wide, again.wide)
    with pytest.raises(ValueError, match="no score columns"):
        ScoreMatrix.from_wide(pd.DataFrame({"model": ["a"], "name": ["x"]}))


def test_grid_feeds_lobo_and_long_feeds_irt(long):
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(40)
    rows = []
    for i, t in enumerate(theta):
        for j in range(5):
            y = 1.2 * (t - 0.3 * j) + rng.normal(0, 0.3)
            rows.append((f"m{i:02d}", f"b{j}", 1 / (1 + np.exp(-y))))
    frame = pd.DataFrame(rows, columns=["model", "item", "score"])
    sm = ScoreMatrix.from_long(frame)
    grid = sm.grid()
    assert grid.n == 40 and grid.benchmarks == ("b0", "b1", "b2", "b3", "b4")
    res = predict.lobo(grid, targets=["b4"], k=1, learners=["ridge"], rungs=["iii_f1"], seed=0)
    assert set(res.metrics["rung"]) == {"iii_f1"}
    panel = irt.panel_from_long(sm.long())
    assert panel.M == 40 and panel.K == 5 and panel.repaired == ()
    fit = irt.fit_single_stage(panel, adam_steps=500)
    assert fit.converged
    table = irt.ability_table(fit, panel=panel)
    assert np.corrcoef(table["theta"], theta)[0, 1] > 0.98
    with pytest.raises(ValueError, match="pass panel="):
        irt.ability_table(fit)


def test_panel_from_long_validates():
    with pytest.raises(KeyError):
        irt.panel_from_long(pd.DataFrame({"model": ["a"], "score": [0.5]}))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        irt.panel_from_long(pd.DataFrame({"model": ["a"], "item": ["b"], "score": [1.5]}))


def test_studies_namespace_exposes_the_three_entry_points():
    assert set(studies.__all__) == {"one_capability", "judge", "price_of_intelligence"}
    out = studies.judge()
    assert out["bank"]["kr20_judge"] == pytest.approx(0.5223, abs=5e-5)
    assert "grid" in out["sweeps"] and "phrasing_gstudy" in out["sweeps"]
