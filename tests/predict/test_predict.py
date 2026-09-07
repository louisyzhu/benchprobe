"""Unit tests for benchprobe.predict (ticket T4). Fast; ridge only; part of the smoke set.

The golden values live in tests/golden. These tests check the LOBO machinery on the vendored grid
with the fast learner and on synthetic data whose structure is known.
"""

import numpy as np
import pandas as pd
import pytest

import benchprobe.io as bio
import benchprobe.predict as p

pytestmark = pytest.mark.smoke

ECON = list(bio.ECONOMIC_BENCHMARKS)


@pytest.fixture(scope="module")
def grid():
    return bio.build_grid(bio.load_snapshot(), "complete_case")


@pytest.fixture(scope="module")
def ridge_result(grid):
    return p.lobo(grid, targets=["gdpval_elo", "tau2"], k=3, learners="ridge", seed=0)


def test_registered_learners_are_the_archives_four():
    learners = p.registered_learners()
    assert list(learners) == ["ridge", "elasticnet", "rf", "gbm"]
    assert learners["ridge"][1] == {"alpha": [0.03, 0.1, 0.3, 1, 3, 10, 30]}
    assert learners["rf"][0].random_state == 0 and learners["gbm"][0].random_state == 0


def test_lobo_metrics_layout(ridge_result):
    m = ridge_result.metrics
    assert list(m.columns) == list(p.METRIC_COLUMNS)
    assert len(m) == 2 * len(p.RUNGS) * 1
    assert set(m.target) == {"gdpval_elo", "tau2"} and set(m.learner) == {"ridge"}
    assert set(m.rung) == set(p.RUNGS)
    assert (m.block == "Economic").all()
    assert np.allclose(m.test_mse, m.test_rmse**2)
    assert ridge_result.n == 96 and ridge_result.grid == "complete_case"


def test_oof_rows_cover_every_model_once(ridge_result):
    y_true, y_pred = ridge_result.oof[("gdpval_elo", "iv_kfac", "ridge")]
    assert y_true.shape == y_pred.shape == (96,)
    # the standardised target has mean ~0 fold by fold; overall it is close to zero
    assert abs(y_true.mean()) < 0.1


def test_lobo_is_deterministic(grid):
    a = p.lobo(grid, targets=["tau2"], learners="ridge", rungs=["ii_meanidx", "iv_kfac"], seed=0)
    b = p.lobo(grid, targets=["tau2"], learners="ridge", rungs=["ii_meanidx", "iv_kfac"], seed=0)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)


def test_lobo_rejects_unknown_target_and_learner(grid):
    with pytest.raises(ValueError, match="not a benchmark"):
        p.lobo(grid, targets=["mmmuPro"], learners="ridge")
    with pytest.raises(ValueError, match="unknown learner"):
        p.lobo(grid, targets=["tau2"], learners="svm")
    with pytest.raises(ValueError, match="unknown learner"):
        p.lobo(grid, targets=["tau2"], learners=["ridge", "svm"])


def test_rung_subsets_are_checked_by_ladder_and_h4_table(grid):
    r = p.lobo(grid, targets=["tau2"], learners="ridge", rungs=["ii_meanidx", "iv_kfac"], seed=0)
    assert list(p.ladder(r, targets=["tau2"]).index) == ["ii_meanidx", "iv_kfac"]
    with pytest.raises(ValueError, match="not in this result"):
        p.h4_table(r, targets=["tau2"], B=5)
    table = p.h4_table(r, baselines=("ii_meanidx",), targets=["tau2"], B=5)
    assert len(table) == 2


def test_rung_v_is_dropped_without_covariates(grid):
    r = p.lobo(grid, targets=["tau2"], learners="ridge", covariates=False)
    assert "v_kfac_cov" not in r.rungs and "iv_kfac" in r.rungs


def test_ladder_layout_and_pooling(ridge_result):
    table = p.ladder(ridge_result, targets=["gdpval_elo", "tau2"])
    assert list(table.columns) == ["learner", "train_rmse", "test_rmse", "test_r2", "runner_up_gap"]
    assert list(table.index) == list(p.RUNGS)
    assert (table.learner == "ridge").all()
    assert table.runner_up_gap.isna().all()  # a single learner has no runner-up
    m = ridge_result.metrics[ridge_result.metrics.rung == "iv_kfac"]
    assert table.loc["iv_kfac", "test_rmse"] == pytest.approx(m.test_rmse.mean())  # mean of RMSEs
    assert table.loc["iv_kfac", "train_rmse"] == pytest.approx(m.train_rmse.mean())
    rms = p.ladder(ridge_result, targets=["gdpval_elo", "tau2"], pooling="rms")
    assert rms.loc["iv_kfac", "test_rmse"] == pytest.approx(np.sqrt(m.test_mse.mean()))
    with pytest.raises(ValueError, match="unknown pooling"):
        p.ladder(ridge_result, targets=["tau2"], pooling="median")


def test_pooled_delta_mse_is_the_mean_squared_error_difference(ridge_result):
    d = p.pooled_delta_mse(ridge_result, targets=["gdpval_elo", "tau2"], B=50, seed=1)
    m = ridge_result.metrics.set_index(["target", "rung"]).test_mse
    expected = np.mean([m[(t, "ii_meanidx")] - m[(t, "iv_kfac")] for t in ["gdpval_elo", "tau2"]])
    assert d.point == pytest.approx(expected)
    assert d.n == 192 and d.draws.shape == (50,) and d.ci[0] <= d.ci[1]
    assert list(d.per_target.index) == ["gdpval_elo", "tau2"]


def test_pooled_delta_mse_is_seeded_and_streams_continue(ridge_result):
    a = p.pooled_delta_mse(ridge_result, targets=["tau2"], B=30, seed=7)
    b = p.pooled_delta_mse(ridge_result, targets=["tau2"], B=30, seed=7)
    np.testing.assert_array_equal(a.draws, b.draws)
    rng = np.random.default_rng(7)
    c = p.pooled_delta_mse(ridge_result, targets=["tau2"], B=30, stream=rng)
    d = p.pooled_delta_mse(ridge_result, targets=["tau2"], B=30, stream=rng)
    np.testing.assert_array_equal(a.draws, c.draws)
    assert not np.array_equal(c.draws, d.draws)
    assert c.seed is None


def test_h4_table_layout(ridge_result):
    table = p.h4_table(
        ridge_result, baselines=("ii_meanidx",), targets=["gdpval_elo", "tau2"], B=20, seed=3
    )
    assert list(table.columns) == [
        "scope",
        "baseline",
        "kmodel",
        "dMSE",
        "ci_lo",
        "ci_hi",
        "excludes_zero",
    ]
    assert list(table.scope) == ["pooled", "gdpval_elo", "tau2"]
    assert (table.excludes_zero == ((table.ci_lo > 0) | (table.ci_hi < 0))).all()


def test_factor_rung_predicts_a_planted_factor_structure():
    """Synthetic: benchmarks driven by two latent factors; the k-factor rung should beat the mean
    index when the target loads on the second factor only."""
    rng = np.random.default_rng(5)
    n = 300
    f = rng.standard_normal((n, 2))
    load = np.array([[0.9, 0.0]] * 5 + [[0.0, 0.9]] * 5)
    x = f @ load.T + 0.3 * rng.standard_normal((n, 10))
    cols = [f"b{j}" for j in range(10)]
    frame = pd.DataFrame(x, columns=cols)
    frame["releaseDate"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    grid = bio.Grid(
        name="synthetic",
        frame=frame,
        benchmarks=tuple(cols),
        labels={c: c for c in cols},
        blocks={c: "A" for c in cols},
        z=bio.zscore(frame, cols),
        days=np.arange(n),
    )
    r = p.lobo(grid, targets=["b9"], k=2, learners="ridge", rungs=["ii_meanidx", "iv_kfac"], seed=0)
    d = p.pooled_delta_mse(r, targets=["b9"], B=100, seed=0)
    assert d.point > 0 and d.ci[0] > 0
