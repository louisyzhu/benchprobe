"""Unit tests for benchprobe.measure (ticket T3). Fast, synthetic; part of the smoke set.

The golden values live in tests/golden; these tests check the estimators' behaviour on data
whose structure is known.
"""

import numpy as np
import pandas as pd
import pytest

import benchprobe.measure as m

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def one_factor():
    """200 × 8 indicators driven by one latent factor plus noise, standardised."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal(200)
    loadings = np.linspace(0.6, 0.9, 8)
    x = np.outer(f, loadings) + rng.standard_normal((200, 8)) * np.sqrt(1 - loadings**2)
    z = (x - x.mean(0)) / x.std(0)
    return pd.DataFrame(z, columns=[f"b{j}" for j in range(8)])


@pytest.fixture(scope="module")
def days():
    return np.linspace(0, 999, 200).round().astype(int)


def test_kmo_is_high_for_a_clean_one_factor_battery(one_factor):
    value = m.kmo(one_factor)
    assert 0.8 < value <= 1.0


def test_kmo_accepts_arrays_and_rejects_nan(one_factor):
    assert m.kmo(one_factor.to_numpy()) == pytest.approx(m.kmo(one_factor))
    bad = one_factor.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m.kmo(bad)


def test_parallel_analysis_retains_one_factor_and_is_seeded(one_factor):
    a = m.parallel_analysis(one_factor, n_iter=200, seed=1)
    b = m.parallel_analysis(one_factor, n_iter=200, seed=1)
    assert a.k_retained == 1
    np.testing.assert_array_equal(a.threshold, b.threshold)
    assert a.observed[0] > a.threshold[0] > 0
    assert np.all(np.diff(a.observed) <= 1e-12)  # descending


def test_efa_shapes_and_sign_alignment(one_factor):
    sol = m.efa(one_factor, k=2, rotation="oblimin")
    assert sol.loadings.shape == (8, 2) and list(sol.loadings.columns) == ["F1", "F2"]
    assert sol.scores.shape == (200, 2)
    assert sol.phi.shape == (2, 2) and np.allclose(np.diag(sol.phi), 1)
    row_mean = one_factor.to_numpy().mean(axis=1)
    for j in range(2):
        c = np.corrcoef(sol.scores[:, j], row_mean)[0, 1]
        assert not np.isfinite(c) or c >= 0
    assert set(np.unique(sol.signs)) <= {-1.0, 1.0}


def test_efa_unrotated_scores_equal_thurstone_projection(one_factor):
    """Unrotated, the structure matrix is the loadings, so transform() and z @ R⁻¹Λ coincide."""
    sol = m.efa(one_factor, k=1, rotation=None, align_to_mean_score=False)
    R = np.corrcoef(one_factor.to_numpy(), rowvar=False)
    W = m.thurstone_weights(R, sol.loadings.to_numpy())
    np.testing.assert_allclose(sol.weights.to_numpy(), W, atol=1e-10)
    np.testing.assert_allclose(sol.scores, one_factor.to_numpy() @ W, atol=1e-8)


def test_efa_structure_is_loadings_times_phi(one_factor):
    sol = m.efa(one_factor, k=2, rotation="oblimin")
    np.testing.assert_allclose(
        sol.structure.to_numpy(), sol.loadings.to_numpy() @ sol.phi.to_numpy(), atol=1e-10
    )


def test_first_factor_share_is_a_fraction_and_large_here(one_factor):
    share = m.first_factor_share(one_factor, k=2)
    assert 0.5 < share <= 1.0


def test_residualise_on_date_removes_the_linear_trend(one_factor, days):
    trended = one_factor + np.outer(days / days.max(), np.ones(8))
    resid = m.residualise_on_date(trended, days)
    assert list(resid.columns) == list(one_factor.columns)
    np.testing.assert_allclose(resid.mean().to_numpy(), 0, atol=1e-10)
    for col in resid.columns:
        assert abs(np.corrcoef(resid[col], days)[0, 1]) < 1e-8


def test_residualise_on_date_checks_lengths(one_factor, days):
    with pytest.raises(ValueError, match="one entry per row"):
        m.residualise_on_date(one_factor, days[:-1])


def test_date_r2_ols_matches_a_closed_form(days):
    y = 0.002 * days + np.sin(days / 50.0)
    r2 = m.date_r2(y, days, form="ols")
    slope, intercept = np.polyfit(days, y, 1)
    expected = 1 - ((y - (slope * days + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 == pytest.approx(expected, abs=1e-10)


def test_date_r2_logistic_fits_a_logistic_curve_better_than_a_line(days):
    y = -1 + 2 / (1 + np.exp(-0.02 * (days - 500)))
    assert m.date_r2(y, days, form="logistic") > 0.999
    assert m.date_r2(y, days, form="logistic") > m.date_r2(y, days, form="ols")


def test_date_r2_rejects_unknown_form_and_constant_scores(days):
    with pytest.raises(ValueError, match="unknown form"):
        m.date_r2(np.sin(days / 40.0), days, form="cubic")
    for form in ("ols", "logistic"):
        with pytest.raises(ValueError, match="constant"):
            m.date_r2(np.ones(len(days)), days, form=form)


def test_efa_scores_match_factor_analyzer_transform(one_factor):
    """The explicit score computation reproduces factor_analyzer.transform (no silent fallback)."""
    FactorAnalyzer, _ = m._factor_analyzer()
    fa = FactorAnalyzer(n_factors=2, rotation="oblimin", method="ml").fit(one_factor.to_numpy())
    expected = fa.transform(one_factor.to_numpy())
    sol = m.efa(one_factor, k=2, rotation="oblimin", align_to_mean_score=False)
    np.testing.assert_allclose(sol.scores, expected, atol=1e-10)


def test_first_factor_share_warns_when_column_zero_is_not_the_largest(monkeypatch, one_factor):
    """The archive's convention (column 0) is kept, but a re-ordered solution is not silent."""
    real = m.efa

    def swapped(z, **kwargs):
        sol = real(z, **kwargs)
        ssl = sol.ssl.iloc[::-1].to_numpy()  # pretend the small factor came first
        return m.Efa(**{**sol.__dict__, "ssl": pd.Series(ssl, index=sol.ssl.index)})

    monkeypatch.setattr(m, "efa", swapped)
    with pytest.warns(RuntimeWarning, match="not the largest"):
        m.first_factor_share(one_factor, k=2)


def test_residualisation_drop_bootstrap_is_seeded_and_ordered(one_factor, days):
    a = m.residualisation_drop_bootstrap(one_factor, days, k=2, B=8, seed=3)
    b = m.residualisation_drop_bootstrap(one_factor, days, k=2, B=8, seed=3)
    np.testing.assert_array_equal(a.draws, b.draws)
    assert a.draws.shape == (8,) and a.B == 8 and a.seed == 3
    assert a.ci[0] <= a.ci[1]
    assert abs(a.point) < 0.5
