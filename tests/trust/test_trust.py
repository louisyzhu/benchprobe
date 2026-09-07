"""Unit tests for benchprobe.trust (ticket T5). Fast; part of the smoke set.

Golden values live in tests/golden/test_judge.py. Here: known small examples, published reference
values for the agreement coefficients, and behaviour on synthetic data.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import benchprobe.trust as t

pytestmark = pytest.mark.smoke

REFERENCE = Path(__file__).with_name("krippendorff_reference.json")


@pytest.fixture(scope="module")
def bank():
    return t.load_bank()


# --------------------------------------------------------------------------------------------
# Bank
# --------------------------------------------------------------------------------------------


def test_bank_layout(bank):
    assert bank.gold.shape == bank.judge.shape == (180, 10)
    assert set(np.unique(bank.gold)) <= {0.0, 1.0} and set(np.unique(bank.judge)) <= {0.0, 1.0}
    assert bank.gold_total.shape == (180,) and bank.judge_total.max() <= 10
    assert len(np.unique(bank.question_id)) == 13
    assert bank.judge_model == "claude-haiku-4-5-20251001"


def test_published_grid_layout():
    grid = t.published_grid()
    assert np.asarray(grid["kr20_mean"]).shape == (5, 5)
    assert grid["measured_error_run"]["judge_error"] == t.MEASURED_ERROR


# --------------------------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------------------------


def test_kr20_hand_example():
    P = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    p = P.mean(0)
    expected = (3 / 2) * (1 - (p * (1 - p)).sum() / P.sum(1).var(ddof=1))
    assert t.kr20(P) == pytest.approx(expected)
    assert np.isnan(t.kr20(np.ones((5, 3))))  # no total variance


def test_kr21_equals_kr20_when_element_difficulties_are_equal():
    rng = np.random.default_rng(0)
    P = (rng.random((500, 8)) < 0.5).astype(float)
    # with equal element rates KR-21 ≈ KR-20 (they differ by the spread of the p_j only)
    assert abs(t.kr21(P.sum(1), 8) - t.kr20(P)) < 0.02


def test_intra_item_rho_is_mean_offdiagonal_phi():
    rng = np.random.default_rng(1)
    P = (rng.random((100, 4)) < 0.5).astype(float)
    C = np.corrcoef(P.T)
    assert t.intra_item_rho(P) == pytest.approx(np.mean(C[np.triu_indices(4, 1)]))


# --------------------------------------------------------------------------------------------
# Beta-binomial and classification
# --------------------------------------------------------------------------------------------


def test_bb_pmf_sums_to_one_and_mle_recovers_parameters():
    assert t.bb_pmf(2.0, 3.0, 10).sum() == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    p = rng.beta(4.0, 6.0, 20000)
    tot = rng.binomial(10, p)
    a, b = t.bb_mle(tot, 10)
    assert a == pytest.approx(4.0, abs=0.4) and b == pytest.approx(6.0, abs=0.6)


def test_bb_gof_shape_and_bounds(bank):
    fit = t.bb_gof(bank.judge_total.astype(int), bank.K)
    assert fit.df >= 1 and 0 <= fit.p <= 1 and fit.chi2 >= 0 and fit.alpha > 0 and fit.beta > 0


def test_livingston_lewis_bounds_and_monotone_cut(bank):
    dc, ca = t.livingston_lewis(bank.judge_total, bank.K, 6)
    assert 0 <= dc <= ca <= 1
    ll = t.livingston_lewis(bank.judge_total, bank.K, 6)
    assert ll.decision_consistency == dc and ll.classification_accuracy == ca


def test_phi_lambda_is_in_unit_interval(bank):
    for cut in (4, 5, 6, 7):
        assert 0 <= t.phi_lambda(bank.judge, cut) <= 1


def test_classification_accuracy_and_agreement_interval(bank):
    acc = t.classification_accuracy(bank.judge_total, bank.gold_total, 6)
    assert 0 <= acc <= 1
    a = t.agreement_with_interval(
        bank.judge_total, bank.gold_total, 6, cluster_ids=bank.question_id, B=50, seed=0
    )
    b = t.agreement_with_interval(
        bank.judge_total, bank.gold_total, 6, cluster_ids=bank.question_id, B=50, seed=0
    )
    assert a.point == acc and a.n_clusters == 13
    np.testing.assert_array_equal(a.draws, b.draws)
    assert a.ci[0] <= a.point <= a.ci[1] or a.ci[0] <= a.ci[1]


def test_cluster_bootstrap_drops_non_finite():
    ids = np.array([0, 0, 1, 1])
    draws = t.cluster_bootstrap(lambda idx: float("nan") if 0 in ids[idx] else 1.0, ids, B=20)
    assert len(draws) < 20 and (draws == 1.0).all()


# --------------------------------------------------------------------------------------------
# Agreement coefficients (new code; reference values)
# --------------------------------------------------------------------------------------------


def test_cohens_kappa_classic_two_by_two():
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert t.cohens_kappa(a, b) == pytest.approx(0.4)
    assert t.cohens_kappa(a, a) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        t.cohens_kappa(a, b[:-1])


def test_krippendorffs_alpha_published_example():
    """Krippendorff (2011), 'Computing Krippendorff's Alpha-Reliability', four observers, twelve
    units: nominal 0.743, interval 0.849 (the published values); ordinal and ratio as computed by
    the reference implementation (the `krippendorff` package, 7 September 2026)."""
    nan = np.nan
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
        [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
    ]
    assert t.krippendorffs_alpha(data, level="nominal") == pytest.approx(0.7434, abs=5e-5)
    assert t.krippendorffs_alpha(data, level="ordinal") == pytest.approx(0.8154, abs=5e-5)
    assert t.krippendorffs_alpha(data, level="interval") == pytest.approx(0.8491, abs=5e-5)
    assert t.krippendorffs_alpha(data, level="ratio") == pytest.approx(0.7974, abs=5e-5)


def test_krippendorffs_alpha_against_reference_implementation():
    """Six random reliability matrices with missing ratings; values from the `krippendorff`
    package (7 September 2026), stored in krippendorff_reference.json."""
    for case in json.loads(REFERENCE.read_text(encoding="utf-8")):
        data = np.array([[np.nan if v is None else v for v in row] for row in case["data"]])
        assert t.krippendorffs_alpha(data, level="nominal") == pytest.approx(
            case["nominal"], abs=1e-9
        )
        assert t.krippendorffs_alpha(data, level="interval") == pytest.approx(
            case["interval"], abs=1e-9
        )


def test_krippendorffs_alpha_edge_cases():
    assert np.isnan(t.krippendorffs_alpha([[1, 1, 1], [1, 1, 1]]))  # a single category
    assert t.krippendorffs_alpha([[1, 2, 3], [1, 2, 3]]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown level"):
        t.krippendorffs_alpha([[1, 2], [1, 2]], level="log")


# --------------------------------------------------------------------------------------------
# G-theory
# --------------------------------------------------------------------------------------------


def test_gstudy_two_way_recovers_planted_components():
    rng = np.random.default_rng(3)
    n, R = 4000, 3
    X = rng.normal(0, 1, (n, 1)) + rng.normal(0, np.sqrt(0.36), (1, R)) + rng.normal(0, 0.3, (n, R))
    g = t.gstudy_two_way(X)
    assert g.v_items == pytest.approx(1.0, abs=0.1)
    assert g.v_resid == pytest.approx(0.09, abs=0.02)
    assert 0 < g.Phi < 1
    with pytest.raises(ValueError):
        t.gstudy_two_way(np.zeros((1, 3)))


# --------------------------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------------------------


def test_simulate_bank_shapes_and_error_rate():
    rng = np.random.default_rng(0)
    G, J = t.simulate_bank(0.4, 0.1, n=2000, K=10, rng=rng)
    assert G.shape == J.shape == (2000, 10)
    assert (G != J).mean() == pytest.approx(0.1, abs=0.01)


def test_two_way_sweep_is_seeded_and_shaped():
    a = t.two_way_sweep(reps=3, seed=5, spreads=[0.2, 0.8], errors=[0.0, 0.1])
    b = t.two_way_sweep(reps=3, seed=5, spreads=[0.2, 0.8], errors=[0.0, 0.1])
    assert a[0].shape == (2, 2)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[0][1, 0] > a[0][0, 0]  # wider bank, higher KR-20 at zero error
