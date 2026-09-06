"""Golden acceptance tests: the locked numbers of *One Capability or Many?* (handoff §6).

Every value below is locked. It comes from the paper (arXiv:2608.29420, unchanged since 19 August
2026) and was verified on 6 September 2026 against the archived tables in
``frontier-ai-economic-validity`` at commit ``946ce845``: ``task1_structure_results.json``,
``hypothesis_adjudication.csv``, ``bootstrap_h2_drop.csv``, ``dedup_r1_comparison.csv``,
``subsample_date_compute.csv``, ``loadings_residualised.csv``, ``h4_bootstrap_dmse.csv``,
``h4_bootstrap_dedup.csv`` and ``lobo_rung_summary.csv``. The archive file each row was checked
against is named in the test's docstring.

Tolerances for KMO, shares, R² and ΔMSE are the handoff's (T1). The others were set at T1 and are
recorded in ``docs/decisions.md``. None of them, and no number here, is changed without a ticket
(rule 1). A failing test is a finding: report the archive value, the recomputed value and the
tolerance (rule 3).

These tests call the SPEC.md §2 signatures. Until the implementing ticket lands, each fails with the
name of the missing function; nothing here is skipped or marked expected-to-fail. Inputs come from
the snapshot vendored under ``benchprobe/data/one_capability_2026-07-06`` (SHA-256 verified by
``benchprobe.io``).
"""

from __future__ import annotations

import functools
import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.golden

# --------------------------------------------------------------------------------------------
# Names (SPEC.md §3)
# --------------------------------------------------------------------------------------------

SNAPSHOT = "one_capability_2026-07-06"
BENCHMARKS = [
    "gpqa",
    "hle",
    "omniscience",
    "tau2",
    "ifbench",
    "lcr",
    "scicode",
    "critpt",
    "terminalbenchHard",
    "gdpval_elo",
    "terminalbenchV21",
    "tauBanking",
]
ECONOMIC = ["gdpval_elo", "terminalbenchV21", "tauBanking", "tau2"]
K = 3  # factors in the exploratory solution and the LOBO rung (iv)

# --------------------------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------------------------

# Handoff §7, T1
TOL_KMO = 0.005
TOL_SHARE_PP = 0.5  # shares of common variance, in percentage points
TOL_R2 = 0.005
TOL_DMSE = 0.002

# Set at T1 (docs/decisions.md); adjust only by ticket
TOL_LOADING = 0.02
TOL_RMSE = 0.005
TOL_DROP_PP = 1.0  # a drop is the difference of two shares that each carry 0.5 pp
TOL_CI_DMSE = 0.005  # bootstrap endpoints: statistically-reproduced tier
TOL_CI_DROP_PP = 2.0  # bootstrap endpoints: statistically-reproduced tier

# --------------------------------------------------------------------------------------------
# Locked values (handoff §6)
# --------------------------------------------------------------------------------------------

GRID_SIZES = {
    "complete_case": 96,  # task1_structure_results.json n_G1; task2_prediction_results.json n_LOBO
    "deduplicated": 89,  # dedup_r1_comparison.csv, h4_bootstrap_dedup.csv
    "compute_known": 58,  # task1_structure_results.json n_subsample; subsample_date_compute.csv
    "economic_dense": 103,  # archive README; open item 1 in docs/reproducibility.md
}

KMO_COMPLETE_CASE = 0.933
FIRST_FACTOR_SHARE_PCT = 74.5
LOGISTIC_DATE_R2 = 0.505

RESIDUALISED_SHARE_PCT = 59.6
RESIDUALISATION_DROP_PP = 14.9
RESIDUALISATION_DROP_CI_PP = (-5.3, 32.7)
RESIDUALISATION_DROP_PP_BY_GRID = {"deduplicated": 24.1, "compute_known": 16.5}

RESIDUALISED_ECONOMIC_LOADINGS = {
    "terminalbenchV21": 1.01,
    "gdpval_elo": 0.84,
    "tauBanking": 0.54,
    "tau2": 0.50,
}
LARGEST_ECONOMIC_CROSS_LOADING = 0.38

LOBO_POOLED_DMSE = 0.037
LOBO_POOLED_DMSE_CI = (0.019, 0.055)
LOBO_POOLED_DMSE_DEDUPLICATED = 0.038

LADDER = {  # economic block, best learner per rung
    "ii_meanidx": {"test_rmse": 0.474, "test_r2": 0.771},  # single-index baseline
    "iv_kfac": {"test_rmse": 0.433, "test_r2": 0.808},  # k-factor
    "iii_f1": {"test_rmse": 0.950},  # first factor alone
}


# --------------------------------------------------------------------------------------------
# Access to the SPEC API, resolved inside the test so a missing function fails that test by name
# --------------------------------------------------------------------------------------------


def api(module: str, name: str):
    """Return ``benchprobe.<module>.<name>`` or fail the calling test with a clear reason."""
    try:
        mod = importlib.import_module(f"benchprobe.{module}")
    except ImportError as exc:  # pragma: no cover - the module stubs exist from T0
        pytest.fail(f"benchprobe.{module} cannot be imported: {exc}")
    fn = getattr(mod, name, None)
    if fn is None:
        pytest.fail(f"benchprobe.{module}.{name} is not implemented (SPEC.md §2)")
    return fn


@functools.lru_cache(maxsize=None)
def grid(name: str):
    load_snapshot = api("io", "load_snapshot")
    build_grid = api("io", "build_grid")
    return build_grid(load_snapshot(SNAPSHOT), name)


@functools.lru_cache(maxsize=None)
def lobo(grid_name: str, learners: str):
    run = api("predict", "lobo")
    return run(
        grid(grid_name),
        targets=tuple(ECONOMIC),
        k=K,
        learners=learners,
        outer_folds=5,
        inner_folds=5,
        seed=0,
    )


def economic_factor(loadings) -> str:
    """The factor column carrying the largest sum of squared loadings over the economic benchmarks.

    Identified by the numbers, not assumed to be ``F1``: a rotation may permute columns without
    changing any value the paper prints.
    """
    ssl = (loadings.loc[ECONOMIC] ** 2).sum(axis=0)
    return str(ssl.idxmax())


def close(actual: float, expected: float, tol: float, label: str) -> None:
    assert abs(actual - expected) <= tol, (
        f"{label}: archive {expected}, recomputed {actual:.6g}, tolerance {tol}"
    )


# --------------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "n"), sorted(GRID_SIZES.items()))
def test_grid_size(name, n):
    """Row counts the §6 rows are stated on (task1_structure_results.json, dedup_r1_comparison.csv,
    subsample_date_compute.csv, archive README). The three grids complete on all twelve benchmarks
    carry them in the archive's order; ``economic_dense`` is counted only (open item 1)."""
    g = grid(name)
    assert g.n == n, f"grid {name}: archive n = {n}, built n = {g.n}"
    if name != "economic_dense":
        assert list(g.benchmarks) == BENCHMARKS


# --------------------------------------------------------------------------------------------
# Measure: structure on the complete-case grid
# --------------------------------------------------------------------------------------------


def test_kmo():
    """KMO 0.933 on the 96-model complete-case grid (task1_structure_results.json kmo_G1 0.9326)."""
    kmo = api("measure", "kmo")
    close(kmo(grid("complete_case").z), KMO_COMPLETE_CASE, TOL_KMO, "KMO")


def test_first_factor_share():
    """First factor carries 74.5 % of common variance, unrotated k = 3 ML solution
    (task1_structure_results.json raw_factor1_communal_share 0.74544)."""
    first_factor_share = api("measure", "first_factor_share")
    share_pct = 100 * first_factor_share(grid("complete_case").z, k=K)
    close(share_pct, FIRST_FACTOR_SHARE_PCT, TOL_SHARE_PP, "first-factor share (pp)")


def test_logistic_date_r2_of_dominant_factor():
    """Dominant factor's scores track release date with logistic-fit R² 0.505
    (hypothesis_adjudication.csv H2(i); paper §Results and Appendix). The dominant factor is the
    oblimin factor with the largest sum of squared loadings. The archived notebook derives only the
    OLS counterpart (0.477); the logistic form is open item 2 in docs/reproducibility.md."""
    efa = api("measure", "efa")
    date_r2 = api("measure", "date_r2")
    g = grid("complete_case")
    solution = efa(g.z, k=K, rotation="oblimin")
    dominant = int(np.argmax(np.asarray(solution.ssl)))
    r2 = date_r2(np.asarray(solution.scores)[:, dominant], g.days, form="logistic")
    close(r2, LOGISTIC_DATE_R2, TOL_R2, "logistic date-R² of the dominant factor")


def test_residualised_first_factor_share():
    """Date-residualised first-factor share 59.6 %
    (task1_structure_results.json residualised_factor1_share_dateonly 0.59635)."""
    residualise_on_date = api("measure", "residualise_on_date")
    first_factor_share = api("measure", "first_factor_share")
    g = grid("complete_case")
    share_pct = 100 * first_factor_share(residualise_on_date(g.z, g.days), k=K)
    close(share_pct, RESIDUALISED_SHARE_PCT, TOL_SHARE_PP, "residualised first-factor share (pp)")


def test_residualisation_drop():
    """Residualising on date drops the first-factor share by 14.9 pp
    (task1_structure_results.json drop_pp_dateonly 14.909)."""
    residualise_on_date = api("measure", "residualise_on_date")
    first_factor_share = api("measure", "first_factor_share")
    g = grid("complete_case")
    raw = 100 * first_factor_share(g.z, k=K)
    res = 100 * first_factor_share(residualise_on_date(g.z, g.days), k=K)
    close(raw - res, RESIDUALISATION_DROP_PP, TOL_DROP_PP, "residualisation drop (pp)")


def test_residualisation_drop_bootstrap_interval():
    """Model-resampling bootstrap of the drop: 95 % interval [−5.3, +32.7] pp
    (bootstrap_h2_drop.csv drop_pp ci_lo −5.337, ci_hi 32.731; B = 2000, default_rng(42)).
    Statistically-reproduced tier unless the archive's draw order is replicated."""
    bootstrap = api("measure", "residualisation_drop_bootstrap")
    g = grid("complete_case")
    result = bootstrap(g.z, g.days, k=K, B=2000, seed=42)
    lo, hi = result.ci
    close(100 * lo, RESIDUALISATION_DROP_CI_PP[0], TOL_CI_DROP_PP, "drop bootstrap lower (pp)")
    close(100 * hi, RESIDUALISATION_DROP_CI_PP[1], TOL_CI_DROP_PP, "drop bootstrap upper (pp)")


@pytest.mark.parametrize(("name", "drop_pp"), sorted(RESIDUALISATION_DROP_PP_BY_GRID.items()))
def test_residualisation_drop_by_grid(name, drop_pp):
    """The same drop on the other two grids: deduplicated base-model grid 24.1 pp
    (dedup_r1_comparison.csv h2ii_drop_pp) and compute-known subsample, date only, 16.5 pp
    (subsample_date_compute.csv date_only drop_vs_raw_pp 16.48)."""
    residualise_on_date = api("measure", "residualise_on_date")
    first_factor_share = api("measure", "first_factor_share")
    g = grid(name)
    raw = 100 * first_factor_share(g.z, k=K)
    res = 100 * first_factor_share(residualise_on_date(g.z, g.days), k=K)
    close(raw - res, drop_pp, TOL_DROP_PP, f"residualisation drop on {name} (pp)")


def test_residualised_economic_loadings():
    """Residualised three-factor oblimin loadings of the economic benchmarks on the economic factor:
    Terminal-Bench v2.1 1.01, GDPval 0.84, τ³-Banking 0.54, τ²-Bench 0.50
    (loadings_residualised.csv; paper Table tab:loadings)."""
    residualise_on_date = api("measure", "residualise_on_date")
    efa = api("measure", "efa")
    g = grid("complete_case")
    solution = efa(residualise_on_date(g.z, g.days), k=K, rotation="oblimin")
    factor = economic_factor(solution.loadings)
    for key, expected in RESIDUALISED_ECONOMIC_LOADINGS.items():
        close(
            float(solution.loadings.loc[key, factor]),
            expected,
            TOL_LOADING,
            f"residualised loading of {key} on the economic factor ({factor})",
        )


def test_largest_economic_cross_loading():
    """Largest cross-loading of an economic benchmark on a non-economic factor: 0.38
    (loadings_residualised.csv τ³-Banking on F3 0.3826; hypothesis_adjudication.csv H3)."""
    residualise_on_date = api("measure", "residualise_on_date")
    efa = api("measure", "efa")
    g = grid("complete_case")
    solution = efa(residualise_on_date(g.z, g.days), k=K, rotation="oblimin")
    factor = economic_factor(solution.loadings)
    others = [c for c in solution.loadings.columns if c != factor]
    largest = float(solution.loadings.loc[ECONOMIC, others].abs().max().max())
    close(largest, LARGEST_ECONOMIC_CROSS_LOADING, TOL_LOADING, "largest economic cross-loading")


# --------------------------------------------------------------------------------------------
# Predict: LOBO on the complete-case grid
# --------------------------------------------------------------------------------------------


def test_lobo_pooled_delta_mse():
    """Pooled economic ΔMSE, mean-index baseline vs k-factor, ridge: +0.037
    (h4_bootstrap_dmse.csv economic_pooled; task2_prediction_results.json 0.037342)."""
    pooled_delta_mse = api("predict", "pooled_delta_mse")
    result = pooled_delta_mse(
        lobo("complete_case", "ridge"),
        baseline="ii_meanidx",
        model="iv_kfac",
        targets=tuple(ECONOMIC),
        learner="ridge",
        B=2000,
        seed=42,
    )
    close(result.point, LOBO_POOLED_DMSE, TOL_DMSE, "pooled ΔMSE")


def test_lobo_pooled_delta_mse_bootstrap_interval():
    """Its percentile bootstrap interval: [+0.019, +0.055]
    (h4_bootstrap_dmse.csv economic_pooled ci_lo, ci_hi; B = 2000, default_rng(42)).
    Statistically-reproduced tier unless the archive's draw order is replicated."""
    pooled_delta_mse = api("predict", "pooled_delta_mse")
    result = pooled_delta_mse(
        lobo("complete_case", "ridge"),
        baseline="ii_meanidx",
        model="iv_kfac",
        targets=tuple(ECONOMIC),
        learner="ridge",
        B=2000,
        seed=42,
    )
    lo, hi = result.ci
    close(lo, LOBO_POOLED_DMSE_CI[0], TOL_CI_DMSE, "pooled ΔMSE bootstrap lower")
    close(hi, LOBO_POOLED_DMSE_CI[1], TOL_CI_DMSE, "pooled ΔMSE bootstrap upper")


def test_lobo_pooled_delta_mse_deduplicated():
    """The same on the 89-model deduplicated grid: +0.038
    (h4_bootstrap_dedup.csv deduplicated_base iv_kfac_k3 dMSE 0.0378)."""
    pooled_delta_mse = api("predict", "pooled_delta_mse")
    result = pooled_delta_mse(
        lobo("deduplicated", "ridge"),
        baseline="ii_meanidx",
        model="iv_kfac",
        targets=tuple(ECONOMIC),
        learner="ridge",
        B=2000,
        seed=42,
    )
    close(result.point, LOBO_POOLED_DMSE_DEDUPLICATED, TOL_DMSE, "pooled ΔMSE, deduplicated grid")


@pytest.mark.slow
@pytest.mark.parametrize("rung", sorted(LADDER))
def test_ladder_economic_block(rung):
    """Economic-block ladder rows at the best of the four registered learners
    (lobo_rung_summary.csv, scope economic): single index (ii) RMSE 0.474, R² 0.771; k-factor (iv)
    RMSE 0.433, R² 0.808; first factor alone (iii) RMSE 0.950. The archive notes rung (iii) can
    settle on a slightly different point across platforms (weakly determined leading direction);
    the tolerance is not widened for it here — a miss is reported as a finding."""
    ladder = api("predict", "ladder")
    table = ladder(lobo("complete_case", "registered"), targets=tuple(ECONOMIC))
    for metric, expected in LADDER[rung].items():
        tol = TOL_RMSE if metric.endswith("rmse") else TOL_R2
        close(float(table.loc[rung, metric]), expected, tol, f"ladder {rung} {metric}")
