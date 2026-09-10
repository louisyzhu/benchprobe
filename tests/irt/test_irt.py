"""Unit tests for benchprobe.irt (ticket T6). Fast, synthetic or metadata-only; part of the smoke
set.

Nothing here fits the real panel — that is ``tests/golden/test_price_of_intelligence.py``, which
takes minutes. These tests check the panel builder, the scale-repair evidence, and that the
estimator recovers parameters it generated itself.
"""

import numpy as np
import pandas as pd
import pytest

import benchprobe.irt as irt
from benchprobe.io import load_snapshot

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(irt.PANEL_SNAPSHOT)


def test_scale_repair_report_flags_exactly_the_recorded_benchmarks(snapshot):
    report = irt.scale_repair_report(snapshot.table)
    flagged = set(report.loc[report["needs_repair"], "benchmark_id"])
    assert flagged == set(irt.SCALE_REPAIR_BENCHMARKS), sorted(
        flagged ^ set(irt.SCALE_REPAIR_BENCHMARKS)
    )
    assert len(report) == 64


def test_the_repair_signature_separates_the_two_groups(snapshot):
    """The evidence, not the conclusion: every flagged score carries the division signature and
    no flagged score exceeds 1; the rest carry it only by coincidence."""
    report = irt.scale_repair_report(snapshot.table).set_index("benchmark_id")
    flagged = report.loc[list(irt.SCALE_REPAIR_BENCHMARKS)]
    assert (flagged["share_on_two_decimal_percent_grid"] == 1.0).all()
    assert (flagged["max_score"] <= 1.0).all()
    assert (flagged["declared_divisor"] == 100.0).all()
    others = report.drop(index=list(irt.SCALE_REPAIR_BENCHMARKS))
    assert (others["declared_divisor"] == 1.0).all()


def test_panel_matches_the_archives_own_cell_counts(snapshot):
    spec = irt.estimation_spec(snapshot)
    panel = irt.build_panel(snapshot)
    assert panel.n_cells == spec["n_cells"]
    assert panel.M == spec["n_models"]
    assert panel.K == spec["n_benchmarks"]
    assert panel.n_squeezed == spec["n_cells_squeezed_from_boundary"]
    assert panel.n_rows_repaired == 221
    assert panel.repaired == tuple(sorted(irt.SCALE_REPAIR_BENCHMARKS))


def test_repair_can_be_switched_off_and_is_recorded(snapshot):
    off = irt.build_panel(snapshot, repair_scale=False)
    assert off.repaired == () and off.n_rows_repaired == 0
    on = irt.build_panel(snapshot)
    assert on.provenance["scale_repaired"] == list(on.repaired)
    assert on.provenance["rows_repaired"] == 221
    # the repaired cells are the only ones that move, and they move by a factor of 100
    changed = np.flatnonzero(
        on.frame["proportion"].to_numpy() != off.frame["proportion"].to_numpy()
    )
    assert len(changed) == 221


def test_a_named_benchmark_without_evidence_is_not_repaired_silently(snapshot, monkeypatch):
    monkeypatch.setattr(
        irt, "SCALE_REPAIR_BENCHMARKS", irt.SCALE_REPAIR_BENCHMARKS + ("mmlu_external",)
    )
    with pytest.raises(ValueError, match="no longer supports the recorded scale repair"):
        irt.build_panel(snapshot)


def test_scale_repair_report_rejects_a_frame_without_the_metadata():
    with pytest.raises(KeyError):
        irt.scale_repair_report(pd.DataFrame({"benchmark_id": ["a"], "score": [0.5]}))


def _synthetic_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    M, K = 60, 6
    theta = rng.standard_normal(M)
    difficulty = np.linspace(-1.0, 1.0, K)
    discrimination = np.linspace(0.8, 1.6, K)
    difficulty[0] = 0.0
    discrimination[0] = 1.0
    rows = []
    for m in range(M):
        for k in range(K):
            y = discrimination[k] * (theta[m] - difficulty[k]) + rng.normal(0, 0.2)
            rows.append((f"m{m}", f"b{k}", float(1 / (1 + np.exp(-y)))))
    frame = pd.DataFrame(rows, columns=["model_id", "benchmark_id", "score"])
    frame["score_scale"] = "proportion"
    frame["score_divisor"] = 1.0
    return frame, theta, difficulty, discrimination


def test_fit_crm_recovers_parameters_it_generated():
    frame, theta, difficulty, discrimination = _synthetic_panel()
    panel = irt.Panel(
        frame=frame,
        models=tuple(sorted(frame.model_id.unique())),
        items=tuple(sorted(frame.benchmark_id.unique())),
        model_of_cell=frame.model_id.str[1:].astype(int).to_numpy(),
        item_of_cell=frame.benchmark_id.str[1:].astype(int).to_numpy(),
        y=np.log(frame.score / (1 - frame.score)).to_numpy(),
        epsilon=0.001,
        n_squeezed=0,
        repaired=(),
        n_rows_repaired=0,
    )
    free_items = np.ones(panel.K, dtype=bool)
    free_items[0] = False
    fit = irt.fit_crm(
        panel, free_models=np.ones(panel.M, dtype=bool), free_items=free_items, adam_steps=800
    )
    assert fit.converged
    assert fit.max_abs_gradient < 1e-4
    assert np.max(np.abs(fit.difficulty - difficulty)) < 0.15
    assert np.max(np.abs(fit.discrimination - discrimination)) < 0.15
    assert np.corrcoef(fit.theta, theta)[0, 1] > 0.99
    assert 0.15 < fit.residual_sd < 0.3
    assert fit.n_free_parameters == panel.M + 2 * int(free_items.sum()) + 1


def test_analytic_gradient_matches_central_differences():
    """The gradient is hand-derived; this is the check the T8 review asked for."""
    frame, *_ = _synthetic_panel(2)
    m_of = frame.model_id.str[1:].astype(int).to_numpy()
    k_of = frame.benchmark_id.str[1:].astype(int).to_numpy()
    y = np.log(frame.score / (1 - frame.score)).to_numpy()
    M, K = 60, 6
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.5, M + 2 * K + 1)
    free_models = np.ones(M, dtype=bool)
    free_items = np.ones(K, dtype=bool)
    free_items[0] = False
    args = (m_of, k_of, y, free_models, free_items, M, K, 25.0, 25.0, 2.25)
    _, g = irt._objective_and_gradient(x, *args)
    h = 1e-6
    fd = np.empty_like(x)
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        fd[j] = (
            irt._objective_and_gradient(xp, *args)[0] - irt._objective_and_gradient(xm, *args)[0]
        ) / (2 * h)
    # fixed coordinates: the analytic gradient is zeroed by design; the objective's own
    # derivative there is the prior term, which is what the finite difference sees
    fixed = np.zeros_like(x, dtype=bool)
    fixed[M] = True  # log a of item 0
    fixed[M + K] = True  # difficulty of item 0
    assert np.max(np.abs(g[~fixed] - fd[~fixed])) < 1e-5 * (1 + np.max(np.abs(fd[~fixed])))
    assert np.all(g[fixed] == 0.0)


def test_convergence_is_judged_on_the_hessian_not_the_optimiser_flag():
    frame, *_ = _synthetic_panel(4)
    panel = irt.Panel(
        frame=frame,
        models=tuple(sorted(frame.model_id.unique())),
        items=tuple(sorted(frame.benchmark_id.unique())),
        model_of_cell=frame.model_id.str[1:].astype(int).to_numpy(),
        item_of_cell=frame.benchmark_id.str[1:].astype(int).to_numpy(),
        y=np.log(frame.score / (1 - frame.score)).to_numpy(),
        epsilon=0.001,
        n_squeezed=0,
        repaired=(),
        n_rows_repaired=0,
    )
    free_items = np.ones(panel.K, dtype=bool)
    free_items[0] = False
    fit = irt.fit_crm(
        panel, free_models=np.ones(panel.M, dtype=bool), free_items=free_items, adam_steps=800
    )
    c = fit.convergence
    assert c.n_negative_eigenvalues == 0
    assert c.min_eigenvalue > 0
    assert abs(c.newton_decrement) < 1e-3
    assert c.max_remaining_theta_step < 1e-3
    assert c.converged and fit.converged
    assert isinstance(c.lbfgs_success, bool)
    assert fit.priors == {
        "theta_prior_sd": 5.0,
        "difficulty_prior_sd": 5.0,
        "log_discrimination_prior_sd": 1.5,
    }
    # a point that is not a minimum must not be called converged
    x_bad = np.zeros(panel.M + 2 * panel.K + 1)
    args = (
        panel.model_of_cell,
        panel.item_of_cell,
        panel.y,
        np.ones(panel.M, dtype=bool),
        free_items,
        panel.M,
        panel.K,
        25.0,
        25.0,
        2.25,
    )
    free = irt._free_index(np.ones(panel.M, dtype=bool), free_items, panel.M, panel.K)
    bad = irt.convergence_diagnostics(x_bad, args, free, panel.M, lbfgs_success=True)
    assert not bad.converged and bad.lbfgs_success


def test_build_panel_refuses_a_score_outside_the_unit_interval(snapshot):
    table = snapshot.table.copy()
    i = table.index[table["score"].notna()][0]
    table.loc[i, "score"] = 1.5
    fake = type(snapshot)(**{**snapshot.__dict__, "table": table})
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        irt.build_panel(fake)


def test_a_penalty_on_a_fixed_parameter_does_not_enter_the_objective():
    """The stage-2 objective excludes priors on held-fixed coordinates; that is what makes it
    agree with the archive's (docs/decisions.md, 2026-09-10)."""
    frame, *_ = _synthetic_panel(1)
    m_of = frame.model_id.str[1:].astype(int).to_numpy()
    k_of = frame.benchmark_id.str[1:].astype(int).to_numpy()
    y = np.log(frame.score / (1 - frame.score)).to_numpy()
    M, K = 60, 6
    x = np.concatenate([np.full(M, 0.3), np.full(K, 0.2), np.full(K, 0.4), [0.0]])
    free_models = np.ones(M, dtype=bool)
    all_free = np.ones(K, dtype=bool)
    some_fixed = all_free.copy()
    some_fixed[:2] = False
    args = (m_of, k_of, y, free_models, None, M, K, 25.0, 25.0, 2.25)
    full = irt._objective_and_gradient(x, *args[:4], all_free, *args[5:])[0]
    part = irt._objective_and_gradient(x, *args[:4], some_fixed, *args[5:])[0]
    dropped = 2 * (0.5 * 0.2**2 / 2.25 + 0.5 * 0.4**2 / 25.0)
    assert full - part == pytest.approx(dropped, abs=1e-12)


def test_estimation_spec_rejects_an_unknown_section(snapshot):
    with pytest.raises(KeyError):
        irt.estimation_spec(snapshot, key="not_a_section")
