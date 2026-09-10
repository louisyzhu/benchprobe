"""Golden acceptance tests for the Price of Intelligence item-response fit, ticket T6.

The archive (Zenodo 10.5281/zenodo.22177190, v1.0.0) ships its frozen panel and its Phase-2
outputs but **no estimation code**. ``benchprobe.irt`` is therefore a re-implementation of the
model the archive states it fitted, and these tests are the evidence for that claim: the same
panel, put through benchprobe's estimator, must land on the archive's published item parameters,
abilities, standard errors, objectives and residual scale.

Reproducibility tier: *recomputed*, for these quantities and no others (rule 7,
docs/reproducibility.md). Nothing here is bit-for-bit — a different optimiser implementation
cannot be — so every tolerance needs a reason that is not "what passed". At T6 the reason given
was reporting precision; the T8 review (Codex) pointed out that the archived CSVs carry full
floats, so that reason was wrong. The reason now, from T6.1's Hessian diagnostics:

**The archive's published solution is not at the exact optimum, and it says by how much.** Its
estimation record gives a Newton decrement of 4.8e-8 and a largest remaining ability step of
3.5e-4; its convergence rule accepts any point whose remaining ability step is under 1e-3.
Evaluating benchprobe's objective *at the archive's published point* reproduces those diagnostics
(gradient 5.487e-5, decrement 4.787e-8, ability step 3.480e-4 — all three to the digits the record
prints; ``test_the_archives_own_diagnostics_reproduce_at_its_published_point``), which establishes
that the two implementations are the same function. Two points on the same function, each within
its remaining Newton step of the optimum, differ by at most the sum of those steps. Measured per
block at T8: ability 3.31e-4 against an archive step of 3.48e-4; difficulty 7.60e-5 against
7.54e-5; log-discrimination 2.65e-5 against 2.55e-5; log residual scale 1.3e-8 against 1.1e-8.
benchprobe's own remaining steps are twenty times smaller. The deviations are the archive's
stopping distance, and the tolerances are set from it:

* abilities — 1e-3, the archive's own convergence resolution on the ability block;
* item parameters and standard errors — 2e-4, about 2.5x the archive's remaining step on the
  item blocks (tightened from 5e-4 at T8);
* residual scale — 1e-7, about 15x its remaining step (tightened from 1e-6 at T8);
* objectives — half a unit of the last printed digit of the record (4 dp);
* test information — 5e-3; derived from the discrimination bound through a_k²/σ², loosely, and
  about 6x the observed worst.

Observed worst deviations, for the record and so that drift is visible: objectives 2.3e-5 and
1.3e-5; residual scale 1.8e-9 and 5.9e-9; item difficulty 7.6e-5, discrimination 1.3e-5;
theta 3.3e-4; se_theta 2.7e-5; test information 8.7e-4.

Two findings are locked here rather than smoothed over (rule 3):

1. **The scale repair.** Four benchmarks declare ``score_divisor=100`` over scores that are
   already proportions. ``test_the_unrepaired_panel_does_not_reproduce_the_archive`` fits the
   panel as its metadata literally says and asserts that it *fails*, so the correction can never
   be quietly dropped and be mistaken for agreement.
2. **The archive's two informations.** ``test_information`` in ``se_theta_structure.csv`` sums
   over a model's distinct benchmarks; the ``se_theta`` in the same row sums over its cells.
   ``benchprobe.irt.ability_table`` reproduces both, and says so.

Adjust nothing here except by ticket (rule 1).
"""

from __future__ import annotations

import hashlib
import importlib

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.golden

SNAPSHOT = "price_of_intelligence_2026-08-09"

# The archived files these tests are checked against, so the gate cannot be moved from src/.
# The manifest itself is pinned too (T8): a file and its manifest entry could otherwise be
# changed together and the loader would relabel rather than fail.
MANIFEST_SHA256 = "8eaea7e7faece363da31bda033d98ac26ba6974b3cde0fb4909785a6bd042177"
ARCHIVED_SHA256 = {
    "benchmark_scores.parquet": (
        "8f92c5f43214f8eb11b9c0977ca30d26bb05ac2984c9a38fa571c48651a2c9a8"
    ),
    "archived/item_parameters.csv": (
        "f77cd0b38a3c1bd6ff70e3de140a35526650afcc9bb45965cc5eaea25d17f3c5"
    ),
    "archived/se_theta_structure.csv": (
        "353ac7260be7fd14e3470621620e1f17b741a31f8d37bcc7f609d132e9644ea9"
    ),
    "archived/theta_estimation.json": (
        "d2b1165c8d1f81dcf0956ff8dd8453d14b75718351f8b003a9a65b15e2db6a4c"
    ),
}

# Panel shape (archive, theta_estimation.json "primary")
N_MODELS = 782
N_BENCHMARKS = 64
N_CELLS = 4605
N_SQUEEZED = 89
N_MODELS_WITH_ANCHOR_CELL = 480
N_ROWS_REPAIRED = 221

# Stage 1: anchors only
OBJECTIVE_STAGE1 = -1044.5802
RESIDUAL_SD_STAGE1 = 0.30762771382713755
N_FREE_STAGE1 = 607

# Stage 2: full panel, anchors held fixed
OBJECTIVE_STAGE2 = -1328.6389
RESIDUAL_SD_STAGE2 = 0.44927472556854087
N_FREE_STAGE2 = 891

TOL_OBJECTIVE = 0.00005
TOL_RESIDUAL_SD = 1e-7  # T8: was 1e-6
TOL_ITEM = 2e-4  # T8: was 5e-4
TOL_THETA = 1e-3
TOL_SE = 2e-4  # T8: was 5e-4
TOL_INFORMATION = 5e-3
TOL_GRADIENT = 1e-3  # the archive records 5.5e-5; this is a convergence bound, not a number

# The archive's convergence diagnostics (theta_estimation.json), reproduced by T6.1
HESSIAN_MIN_EIG = {"stage1": 0.039999999999931, "stage2": 0.08614290139459405}
HESSIAN_MAX_EIG = {"stage1": 19746.991769118642, "stage2": 18740.29143449455}
CONDITION_NUMBER = {"stage1": 493674.7942288176, "stage2": 217548.87670489596}
TOL_EIG_RELATIVE = 1e-4
# and the diagnostics the archive recorded *at its own published point*, which benchprobe's
# objective must reproduce when evaluated there
ARCHIVE_POINT_MAX_ABS_GRADIENT = 5.486796555786988e-05
ARCHIVE_POINT_NEWTON_DECREMENT = 4.787302219616923e-08
ARCHIVE_POINT_MAX_THETA_STEP = 0.00034798423554912737
TOL_ARCHIVE_POINT_RELATIVE = 1e-2


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
        f"{label}: archive {expected}, recomputed {actual:.10g}, tolerance {tol}"
    )


def worst(archive: pd.Series, ours: pd.Series, tol: float, label: str) -> None:
    """Largest absolute deviation, with a NaN anywhere counted as failure.

    ``Series.max()`` skips NaN, so without the explicit check a partly-NaN recomputation could
    pass on its finite remainder (T8, Codex review, defect 1).
    """
    ours = ours.reindex(archive.index)
    bad = ours[~np.isfinite(ours.to_numpy(dtype=float))]
    assert bad.empty, (
        f"{label}: {len(bad)} non-finite recomputed value(s), first at {bad.index[0]!r}"
    )
    assert np.isfinite(archive.to_numpy(dtype=float)).all(), (
        f"{label}: archive has non-finite values"
    )
    deviation = (archive - ours).abs()
    row = deviation.idxmax()
    assert deviation.max() <= tol, (
        f"{label}: worst at {row!r} — archive {archive[row]!r}, recomputed {ours[row]!r}, "
        f"deviation {deviation.max():.6g}, tolerance {tol}"
    )


@pytest.fixture(scope="module")
def snapshot():
    return api("io", "load_snapshot")(SNAPSHOT)


@pytest.fixture(scope="module")
def fit(snapshot):
    build_panel = api("irt", "build_panel")
    two_stage_link = api("irt", "two_stage_link")
    spec = api("irt", "estimation_spec")(snapshot)
    return two_stage_link(build_panel(snapshot), spec)


def test_the_archived_files_are_the_ones_these_numbers_came_from(snapshot):
    manifest = hashlib.sha256((snapshot.folder / "MANIFEST.json").read_bytes()).hexdigest()
    assert manifest == MANIFEST_SHA256, f"MANIFEST.json: {manifest}, expected {MANIFEST_SHA256}"
    for name, expected in ARCHIVED_SHA256.items():
        digest = hashlib.sha256(snapshot.file(name).read_bytes()).hexdigest()
        assert digest == expected, f"{name}: {digest}, expected {expected}"


def test_panel_shape(fit):
    panel = fit.panel
    assert panel.M == N_MODELS
    assert panel.K == N_BENCHMARKS
    assert panel.n_cells == N_CELLS
    assert panel.n_squeezed == N_SQUEEZED
    assert panel.n_rows_repaired == N_ROWS_REPAIRED
    assert fit.n_models_with_anchor_cell == N_MODELS_WITH_ANCHOR_CELL


def test_stage1_reproduces_the_anchor_fit(fit):
    close(fit.stage1.objective, OBJECTIVE_STAGE1, TOL_OBJECTIVE, "stage 1 objective")
    close(fit.stage1.residual_sd, RESIDUAL_SD_STAGE1, TOL_RESIDUAL_SD, "stage 1 residual sd")
    assert fit.stage1.n_free_parameters == N_FREE_STAGE1
    assert fit.stage1.max_abs_gradient <= TOL_GRADIENT


def test_stage2_reproduces_the_full_panel_fit(fit):
    close(fit.stage2.objective, OBJECTIVE_STAGE2, TOL_OBJECTIVE, "stage 2 objective")
    close(fit.stage2.residual_sd, RESIDUAL_SD_STAGE2, TOL_RESIDUAL_SD, "stage 2 residual sd")
    assert fit.stage2.n_free_parameters == N_FREE_STAGE2
    assert fit.stage2.max_abs_gradient <= TOL_GRADIENT


def _relative(actual: float, expected: float, tol: float, label: str) -> None:
    assert abs(actual - expected) <= tol * abs(expected), (
        f"{label}: archive {expected!r}, recomputed {actual!r}, relative tolerance {tol}"
    )


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_convergence_on_the_archives_criterion(fit, stage):
    """T6.1: the Hessian spectrum and Newton decrement on the free block, judged by the
    archive's rule (all eigenvalues positive, decrement < 1e-3, ability step < 1e-3) and compared
    with the four diagnostics the archive records for each stage."""
    c = getattr(fit, stage).convergence
    assert c.n_negative_eigenvalues == 0
    _relative(c.min_eigenvalue, HESSIAN_MIN_EIG[stage], TOL_EIG_RELATIVE, f"{stage} min eigenvalue")
    _relative(c.max_eigenvalue, HESSIAN_MAX_EIG[stage], TOL_EIG_RELATIVE, f"{stage} max eigenvalue")
    _relative(c.condition_number, CONDITION_NUMBER[stage], TOL_EIG_RELATIVE, f"{stage} condition")
    assert abs(c.newton_decrement) < 1e-3 and c.max_remaining_theta_step < 1e-3
    assert c.converged, c


def test_the_archives_own_diagnostics_reproduce_at_its_published_point(fit, snapshot):
    """The check that breaks the shared-dependence loop the T8 review pointed at.

    The archive records the gradient, Newton decrement and remaining ability step *at its own
    published solution*. Assemble that solution from the published tables, evaluate benchprobe's
    objective there, and those three numbers must come back. They depend on the objective and its
    curvature, not on any optimiser, so agreement here says the two implementations are the same
    function — independently of whether either optimiser found the optimum.
    """
    irt = importlib.import_module("benchprobe.irt")
    panel = fit.panel
    ip = pd.read_csv(snapshot.file("archived/item_parameters.csv")).set_index("benchmark_id")
    se = pd.read_csv(snapshot.file("archived/se_theta_structure.csv")).set_index("model_id")
    ip = ip.reindex(list(panel.items))
    se = se.reindex(list(panel.models))
    x = irt._packed(
        se["theta"].to_numpy(dtype=float),
        np.log(ip["discrimination"].to_numpy(dtype=float)),
        ip["difficulty"].to_numpy(dtype=float),
        float(np.log(RESIDUAL_SD_STAGE2)),
    )
    priors = fit.stage2.priors
    args = (
        panel.model_of_cell,
        panel.item_of_cell,
        panel.y,
        fit.stage2.free_models,
        fit.stage2.free_items,
        panel.M,
        panel.K,
        priors["theta_prior_sd"] ** 2,
        priors["difficulty_prior_sd"] ** 2,
        priors["log_discrimination_prior_sd"] ** 2,
    )
    free = irt._free_index(fit.stage2.free_models, fit.stage2.free_items, panel.M, panel.K)
    d = irt.convergence_diagnostics(x, args, free, panel.M, lbfgs_success=True)
    _relative(
        d.max_abs_gradient, ARCHIVE_POINT_MAX_ABS_GRADIENT, TOL_ARCHIVE_POINT_RELATIVE, "gradient"
    )
    _relative(
        d.newton_decrement, ARCHIVE_POINT_NEWTON_DECREMENT, TOL_ARCHIVE_POINT_RELATIVE, "decrement"
    )
    _relative(
        d.max_remaining_theta_step, ARCHIVE_POINT_MAX_THETA_STEP, TOL_ARCHIVE_POINT_RELATIVE, "step"
    )
    # The objective at the archive's point is above ours (ours is the lower point) and within the
    # objective tolerance. Observed at T8: 8.8e-6 above — more than the 2.4e-8 its own Newton
    # decrement predicts is available, an observation recorded in docs/decisions.md and not
    # explained; it is not an archive number and nothing is locked on it beyond these bounds.
    f_archive = irt._objective_and_gradient(x, *args)[0]
    assert 0.0 <= f_archive - fit.stage2.objective <= TOL_OBJECTIVE, (
        f"objective at the archive's point {f_archive!r} vs ours {fit.stage2.objective!r}"
    )


def test_item_parameters_match_the_published_table(fit, snapshot):
    archive = pd.read_csv(snapshot.file("archived/item_parameters.csv")).set_index("benchmark_id")
    ours = fit.item_parameters().set_index("benchmark_id")
    assert list(ours.index) == list(archive.index)
    worst(archive["difficulty"], ours["difficulty"], TOL_ITEM, "item difficulty")
    worst(archive["discrimination"], ours["discrimination"], TOL_ITEM, "item discrimination")
    assert archive["is_anchor"].equals(ours["is_anchor"])
    assert archive["is_reference"].equals(ours["is_reference"])
    assert archive["n_cells"].equals(ours["n_cells"])


def test_abilities_and_standard_errors_match_the_published_table(fit, snapshot):
    archive = pd.read_csv(snapshot.file("archived/se_theta_structure.csv")).set_index("model_id")
    ours = api("irt", "ability_table")(fit).set_index("model_id")
    assert set(ours.index) == set(archive.index)
    ours = ours.reindex(archive.index)
    assert archive["n_cells"].equals(ours["n_cells"])
    assert archive["n_benchmarks"].equals(ours["n_benchmarks"])
    worst(archive["theta"], ours["theta"], TOL_THETA, "ability")
    worst(archive["se_theta"], ours["se_theta"], TOL_SE, "se(theta)")
    worst(
        archive["se_predicted_from_information"],
        ours["se_predicted_from_information"],
        TOL_SE,
        "se predicted from information",
    )
    worst(archive["mean_discrimination"], ours["mean_discrimination"], TOL_ITEM, "mean a")
    worst(archive["test_information"], ours["test_information"], TOL_INFORMATION, "information")


def test_ability_ranking_is_preserved(fit, snapshot):
    """Abilities are used as a ranking, and a ranking has no decimal tolerance (T8, Codex).

    Every pair of models the archive orders by more than the ability tolerance must be ordered the
    same way here; pairs closer than that are inside the optimiser's resolution and may legitimately
    swap. The count of such swaps is asserted small and reported.
    """
    archive = pd.read_csv(snapshot.file("archived/se_theta_structure.csv")).set_index("model_id")
    ours = api("irt", "ability_table")(fit).set_index("model_id").reindex(archive.index)
    a = archive["theta"].to_numpy(dtype=float)
    b = ours["theta"].to_numpy(dtype=float)
    order = np.argsort(a)
    a, b = a[order], b[order]
    # pairwise sign agreement, all 782*781/2 pairs
    da = a[None, :] - a[:, None]
    db = b[None, :] - b[:, None]
    upper = np.triu_indices(len(a), k=1)
    da, db = da[upper], db[upper]
    resolvable = np.abs(da) > TOL_THETA
    reversed_resolvable = int(np.sum(resolvable & (np.sign(da) != np.sign(db))))
    reversed_all = int(np.sum(np.sign(da) != np.sign(db)))
    assert reversed_resolvable == 0, (
        f"{reversed_resolvable} model pairs separated by more than {TOL_THETA} in the archive are "
        "ordered the other way by benchprobe"
    )
    assert reversed_all <= 5, f"{reversed_all} pair reversals in total (all inside {TOL_THETA})"
    from scipy.stats import spearmanr

    rho = float(spearmanr(a, b).statistic)
    assert rho >= 0.999999, f"Spearman rho {rho:.8f}"


def test_the_archives_two_informations_really_do_differ(snapshot):
    """Finding, locked so it is not mistaken for a benchprobe defect: the published
    ``test_information`` cannot be the information behind the published ``se_theta``."""
    archive = pd.read_csv(snapshot.file("archived/se_theta_structure.csv"))
    prior_precision = 1.0 / 5.0**2
    implied = 1.0 / np.sqrt(archive["test_information"] + prior_precision)
    deviation = (implied - archive["se_theta"]).abs()
    assert deviation.max() > 0.1, (
        "the archive's se_theta now agrees with its own test_information column; the finding "
        f"this test records has gone away (worst deviation {deviation.max():.6g})"
    )
    single = archive[archive["n_cells"] == archive["n_benchmarks"]]
    assert (
        1.0 / np.sqrt(single["test_information"] + prior_precision) - single["se_theta"]
    ).abs().max() <= 1e-9, (
        "the two informations should coincide exactly where a model has no repeated benchmark"
    )


@pytest.mark.slow
def test_the_unrepaired_panel_does_not_reproduce_the_archive(snapshot):
    """Fit the panel as its recorded ``score_divisor`` literally says, and confirm it misses.

    This is the evidence for the correction ``build_panel`` applies by default. If this test ever
    passes agreement, the repair is no longer needed and T6's premise is wrong — report it, do
    not delete the test (rule 1).
    """
    build_panel = api("irt", "build_panel")
    two_stage_link = api("irt", "two_stage_link")
    spec = api("irt", "estimation_spec")(snapshot)
    unrepaired = two_stage_link(build_panel(snapshot, repair_scale=False), spec)
    archive = pd.read_csv(snapshot.file("archived/item_parameters.csv")).set_index("benchmark_id")
    ours = unrepaired.item_parameters().set_index("benchmark_id")
    deviation = (archive["discrimination"] - ours["discrimination"]).abs()
    assert deviation.max() > 10 * TOL_ITEM, (
        "fitting the panel as its metadata literally says now reproduces the archive's item "
        f"parameters (worst deviation {deviation.max():.6g}); the scale repair is unnecessary"
    )
    assert abs(unrepaired.stage2.residual_sd - RESIDUAL_SD_STAGE2) > 100 * TOL_RESIDUAL_SD
