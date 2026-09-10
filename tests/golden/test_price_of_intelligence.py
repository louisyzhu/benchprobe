"""Golden acceptance tests for the Price of Intelligence item-response fit, ticket T6.

The archive (Zenodo 10.5281/zenodo.22177190, v1.0.0) ships its frozen panel and its Phase-2
outputs but **no estimation code**. ``benchprobe.irt`` is therefore a re-implementation of the
model the archive states it fitted, and these tests are the evidence for that claim: the same
panel, put through benchprobe's estimator, must land on the archive's published item parameters,
abilities, standard errors, objectives and residual scale.

Reproducibility tier: *recomputed*, for these quantities and no others (rule 7,
docs/reproducibility.md). Nothing here is bit-for-bit — a different optimiser implementation
cannot be — so the tolerances are declared from the precision at which each quantity is reported
and used, not from what happens to pass:

* objectives — half a unit of the last printed digit of the archive's record (4 dp);
* residual scale — 1e-6, three orders inside the six figures the papers quote;
* item parameters and standard errors — 5e-4, half a unit of the last figure any table prints;
* abilities — 1e-3, the precision at which theta is reported and ranked;
* test information — 5e-3, likewise.

Observed at T6 on this repository's locked environment, for the record and so that drift is
visible: objectives 2.3e-5 and 1.3e-5; residual scale 1.8e-9 and 5.9e-9; item difficulty 7.6e-5,
discrimination 1.3e-5; theta 3.3e-4; se_theta 2.7e-5; test information 8.7e-4.

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
TOL_RESIDUAL_SD = 1e-6
TOL_ITEM = 5e-4
TOL_THETA = 1e-3
TOL_SE = 5e-4
TOL_INFORMATION = 5e-3
TOL_GRADIENT = 1e-3  # the archive records 5.5e-5; this is the convergence claim, not a number


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
    assert fit.stage1.converged


def test_stage2_reproduces_the_full_panel_fit(fit):
    close(fit.stage2.objective, OBJECTIVE_STAGE2, TOL_OBJECTIVE, "stage 2 objective")
    close(fit.stage2.residual_sd, RESIDUAL_SD_STAGE2, TOL_RESIDUAL_SD, "stage 2 residual sd")
    assert fit.stage2.n_free_parameters == N_FREE_STAGE2
    assert fit.stage2.max_abs_gradient <= TOL_GRADIENT
    assert fit.stage2.converged


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
