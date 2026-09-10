"""Item-response estimation for the benchmark-score panel (sits under the measure layer).

SPEC.md §1 scope, ticket T6. The model is the one the Price of Intelligence archive states it
fitted — **Samejima's continuous response model on logit scores, homoscedastic** — not the
two-parameter or graded-response models the T0 skeleton guessed at; SPEC.md was corrected at T6
to say so.

The archive (Zenodo 10.5281/zenodo.22177190, v1.0.0) ships its frozen panel and its Phase-2
outputs but **no estimation code**. Everything here is therefore a re-implementation written from
the archive's stated model, optimiser and penalties, and validated against its published numbers;
``docs/reproducibility.md`` records it at the *recomputed* tier only for the quantities
``tests/golden/test_price_of_intelligence.py`` locks.

Parameterisation, following the archive's estimation record
(``archived/theta_estimation.json``):

* ability ``theta_m`` for each model, difficulty ``d_k`` and log-discrimination ``log a_k`` for
  each benchmark, and one log residual scale ``log sigma`` shared by all cells;
* observed proportion ``p`` squeezed off the boundary by ``epsilon`` and mapped by ``logit``;
* penalised negative log-likelihood, with Gaussian penalties of the recorded prior scales;
* the reference benchmark's difficulty and log-discrimination held at zero to fix location and
  scale, and a two-stage fixed-parameter anchor link: stage 1 fits abilities and item parameters
  on the anchor cells only, stage 2 refits the whole panel with the anchors held fixed;
* optimiser Adam (3000 steps, lr 0.05) then L-BFGS, exactly as recorded. Adam starts from zeros,
  so no random number is drawn and the recorded ``seed`` does not enter this estimator; nothing
  here is stochastic.

**The scale repair.** Four benchmarks in the frozen panel carry ``score_scale="percentage"`` and
``score_divisor=100`` while their stored scores are already proportions. Dividing by the recorded
divisor a second time reproduces neither the archive's item parameters nor its residual scale;
not dividing reproduces both. :func:`scale_repair_report` is the evidence, computed from the
panel rather than asserted, and :func:`build_panel` applies the repair by default and records it.
See ``docs/decisions.md``, 2026-09-10, and rule 3: this moves no published number — it is what
makes the published numbers reproduce.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit

from benchprobe import __version__
from benchprobe.io import Snapshot, load_snapshot

__all__ = [
    "PANEL_SNAPSHOT",
    "SCALE_REPAIR_BENCHMARKS",
    "Convergence",
    "CrmFit",
    "Panel",
    "TwoStageFit",
    "ability_table",
    "build_panel",
    "convergence_diagnostics",
    "estimation_spec",
    "fit_crm",
    "fit_single_stage",
    "panel_from_long",
    "scale_repair_report",
    "two_stage_link",
]

PANEL_SNAPSHOT = "price_of_intelligence_2026-08-09"

SCALE_REPAIR_BENCHMARKS: tuple[str, ...] = (
    "aider_polyglot_external",
    "forecastbench_external",
    "live_bench_external",
    "os_world_external",
)
"""Benchmarks whose recorded ``score_divisor`` describes the source page, not the stored value.

Named rather than detected so that the set is reviewable and cannot silently grow; the detection
that identified them is :func:`scale_repair_report`, which every run can re-check.
"""


# --------------------------------------------------------------------------------------------
# The panel, and the scale repair
# --------------------------------------------------------------------------------------------


def estimation_spec(snapshot: Snapshot | None = None, key: str = "primary") -> dict[str, Any]:
    """The archive's own estimation record (``archived/theta_estimation.json``), section ``key``."""
    snap = snapshot if snapshot is not None else load_snapshot(PANEL_SNAPSHOT)
    record = json.loads(snap.file("archived/theta_estimation.json").read_text(encoding="utf-8"))
    if key not in record:
        raise KeyError(f"{key!r} is not a section of the archive's estimation record")
    return record[key]


def scale_repair_report(table: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark evidence on whether a recorded ``score_divisor`` has already been applied.

    For each benchmark the report gives the declared scale and divisor, the largest stored score,
    and ``share_on_two_decimal_percent_grid``: the fraction of stored scores that are *bit-for-bit*
    equal to ``round(score * 100, 2) / 100``.

    What that test proves, and what it does not (T8, Codex review): it is a test of **grid
    membership** — that the stored value is the double nearest to some k/10000 — not of the
    value's history. A native ratio of counts lands on that grid whenever its denominator divides
    10000, and about 60 % of the panel's other cells do. The evidence for the repair is therefore
    not any single cell but the joint pattern: all 221 cells of the four declared-percentage
    benchmarks on the grid, against a 60 % background rate (a chance of order 1e-50), together with
    no score above 1 and the published fit reproducing only when the divisor is not applied. The
    name of the column says what is measured; ``needs_repair`` is true where the divisor is not 1,
    no score exceeds 1, and every score lies on the grid.

    The report is descriptive. Nothing in it changes a number; :func:`build_panel` decides.
    """
    required = {"benchmark_id", "score", "score_scale", "score_divisor"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"panel is missing {sorted(missing)}")
    scored = table[table["score"].notna()]
    rows = []
    for benchmark, part in scored.groupby("benchmark_id", sort=True):
        values = part["score"].to_numpy(dtype=float)
        divisor = part["score_divisor"].dropna().unique()
        divisor_value = float(divisor[0]) if len(divisor) == 1 else float("nan")
        signature = float(np.mean(values == np.round(values * 100.0, 2) / 100.0))
        declared = sorted(set(part["score_scale"].dropna().astype(str)))
        rows.append(
            {
                "benchmark_id": benchmark,
                "n_scored": int(len(values)),
                "declared_scale": declared[0] if len(declared) == 1 else "|".join(declared),
                "declared_divisor": divisor_value,
                "max_score": float(np.max(values)),
                "share_on_two_decimal_percent_grid": signature,
                "needs_repair": bool(
                    divisor_value not in (1.0,)
                    and np.isfinite(divisor_value)
                    and np.max(values) <= 1.0
                    and signature == 1.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark_id", ignore_index=True)


@dataclass(frozen=True, eq=False)
class Panel:
    """A scored, logit-transformed panel ready for :func:`fit_crm`."""

    frame: pd.DataFrame
    models: tuple[str, ...]
    items: tuple[str, ...]
    model_of_cell: np.ndarray
    item_of_cell: np.ndarray
    y: np.ndarray
    epsilon: float
    n_squeezed: int
    repaired: tuple[str, ...]
    n_rows_repaired: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(len(self.y))

    @property
    def M(self) -> int:
        return len(self.models)

    @property
    def K(self) -> int:
        return len(self.items)

    def index_of_item(self, name: str) -> int:
        try:
            return self.items.index(name)
        except ValueError as exc:  # pragma: no cover - message is the point
            raise KeyError(f"{name!r} is not a benchmark of this panel") from exc


def build_panel(
    snapshot: Snapshot | None = None,
    *,
    epsilon: float | None = None,
    repair_scale: bool = True,
) -> Panel:
    """Scored cells of the frozen panel, divided by their scale, squeezed and logit-transformed.

    ``repair_scale`` (default true) suspends the recorded divisor for the benchmarks of
    :data:`SCALE_REPAIR_BENCHMARKS`, but only for those the panel's own evidence still flags
    (:func:`scale_repair_report`); a named benchmark whose evidence no longer supports the repair
    raises rather than being repaired silently. Pass ``repair_scale=False`` to reproduce what the
    recorded metadata says, which does *not* reproduce the archive's published fit.
    """
    snap = snapshot if snapshot is not None else load_snapshot(PANEL_SNAPSHOT)
    spec = estimation_spec(snap)
    eps = float(spec["epsilon_squeeze"]) if epsilon is None else float(epsilon)

    frame = snap.table
    frame = frame[frame["score"].notna()].copy()
    divisor = frame["score_divisor"].astype(float)

    repaired: tuple[str, ...] = ()
    n_rows_repaired = 0
    if repair_scale:
        report = scale_repair_report(snap.table).set_index("benchmark_id")
        unsupported = [
            b
            for b in SCALE_REPAIR_BENCHMARKS
            if b not in report.index or not bool(report.loc[b, "needs_repair"])
        ]
        if unsupported:
            raise ValueError(
                "the panel no longer supports the recorded scale repair for "
                f"{unsupported}; re-derive it before fitting (docs/decisions.md, 2026-09-10)"
            )
        mask = frame["benchmark_id"].isin(SCALE_REPAIR_BENCHMARKS)
        divisor = divisor.where(~mask, 1.0)
        repaired = tuple(sorted(SCALE_REPAIR_BENCHMARKS))
        n_rows_repaired = int(mask.sum())

    proportion = frame["score"].to_numpy(dtype=float) / divisor.to_numpy(dtype=float)
    outside = ~((proportion >= 0.0) & (proportion <= 1.0))
    if outside.any():
        bad = frame.loc[outside, ["model_id", "benchmark_id", "score", "score_divisor"]].head(5)
        raise ValueError(
            f"{int(outside.sum())} cell(s) fall outside [0, 1] after dividing by the recorded "
            f"scale; the panel is not the archive's, or a divisor is wrong. First rows:\n"
            f"{bad.to_string(index=False)}"
        )
    # The archive reports 89 cells "squeezed from the boundary" and its estimation note says they
    # "sat at exactly 0 or 1" (docs/phase2-1-estimation.md §5). Counted on that definition the
    # panel has 88: one cell (glm-5.2_unknown on gbaeval_external, score 1/5807 = 0.000172) lies
    # inside epsilon without being on the boundary, and the archive's own count includes it. So the
    # archive counted the cells its clip moved, and the prose is off by one. benchprobe reproduces
    # the recorded count, on the rule that reproduces it (T8, Codex review; docs/decisions.md).
    n_squeezed = int(np.sum((proportion <= eps) | (proportion >= 1.0 - eps)))
    proportion = np.clip(proportion, eps, 1.0 - eps)

    models = tuple(sorted(frame["model_id"].unique()))
    items = tuple(sorted(frame["benchmark_id"].unique()))
    model_index = {m: i for i, m in enumerate(models)}
    item_index = {b: i for i, b in enumerate(items)}
    frame = frame.assign(
        proportion=proportion,
        y=logit(proportion),
        _m=frame["model_id"].map(model_index).to_numpy(dtype=int),
        _k=frame["benchmark_id"].map(item_index).to_numpy(dtype=int),
    )
    return Panel(
        frame=frame,
        models=models,
        items=items,
        model_of_cell=frame["_m"].to_numpy(dtype=int),
        item_of_cell=frame["_k"].to_numpy(dtype=int),
        y=frame["y"].to_numpy(dtype=float),
        epsilon=eps,
        n_squeezed=n_squeezed,
        repaired=repaired,
        n_rows_repaired=n_rows_repaired,
        provenance={
            **snap.provenance(),
            "scale_repaired": list(repaired),
            "rows_repaired": n_rows_repaired,
            "epsilon_squeeze": eps,
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "benchprobe": __version__,
        },
    )


def panel_from_long(
    frame: pd.DataFrame,
    *,
    model: str = "model",
    item: str = "item",
    score: str = "score",
    epsilon: float = 1e-3,
) -> Panel:
    """A :class:`Panel` from your own long data (one row per model × item cell, scores in [0, 1]).

    The door for data that is not the vendored archive: no snapshot, no scale metadata, no repair.
    Scores at exactly 0 or 1 are squeezed inward by ``epsilon`` so the logit is defined, and the
    count is recorded in ``n_squeezed``. Use :class:`benchprobe.scores.ScoreMatrix` to get here from
    wide data (``ScoreMatrix.long()``).
    """
    missing = {model, item, score} - set(frame.columns)
    if missing:
        raise KeyError(f"long frame is missing columns {sorted(missing)}")
    f = frame[[model, item, score]].rename(
        columns={model: "model_id", item: "benchmark_id", score: "score"}
    )
    f = f[f["score"].notna()].copy()
    p = f["score"].to_numpy(dtype=float)
    if p.size == 0:
        raise ValueError("no scored cells")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(f"scores must lie in [0, 1] (got {p.min():.4g} to {p.max():.4g})")
    eps = float(epsilon)
    n_squeezed = int(np.sum((p <= eps) | (p >= 1.0 - eps)))
    p = np.clip(p, eps, 1.0 - eps)
    models = tuple(sorted(f["model_id"].unique()))
    items = tuple(sorted(f["benchmark_id"].unique()))
    mi = {m: i for i, m in enumerate(models)}
    ki = {b: i for i, b in enumerate(items)}
    f = f.assign(
        proportion=p,
        y=logit(p),
        _m=f["model_id"].map(mi).to_numpy(dtype=int),
        _k=f["benchmark_id"].map(ki).to_numpy(dtype=int),
    )
    return Panel(
        frame=f,
        models=models,
        items=items,
        model_of_cell=f["_m"].to_numpy(dtype=int),
        item_of_cell=f["_k"].to_numpy(dtype=int),
        y=f["y"].to_numpy(dtype=float),
        epsilon=eps,
        n_squeezed=n_squeezed,
        repaired=(),
        n_rows_repaired=0,
        provenance={
            "source": "user data (panel_from_long)",
            "epsilon_squeeze": eps,
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "benchprobe": __version__,
        },
    )


# --------------------------------------------------------------------------------------------
# The continuous response model
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Convergence:
    """The archive's convergence criterion, evaluated on the free parameter block (T6.1).

    Judged on the Hessian spectrum and the Newton decrement rather than on a raw gradient, for
    the reason the archive gives (``docs/phase2-1-estimation.md`` §4): curvature differs by orders
    of magnitude across parameter blocks, so an absolute gradient bound is slack on one block and
    punishing on another, and an earlier specification of theirs reported a small gradient at a
    saddle point. ``converged`` is true when every eigenvalue on the free block is positive, the
    Newton decrement ``gᵀH⁻¹g`` is below 1e-3, and the largest remaining Newton step on any ability
    is below 1e-3 — the archive's stated rule, verbatim.
    """

    min_eigenvalue: float
    max_eigenvalue: float
    n_negative_eigenvalues: int
    condition_number: float
    newton_decrement: float
    predicted_objective_improvement: float
    max_remaining_theta_step: float
    max_abs_gradient: float
    converged: bool
    lbfgs_success: bool

    criterion: str = (
        "all Hessian eigenvalues positive on the free block AND |Newton decrement| < 0.001 AND "
        "largest remaining theta step < 0.001"
    )


@dataclass(frozen=True, eq=False)
class CrmFit:
    """One penalised maximum-a-posteriori fit of the continuous response model.

    ``converged`` is the archive's criterion (:class:`Convergence`), not the optimiser's own
    success flag; the flag is kept as ``convergence.lbfgs_success``. ``priors`` are the penalty
    scales this fit was made with, so anything derived from it (standard errors, in particular)
    uses the same prior rather than a global default (T8, Codex review, defect 9).
    """

    theta: np.ndarray
    difficulty: np.ndarray
    discrimination: np.ndarray
    log_sigma: float
    objective: float
    initial_objective: float
    after_adam_objective: float
    max_abs_gradient: float
    n_free_parameters: int
    n_lbfgs_closure_evaluations: int
    convergence: Convergence
    priors: dict[str, float]
    models: tuple[str, ...]
    items: tuple[str, ...]
    free_models: np.ndarray
    free_items: np.ndarray

    @property
    def converged(self) -> bool:
        return self.convergence.converged

    @property
    def residual_sd(self) -> float:
        return float(np.exp(self.log_sigma))

    def item_parameters(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "benchmark_id": list(self.items),
                "difficulty": self.difficulty,
                "discrimination": self.discrimination,
            }
        )


def _packed(theta: np.ndarray, log_a: np.ndarray, difficulty: np.ndarray, log_sigma: float):
    return np.concatenate([theta, log_a, difficulty, [log_sigma]])


def _objective_and_gradient(
    x: np.ndarray,
    m: np.ndarray,
    k: np.ndarray,
    y: np.ndarray,
    free_models: np.ndarray,
    free_items: np.ndarray,
    M: int,
    K: int,
    theta_var: float,
    difficulty_var: float,
    log_discrimination_var: float,
) -> tuple[float, np.ndarray]:
    """Penalised negative log-likelihood and its analytic gradient.

    ``x`` is ``[theta (M), log a (K), difficulty (K), log sigma]``. Fixed coordinates keep their
    value by having their gradient zeroed, which is what "fixed-parameter linking" means here.
    """
    theta = x[:M]
    log_a = x[M : M + K]
    difficulty = x[M + K : M + 2 * K]
    log_sigma = x[-1]

    a = np.exp(log_a)
    inv_var = np.exp(-2.0 * log_sigma)
    residual = y - a[k] * (theta[m] - difficulty[k])
    n = len(y)
    weighted = residual * inv_var

    # The penalty is carried by the free coordinates only. A Gaussian penalty on a parameter that
    # is held fixed is an additive constant: it cannot move the optimum, and counting it would
    # make the reported objective depend on which parameters happen to be fixed at this stage.
    # Excluding it is what reproduces the archive's stage-2 objective; including it leaves ours
    # exactly 1.8839 higher, the penalty carried by the nine fixed non-reference anchors
    # (docs/decisions.md, 2026-09-10).
    value = (
        n * log_sigma
        + 0.5 * float(np.sum(residual * residual)) * inv_var
        + 0.5 * float(np.sum(theta[free_models] ** 2)) / theta_var
        + 0.5 * float(np.sum(difficulty[free_items] ** 2)) / difficulty_var
        + 0.5 * float(np.sum(log_a[free_items] ** 2)) / log_discrimination_var
    )

    g_theta = np.bincount(m, weights=-weighted * a[k], minlength=M) + theta / theta_var
    g_difficulty = (
        np.bincount(k, weights=weighted * a[k], minlength=K) + difficulty / difficulty_var
    )
    g_log_a = (
        np.bincount(k, weights=-weighted * a[k] * (theta[m] - difficulty[k]), minlength=K)
        + log_a / log_discrimination_var
    )
    g_theta[~free_models] = 0.0
    g_difficulty[~free_items] = 0.0
    g_log_a[~free_items] = 0.0
    g_log_sigma = n - float(np.sum(residual * residual)) * inv_var
    return value, np.concatenate([g_theta, g_log_a, g_difficulty, [g_log_sigma]])


def _adam(x: np.ndarray, args: tuple, steps: int, lr: float) -> np.ndarray:
    """The archive's stated first stage: Adam, 3000 steps, lr 0.05, from the given start."""
    first = np.zeros_like(x)
    second = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        _, gradient = _objective_and_gradient(x, *args)
        first = b1 * first + (1.0 - b1) * gradient
        second = b2 * second + (1.0 - b2) * gradient * gradient
        x = x - lr * (first / (1.0 - b1**t)) / (np.sqrt(second / (1.0 - b2**t)) + eps)
    return x


def _free_index(free_models: np.ndarray, free_items: np.ndarray, M: int, K: int) -> np.ndarray:
    """Positions in the packed vector of the coordinates a fit may move."""
    theta = np.flatnonzero(free_models)
    log_a = M + np.flatnonzero(free_items)
    difficulty = M + K + np.flatnonzero(free_items)
    return np.concatenate([theta, log_a, difficulty, [M + 2 * K]])


def convergence_diagnostics(
    x: np.ndarray, args: tuple, free: np.ndarray, M: int, *, lbfgs_success: bool, step: float = 1e-5
) -> Convergence:
    """Hessian spectrum and Newton decrement on the free block, by central differences of the
    analytic gradient.

    The gradient is exact, so the Hessian error is O(step²) — far below the scale of anything
    judged here (eigenvalues of order 1e-2 and up; a decrement threshold of 1e-3). The free block
    is at most a few thousand coordinates, so the dense matrix is cheap.
    """
    n = len(free)
    H = np.empty((n, n))
    for j, idx in enumerate(free):
        xp = x.copy()
        xp[idx] += step
        xm = x.copy()
        xm[idx] -= step
        gp = _objective_and_gradient(xp, *args)[1][free]
        gm = _objective_and_gradient(xm, *args)[1][free]
        H[:, j] = (gp - gm) / (2.0 * step)
    H = 0.5 * (H + H.T)
    g = _objective_and_gradient(x, *args)[1]
    g_free = g[free]
    eig = np.linalg.eigvalsh(H)
    n_negative = int(np.sum(eig <= 0.0))
    if n_negative == 0:
        newton_step = np.linalg.solve(H, g_free)
        decrement = float(g_free @ newton_step)
    else:  # not a local minimum; the decrement is undefined as a distance to one
        newton_step = np.full(n, np.nan)
        decrement = float("nan")
    theta_positions = free < M
    max_theta_step = (
        float(np.max(np.abs(newton_step[theta_positions]))) if theta_positions.any() else 0.0
    )
    converged = bool(
        n_negative == 0
        and abs(decrement) < 1e-3
        and max_theta_step < 1e-3
        and np.isfinite(decrement)
    )
    return Convergence(
        min_eigenvalue=float(eig.min()),
        max_eigenvalue=float(eig.max()),
        n_negative_eigenvalues=n_negative,
        condition_number=float(eig.max() / eig.min()) if eig.min() > 0 else float("inf"),
        newton_decrement=decrement,
        predicted_objective_improvement=0.5 * decrement,
        max_remaining_theta_step=max_theta_step,
        max_abs_gradient=float(np.max(np.abs(g))),
        converged=converged,
        lbfgs_success=bool(lbfgs_success),
    )


def fit_crm(
    panel: Panel,
    *,
    free_models: np.ndarray,
    free_items: np.ndarray,
    cells: np.ndarray | None = None,
    start: np.ndarray | None = None,
    priors: dict[str, float] | None = None,
    adam_steps: int = 3000,
    lr: float = 0.05,
    maxiter: int = 50_000,
) -> CrmFit:
    """Fit the continuous response model by Adam then L-BFGS, as the archive records.

    ``free_models`` and ``free_items`` are boolean masks; everything else keeps its starting
    value. ``cells`` restricts the likelihood to a subset of the panel's rows (stage 1 uses the
    anchor cells). ``priors`` defaults to the archive's recorded ``penalty_scales``.
    """
    scales = priors or {
        "theta_prior_sd": 5.0,
        "difficulty_prior_sd": 5.0,
        "log_discrimination_prior_sd": 1.5,
    }
    m = panel.model_of_cell if cells is None else panel.model_of_cell[cells]
    k = panel.item_of_cell if cells is None else panel.item_of_cell[cells]
    y = panel.y if cells is None else panel.y[cells]
    args = (
        m,
        k,
        y,
        np.asarray(free_models, dtype=bool),
        np.asarray(free_items, dtype=bool),
        panel.M,
        panel.K,
        float(scales["theta_prior_sd"]) ** 2,
        float(scales["difficulty_prior_sd"]) ** 2,
        float(scales["log_discrimination_prior_sd"]) ** 2,
    )
    x0 = np.zeros(panel.M + 2 * panel.K + 1) if start is None else np.asarray(start, float).copy()
    initial_objective = float(_objective_and_gradient(x0, *args)[0])
    after_adam = _adam(x0, args, adam_steps, lr)
    after_adam_objective = float(_objective_and_gradient(after_adam, *args)[0])
    result = minimize(
        lambda z: _objective_and_gradient(z, *args),
        after_adam,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-12},
    )
    _, gradient = _objective_and_gradient(result.x, *args)
    n_free = int(np.sum(free_models)) + 2 * int(np.sum(free_items)) + 1
    free = _free_index(args[3], args[4], panel.M, panel.K)
    diagnostics = convergence_diagnostics(
        result.x, args, free, panel.M, lbfgs_success=bool(result.success)
    )
    return CrmFit(
        theta=result.x[: panel.M].copy(),
        difficulty=result.x[panel.M + panel.K : panel.M + 2 * panel.K].copy(),
        discrimination=np.exp(result.x[panel.M : panel.M + panel.K]),
        log_sigma=float(result.x[-1]),
        objective=float(result.fun),
        initial_objective=initial_objective,
        after_adam_objective=after_adam_objective,
        max_abs_gradient=float(np.max(np.abs(gradient))),
        n_free_parameters=n_free,
        n_lbfgs_closure_evaluations=int(result.nfev),
        convergence=diagnostics,
        priors={k: float(v) for k, v in scales.items()},
        models=panel.models,
        items=panel.items,
        free_models=np.asarray(free_models, dtype=bool).copy(),
        free_items=np.asarray(free_items, dtype=bool).copy(),
    )


def fit_single_stage(
    panel: Panel,
    *,
    reference_item: str | None = None,
    priors: dict[str, float] | None = None,
    adam_steps: int = 3000,
    lr: float = 0.05,
) -> CrmFit:
    """One fit of the whole panel with every ability and item free except the reference item,
    whose difficulty and log-discrimination are held at zero to fix the scale.

    This is the model of :func:`two_stage_link` without the anchor design, for data that has no
    ratified anchor set. ``reference_item`` defaults to the item scored on the most models, the
    archive's own rule for choosing one.
    """
    if reference_item is None:
        counts = np.bincount(panel.item_of_cell, minlength=panel.K)
        reference_item = panel.items[int(np.argmax(counts))]
    ref = panel.index_of_item(reference_item)
    free_items = np.ones(panel.K, dtype=bool)
    free_items[ref] = False
    return fit_crm(
        panel,
        free_models=np.ones(panel.M, dtype=bool),
        free_items=free_items,
        priors=priors,
        adam_steps=adam_steps,
        lr=lr,
    )


# --------------------------------------------------------------------------------------------
# Two-stage fixed-parameter anchor linking
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TwoStageFit:
    """The archive's linking design: anchors first, then the panel with the anchors held fixed."""

    stage1: CrmFit
    stage2: CrmFit
    panel: Panel
    anchors: tuple[str, ...]
    reference_item: str
    n_models_with_anchor_cell: int

    def item_parameters(self) -> pd.DataFrame:
        """Difficulty and discrimination per benchmark, with the anchor and reference flags."""
        table = self.stage2.item_parameters()
        table["is_anchor"] = table["benchmark_id"].isin(self.anchors)
        table["is_reference"] = table["benchmark_id"] == self.reference_item
        counts = self.panel.frame["benchmark_id"].value_counts()
        table["n_cells"] = table["benchmark_id"].map(counts).astype(int)
        return table

    def summary(self) -> dict[str, Any]:
        return {
            "model": "Samejima continuous response model on logit scores, homoscedastic",
            "n_models": self.panel.M,
            "n_benchmarks": self.panel.K,
            "n_cells": self.panel.n_cells,
            "n_cells_squeezed_from_boundary": self.panel.n_squeezed,
            "n_models_with_anchor_cell": self.n_models_with_anchor_cell,
            "reference_item": self.reference_item,
            "anchors": list(self.anchors),
            "stage1": _stage_summary(self.stage1),
            "stage2": _stage_summary(self.stage2),
            "provenance": self.panel.provenance,
        }


def _stage_summary(fit: CrmFit) -> dict[str, Any]:
    c = fit.convergence
    return {
        "final_objective": fit.objective,
        "residual_sd": fit.residual_sd,
        "n_free_parameters": fit.n_free_parameters,
        "max_abs_gradient": fit.max_abs_gradient,
        "hessian_min_eigenvalue": c.min_eigenvalue,
        "hessian_max_eigenvalue": c.max_eigenvalue,
        "hessian_n_negative_eigenvalues": c.n_negative_eigenvalues,
        "hessian_condition_number": c.condition_number,
        "newton_decrement": c.newton_decrement,
        "max_remaining_theta_step": c.max_remaining_theta_step,
        "converged": c.converged,
        "convergence_criterion": c.criterion,
    }


def two_stage_link(
    panel: Panel | None = None,
    spec: dict[str, Any] | None = None,
    *,
    adam_steps: int = 3000,
    lr: float = 0.05,
) -> TwoStageFit:
    """Reproduce the archive's two-stage fixed-parameter link on ``panel``.

    Stage 1 frees the abilities of models that have at least one anchor cell and fits on the
    anchor cells alone; stage 2 starts from stage 1, frees every ability, holds the anchors'
    item parameters fixed, and fits the whole panel. The reference benchmark's difficulty and
    log-discrimination are held at zero throughout, which is what fixes location and scale.
    """
    working = panel if panel is not None else build_panel()
    record = spec if spec is not None else estimation_spec()
    anchors = tuple(record["anchors"])
    reference = str(record["reference_item"])
    priors = record["penalty_scales"]

    reference_index = working.index_of_item(reference)
    anchor_index = np.array(sorted(working.index_of_item(a) for a in anchors), dtype=int)

    free_items_stage1 = np.ones(working.K, dtype=bool)
    free_items_stage1[reference_index] = False

    on_anchor = np.isin(working.item_of_cell, anchor_index)
    free_models_stage1 = np.zeros(working.M, dtype=bool)
    free_models_stage1[np.unique(working.model_of_cell[on_anchor])] = True

    stage1 = fit_crm(
        working,
        free_models=free_models_stage1,
        free_items=free_items_stage1,
        cells=on_anchor,
        priors=priors,
        adam_steps=adam_steps,
        lr=lr,
    )

    free_items_stage2 = free_items_stage1.copy()
    free_items_stage2[anchor_index] = False
    stage2 = fit_crm(
        working,
        free_models=np.ones(working.M, dtype=bool),
        free_items=free_items_stage2,
        start=_packed(
            stage1.theta,
            np.log(stage1.discrimination),
            stage1.difficulty,
            stage1.log_sigma,
        ),
        priors=priors,
        adam_steps=adam_steps,
        lr=lr,
    )
    return TwoStageFit(
        stage1=stage1,
        stage2=stage2,
        panel=working,
        anchors=anchors,
        reference_item=reference,
        n_models_with_anchor_cell=int(np.sum(free_models_stage1)),
    )


# --------------------------------------------------------------------------------------------
# Abilities and their standard errors
# --------------------------------------------------------------------------------------------


def ability_table(
    fit: TwoStageFit | CrmFit,
    *,
    panel: Panel | None = None,
    theta_prior_sd: float | None = None,
) -> pd.DataFrame:
    """Per-model ability, test information and standard error, on the archive's two definitions.

    For the homoscedastic continuous response model the information a benchmark contributes about
    an ability does not depend on the ability: it is ``a_k**2 / sigma**2``.

    The archive sums that over two different sets, and this function reproduces both because its
    published file does:

    * ``test_information`` and ``se_predicted_from_information`` sum over the model's **distinct
      benchmarks**;
    * ``se_theta`` — the standard error the papers quote — sums over the model's **cells** and
      adds the ability prior's precision ``1 / theta_prior_sd**2``.

    148 of the 782 models are scored more than once on some benchmark, so for those the two
    differ, and ``se_predicted_from_information`` is not the likelihood-only counterpart of
    ``se_theta`` that its name implies. Reported, not reconciled (rule 3); see
    ``docs/decisions.md``, 2026-09-10. ``information_basis`` records which set each column used.
    """
    if isinstance(fit, TwoStageFit):
        panel, final = fit.panel, fit.stage2
    else:
        if panel is None:
            raise ValueError("pass panel= when fit is a single CrmFit")
        final = fit
    prior_sd = (
        float(theta_prior_sd)
        if theta_prior_sd is not None
        else float(final.priors["theta_prior_sd"])
    )
    a = final.discrimination
    inv_var = float(np.exp(-2.0 * final.log_sigma))

    n_cells = np.bincount(panel.model_of_cell, minlength=panel.M)
    information_cells = np.bincount(
        panel.model_of_cell, weights=a[panel.item_of_cell] ** 2 * inv_var, minlength=panel.M
    )

    distinct = panel.frame[["model_id", "benchmark_id"]].drop_duplicates()
    dm = distinct["model_id"].map({m: i for i, m in enumerate(panel.models)}).to_numpy(dtype=int)
    dk = distinct["benchmark_id"].map({b: i for i, b in enumerate(panel.items)}).to_numpy(dtype=int)
    n_benchmarks = np.bincount(dm, minlength=panel.M)
    information_distinct = np.bincount(dm, weights=a[dk] ** 2 * inv_var, minlength=panel.M)
    mean_discrimination = np.bincount(dm, weights=a[dk], minlength=panel.M) / np.maximum(
        n_benchmarks, 1
    )

    return pd.DataFrame(
        {
            "model_id": list(panel.models),
            "n_cells": n_cells.astype(int),
            "n_benchmarks": n_benchmarks.astype(int),
            "mean_discrimination": mean_discrimination,
            "theta": final.theta,
            "test_information": information_distinct,
            "se_predicted_from_information": 1.0 / np.sqrt(information_distinct),
            "se_theta": 1.0 / np.sqrt(information_cells + 1.0 / prior_sd**2),
            "information_basis": "test_information: distinct benchmarks; se_theta: cells + prior",
        }
    )
