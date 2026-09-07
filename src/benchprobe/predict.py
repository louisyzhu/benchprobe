"""Predict layer: held-out designs and uncertainty for criterion studies.

SPEC.md §1 scope: leave-one-benchmark-out with factor re-estimation inside every fold, pooled
ΔMSE with bootstrap intervals, general-index baseline. Extracted from
``frontier-ai-economic-validity`` (``notebook/analysis.ipynb``, Task-2 cells "Leakage-safe rung
builder", "LOBO nested-CV loop", "H4 — bootstrap ΔMSE" and the R1 deduplicated re-run) under
ticket T4; signatures in SPEC.md §2. Hold-out by task, model, family or context and
effective-sample-size reporting (§1, "thesis pipeline needs") wait for a ticket that can name the
thesis design.

Every fit is deterministic: the outer folds are ``KFold(shuffle=True, random_state=seed)``, the
tree learners carry ``random_state=0`` as registered, and the bootstraps draw from
``numpy.random.default_rng(seed)`` in the archive's order.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from benchprobe.measure import _factor_analyzer, thurstone_weights

__all__ = [
    "DeltaMse",
    "LoboResult",
    "METRIC_COLUMNS",
    "RUNGS",
    "h4_table",
    "ladder",
    "lobo",
    "pooled_delta_mse",
    "registered_learners",
]

RUNGS: tuple[str, ...] = ("i_date", "ii_meanidx", "iii_f1", "iv_kfac", "v_kfac_cov")
"""The archive's predictor ladder: standardised release date; mean of the standardised predictors
(the general-index baseline); the first factor score alone; all k factor scores; the factor scores
plus reasoning flag, open-weights flag, log10 parameters and a missing-parameters indicator."""

METRIC_COLUMNS: tuple[str, ...] = (
    "target",
    "block",
    "rung",
    "learner",
    "train_rmse",
    "test_rmse",
    "train_mae",
    "test_mae",
    "train_r2",
    "test_r2",
    "test_mse",
)
"""Columns of ``LoboResult.metrics``, the archive's ``lobo_metrics_full.csv`` layout."""

COVARIATE_COLUMNS: tuple[str, ...] = ("isReasoning", "isOpenWeights", "totalParameters")


def registered_learners() -> dict[str, tuple[Any, dict[str, list]]]:
    """The four registered learners and their grids, verbatim from the archive."""
    return {
        "ridge": (Ridge(), {"alpha": [0.03, 0.1, 0.3, 1, 3, 10, 30]}),
        "elasticnet": (
            ElasticNet(max_iter=5000),
            {"alpha": [0.03, 0.1, 0.3, 1], "l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "rf": (
            RandomForestRegressor(random_state=0),
            {"n_estimators": [300], "max_depth": [None, 4], "min_samples_leaf": [1, 3]},
        ),
        "gbm": (
            GradientBoostingRegressor(random_state=0),
            {"n_estimators": [200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        ),
    }


def _resolve_learners(learners) -> dict[str, tuple[Any, dict[str, list]]]:
    registered = registered_learners()
    if learners == "registered":
        return registered
    if isinstance(learners, str):
        if learners not in registered:
            raise ValueError(f"unknown learner {learners!r}; registered: {', '.join(registered)}")
        return {learners: registered[learners]}
    if isinstance(learners, Mapping):
        return dict(learners)
    if isinstance(learners, Sequence):
        return {name: registered[name] for name in learners}
    raise TypeError(
        "learners must be 'registered', a learner name, a sequence of names or a mapping"
    )


def _covariate_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Rung (v) covariates as the archive builds them: reasoning flag, open-weights flag,
    log10 parameters, and an indicator for missing parameters."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logparams = np.log10(frame["totalParameters"].astype(float).to_numpy())
    reason = frame["isReasoning"].astype(float).to_numpy()
    open_w = frame["isOpenWeights"].astype(float).to_numpy()
    return np.column_stack([reason, open_w, logparams, np.isnan(logparams).astype(float)])


def _build_rungs(Ztr, Zte, days_tr, days_te, cov_tr, cov_te, *, k: int, rungs: Sequence[str]):
    """The archive's leakage-safe rung builder: everything is fitted on the training rows only."""
    FactorAnalyzer, _ = _factor_analyzer()
    mu, sd = Ztr.mean(0), Ztr.std(0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Ztr_s, Zte_s = (Ztr - mu) / sd, (Zte - mu) / sd
    mi_tr, mi_te = Ztr_s.mean(1, keepdims=True), Zte_s.mean(1, keepdims=True)
    train: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    if "i_date" in rungs:
        dmu = days_tr.mean()
        dsd = days_tr.std(ddof=0) or 1
        train["i_date"] = ((days_tr - dmu) / dsd).reshape(-1, 1)
        test["i_date"] = ((days_te - dmu) / dsd).reshape(-1, 1)
    if "ii_meanidx" in rungs:
        train["ii_meanidx"], test["ii_meanidx"] = mi_tr, mi_te
    if any(r in rungs for r in ("iii_f1", "iv_kfac", "v_kfac_cov")):
        fa = FactorAnalyzer(n_factors=k, rotation="oblimin", method="ml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(Ztr_s)
        W = thurstone_weights(np.corrcoef(Ztr_s, rowvar=False), fa.loadings_)
        Ftr, Fte = Ztr_s @ W, Zte_s @ W
        for c in range(k):
            if np.corrcoef(Ftr[:, c], mi_tr.ravel())[0, 1] < 0:
                Ftr[:, c] *= -1
                Fte[:, c] *= -1
        if "iii_f1" in rungs:
            train["iii_f1"], test["iii_f1"] = Ftr[:, :1], Fte[:, :1]
        if "iv_kfac" in rungs:
            train["iv_kfac"], test["iv_kfac"] = Ftr, Fte
        if "v_kfac_cov" in rungs:
            train["v_kfac_cov"] = np.hstack([Ftr, cov_tr])
            test["v_kfac_cov"] = np.hstack([Fte, cov_te])
    return train, test


@dataclass(frozen=True)
class LoboResult:
    metrics: pd.DataFrame
    """One row per (target, rung, learner), columns ``METRIC_COLUMNS``."""
    oof: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]
    """``(target, rung, learner)`` → out-of-fold ``(y_true, y_pred)``, targets standardised on
    training-fold statistics, rows in outer-fold order."""
    grid: str
    n: int
    targets: tuple[str, ...]
    rungs: tuple[str, ...]
    learners: tuple[str, ...]
    k: int
    outer_folds: int
    inner_folds: int
    seed: int


def lobo(
    grid,
    *,
    targets: Sequence[str],
    k: int = 3,
    learners="registered",
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 0,
    covariates: bool = True,
    rungs: Sequence[str] = RUNGS,
) -> LoboResult:
    """Leave-one-benchmark-out prediction with factor re-estimation inside every outer fold.

    Each target in ``targets`` is predicted from the grid's other benchmarks. Per outer fold
    (``KFold(outer_folds, shuffle=True, random_state=seed)``) the predictors are standardised on the
    training rows, the ``k``-factor oblimin ML solution is fitted on the training rows and projected
    to the held-out rows by Thurstone weights ``R⁻¹Λ``, and every rung × learner is fitted with
    ``GridSearchCV(cv=inner_folds, scoring="neg_mean_squared_error")`` on the standardised target.
    Rung ``v_kfac_cov`` needs the covariate columns and ``covariates=True``; missing log-parameters
    are imputed with the training-fold mean, as the archive does.
    """
    frame = grid.frame
    benchmarks = list(grid.benchmarks)
    learner_map = _resolve_learners(learners)
    rungs = tuple(rungs)
    if "v_kfac_cov" in rungs and (
        not covariates or any(c not in frame.columns for c in COVARIATE_COLUMNS)
    ):
        rungs = tuple(r for r in rungs if r != "v_kfac_cov")
    days = np.asarray(grid.days, dtype=float)
    cov = _covariate_matrix(frame) if "v_kfac_cov" in rungs else np.zeros((len(frame), 0))
    outer = KFold(outer_folds, shuffle=True, random_state=seed)
    oof: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for target in targets:
        if target not in benchmarks:
            raise ValueError(f"{target!r} is not a benchmark of grid {grid.name!r}")
        predictors = [c for c in benchmarks if c != target]
        Z = frame[predictors].astype(float).to_numpy()
        y = frame[target].astype(float).to_numpy()
        if not (np.isfinite(Z).all() and np.isfinite(y).all() and np.isfinite(days).all()):
            raise ValueError(f"grid {grid.name!r} carries NaN in the LOBO inputs for {target!r}")
        for rung in rungs:
            for name, (estimator, param_grid) in learner_map.items():
                yt, yp, rt, rp = [], [], [], []
                for tr, te in outer.split(Z):
                    ymu = y[tr].mean()
                    ysd = y[tr].std(ddof=0) or 1
                    Rtr, Rte = _build_rungs(
                        Z[tr], Z[te], days[tr], days[te], cov[tr], cov[te], k=k, rungs=[rung]
                    )
                    Xtr, Xte = Rtr[rung].copy(), Rte[rung].copy()
                    if rung == "v_kfac_cov":
                        lp = Xtr.shape[1] - 2
                        m = np.nanmean(Xtr[:, lp])
                        Xtr[np.isnan(Xtr[:, lp]), lp] = m
                        Xte[np.isnan(Xte[:, lp]), lp] = m
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        search = GridSearchCV(
                            estimator,
                            param_grid,
                            cv=inner_folds,
                            scoring="neg_mean_squared_error",
                        ).fit(Xtr, (y[tr] - ymu) / ysd)
                    best = search.best_estimator_
                    yp.append(best.predict(Xte))
                    yt.append((y[te] - ymu) / ysd)
                    rp.append(best.predict(Xtr))
                    rt.append((y[tr] - ymu) / ysd)
                YT, YP = np.concatenate(yt), np.concatenate(yp)
                RT, RP = np.concatenate(rt), np.concatenate(rp)
                oof[(target, rung, name)] = (YT, YP)
                records.append(
                    {
                        "target": target,
                        "block": grid.blocks.get(target, ""),
                        "rung": rung,
                        "learner": name,
                        "train_rmse": float(np.sqrt(mean_squared_error(RT, RP))),
                        "test_rmse": float(np.sqrt(mean_squared_error(YT, YP))),
                        "train_mae": float(mean_absolute_error(RT, RP)),
                        "test_mae": float(mean_absolute_error(YT, YP)),
                        "train_r2": float(r2_score(RT, RP)),
                        "test_r2": float(r2_score(YT, YP)),
                        "test_mse": float(mean_squared_error(YT, YP)),
                    }
                )
    metrics = pd.DataFrame.from_records(records, columns=list(METRIC_COLUMNS))
    return LoboResult(
        metrics=metrics,
        oof=oof,
        grid=grid.name,
        n=int(len(frame)),
        targets=tuple(targets),
        rungs=rungs,
        learners=tuple(learner_map),
        k=k,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )


def ladder(
    result: LoboResult, *, targets: Sequence[str], pooling: str = "mean_rmse"
) -> pd.DataFrame:
    """The ladder table: per rung, the best learner by pooled test RMSE over ``targets``.

    ``pooling="mean_rmse"`` (default) is the arithmetic mean of the per-target RMSEs, which is how
    the archived ``lobo_rung_summary.csv`` and the paper's ladder table are pooled (every row of
    that table reproduces from ``lobo_metrics_full.csv`` this way and no other; see
    ``docs/decisions.md``, T4). ``pooling="rms"`` is the square root of the mean per-target MSE,
    which the archive notebook's ``_ladder`` helper uses. Train RMSE is pooled the same way as test
    RMSE; test R² is the mean across targets; ``runner_up_gap`` is the pooled test-RMSE gap to the
    second-best learner (NaN with a single learner).
    """
    if pooling not in ("mean_rmse", "rms"):
        raise ValueError(f"unknown pooling {pooling!r}; use 'mean_rmse' or 'rms'")
    e = result.metrics[result.metrics.target.isin(list(targets))]
    if e.empty:
        raise ValueError("no metrics for the requested targets")

    def pool(values: pd.Series) -> float:
        v = values.to_numpy(dtype=float)
        return float(v.mean()) if pooling == "mean_rmse" else float(np.sqrt((v**2).mean()))

    rows = {}
    for rung in result.rungs:
        by_learner = e[e.rung == rung].groupby("learner").test_rmse.agg(pool).sort_values()
        best = by_learner.index[0]
        b = e[(e.rung == rung) & (e.learner == best)]
        rows[rung] = {
            "learner": best,
            "train_rmse": pool(b.train_rmse),
            "test_rmse": float(by_learner.iloc[0]),
            "test_r2": float(b.test_r2.mean()),
            "runner_up_gap": (
                float(by_learner.iloc[1] - by_learner.iloc[0]) if len(by_learner) > 1 else np.nan
            ),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "rung"
    return table


@dataclass(frozen=True)
class DeltaMse:
    point: float
    ci: tuple[float, float]
    per_target: pd.DataFrame
    """Per target: ``dMSE`` (mean squared-error difference over its out-of-fold rows) and ``n``."""
    draws: np.ndarray
    n: int
    B: int
    seed: int | None
    baseline: str
    model: str
    learner: str


def _pooled_differences(result: LoboResult, baseline: str, model: str, targets, learner: str):
    per_target = []
    parts = []
    for t in targets:
        yb, pb = result.oof[(t, baseline, learner)]
        yk, pk = result.oof[(t, model, learner)]
        d = (yb - pb) ** 2 - (yk - pk) ** 2
        parts.append(d)
        per_target.append({"target": t, "dMSE": float(d.mean()), "n": int(len(d))})
    return np.concatenate(parts), pd.DataFrame(per_target).set_index("target")


def pooled_delta_mse(
    result: LoboResult,
    *,
    baseline: str = "ii_meanidx",
    model: str = "iv_kfac",
    targets: Sequence[str],
    learner: str = "ridge",
    B: int = 2000,
    seed: int = 42,
    stream: np.random.Generator | None = None,
) -> DeltaMse:
    """Pooled ΔMSE = MSE(baseline) − MSE(model) over the concatenated out-of-fold rows of
    ``targets``, with a percentile bootstrap over those rows (``rng.integers(0, n, n)`` per
    replicate, the archive's draw). Positive favours ``model``.

    ``stream`` lets a caller replicate a multi-comparison sequence from one generator (see
    :func:`h4_table`); otherwise a fresh ``default_rng(seed)`` is used, which reproduces the
    archive's interval exactly when this comparison was the first drawn from its stream.
    """
    d, per_target = _pooled_differences(result, baseline, model, targets, learner)
    rng = stream if stream is not None else np.random.default_rng(seed)
    n = len(d)
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return DeltaMse(
        point=float(d.mean()),
        ci=(float(lo), float(hi)),
        per_target=per_target,
        draws=draws,
        n=n,
        B=B,
        seed=None if stream is not None else seed,
        baseline=baseline,
        model=model,
        learner=learner,
    )


def h4_table(
    result: LoboResult,
    *,
    baselines: Sequence[str] = ("i_date", "ii_meanidx", "iii_f1"),
    model: str = "iv_kfac",
    targets: Sequence[str],
    learner: str = "ridge",
    B: int = 2000,
    seed: int = 42,
    per_target_baselines: Sequence[str] = ("ii_meanidx",),
) -> pd.DataFrame:
    """The archive's H4 table (``h4_bootstrap_dmse.csv`` layout), drawn from one stream in the
    order of the archive's H4 cell: the pooled comparison against each baseline in turn, then, per
    target, the comparison against each of ``per_target_baselines``. Columns ``scope, baseline,
    kmodel, dMSE, ci_lo, ci_hi, excludes_zero``. The archived table itself is one the notebook reads
    rather than derives, so its intervals are reproduced statistically, not exactly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for baseline in baselines:
        r = pooled_delta_mse(
            result,
            baseline=baseline,
            model=model,
            targets=targets,
            learner=learner,
            B=B,
            stream=rng,
        )
        rows.append(
            {
                "scope": "pooled",
                "baseline": baseline,
                "kmodel": model,
                "dMSE": r.point,
                "ci_lo": r.ci[0],
                "ci_hi": r.ci[1],
                "excludes_zero": bool(r.ci[0] > 0 or r.ci[1] < 0),
            }
        )
    for t in targets:
        for baseline in per_target_baselines:
            r = pooled_delta_mse(
                result,
                baseline=baseline,
                model=model,
                targets=[t],
                learner=learner,
                B=B,
                stream=rng,
            )
            rows.append(
                {
                    "scope": t,
                    "baseline": baseline,
                    "kmodel": model,
                    "dMSE": r.point,
                    "ci_lo": r.ci[0],
                    "ci_hi": r.ci[1],
                    "excludes_zero": bool(r.ci[0] > 0 or r.ci[1] < 0),
                }
            )
    return pd.DataFrame(rows)
