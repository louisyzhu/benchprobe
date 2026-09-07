"""Measure layer: factor structure and incremental validity.

SPEC.md §1 scope: KMO, Horn's parallel analysis, maximum-likelihood EFA with oblimin rotation,
factor scores, Thurstone weights W = R⁻¹Λ, date-residualisation. Extracted from
``frontier-ai-economic-validity`` (``notebook/analysis.ipynb``, Task-1 cells and the
"Registration-honest robustness statistics" cell) under ticket T3; signatures in SPEC.md §2.

Every estimator takes an indicator matrix (models × benchmarks; a DataFrame or a 2-D array) and
returns numbers, never prints. The ML factor solution comes from ``factor_analyzer``, as in the
archive; ``factor_analyzer`` 0.5.1 calls scikit-learn's ``check_array`` with the keyword that
scikit-learn ≥ 1.7 renamed, so the archive's one-line shim is applied once, at first use.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

__all__ = [
    "DropBootstrap",
    "Efa",
    "ParallelAnalysis",
    "date_r2",
    "efa",
    "first_factor_share",
    "kmo",
    "parallel_analysis",
    "residualisation_drop_bootstrap",
    "residualise_on_date",
    "thurstone_weights",
]

_SHIMMED = False


def _factor_analyzer():
    """Import factor_analyzer with the archive's scikit-learn compatibility shim applied once."""
    global _SHIMMED
    import factor_analyzer.factor_analyzer as fam
    import sklearn.utils as sku

    if not _SHIMMED:
        original = sku.check_array

        def check_array(*args: Any, **kwargs: Any):
            if "force_all_finite" in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return original(*args, **kwargs)

        fam.check_array = check_array
        _SHIMMED = True
    return fam.FactorAnalyzer, fam.calculate_kmo


def _matrix(z) -> tuple[np.ndarray, list[str], Any]:
    """Values, column names and row index of an indicator matrix given as DataFrame or array."""
    if isinstance(z, pd.DataFrame):
        return z.to_numpy(dtype=float), [str(c) for c in z.columns], z.index
    a = np.asarray(z, dtype=float)
    if a.ndim != 2:
        raise ValueError("an indicator matrix must be two-dimensional (models × benchmarks)")
    return a, [f"x{j}" for j in range(a.shape[1])], pd.RangeIndex(a.shape[0])


def _check_finite(a: np.ndarray, what: str) -> None:
    if not np.isfinite(a).all():
        raise ValueError(f"{what} contains NaN or infinite values")


# --------------------------------------------------------------------------------------------
# KMO and parallel analysis
# --------------------------------------------------------------------------------------------


def kmo(z) -> float:
    """Overall Kaiser–Meyer–Olkin measure of sampling adequacy (archive: ``calculate_kmo``)."""
    a, _, _ = _matrix(z)
    _check_finite(a, "indicator matrix")
    _, calculate_kmo = _factor_analyzer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, overall = calculate_kmo(a)
    return float(overall)


@dataclass(frozen=True)
class ParallelAnalysis:
    observed: np.ndarray
    """Eigenvalues of the correlation matrix, descending."""
    threshold: np.ndarray
    """The ``percentile`` of the random eigenvalues, position by position."""
    k_retained: int
    n_iter: int
    seed: int
    percentile: float


def parallel_analysis(
    z, *, n_iter: int = 1000, seed: int = 42, percentile: float = 95
) -> ParallelAnalysis:
    """Horn's parallel analysis on the correlation matrix, as the archive computes it.

    ``n_iter`` standard-normal matrices of the same shape are drawn from
    ``numpy.random.default_rng(seed)``; a factor is retained while its observed eigenvalue exceeds
    the ``percentile`` of the random eigenvalues at the same position.
    """
    a, _, _ = _matrix(z)
    _check_finite(a, "indicator matrix")
    n, p = a.shape
    rng = np.random.default_rng(seed)
    observed = np.linalg.eigvalsh(np.corrcoef(a, rowvar=False))[::-1]
    random = np.array(
        [
            np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False))[::-1]
            for _ in range(n_iter)
        ]
    )
    threshold = np.percentile(random, percentile, axis=0)
    return ParallelAnalysis(
        observed=observed,
        threshold=threshold,
        k_retained=int((observed > threshold).sum()),
        n_iter=n_iter,
        seed=seed,
        percentile=percentile,
    )


# --------------------------------------------------------------------------------------------
# Exploratory factor analysis
# --------------------------------------------------------------------------------------------


def thurstone_weights(R, loadings) -> np.ndarray:
    """Regression (Thurstone) factor-score weights ``W = R⁻¹Λ`` by pseudo-inverse.

    This is the weighting the archive's LOBO path uses on the pattern loadings.
    """
    return np.linalg.pinv(np.asarray(R, dtype=float)) @ np.asarray(loadings, dtype=float)


@dataclass(frozen=True)
class Efa:
    loadings: pd.DataFrame
    """Pattern loadings, benchmarks × factors."""
    structure: pd.DataFrame
    """Structure matrix ``ΛΦ`` (equal to the pattern loadings when unrotated or orthogonal)."""
    phi: pd.DataFrame
    """Factor correlations (identity when unrotated or orthogonal)."""
    uniquenesses: pd.Series
    ssl: pd.Series
    """Sum of squared pattern loadings per factor."""
    weights: pd.DataFrame
    """Thurstone weights on the pattern loadings, ``R⁻¹Λ``."""
    scores: np.ndarray
    """Regression factor scores ``z R⁻¹ S`` (what ``factor_analyzer.transform`` computes)."""
    signs: np.ndarray
    """Sign applied to each factor (+1 or −1) by mean-score alignment."""
    k: int
    rotation: str | None
    method: str


def efa(
    z,
    *,
    k: int = 3,
    rotation: str | None = "oblimin",
    method: str = "ml",
    align_to_mean_score: bool = True,
) -> Efa:
    """Exploratory factor analysis as the archive fits it (``FactorAnalyzer(n_factors=k, rotation,
    method)``), with the archive's sign convention.

    With ``align_to_mean_score`` each factor's sign is chosen so that its regression scores
    correlate positively with the row-mean of the indicator matrix; the sign is applied to the
    scores, the loadings, the structure matrix, the weights and the factor correlations together,
    so the solution stays internally consistent.
    """
    a, names, index = _matrix(z)
    _check_finite(a, "indicator matrix")
    FactorAnalyzer, _ = _factor_analyzer()
    fa = FactorAnalyzer(n_factors=k, rotation=rotation, method=method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa.fit(a)
        scores = fa.transform(a)
    loadings = np.asarray(fa.loadings_, dtype=float)
    structure = (
        np.asarray(fa.structure_, dtype=float) if fa.structure_ is not None else loadings.copy()
    )
    phi = np.asarray(fa.phi_, dtype=float) if getattr(fa, "phi_", None) is not None else np.eye(k)
    signs = np.ones(k)
    if align_to_mean_score:
        row_mean = a.mean(axis=1)
        for j in range(k):
            c = np.corrcoef(scores[:, j], row_mean)[0, 1]
            if np.isfinite(c) and c < 0:
                signs[j] = -1.0
    scores = scores * signs
    loadings = loadings * signs
    structure = structure * signs
    phi = (signs[:, None] * phi) * signs[None, :]
    weights = thurstone_weights(np.asarray(fa.corr_, dtype=float), loadings)
    factors = [f"F{j + 1}" for j in range(k)]
    return Efa(
        loadings=pd.DataFrame(loadings, index=names, columns=factors),
        structure=pd.DataFrame(structure, index=names, columns=factors),
        phi=pd.DataFrame(phi, index=factors, columns=factors),
        uniquenesses=pd.Series(np.asarray(fa.get_uniquenesses(), dtype=float), index=names),
        ssl=pd.Series((loadings**2).sum(axis=0), index=factors),
        weights=pd.DataFrame(weights, index=names, columns=factors),
        scores=scores,
        signs=signs,
        k=k,
        rotation=rotation,
        method=method,
    )


def first_factor_share(z, *, k: int = 3) -> float:
    """Share of common variance on the first factor of the unrotated ``k``-factor ML solution.

    The archive's ``factor1_share``: ``ssl[0] / ssl.sum()`` on the unrotated loadings, as a
    fraction. The first column of the unrotated solution is what the archive reports; it is not
    re-ordered here.
    """
    solution = efa(z, k=k, rotation=None, method="ml", align_to_mean_score=False)
    ssl = solution.ssl.to_numpy()
    return float(ssl[0] / ssl.sum())


# --------------------------------------------------------------------------------------------
# Release-date residualisation and date fits
# --------------------------------------------------------------------------------------------


def _days(days) -> np.ndarray:
    d = np.asarray(days, dtype=float).reshape(-1)
    _check_finite(d, "days")
    return d


def residualise_on_date(z, days) -> pd.DataFrame:
    """Ordinary-least-squares residual of every column on ``days`` (with intercept).

    The archive residualises the indicators and re-estimates the factor solution on the residuals
    (Task-1 cell "Residualise on release date — H2").
    """
    a, names, index = _matrix(z)
    _check_finite(a, "indicator matrix")
    d = _days(days)
    if len(d) != a.shape[0]:
        raise ValueError("days must have one entry per row of the indicator matrix")
    x = d.reshape(-1, 1)
    residual = np.empty_like(a)
    for j in range(a.shape[1]):
        model = LinearRegression().fit(x, a[:, j])
        residual[:, j] = a[:, j] - model.predict(x)
    return pd.DataFrame(residual, index=index, columns=names)


def _logistic4(t, upper, rate, midpoint, floor):
    return floor + upper / (1.0 + np.exp(-rate * (t - midpoint)))


def date_r2(x, days, *, form: str = "ols") -> float:
    """R² of a one-dimensional score vector on ``days``.

    ``form="ols"`` is a straight line (the archive's Task-1 computation). ``form="logistic"`` is
    the four-parameter logistic ``floor + upper / (1 + exp(-rate (t - midpoint)))`` fitted by
    least squares over a small grid of starting values, the best fit kept; on the archive's grid it
    reproduces the paper's 0.505 / 0.364 / 0.286 for the three oblimin factors
    (``docs/decisions.md``, T3). ``R² = 1 - SS_res / SS_tot`` in both cases.
    """
    y = np.asarray(x, dtype=float).reshape(-1)
    d = _days(days)
    if len(y) != len(d):
        raise ValueError("x and days must have the same length")
    _check_finite(y, "x")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if form == "ols":
        model = LinearRegression().fit(d.reshape(-1, 1), y)
        return float(model.score(d.reshape(-1, 1), y))
    if form != "logistic":
        raise ValueError(f"unknown form {form!r}; use 'ols' or 'logistic'")
    t = (d - d.mean()) / (
        d.std() or 1.0
    )  # standardised time conditions the optimiser; R² invariant
    span = float(y.max() - y.min()) or 1.0
    best = -np.inf
    for rate0 in (0.5, 1.0, 2.0, 5.0):
        for mid0 in (-1.0, 0.0, 1.0):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params, _ = curve_fit(
                        _logistic4, t, y, p0=[span, rate0, mid0, float(y.min())], maxfev=20000
                    )
            except (RuntimeError, ValueError):
                continue
            ss_res = float(((y - _logistic4(t, *params)) ** 2).sum())
            best = max(best, 1.0 - ss_res / ss_tot)
    if not np.isfinite(best):
        raise RuntimeError("the logistic fit did not converge from any starting value")
    return float(best)


@dataclass(frozen=True)
class DropBootstrap:
    point: float
    """Full-sample drop in first-factor share after date residualisation, as a fraction."""
    ci: tuple[float, float]
    """2.5th and 97.5th percentiles of the bootstrap draws, as fractions."""
    draws: np.ndarray
    B: int
    seed: int
    k: int


def residualisation_drop_bootstrap(
    z, days, *, k: int = 3, B: int = 2000, seed: int = 42
) -> DropBootstrap:
    """Model-resampling bootstrap of the residualisation drop (archive, H2(ii) check).

    Rows are resampled with replacement ``B`` times from ``numpy.random.default_rng(seed)``; on
    each draw the first-factor share is recomputed before and after date residualisation and the
    difference recorded. The archive draws ``rng.integers(0, n, n)`` per replicate, in this order.
    """
    a, names, index = _matrix(z)
    d = _days(days)
    frame = pd.DataFrame(a, index=index, columns=names)
    point = first_factor_share(frame, k=k) - first_factor_share(residualise_on_date(frame, d), k=k)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    draws = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        sample = pd.DataFrame(a[idx], columns=names)
        draws[b] = first_factor_share(sample, k=k) - first_factor_share(
            residualise_on_date(sample, d[idx]), k=k
        )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return DropBootstrap(
        point=float(point), ci=(float(lo), float(hi)), draws=draws, B=B, seed=seed, k=k
    )
