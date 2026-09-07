"""Trust layer: reliability and grader agreement.

SPEC.md §1 scope: KR-20, dependability index, Livingston–Lewis classification accuracy,
Krippendorff's α, Cohen's κ, generalisability-theory variance components, human-vs-grader
agreement with bootstrap intervals. Extracted from ``llm-judge-reliability`` (``estimators.py``,
``sweeps.py``, commit ``126d1bce``) under ticket T5; signatures in SPEC.md §2.

The estimators take element-level verdict matrices of shape ``(n_items, K)`` with 0/1 entries, or
total scores of shape ``(n_items,)``. The simulations carry the archive's seeds and draw order, so
``sweeps()`` reproduces the archive's ``sweeps.py`` output; the *published* grid
(``sweep_grid.json``) came from a different random stream and is reproduced statistically.
Krippendorff's α and Cohen's κ are not in the archive; they are new code under this ticket, checked
against published reference values and scikit-learn.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from scipy.stats import beta as beta_dist
from scipy.stats import binom
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import cohen_kappa_score

from benchprobe.io import Snapshot, load_snapshot

__all__ = [
    "BANK_SPREADS",
    "JUDGE_ERRORS",
    "K_DEFAULT",
    "MEASURED_ERROR",
    "N_ITEMS",
    "Agreement",
    "Bank",
    "BetaBinomialFit",
    "GStudy",
    "LivingstonLewis",
    "agreement_with_interval",
    "bank_summary",
    "bb_gof",
    "bb_mle",
    "bb_pmf",
    "classification_accuracy",
    "cluster_bootstrap",
    "cohens_kappa",
    "gstudy_two_way",
    "intra_item_rho",
    "kr20",
    "kr21",
    "krippendorffs_alpha",
    "livingston_lewis",
    "ll_estimand_control",
    "load_bank",
    "phi_control",
    "phi_lambda",
    "phrasing_gstudy",
    "published_grid",
    "simulate_bank",
    "sweeps",
    "two_way_sweep",
]

DEFAULT_SNAPSHOT = "judge_2026-08-31"
K_DEFAULT = 10
N_ITEMS = 210
MEASURED_ERROR = 0.0472
"""Per-element judge error measured on the real bank (archive ``sweeps.py``)."""
BANK_SPREADS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
JUDGE_ERRORS: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30)


# --------------------------------------------------------------------------------------------
# The real bank
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Bank:
    """Element-level verdicts of the judged items: ``gold`` and ``judge`` are ``(n, K)`` 0/1."""

    gold: np.ndarray
    judge: np.ndarray
    question_id: np.ndarray
    item_id: np.ndarray
    K: int
    n_items_total: int
    judge_model: str

    @property
    def n(self) -> int:
        return int(self.gold.shape[0])

    @property
    def gold_total(self) -> np.ndarray:
        return self.gold.sum(axis=1)

    @property
    def judge_total(self) -> np.ndarray:
        return self.judge.sum(axis=1)


def load_bank(snapshot: Snapshot | None = None, *, K: int = K_DEFAULT) -> Bank:
    """The archive's ``load_bank``: rows with judge verdicts, as gold and judge element matrices."""
    snapshot = snapshot if snapshot is not None else load_snapshot(DEFAULT_SNAPSHOT)
    df = snapshot.table
    g = [f"gold_e{i + 1}" for i in range(K)]
    j = [f"judge_e{i + 1}" for i in range(K)]
    scored = df.dropna(subset=j)
    models = [m for m in scored["judge_model"].dropna().unique()] if "judge_model" in df else []
    return Bank(
        gold=scored[g].to_numpy(float),
        judge=scored[j].to_numpy(float),
        question_id=scored["question_id"].to_numpy(),
        item_id=scored["item_id"].to_numpy(),
        K=K,
        n_items_total=int(len(df)),
        judge_model=", ".join(map(str, models)),
    )


def published_grid(snapshot: Snapshot | None = None) -> dict[str, Any]:
    """The published simulation values (``sweep_grid.json``, hash-verified) shipped with the
    snapshot: the two-way KR-20 grid and the 60-replicate run at the measured error rate."""
    snapshot = snapshot if snapshot is not None else load_snapshot(DEFAULT_SNAPSHOT)
    return json.loads(snapshot.file("sweep_grid.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# Reliability (archive estimators.py, verbatim)
# --------------------------------------------------------------------------------------------


def kr20(P) -> float:
    """Kuder–Richardson 20 over ``K`` binary elements; ``P`` is ``(n_items, K)``.

    NaN when the totals have no variance, as the archive returns.
    """
    P = np.asarray(P, float)
    n, K = P.shape
    p = P.mean(axis=0)
    tot_var = P.sum(axis=1).var(ddof=1)
    if tot_var <= 0:
        return float("nan")
    return float((K / (K - 1)) * (1 - (p * (1 - p)).sum() / tot_var))


def kr21(tot, K: int) -> float:
    """KR-21, computable from totals alone; reported by the archive for comparison only."""
    tot = np.asarray(tot, float)
    m, v = tot.mean(), tot.var(ddof=1)
    return float((K / (K - 1)) * (1 - (m * (K - m)) / (K * v)))


def intra_item_rho(P) -> float:
    """Mean pairwise phi between elements within an item."""
    P = np.asarray(P, float)
    K = P.shape[1]
    C = np.corrcoef(P.T)
    iu = np.triu_indices(K, 1)
    return float(np.nanmean(C[iu]))


# --------------------------------------------------------------------------------------------
# Beta-binomial fit and classification consistency (archive estimators.py, verbatim)
# --------------------------------------------------------------------------------------------


def _bb_nll(par, tot, K):
    a, b = np.exp(par)
    x = np.asarray(tot, float)
    ll = betaln(x + a, K - x + b) - betaln(a, b)
    return -ll.sum()


def bb_mle(tot, K: int) -> tuple[float, float]:
    """Beta-binomial maximum-likelihood fit to totals; returns ``(alpha, beta)``."""
    tot = np.asarray(tot, float)
    m, v = tot.mean() / K, tot.var(ddof=1) / (K**2)
    s = max(m * (1 - m) / v - 1, 0.1) if v > 0 else 1.0
    x0 = np.log([max(m * s, 1e-3), max((1 - m) * s, 1e-3)])
    res = minimize(_bb_nll, x0, args=(tot, K), method="Nelder-Mead")
    a, b = np.exp(res.x)
    return float(a), float(b)


def bb_pmf(a: float, b: float, K: int) -> np.ndarray:
    """Beta-binomial probability mass over ``0..K``."""
    x = np.arange(K + 1)
    logc = gammaln(K + 1) - gammaln(x + 1) - gammaln(K - x + 1)
    return np.exp(logc + betaln(x + a, K - x + b) - betaln(a, b))


class BetaBinomialFit(NamedTuple):
    chi2: float
    df: int
    p: float
    alpha: float
    beta: float


def bb_gof(tot, K: int, min_expected: float = 5) -> BetaBinomialFit:
    """Chi-square goodness of fit of the MLE beta-binomial to observed totals.

    Adjacent cells are pooled until every expected count reaches ``min_expected``; degrees of
    freedom are cells − 1 − 2, as the archive computes.
    """
    tot = np.asarray(tot, int)
    n = len(tot)
    a, b = bb_mle(tot, K)
    exp = bb_pmf(a, b, K) * n
    obs = np.bincount(tot, minlength=K + 1).astype(float)
    o: list[float] = []
    e: list[float] = []
    co = ce = 0.0
    for i in range(K + 1):
        co += obs[i]
        ce += exp[i]
        if ce >= min_expected:
            o.append(co)
            e.append(ce)
            co = ce = 0.0
    if ce > 0:
        if o:
            o[-1] += co
            e[-1] += ce
        else:
            o.append(co)
            e.append(ce)
    oa, ea = np.array(o), np.array(e)
    chi2 = float(((oa - ea) ** 2 / ea).sum())
    df = int(len(oa) - 1 - 2)
    return BetaBinomialFit(chi2, df, float(chi2_dist.sf(chi2, df)), a, b)


class LivingstonLewis(NamedTuple):
    decision_consistency: float
    classification_accuracy: float


def livingston_lewis(tot, K: int, cut: int, n_grid: int | None = None) -> LivingstonLewis:
    """Livingston–Lewis decision consistency and classification accuracy at ``cut``.

    The accuracy is indexed to the examinee's own true score on this instrument, not to any
    external criterion (archive, Section 4 of the paper).
    """
    tot = np.asarray(tot, float)
    a, b = bb_mle(tot, K)
    xs = np.arange(K + 1)
    grid = np.linspace(1e-4, 1 - 1e-4, n_grid or 400)
    w = beta_dist.pdf(grid, a, b)
    w = w / w.sum()
    L = binom.pmf(xs[:, None], K, grid[None, :])
    pass_obs = (xs >= cut).astype(float)
    p_pass_given_true = (L * pass_obs[:, None]).sum(axis=0)
    true_pass = (grid * K >= cut).astype(float)
    ca = float(
        (w * (p_pass_given_true * true_pass + (1 - p_pass_given_true) * (1 - true_pass))).sum()
    )
    dc = float((w * (p_pass_given_true**2 + (1 - p_pass_given_true) ** 2)).sum())
    return LivingstonLewis(dc, ca)


def phi_lambda(P, cut: int) -> float:
    """Generalisability dependability index Φ(λ) for absolute decisions at cut ``λ = cut/K``.

    A ratio of variance components, not a classification probability (archive, Section 4).
    """
    P = np.asarray(P, float)
    n, K = P.shape
    lam = cut / K
    X = P.mean(axis=1)
    grand = P.mean()
    v_p = max(X.var(ddof=1) - P.var(ddof=1) / K, 0.0)
    v_e = P.var(ddof=1) / K
    return float((v_p + (grand - lam) ** 2) / (v_p + (grand - lam) ** 2 + v_e))


def classification_accuracy(judge_tot, ref_tot, cut: int) -> float:
    """Agreement of pass/fail decisions between judge totals and a reference at ``cut``."""
    return float(np.mean((np.asarray(judge_tot) >= cut) == (np.asarray(ref_tot) >= cut)))


def cluster_bootstrap(
    stat_fn: Callable[[np.ndarray], float], cluster_ids, B: int = 2000, seed: int = 0
) -> np.ndarray:
    """Bootstrap ``stat_fn(idx)`` by resampling clusters with replacement (archive draw order).

    Non-finite statistics are dropped, so the result may be shorter than ``B``.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    out = []
    for _ in range(B):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(cluster_ids == c)[0] for c in drawn])
        v = stat_fn(idx)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


@dataclass(frozen=True, eq=False)
class Agreement:
    point: float
    ci: tuple[float, float]
    draws: np.ndarray
    cut: int
    n_clusters: int
    B: int
    seed: int


def agreement_with_interval(
    judge_tot,
    ref_tot,
    cut: int,
    *,
    cluster_ids,
    B: int = 2000,
    seed: int = 0,
) -> Agreement:
    """Human-vs-grader pass/fail agreement at ``cut`` with a cluster-bootstrap 95 % interval."""
    judge_tot = np.asarray(judge_tot, float)
    ref_tot = np.asarray(ref_tot, float)
    point = classification_accuracy(judge_tot, ref_tot, cut)
    draws = cluster_bootstrap(
        lambda idx: classification_accuracy(judge_tot[idx], ref_tot[idx], cut),
        cluster_ids,
        B=B,
        seed=seed,
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Agreement(
        point=point,
        ci=(float(lo), float(hi)),
        draws=draws,
        cut=cut,
        n_clusters=int(len(np.unique(np.asarray(cluster_ids)))),
        B=B,
        seed=seed,
    )


# --------------------------------------------------------------------------------------------
# Agreement coefficients (new under T5; not in the archive)
# --------------------------------------------------------------------------------------------


def cohens_kappa(a, b, *, weights: str | None = None) -> float:
    """Cohen's κ between two raters' labels (``weights`` None, ``"linear"`` or ``"quadratic"``).

    Delegates to ``sklearn.metrics.cohen_kappa_score``.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be one-dimensional and of the same length")
    return float(cohen_kappa_score(a, b, weights=weights))


def krippendorffs_alpha(data, *, level: str = "nominal") -> float:
    """Krippendorff's α for a reliability matrix ``data`` of shape (raters, units).

    Missing ratings are NaN (or None). ``level`` is ``"nominal"``, ``"ordinal"``, ``"interval"`` or
    ``"ratio"``; ordinal values are ranks over the observed categories. Units with fewer than two
    ratings do not enter the coincidence matrix. Follows Krippendorff (2011), "Computing
    Krippendorff's Alpha-Reliability"; reproduces that paper's worked example (nominal 0.743,
    interval 0.849).
    """
    arr = np.array(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be two-dimensional: raters × units")
    if level not in ("nominal", "ordinal", "interval", "ratio"):
        raise ValueError(f"unknown level {level!r}")
    values = np.unique(arr[~np.isnan(arr)])
    if values.size < 2:
        return float("nan")
    index = {v: i for i, v in enumerate(values)}
    c = values.size
    o = np.zeros((c, c))
    for u in range(arr.shape[1]):
        col = arr[:, u]
        col = col[~np.isnan(col)]
        m_u = col.size
        if m_u < 2:
            continue
        idx = np.array([index[v] for v in col])
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    o[idx[i], idx[j]] += 1.0 / (m_u - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    if level == "nominal":
        delta = 1.0 - np.eye(c)
    elif level == "interval":
        delta = (values[:, None] - values[None, :]) ** 2
    elif level == "ratio":
        s = values[:, None] + values[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(s > 0, ((values[:, None] - values[None, :]) / s) ** 2, 0.0)
    else:  # ordinal: squared difference of cumulative counts, Krippendorff (2011) eq. for ordinal
        cum = np.cumsum(n_c)
        delta = np.zeros((c, c))
        for i in range(c):
            for j in range(c):
                lo, hi = min(i, j), max(i, j)
                delta[i, j] = (cum[hi] - (cum[lo] - n_c[lo]) - (n_c[lo] + n_c[hi]) / 2) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    if d_e == 0:
        return float("nan")
    return float(1.0 - d_o / d_e)


# --------------------------------------------------------------------------------------------
# Generalisability theory (archive sweeps.py: p × phrasing G-study by two-way random-effects ANOVA)
# --------------------------------------------------------------------------------------------


class GStudy(NamedTuple):
    v_items: float
    v_facet: float
    v_resid: float
    Phi: float


def gstudy_two_way(X) -> GStudy:
    """Variance components of a fully crossed items × facet design (one observation per cell).

    Two-way random-effects ANOVA on an ``(n, R)`` matrix; negative estimates are truncated at zero.
    ``Phi`` is the dependability for absolute decisions with one condition of the facet,
    ``v_p / (v_p + (v_facet + v_e) / R)``, as the archive's phrasing G-study computes it.
    """
    X = np.asarray(X, float)
    n, R = X.shape
    if n < 2 or R < 2:
        raise ValueError("a two-way G-study needs at least two items and two facet conditions")
    gm = X.mean()
    rm = X.mean(axis=1, keepdims=True)
    cm = X.mean(axis=0, keepdims=True)
    ms_p = R * ((rm - gm) ** 2).sum() / (n - 1)
    ms_f = n * ((cm - gm) ** 2).sum() / (R - 1)
    ms_e = ((X - rm - cm + gm) ** 2).sum() / ((n - 1) * (R - 1))
    vp = max((ms_p - ms_e) / R, 0.0)
    vf = max((ms_f - ms_e) / n, 0.0)
    ve = max(ms_e, 0.0)
    phi = vp / (vp + (vf + ve) / R) if vp > 0 else float("nan")
    return GStudy(float(vp), float(vf), float(ve), float(phi))


# --------------------------------------------------------------------------------------------
# Real-bank summary (the quantities the archive README states reproduce bit-for-bit)
# --------------------------------------------------------------------------------------------


def bank_summary(bank: Bank) -> dict[str, float]:
    """KR-20 on judge and gold verdicts, per-element error, judge–gold correlation, leniency, and
    the beta-binomial goodness of fit on judge totals."""
    gt, jt = bank.gold_total, bank.judge_total
    fit = bb_gof(jt.astype(int), bank.K)
    return {
        "n_judged": bank.n,
        "kr20_judge": kr20(bank.judge),
        "kr20_gold": kr20(bank.gold),
        "per_element_error": float((bank.gold != bank.judge).mean()),
        "judge_gold_correlation": float(np.corrcoef(gt, jt)[0, 1]),
        "leniency_elements": float((jt - gt).mean()),
        "bb_chi2": fit.chi2,
        "bb_df": fit.df,
        "bb_p": fit.p,
        "bb_alpha": fit.alpha,
        "bb_beta": fit.beta,
    }


# --------------------------------------------------------------------------------------------
# Simulations (archive sweeps.py, verbatim; seeds and draw order preserved)
# --------------------------------------------------------------------------------------------


def simulate_bank(spread: float, err: float, n: int = N_ITEMS, K: int = K_DEFAULT, rng=None):
    """One synthetic bank: ``spread`` sets between-item true-score variance, ``err`` the per-element
    probability that the judge flips a verdict. Returns ``(gold, judge)`` matrices."""
    rng = rng or np.random.default_rng(0)
    lo, hi = 0.5 - spread / 2, 0.5 + spread / 2
    rates = rng.uniform(lo, hi, n)
    Gm = (rng.random((n, K)) < rates[:, None]).astype(float)
    flip = rng.random((n, K)) < err
    Jm = np.where(flip, 1 - Gm, Gm)
    return Gm, Jm


def two_way_sweep(
    reps: int = 60,
    seed: int = 11,
    *,
    spreads: Sequence[float] = BANK_SPREADS,
    errors: Sequence[float] = JUDGE_ERRORS,
    n: int = N_ITEMS,
    K: int = K_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid of KR-20 (mean, SD over ``reps``) over bank spread × judge error, one random stream in
    the archive's order. The archive's defaults reproduce its ``sweeps.py`` grid; the published
    grid came from another stream (statistically reproduced)."""
    rng = np.random.default_rng(seed)
    mean = np.zeros((len(spreads), len(errors)))
    sd = np.zeros_like(mean)
    for i, s in enumerate(spreads):
        for j, e in enumerate(errors):
            vals = [kr20(simulate_bank(s, e, n=n, K=K, rng=rng)[1]) for _ in range(reps)]
            arr = np.array([v for v in vals if np.isfinite(v)])
            mean[i, j], sd[i, j] = arr.mean(), arr.std(ddof=1)
    return mean, sd


def phi_control(seed: int = 101, n: int = 4000, K: int = K_DEFAULT) -> dict[int, dict[str, float]]:
    """Control where the beta-binomial holds exactly, so Livingston–Lewis is accurate; any residual
    Φ-vs-accuracy gap is a mismatch of quantities (archive)."""
    rng = np.random.default_rng(seed)
    p = rng.beta(5.452, 3.663, n)
    P = (rng.random((n, K)) < p[:, None]).astype(float)
    tot = P.sum(axis=1)
    out: dict[int, dict[str, float]] = {}
    for cut in (4, 5, 6, 7):
        _, ca_ll = livingston_lewis(tot, K, cut)
        ca_true = float(np.mean((tot >= cut) == (p * K >= cut)))
        phi = phi_lambda(P, cut)
        out[cut] = dict(
            ll=ca_ll, true=ca_true, gap_ll=ca_true - ca_ll, phi=phi, gap_phi=ca_true - phi
        )
    return out


def ll_estimand_control(
    seed: int = 303, n: int = 4000, K: int = K_DEFAULT
) -> dict[int, dict[str, float]]:
    """The judge's true score is not a deterministic function of gold: accuracy against the judge's
    own true score versus accuracy against gold (archive)."""
    rng = np.random.default_rng(seed)
    gold = rng.binomial(K, 0.55, n).astype(float)
    judge_true = np.clip(0.9 * gold + 0.9 + rng.normal(0, 1.1, n), 0, K)
    obs = np.clip(np.round(judge_true + rng.normal(0, 0.85, n)), 0, K)
    out: dict[int, dict[str, float]] = {}
    for cut in (4, 5, 6, 7):
        own = classification_accuracy(obs, judge_true, cut)
        ext = classification_accuracy(obs, gold, cut)
        out[cut] = dict(vs_own_true=own, vs_gold=ext, gap=own - ext)
    return out


def phrasing_gstudy(
    v_phrasing: float,
    v_resid: float,
    n: int = N_ITEMS,
    R: int = 3,
    sims: int = 200,
    seed: int = 7,
) -> dict[str, float]:
    """p × phrasing G-study by simulation: mean variance components and Φ over ``sims`` draws of a
    fully crossed design with the given phrasing and residual variances (archive)."""
    rng = np.random.default_rng(seed)
    comps = []
    for _ in range(sims):
        p = rng.normal(0, 1, (n, 1))
        ph = rng.normal(0, np.sqrt(v_phrasing), (1, R))
        e = rng.normal(0, np.sqrt(v_resid), (n, R))
        comps.append(tuple(gstudy_two_way(p + ph + e)))
    m = np.nanmean(np.array(comps, dtype=float), axis=0)
    return dict(v_items=float(m[0]), v_phrasing=float(m[1]), v_resid=float(m[2]), Phi=float(m[3]))


def sweeps() -> dict[str, Any]:
    """Every simulated quantity of the archive's ``sweeps.py`` main block, in its order."""
    mean, sd = two_way_sweep()
    return {
        "grid": {
            "rows_bank_spread": list(BANK_SPREADS),
            "cols_judge_error": list(JUDGE_ERRORS),
            "kr20_mean": mean.tolist(),
            "kr20_sd": sd.tolist(),
        },
        "range_at_measured_error": {
            "judge_error": MEASURED_ERROR,
            "kr20_min": float(mean[:, 1].min()),
            "kr20_max": float(mean[:, 1].max()),
        },
        "judge_error_axis_at_widest_bank": float(mean[-1].max() - mean[-1].min()),
        "bank_axis_at_zero_error": float(mean[:, 0].max() - mean[:, 0].min()),
        "phi_control": phi_control(),
        "ll_estimand_control": ll_estimand_control(),
        "phrasing_gstudy": {
            "irrelevant": phrasing_gstudy(0.000, 0.090),
            "mild": phrasing_gstudy(0.039, 0.090),
            "large": phrasing_gstudy(0.362, 0.090),
            "deterministic_given_phrasing": phrasing_gstudy(0.175, 0.000),
        },
    }
