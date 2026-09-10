"""benchprobe on data that is not one of the vendored studies.

Run: uv run python examples/quickstart.py

The data here are simulated (a one-factor battery of eight benchmarks over 120 models, scores as
proportions, with release dates) so the example runs anywhere in a second. Replace the first block
with your own long or wide table; nothing below it changes.
"""

import warnings

import numpy as np
import pandas as pd

import benchprobe.irt as irt
import benchprobe.measure as measure
import benchprobe.predict as predict
import benchprobe.trust as trust
from benchprobe.scores import ScoreMatrix

warnings.filterwarnings("ignore", message="No rotation will be performed")

# ---- 1. your data ------------------------------------------------------------------------
rng = np.random.default_rng(0)
n_models, benchmarks = 120, [f"bench_{j}" for j in range(8)]
ability = rng.standard_normal(n_models)
loading = np.linspace(0.5, 1.5, 8)
difficulty = np.linspace(-1, 1, 8)
logit = (
    ability[:, None] * loading[None, :] - difficulty[None, :] + rng.normal(0, 0.4, (n_models, 8))
)
scores = 1 / (1 + np.exp(-logit))
scores[rng.random(scores.shape) < 0.15] = np.nan  # an incomplete leaderboard, like real ones
long = (
    pd.DataFrame(scores, columns=benchmarks)
    .assign(model=[f"model_{i:03d}" for i in range(n_models)])
    .melt(id_vars="model", var_name="item", value_name="score")
    .dropna()
)
long["release_date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
    long["model"].str[-3:].astype(int) * 9, unit="D"
)

# ---- 2. the matrix ----------------------------------------------------------------------
sm = ScoreMatrix.from_long(
    long, model="model", item="item", score="score", release_date="release_date"
)
print(f"{sm.shape[0]} models × {sm.shape[1]} benchmarks; coverage:\n{sm.coverage().round(3)}\n")
grid = sm.complete()  # rows scored on every benchmark
z = (grid - grid.mean()) / grid.std(ddof=0)

# ---- 3. measure: is it one construct? ---------------------------------------------------
print(f"KMO {measure.kmo(z):.3f}")
pa = measure.parallel_analysis(z, seed=0)
print(f"parallel analysis retains {pa.k_retained} factor(s)")
print(f"first-factor share of common variance {100 * measure.first_factor_share(z, k=3):.1f} %")
sol = measure.efa(z, k=2)
print("two-factor oblimin loadings:\n", sol.loadings.round(2))

# ---- 4. trust: reliability of a binary-scored item bank ---------------------------------
item_quality = rng.normal(0, 1, (200, 1))  # 200 items × 10 scored elements, gold verdicts
bank = (rng.random((200, 10)) < 1 / (1 + np.exp(-item_quality))).astype(int)
print(f"KR-20 on a simulated bank {trust.kr20(bank):.3f}")

# ---- 5. predict: does the battery predict a held-out criterion? -------------------------
res = predict.lobo(
    sm.grid(), targets=["bench_7"], k=1, learners=["ridge"], seed=0, rungs=["i_date", "iii_f1"]
)
print(
    res.metrics[["target", "rung", "learner", "test_rmse", "test_r2"]]
    .round(3)
    .to_string(index=False)
)

# ---- 6. irt: a common scale with standard errors ----------------------------------------
panel = irt.panel_from_long(sm.long(), model="model", item="item", score="score")
fit = irt.fit_single_stage(panel)
print(f"CRM fit converged={fit.converged}, residual sd {fit.residual_sd:.3f}")
print(fit.item_parameters().round(3).head())
print(irt.ability_table(fit, panel=panel)[["model_id", "theta", "se_theta"]].round(3).head())
