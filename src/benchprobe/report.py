"""Report layer: tables regenerated from run outputs, every caption stating the boundary.

SPEC.md §1 scope. Ticket T7 implements the one-command reproduction of the twelve One Capability
result tables the archive notebook reads rather than derives (archive README, "Tables that ship
without notebook derivation"). Every table this module writes carries a caption (rule 8) naming its
tier — *recomputed*, *statistically reproduced* or *regenerated* (rule 7) — and, where recomputed,
the largest deviation from the archived table and the tolerance it was checked against.

Run: ``python -m benchprobe.report one-capability --out runs/one_capability [--full-ladder]
[--config config.json]``. The fast path recomputes everything except the four-learner ladder
(``lobo_rung_summary.csv``), which it regenerates from the archive with that stated;
``--full-ladder`` refits the four learners on all twelve targets (about forty minutes on two
cores). Bootstrap intervals are judged against their own Monte-Carlo spread: an archived endpoint
counts as statistically reproduced when it lies within three standard deviations (over re-runs of
the bootstrap with other seeds) of benchprobe's, after allowing for the archive's rounding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LinearRegression

from benchprobe import __version__
from benchprobe import io as bio
from benchprobe import measure as m
from benchprobe import predict as p

__all__ = [
    "ONE_CAPABILITY_TABLES",
    "RunRecord",
    "TableResult",
    "caption",
    "compare",
    "interval_agreement",
    "default_config",
    "load_config",
    "main",
    "reproduce_one_capability",
]

RECOMPUTED = "recomputed"
STATISTICAL = "statistically reproduced"
REGENERATED = "regenerated"

ONE_CAPABILITY_TABLES: tuple[str, ...] = (
    "loadings_raw.csv",
    "loadings_residualised.csv",
    "task1_structure_results.json",
    "cluster_validation.csv",
    "lobo_rung_summary.csv",
    "h4_bootstrap_dmse.csv",
    "task2_error_analysis_gdpval.csv",
    "k_selection_evidence.csv",
    "ksweep_rung_iv.csv",
    "efa_factor_correlations.csv",
    "efa_uniquenesses.csv",
    "eda_distribution_stats.csv",
)
"""The twelve archived tables, in the archive README's order."""

K = 3
ECON = list(bio.ECONOMIC_BENCHMARKS)
# Literature calibration constants the archive stores alongside its own results.
KEARNS_CALIBRATION = {"kearns_calibration_raw": 0.72, "kearns_calibration_corrected": 0.41}


@dataclass(eq=False)
class TableResult:
    name: str
    tier: str
    path: str
    caption: str
    max_abs_deviation: float | None = None
    tolerance: float | None = None
    within_tolerance: bool | None = None
    notes: list[str] = field(default_factory=list)
    regenerated_columns: list[str] = field(default_factory=list)
    """Columns copied from the archive inside an otherwise recomputed table (T8)."""
    judged: bool = True


@dataclass(eq=False)
class RunRecord:
    study: str
    out_dir: str
    tables: list[TableResult]
    provenance: dict[str, Any]
    config: dict[str, Any]
    started_at: str
    finished_at: str

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "table": t.name,
                "tier": t.tier,
                "max_abs_deviation": t.max_abs_deviation,
                "tolerance": t.tolerance,
                "within_tolerance": t.within_tolerance,
                "notes": "; ".join(t.notes),
            }
            for t in self.tables
        ]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Captions and comparisons
# --------------------------------------------------------------------------------------------


def caption(
    name: str,
    tier: str,
    *,
    study: str,
    snapshot: bio.Snapshot,
    deviation: float | None = None,
    tolerance: float | None = None,
    notes: list[str] | None = None,
    within: bool | None = None,
    judged: bool = True,
    regenerated_columns: list[str] | None = None,
) -> str:
    """The boundary statement every emitted table carries (rules 7 and 8). A table whose
    deviation exceeds its tolerance says so in the caption instead of asserting its tier; a table
    whose criterion was not applied (``judged=False``, quick runs) says *that* first, before
    naming the tier it would have been judged against; and a table some of whose columns were
    copied from the archive names them in its opening clause, not in a trailing note
    (T8, Codex review, defect 8)."""
    src = snapshot.manifest.get("source", {})
    origin = f"{src.get('repository', '?')} @ {str(src.get('commit', '?'))[:8]}"
    if not judged and tier in (RECOMPUTED, STATISTICAL):
        text = (
            f"{name} — NOT JUDGED in this run (quick mode: shrunk bootstraps, criterion not "
            f"applied) by benchprobe {__version__} from snapshot {snapshot.name} "
            f"(SHA-256 {snapshot.sha256[:12]}…) for {study}; intended tier {tier}; point values "
            f"are the recomputed ones. Archived table: {origin}."
        )
        for note in notes or []:
            text += f" {note}"
        return text
    if within is False and tier in (RECOMPUTED, STATISTICAL):
        text = (
            f"{name} — NOT reproduced within tolerance by benchprobe {__version__} from snapshot "
            f"{snapshot.name} (SHA-256 {snapshot.sha256[:12]}…) for {study}: intended tier "
            f"{tier}; deviation {deviation:.3g} against tolerance {tolerance:g}. Reported, not "
            f"adjusted (rules 1 and 3). Archived table: {origin}."
        )
        for note in notes or []:
            text += f" {note}"
        return text
    if tier == RECOMPUTED:
        if regenerated_columns:
            text = (
                f"{name} — recomputed by benchprobe {__version__} EXCEPT "
                f"{', '.join(regenerated_columns)}, which are regenerated (copied from the "
                f"archive, not recomputed, and not compared); from snapshot {snapshot.name} "
                f"(SHA-256 {snapshot.sha256[:12]}…) for {study}"
            )
        else:
            text = (
                f"{name} — recomputed by benchprobe {__version__} from snapshot {snapshot.name} "
                f"(SHA-256 {snapshot.sha256[:12]}…) for {study}"
            )
        if deviation is not None:
            text += (
                f"; largest absolute deviation from the archived table {deviation:.3g} "
                f"(tolerance {tolerance:g})"
            )
        text += f". Archived table: {origin}."
    elif tier == STATISTICAL:
        text = (
            f"{name} — statistically reproduced by benchprobe {__version__} from snapshot "
            f"{snapshot.name} (SHA-256 {snapshot.sha256[:12]}…) for {study}: point values "
            f"recomputed, bootstrap intervals from a random stream the archive does not record"
        )
        if deviation is not None:
            text += (
                f"; worst interval endpoint {deviation:.2f} Monte-Carlo standard deviations from "
                f"the archived value (tolerance {tolerance:g})"
            )
        text += f". Archived table: {origin}."
    elif tier == REGENERATED:
        text = (
            f"{name} — regenerated: copied unchanged from the archive ({origin}) into this run on "
            f"snapshot {snapshot.name}; not recomputed by benchprobe {__version__}."
        )
    else:
        raise ValueError(f"unknown tier {tier!r}")
    for note in notes or []:
        text += f" {note}"
    return text


def compare(
    new: pd.DataFrame,
    archived: pd.DataFrame,
    *,
    keys: list[str],
    numeric: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Largest absolute deviation over the numeric columns shared by two tables joined on ``keys``,
    and the per-cell deviation table. Rows must match one to one."""
    if numeric is None:
        numeric = [
            c
            for c in archived.columns
            if c not in keys
            and c not in (exclude or [])
            and pd.api.types.is_numeric_dtype(archived[c])
            and c in new.columns
        ]
    for label, frame in (("archived", archived), ("new", new)):
        dup = frame.duplicated(subset=keys, keep=False)
        if dup.any():
            raise ValueError(
                f"{label} table has {int(dup.sum())} rows sharing a key on {keys}; a many-to-many "
                f"merge would compare cells that do not correspond:\n"
                f"{frame.loc[dup, keys].head().to_string(index=False)}"
            )
    merged = archived.merge(
        new, on=keys, suffixes=("_archived", "_new"), how="outer", indicator=True
    )
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", keys + ["_merge"]]
        raise ValueError(f"rows do not match one to one on {keys}:\n{missing.to_string()}")
    if not numeric:
        raise ValueError(
            f"no numeric columns in common to compare on {keys}; a comparison over zero cells "
            "would report a deviation of zero"
        )
    dev = pd.DataFrame({k: merged[k] for k in keys})
    for c in numeric:
        dev[c] = (merged[f"{c}_archived"].astype(float) - merged[f"{c}_new"].astype(float)).abs()
    values = dev[numeric].to_numpy(dtype=float) if numeric else np.zeros((0, 0))
    if values.size and not np.isfinite(values).all():
        worst = float("nan")  # a NaN cell is a failure, never a zero deviation
    else:
        worst = float(values.max()) if values.size else 0.0
    return worst, dev


def interval_agreement(
    archived: np.ndarray, ours: np.ndarray, reruns: np.ndarray, *, rounding: float
) -> tuple[float, np.ndarray]:
    """How far archived bootstrap endpoints sit from benchprobe's, in Monte-Carlo standard
    deviations. ``reruns`` is ``(n_seeds, n_cells)`` from the same bootstrap under other seeds;
    ``rounding`` is the archive's half-unit of precision, subtracted from each deviation first.
    Returns the largest z and the per-cell z (0 where the deviation is inside the rounding)."""
    archived = np.asarray(archived, float)
    ours = np.asarray(ours, float)
    sd = reruns.std(axis=0, ddof=1)
    diff = np.abs(archived - ours)
    excess = np.maximum(diff - rounding, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(excess > 0, excess / sd, 0.0)
    # a NaN on either side, or a NaN spread, is a failure, never agreement
    bad = ~np.isfinite(archived) | ~np.isfinite(ours) | ~np.isfinite(sd) | ~np.isfinite(z)
    z = np.where(bad, np.inf, z)
    return float(np.max(z)) if z.size else 0.0, z


MC_Z_TOLERANCE = 3.0


def _flag_flips(new: pd.DataFrame, archived: pd.DataFrame, keys: list[str]) -> list[str]:
    """Rows whose archived ``excludes_zero`` conclusion differs from the recomputed one."""
    merged = archived.merge(new, on=keys, suffixes=("_archived", "_new"))
    flips = merged[merged["excludes_zero_archived"] != merged["excludes_zero_new"]]
    return [" / ".join(str(r[k]) for k in keys) for _, r in flips.iterrows()]


def _read_archived(snapshot: bio.Snapshot, name: str):
    path = snapshot.file(f"archived/{name}")
    if name.endswith(".json"):
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------------
# Configuration (SPEC §1 io: config-driven runs)
# --------------------------------------------------------------------------------------------


def default_config() -> dict[str, Any]:
    return {
        "study": "one_capability",
        "snapshot": bio.DEFAULT_SNAPSHOT,
        "out_dir": "runs/one_capability",
        "full_ladder": False,
        "k": K,
        "seed": 0,
        "bootstrap_B": 2000,
        "bootstrap_seed": 42,
        "parallel_analysis_iter": 1000,
        "parallel_analysis_seed": 42,
    }


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Defaults, updated by the JSON file at ``path`` (unknown keys are an error)."""
    cfg = default_config()
    if path is None:
        return cfg
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(user) - set(cfg))
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    cfg.update(user)
    return cfg


# --------------------------------------------------------------------------------------------
# One Capability: the twelve tables
# --------------------------------------------------------------------------------------------


def _write(out: Path, name: str, obj) -> Path:
    path = out / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _loadings_table(sol: m.Efa, labels: dict[str, str], blocks: dict[str, str], order):
    tab = sol.loadings.copy()
    tab.index = [labels[k] for k in tab.index]
    tab.insert(0, "block", [blocks[k] for k in sol.loadings.index])
    tab = tab.reset_index().rename(columns={"index": ""})
    return tab[[""] + list(order)]


def reproduce_one_capability(
    out_dir: str | Path,
    *,
    full_ladder: bool = False,
    k: int = K,
    seed: int = 0,
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 42,
    parallel_analysis_iter: int = 1000,
    parallel_analysis_seed: int = 42,
    snapshot: bio.Snapshot | None = None,
    quick: bool = False,
) -> RunRecord:
    """Write the twelve One Capability tables to ``out_dir`` with captions and a comparison against
    the archived copies.

    ``quick`` shrinks the bootstraps (for tests) and is recorded in the provenance; it never changes
    which tables are recomputed. ``full_ladder`` refits the four registered learners on all twelve
    targets for ``lobo_rung_summary.csv``; otherwise that table is regenerated and says so.
    """
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in (*ONE_CAPABILITY_TABLES, "captions.md", "comparison.csv", "provenance.json"):
        if (out / stale).exists():
            (out / stale).unlink()  # only files this run writes; nothing else is touched
    snap = snapshot if snapshot is not None else bio.load_snapshot(bio.DEFAULT_SNAPSHOT)
    study = "One Capability or Many? (arXiv:2608.29420)"
    n_reruns = 3 if quick else 20
    if quick:
        bootstrap_B = min(bootstrap_B, 50)
        parallel_analysis_iter = min(parallel_analysis_iter, 100)
    results: list[TableResult] = []

    def record(name, tier, path, dev=None, tol=None, notes=None, judged=True, regenerated=None):
        notes = list(notes or [])
        regenerated = list(regenerated or [])
        within = None if (not judged or dev is None) else bool(dev <= tol)
        cap = caption(
            name,
            tier,
            study=study,
            snapshot=snap,
            deviation=dev,
            tolerance=tol,
            notes=notes,
            within=within,
            judged=judged,
            regenerated_columns=regenerated,
        )
        results.append(
            TableResult(
                name=name,
                tier=tier,
                path=str(path),
                caption=cap,
                max_abs_deviation=dev,
                tolerance=tol,
                within_tolerance=within,
                notes=notes,
                regenerated_columns=regenerated,
                judged=judged,
            )
        )

    labels, blocks = bio.LABELS, bio.BLOCKS
    g1 = bio.build_grid(snap, "complete_case")
    g2 = bio.build_grid(snap, "dense9")
    ck = bio.build_grid(snap, "compute_known")

    # ---- structure on the complete-case grid --------------------------------------------
    raw_sol = m.efa(g1.z, k=k, rotation="oblimin")
    zres = m.residualise_on_date(g1.z, g1.days)
    res_sol = m.efa(zres, k=k, rotation="oblimin")

    # 1. loadings_raw.csv
    name = "loadings_raw.csv"
    tab = _loadings_table(raw_sol, labels, blocks, ["block", "F1", "F2", "F3"][: 1 + k])
    arch = _read_archived(snap, name).rename(columns={"Unnamed: 0": ""})
    dev, _ = compare(tab, arch, keys=[""])
    record(name, RECOMPUTED, _write(out, name, tab), dev, 5e-4)

    # 2. loadings_residualised.csv (archive column order: F1..Fk then block)
    name = "loadings_residualised.csv"
    tab = _loadings_table(res_sol, labels, blocks, ["F1", "F2", "F3"][:k] + ["block"])
    arch = _read_archived(snap, name).rename(columns={"Unnamed: 0": ""})
    dev, _ = compare(tab, arch, keys=[""])
    record(name, RECOMPUTED, _write(out, name, tab), dev, 5e-4)

    # 3. task1_structure_results.json
    name = "task1_structure_results.json"
    arch = _read_archived(snap, name)
    pa_raw = m.parallel_analysis(g1.z, n_iter=parallel_analysis_iter, seed=parallel_analysis_seed)
    pa_res = m.parallel_analysis(zres, n_iter=parallel_analysis_iter, seed=parallel_analysis_seed)
    raw_share = m.first_factor_share(g1.z, k=k)
    res_share = m.first_factor_share(zres, k=k)
    ck_raw = m.first_factor_share(ck.z, k=k)
    ck_x = np.column_stack([ck.days, np.log10(ck.frame["totalParameters"].astype(float))])
    ck_resid = np.empty_like(ck.z.to_numpy())
    for j in range(ck.z.shape[1]):
        lr = LinearRegression().fit(ck_x, ck.z.to_numpy()[:, j])
        ck_resid[:, j] = ck.z.to_numpy()[:, j] - lr.predict(ck_x)
    ck_res = m.first_factor_share(pd.DataFrame(ck_resid, columns=ck.z.columns), k=k)
    date_r2 = [m.date_r2(g1.z[c].to_numpy(), g1.days, form="ols") for c in g1.benchmarks]
    dominant = int(np.argmax(raw_sol.ssl.to_numpy()))
    dom_r2 = m.date_r2(raw_sol.scores[:, dominant], g1.days, form="ols")
    spearman = snap.table[list(bio.PRIMARY_BENCHMARKS)].corr(method="spearman").to_numpy()
    iu = np.triu_indices(len(bio.PRIMARY_BENCHMARKS), 1)
    zrank = g1.z.rank().apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    p_dim = len(g1.benchmarks)
    task1 = {
        "pca_pc1_share": float(pa_raw.observed[0] / p_dim),
        "pca_evr_top5": [float(v / p_dim) for v in pa_raw.observed[:5]],
        "parallel_analysis_k_raw": pa_raw.k_retained,
        "parallel_analysis_k_residualised": pa_res.k_retained,
        "raw_factor1_communal_share": raw_share,
        "residualised_factor1_share_dateonly": res_share,
        "drop_pp_dateonly": 100 * (raw_share - res_share),
        "subsample_raw_share": ck_raw,
        "subsample_res_share": ck_res,
        "subsample_drop_pp": 100 * (ck_raw - ck_res),
        "mean_date_R2": float(np.mean(date_r2)),
        "dominant_factor_date_R2": dom_r2,
        **KEARNS_CALIBRATION,
        "n_G1": g1.n,
        "n_G2": g2.n,
        "n_subsample": ck.n,
        "kmo_G1": m.kmo(g1.z),
        "mean_offdiag_spearman": float(spearman[iu].mean()),
        "rank_efa_factor1_share": m.first_factor_share(zrank, k=k),
        "logit_efa_factor1_share": arch["logit_efa_factor1_share"],
    }
    regenerated_fields = ["logit_efa_factor1_share", *KEARNS_CALIBRATION]
    recomputed_fields = [f for f in task1 if f not in regenerated_fields]
    dev = max(
        float(np.max(np.abs(np.asarray(task1[f], dtype=float) - np.asarray(arch[f], dtype=float))))
        for f in recomputed_fields
    )
    notes = [
        "The logit transform behind logit_efa_factor1_share is not in the archived notebook; the "
        "Kearns figures are literature constants."
    ]
    record(
        name, RECOMPUTED, _write(out, name, task1), dev, 5e-4, notes, regenerated=regenerated_fields
    )

    # 4. cluster_validation.csv — clustering is outside SPEC §1; copied
    name = "cluster_validation.csv"
    arch = _read_archived(snap, name)
    record(
        name,
        REGENERATED,
        _write(out, name, arch),
        notes=["Model clustering is not in benchprobe's scope (SPEC.md §1)."],
    )

    # ---- LOBO on the complete-case grid ---------------------------------------------------
    ridge_econ = p.lobo(g1, targets=ECON, k=k, learners="ridge", seed=seed)

    # 5. lobo_rung_summary.csv
    name = "lobo_rung_summary.csv"
    arch = _read_archived(snap, name)
    if full_ladder:
        full = p.lobo(g1, targets=list(g1.benchmarks), k=k, learners="registered", seed=seed)
        rows = []
        for scope, targets in (("all12", list(g1.benchmarks)), ("economic", ECON)):
            lad = p.ladder(full, targets=targets, pooling="mean_rmse")
            for rung, r in lad.iterrows():
                rows.append(
                    {
                        "scope": scope,
                        "rung": rung,
                        "best_learner": r["learner"],
                        "train_rmse": round(float(r["train_rmse"]), 3),
                        "test_rmse": round(float(r["test_rmse"]), 3),
                        "test_r2": round(float(r["test_r2"]), 3),
                    }
                )
        tab = pd.DataFrame(rows)
        dev, _ = compare(tab, arch, keys=["scope", "rung"])
        learners_match = (
            tab.set_index(["scope", "rung"]).best_learner
            == arch.set_index(["scope", "rung"]).best_learner
        ).all()
        record(
            name,
            RECOMPUTED,
            _write(out, name, tab),
            dev,
            1e-3,
            [
                "Pooling: arithmetic mean of per-target RMSEs — inferred, not extracted. The "
                "archive ships no code that produces this table; mean-of-RMSE is the rule that "
                "reproduces all ten of its rows from lobo_metrics_full.csv and root-mean-square "
                "(what the notebook's own _ladder helper uses) reproduces none (docs/decisions.md, "
                "T4).",
                "Best-learner column "
                + ("matches the archive." if learners_match else "DIFFERS from the archive."),
            ],
        )
    else:
        record(
            name,
            REGENERATED,
            _write(out, name, arch),
            notes=["Fast path: the four-learner ladder was not refitted (use --full-ladder)."],
        )

    # 6. h4_bootstrap_dmse.csv
    name = "h4_bootstrap_dmse.csv"
    arch = _read_archived(snap, name)
    h4 = p.h4_table(
        ridge_econ,
        baselines=("i_date", "ii_meanidx", "iii_f1"),
        model="iv_kfac",
        targets=ECON,
        learner="ridge",
        B=bootstrap_B,
        seed=bootstrap_seed,
        per_target_baselines=("ii_meanidx", "i_date"),
    )
    h4["scope"] = h4["scope"].map(lambda s: "economic_pooled" if s == "pooled" else labels[s])
    keys = ["scope", "baseline", "kmodel"]
    reruns = np.array(
        [
            p.h4_table(
                ridge_econ,
                baselines=("i_date", "ii_meanidx", "iii_f1"),
                model="iv_kfac",
                targets=ECON,
                learner="ridge",
                B=bootstrap_B,
                seed=bootstrap_seed + 1 + r,
                per_target_baselines=("ii_meanidx", "i_date"),
            )[["ci_lo", "ci_hi"]].to_numpy()
            for r in range(n_reruns)
        ]
    )  # (n_reruns, rows, 2) in h4's row order
    aligned = arch.merge(h4, on=keys, suffixes=("_archived", "_new"))
    if len(aligned) != len(arch) or len(aligned) != len(h4):
        raise ValueError("h4 rows do not align with the archived table")
    order = h4.set_index(keys).index.get_indexer(aligned.set_index(keys).index)
    reruns = reruns[:, order, :].reshape(n_reruns, -1)
    max_z, _ = interval_agreement(
        aligned[["ci_lo_archived", "ci_hi_archived"]].to_numpy().ravel(),
        aligned[["ci_lo_new", "ci_hi_new"]].to_numpy().ravel(),
        reruns,
        rounding=0.0005,
    )
    for c in ("dMSE", "ci_lo", "ci_hi"):
        h4[c] = h4[c].round(3)
    dev_pt, _ = compare(h4, arch, keys=keys, numeric=["dMSE"])
    dev_ci, _ = compare(h4, arch, keys=keys, numeric=["ci_lo", "ci_hi"])
    flips = _flag_flips(h4, arch, keys)
    if dev_pt > 1e-3:
        max_z = float("inf")  # a point estimate off at the printed precision is not reproduced
    if flips:
        max_z = float("inf")
    notes = [
        f"Point estimates deviate at most {dev_pt:.3g} from the archive (both rounded to 3 dp,"
        f" tolerance 1e-3); interval endpoints at most {dev_ci:.3g}, which is {max_z:.2f}"
        f" Monte-Carlo standard deviations at the worst cell ({n_reruns} re-runs of the bootstrap"
        " under other seeds; the deviation column carries this z, the tolerance is 3)."
    ]
    notes.append(
        "excludes_zero agrees with the archive on every row."
        if not flips
        else f"excludes_zero DIFFERS from the archive on: {'; '.join(flips)}."
    )
    record(name, STATISTICAL, _write(out, name, h4), max_z, MC_Z_TOLERANCE, notes, judged=not quick)

    # 7. task2_error_analysis_gdpval.csv
    name = "task2_error_analysis_gdpval.csv"
    arch = _read_archived(snap, name)
    yt, yp = ridge_econ.oof[("gdpval_elo", "iv_kfac", "ridge")]
    idx = ridge_econ.oof_index["gdpval_elo"]
    err = pd.DataFrame(
        {
            "model": g1.frame["name"].to_numpy()[idx],
            "observed_z": yt,
            "predicted_z": yp,
            "residual": yt - yp,
        }
    ).sort_values("residual", kind="stable")
    err = err.reset_index(drop=True)
    dev, _ = compare(err, arch, keys=["model"])
    record(
        name,
        RECOMPUTED,
        _write(out, name, err),
        dev,
        5e-4,
        ["Out-of-fold ridge predictions on rung iv (k-factor), sorted by residual."],
    )

    # 8. k_selection_evidence.csv
    name = "k_selection_evidence.csv"
    arch = _read_archived(snap, name)
    obs, thr = pa_raw.observed, pa_raw.threshold
    kaiser = int((obs > 1).sum())
    ksel = pd.DataFrame(
        [
            {
                "criterion": "parallel_analysis",
                "selected_k": pa_raw.k_retained,
                "detail": (
                    f"only obs eig1 ({obs[0]:.2f}) > rand95 ({thr[0]:.2f}); "
                    f"eig2 ({obs[1]:.2f})<{thr[1]:.2f}"
                ),
            },
            {
                "criterion": "kaiser_eig_gt_1",
                "selected_k": kaiser,
                "detail": f"{kaiser} eigenvalue{'s' if kaiser != 1 else ''} >1",
            },
            {
                "criterion": "scree_elbow",
                "selected_k": 1,
                "detail": f"sharp elbow after PC1 ({obs[0]:.2f} -> {obs[1]:.2f})",
            },
            arch.set_index("criterion").loc["BIC_min"].to_dict() | {"criterion": "BIC_min"},
        ]
    )[["criterion", "selected_k", "detail"]]
    recomputed_rows = ["parallel_analysis", "kaiser_eig_gt_1", "scree_elbow"]
    dev, _ = compare(
        ksel[ksel.criterion.isin(recomputed_rows)],
        arch[arch.criterion.isin(recomputed_rows)],
        keys=["criterion"],
        numeric=["selected_k"],
    )
    record(
        name,
        RECOMPUTED,
        _write(out, name, ksel),
        dev,
        0,
        [
            "Three of the four rows are recomputed and compared (parallel_analysis, "
            "kaiser_eig_gt_1, scree_elbow); the BIC_min row is copied from the archive because the "
            "BIC computation is not in the archived notebook, and is excluded from the comparison "
            "rather than compared against itself. The detail column is text written in the "
            "archive's format and is not compared; only selected_k is."
        ],
        regenerated=["row BIC_min"],
    )

    # 9. ksweep_rung_iv.csv
    name = "ksweep_rung_iv.csv"
    arch = _read_archived(snap, name)
    rows = []
    results_by_k = []
    for kk in (2, 3, 4, 5):
        res_k = (
            ridge_econ
            if kk == k
            else p.lobo(
                g1, targets=ECON, k=kk, learners="ridge", seed=seed, rungs=["ii_meanidx", "iv_kfac"]
            )
        )
        results_by_k.append(res_k)
        e = res_k.metrics[(res_k.metrics.rung == "iv_kfac") & (res_k.metrics.learner == "ridge")]
        d = p.pooled_delta_mse(
            res_k,
            baseline="ii_meanidx",
            model="iv_kfac",
            targets=ECON,
            learner="ridge",
            B=bootstrap_B,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                "k": kk,
                "pooled_test_r2": float(e.test_r2.mean()),
                "pooled_test_rmse": float(np.sqrt(e.test_mse.mean())),
                "dMSE_vs_meanidx": d.point,
                "ci_lo": d.ci[0],
                "ci_hi": d.ci[1],
                "excludes_zero": bool(d.ci[0] > 0 or d.ci[1] < 0),
                "note": arch.set_index("k").loc[kk, "note"] if kk in arch.k.values else "",
            }
        )
    ksweep = pd.DataFrame(rows)
    reruns = np.array(
        [
            [
                p.pooled_delta_mse(
                    res_k,
                    baseline="ii_meanidx",
                    model="iv_kfac",
                    targets=ECON,
                    learner="ridge",
                    B=bootstrap_B,
                    seed=bootstrap_seed + 1 + r,
                ).ci
                for res_k in results_by_k
            ]
            for r in range(n_reruns)
        ]
    )  # (n_reruns, 4, 2) in results_by_k order, i.e. ksweep row order
    aligned = arch.merge(ksweep, on=["k"], suffixes=("_archived", "_new"))
    if len(aligned) != len(arch) or len(aligned) != len(ksweep):
        raise ValueError("k-sweep rows do not align with the archived table")
    order = ksweep.set_index("k").index.get_indexer(aligned.set_index("k").index)
    reruns = reruns[:, order, :].reshape(n_reruns, -1)
    max_z, _ = interval_agreement(
        aligned[["ci_lo_archived", "ci_hi_archived"]].to_numpy().ravel(),
        aligned[["ci_lo_new", "ci_hi_new"]].to_numpy().ravel(),
        reruns,
        rounding=0.00005,
    )
    for c in ("pooled_test_r2", "pooled_test_rmse", "dMSE_vs_meanidx", "ci_lo", "ci_hi"):
        ksweep[c] = ksweep[c].round(4)
    dev_pt, _ = compare(
        ksweep, arch, keys=["k"], numeric=["pooled_test_r2", "pooled_test_rmse", "dMSE_vs_meanidx"]
    )
    dev_ci, _ = compare(ksweep, arch, keys=["k"], numeric=["ci_lo", "ci_hi"])
    flips = _flag_flips(ksweep, arch, ["k"])
    if dev_pt > 1e-3 or flips:
        max_z = float("inf")
    notes = [
        f"Point columns deviate at most {dev_pt:.3g} from the archive (rounded to 4 dp, tolerance"
        f" 1e-3); interval endpoints at most {dev_ci:.3g}, {max_z:.2f} Monte-Carlo standard"
        f" deviations at the worst cell ({n_reruns} re-runs; the deviation column carries this z,"
        " the tolerance is 3). pooled_test_rmse is the root of the mean per-target MSE, as the"
        " archived table has it. The note column is copied from the archive.",
        "excludes_zero agrees with the archive on every row."
        if not flips
        else f"excludes_zero DIFFERS from the archive on k = {', '.join(flips)}.",
    ]
    record(
        name, STATISTICAL, _write(out, name, ksweep), max_z, MC_Z_TOLERANCE, notes, judged=not quick
    )

    # 10. efa_factor_correlations.csv (residualised solution)
    name = "efa_factor_correlations.csv"
    arch = _read_archived(snap, name).rename(columns={"Unnamed: 0": ""})
    phi = res_sol.phi.round(4).reset_index().rename(columns={"index": ""})
    dev, _ = compare(phi, arch, keys=[""])
    record(name, RECOMPUTED, _write(out, name, phi), dev, 5e-4)

    # 11. efa_uniquenesses.csv (residualised solution)
    name = "efa_uniquenesses.csv"
    arch = _read_archived(snap, name)
    uniq = pd.DataFrame(
        {
            "benchmark": res_sol.uniquenesses.index,
            "uniqueness": res_sol.uniquenesses.round(4).to_numpy(),
            "communality": (1 - res_sol.uniquenesses).round(4).to_numpy(),
        }
    )
    dev, _ = compare(uniq, arch, keys=["benchmark"])
    record(name, RECOMPUTED, _write(out, name, uniq), dev, 5e-4)

    # 12. eda_distribution_stats.csv
    name = "eda_distribution_stats.csv"
    arch = _read_archived(snap, name)
    rows = []
    for key in bio.PRIMARY_BENCHMARKS:
        x = snap.table[key].dropna().astype(float)
        rows.append(
            {
                "benchmark": labels[key],
                "block": blocks[key],
                "n": int(len(x)),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "skew": float(skew(x)),
                "is_bounded_0_100": bool(x.min() >= 0 and x.max() <= 1),
            }
        )
    eda = pd.DataFrame(rows).sort_values("skew", kind="stable").reset_index(drop=True)
    dev, _ = compare(eda, arch, keys=["benchmark"])
    record(name, RECOMPUTED, _write(out, name, eda), dev, 5e-4)

    # ---- captions, comparison, provenance -------------------------------------------------
    finished = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    config = {
        "full_ladder": full_ladder,
        "k": k,
        "seed": seed,
        "bootstrap_B": bootstrap_B,
        "bootstrap_seed": bootstrap_seed,
        "parallel_analysis_iter": parallel_analysis_iter,
        "parallel_analysis_seed": parallel_analysis_seed,
        "quick": quick,
    }
    rec = RunRecord(
        study=study,
        out_dir=str(out),
        tables=results,
        provenance=snap.provenance(),
        config=config,
        started_at=started,
        finished_at=finished,
    )
    lines = [f"# {study} — twelve archived tables, reproduced by benchprobe {__version__}", ""]
    lines.append(
        f"Snapshot {snap.name} (SHA-256 {snap.sha256}); run {started} → {finished}; "
        f"config {json.dumps(config)}."
    )
    lines.append("")
    for i, t in enumerate(results, 1):
        lines.append(f"{i}. {t.caption}")
    (out / "captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec.summary().to_csv(out / "comparison.csv", index=False)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "study": study,
                "benchprobe": __version__,
                "snapshot": rec.provenance,
                "config": config,
                "started_at": started,
                "finished_at": finished,
                "tables": [asdict(t) for t in results],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return rec


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchprobe.report",
        description="Reproduce a study's archived tables with captions stating the boundary.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    oc = sub.add_parser("one-capability", help="the twelve One Capability tables")
    oc.add_argument("--out", help="output directory (default from config)")
    oc.add_argument("--config", help="JSON config file (keys as in report.default_config())")
    oc.add_argument("--full-ladder", action="store_true", help="refit the four-learner ladder")
    oc.add_argument(
        "--quick",
        action="store_true",
        help="shrink the bootstraps (tests and demos only; recorded in the provenance)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.out:
        cfg["out_dir"] = args.out
    if args.full_ladder:
        cfg["full_ladder"] = True
    rec = reproduce_one_capability(
        cfg["out_dir"],
        full_ladder=cfg["full_ladder"],
        k=cfg["k"],
        seed=cfg["seed"],
        bootstrap_B=cfg["bootstrap_B"],
        bootstrap_seed=cfg["bootstrap_seed"],
        parallel_analysis_iter=cfg["parallel_analysis_iter"],
        parallel_analysis_seed=cfg["parallel_analysis_seed"],
        snapshot=bio.load_snapshot(cfg["snapshot"]),
        quick=args.quick,
    )
    print(rec.summary().to_string(index=False))
    print(
        f"\nwritten to {rec.out_dir}: {len(rec.tables)} tables, captions.md, comparison.csv, "
        "provenance.json"
    )
    bad = [t.name for t in rec.tables if t.within_tolerance is False]
    if bad:
        print(f"OUTSIDE TOLERANCE: {', '.join(bad)} — report, do not adjust (rules 1 and 3)")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
