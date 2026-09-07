"""Input layer: hash-pinned snapshot loading, provenance record, coverage rules, named grids.

SPEC.md §1 scope; signatures in SPEC.md §2 (T2). Snapshots vendored under
``benchprobe/data/<name>/`` carry a ``MANIFEST.json`` whose SHA-256 is verified on every load; a
mismatch raises :class:`SnapshotIntegrityError` and is never downgraded to a warning. The named
grids (SPEC.md §3) reproduce the row selections of the One Capability archive
(``frontier-ai-economic-validity``, ``notebook/analysis.ipynb``), including its base-model
deduplication, verbatim.

Config-driven runs (SPEC.md §1) live in ``benchprobe.report`` (a JSON config, T7).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchprobe import __version__

__all__ = [
    "BLOCKS",
    "DENSE9_BENCHMARKS",
    "ECONOMIC_BENCHMARKS",
    "ENV_DATA_DIR",
    "GRIDS",
    "LABELS",
    "PRIMARY_BENCHMARKS",
    "Grid",
    "Snapshot",
    "SnapshotIntegrityError",
    "base_model_key",
    "benchmark_coverage",
    "build_grid",
    "days_since_earliest_release",
    "deduplicate_by_base_model",
    "list_snapshots",
    "load_snapshot",
    "model_coverage",
    "sha256_of",
    "zscore",
]

ENV_DATA_DIR = "BENCHPROBE_DATA_DIR"
"""Environment variable naming a directory of ``<name>/MANIFEST.json`` snapshot folders."""

DEFAULT_SNAPSHOT = "one_capability_2026-07-06"

# Archive keys, labels and taxonomy blocks (notebook, Task-1 cell "Load data, taxonomy, and build
# analysis grids"); the order is the archive's.
PRIMARY_BENCHMARKS: tuple[str, ...] = (
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
)
DENSE9_BENCHMARKS: tuple[str, ...] = tuple(
    b for b in PRIMARY_BENCHMARKS if b not in ("gdpval_elo", "terminalbenchV21", "tauBanking")
)
ECONOMIC_BENCHMARKS: tuple[str, ...] = ("gdpval_elo", "terminalbenchV21", "tauBanking", "tau2")
LABELS: dict[str, str] = {
    "gdpval_elo": "GDPval (Elo)",
    "terminalbenchV21": "Terminal-Bench v2.1",
    "tauBanking": "τ³-Banking",
    "tau2": "τ²-Bench",
    "gpqa": "GPQA Diamond",
    "hle": "HLE",
    "omniscience": "AA-Omniscience",
    "scicode": "SciCode",
    "critpt": "CritPt",
    "terminalbenchHard": "Terminal-Bench Hard",
    "lcr": "AA-LCR",
    "ifbench": "IFBench",
}
BLOCKS: dict[str, str] = {
    "gdpval_elo": "Economic",
    "terminalbenchV21": "Economic",
    "tauBanking": "Economic",
    "tau2": "Economic",
    "gpqa": "Academic",
    "hle": "Academic",
    "omniscience": "Academic",
    "scicode": "Scientific-coding",
    "critpt": "Scientific-coding",
    "terminalbenchHard": "Scientific-coding",
    "lcr": "Long-context",
    "ifbench": "Instruction-following",
}
GRIDS: tuple[str, ...] = (
    "complete_case",
    "dense9",
    "economic_dense",
    "deduplicated",
    "compute_known",
)

DATE_COLUMN = "releaseDate"


class SnapshotIntegrityError(ValueError):
    """A vendored or user-supplied snapshot does not match its manifest."""


@dataclass(frozen=True, eq=False)
class Snapshot:
    """A loaded, hash-verified snapshot table with its manifest."""

    name: str
    table: pd.DataFrame
    sha256: str
    manifest: dict[str, Any]
    origin: str
    """``"vendored"`` or ``"env:BENCHPROBE_DATA_DIR"`` or ``"data_dir"`` (explicit argument)."""
    loaded_at: str
    folder: Path
    """The snapshot folder every manifest file was verified in."""

    def file(self, name: str) -> Path:
        """Path of a manifest-listed, hash-verified file of this snapshot."""
        if name not in self.manifest.get("files", {}):
            raise KeyError(f"{name!r} is not a file of snapshot {self.name!r}")
        return self.folder / name

    def provenance(self) -> dict[str, Any]:
        """A record of what was loaded, from where, with which hash; serialisable as JSON."""
        table_file = self.manifest["table"]
        entry = self.manifest["files"][table_file]
        return {
            "snapshot": self.name,
            "table": table_file,
            "sha256": self.sha256,
            "bytes": entry["bytes"],
            "rows": int(len(self.table)),
            "columns": int(self.table.shape[1]),
            "verified_against_manifest": True,
            "origin": self.origin,
            "source": self.manifest.get("source", {}),
            "upstream": self.manifest.get("upstream", []),
            "loaded_at": self.loaded_at,
            "benchprobe": __version__,
        }


@dataclass(frozen=True, eq=False)
class Grid:
    """A named analysis grid: rows, indicator matrix, and the release-date covariate."""

    name: str
    frame: pd.DataFrame
    benchmarks: tuple[str, ...]
    labels: dict[str, str]
    blocks: dict[str, str]
    z: pd.DataFrame
    days: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.frame))


# --------------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------------


def sha256_of(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vendored_root() -> Path:
    return Path(str(files("benchprobe").joinpath("data")))


def _resolve_dir(name: str, data_dir: str | os.PathLike | None) -> tuple[Path, str]:
    if data_dir is not None:
        return Path(data_dir) / name, "data_dir"
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env) / name, f"env:{ENV_DATA_DIR}"
    return _vendored_root() / name, "vendored"


def list_snapshots(data_dir: str | os.PathLike | None = None) -> list[str]:
    """Names of the snapshot folders (those carrying a ``MANIFEST.json``) that would be searched."""
    root, _ = _resolve_dir("", data_dir)
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/MANIFEST.json"))


def load_snapshot(
    name: str = DEFAULT_SNAPSHOT, *, data_dir: str | os.PathLike | None = None
) -> Snapshot:
    """Load snapshot ``name`` and verify every file its manifest lists.

    ``data_dir`` defaults to ``$BENCHPROBE_DATA_DIR``, then to the copies vendored with the package.
    Every file named in the manifest's ``files`` must exist and hash to the recorded SHA-256 and
    size; otherwise :class:`SnapshotIntegrityError` is raised.
    """
    folder, origin = _resolve_dir(name, data_dir)
    manifest_path = folder / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no MANIFEST.json for snapshot {name!r} under {folder.parent}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("name") != name:
        raise SnapshotIntegrityError(
            f"manifest name {manifest.get('name')!r} does not match folder {name!r}"
        )
    table_file = manifest.get("table")
    if not table_file or table_file not in manifest.get("files", {}):
        raise SnapshotIntegrityError(f"manifest for {name!r} names no table among its files")
    digests: dict[str, str] = {}
    for filename, expected in manifest["files"].items():
        path = folder / filename
        if not path.is_file():
            raise SnapshotIntegrityError(
                f"{name}: {filename} is listed in the manifest but missing"
            )
        digest = sha256_of(path)
        if digest != expected["sha256"]:
            raise SnapshotIntegrityError(
                f"{name}: {filename} hashes to {digest}, manifest says {expected['sha256']}"
            )
        if "bytes" in expected and path.stat().st_size != expected["bytes"]:
            raise SnapshotIntegrityError(
                f"{name}: {filename} is {path.stat().st_size} bytes, manifest says "
                f"{expected['bytes']}"
            )
        digests[filename] = digest
    table = pd.read_csv(folder / table_file)
    if DATE_COLUMN in table.columns:
        table[DATE_COLUMN] = pd.to_datetime(table[DATE_COLUMN], errors="coerce")
    return Snapshot(
        name=name,
        table=table,
        sha256=digests[table_file],
        manifest=manifest,
        origin=origin,
        loaded_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        folder=folder,
    )


# --------------------------------------------------------------------------------------------
# Coverage rules (archive, Phase 1 "Density audit and pre-registered inclusion rules")
# --------------------------------------------------------------------------------------------


def benchmark_coverage(
    table: pd.DataFrame, benchmarks: list[str] | tuple[str, ...], *, min_models: int = 60
) -> pd.DataFrame:
    """Per-benchmark model counts and the keep decision (kept at ``min_models`` models or more)."""
    counts = table[list(benchmarks)].notna().sum(axis=0)
    return pd.DataFrame(
        {
            "benchmark": list(benchmarks),
            "n_models": counts.to_numpy(dtype=int),
            "kept": (counts >= min_models).to_numpy(),
        }
    )


def model_coverage(
    table: pd.DataFrame, benchmarks: list[str] | tuple[str, ...], *, min_benchmarks: int = 8
) -> pd.DataFrame:
    """Rows scored on at least ``min_benchmarks`` of ``benchmarks``, with an ``n_bench`` count."""
    n_bench = table[list(benchmarks)].notna().sum(axis=1)
    kept = table.loc[n_bench >= min_benchmarks].copy()
    kept["n_bench"] = n_bench.loc[kept.index].to_numpy(dtype=int)
    return kept


# --------------------------------------------------------------------------------------------
# Grids (archive, Task-1 cell "Load data, taxonomy, and build analysis grids"; R1 block)
# --------------------------------------------------------------------------------------------


def zscore(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], *, ddof: int = 0):
    """Column-wise standardisation over the rows of ``frame`` (population SD by default).

    A column with zero variance cannot be standardised and raises, rather than becoming NaN.
    """
    x = frame[list(columns)].astype(float)
    sd = x.std(ddof=ddof)
    constant = [str(c) for c in sd.index[(sd == 0) | sd.isna()]]
    if constant:
        raise ValueError(f"zero-variance column(s) cannot be standardised: {', '.join(constant)}")
    return (x - x.mean()) / sd


def days_since_earliest_release(
    frame: pd.DataFrame,
    reference: pd.Timestamp,
    *,
    column: str = DATE_COLUMN,
    strict: bool = True,
) -> np.ndarray:
    """Days from ``reference`` (the earliest release date in the full snapshot) per row.

    Integer-valued. With ``strict`` (the default for analysis grids) a missing release date raises;
    otherwise the array is float with NaN for those rows.
    """
    days = (pd.to_datetime(frame[column]) - reference).dt.days
    if days.isna().any():
        if strict:
            raise ValueError("release date missing for some rows of the grid")
        return days.to_numpy(dtype=float)
    return days.to_numpy(dtype=int)


def base_model_key(name: str) -> str:
    """The archive's base-model key: strip reasoning, effort and preview suffixes from a name.

    Verbatim from the archive's R1 deduplication block.
    """
    b = re.sub(
        r"\s*\((?:Non-reasoning|Reasoning|high|medium|low|minimal|Preview|preview)[^)]*\)",
        "",
        str(name),
    )
    b = re.sub(r"\s+(high|medium|low|minimal)$", "", b, flags=re.I)
    return b.strip()


def deduplicate_by_base_model(table: pd.DataFrame) -> pd.DataFrame:
    """One row per base model, as the archive's R1 block computes it.

    Configurations are keyed by :func:`base_model_key`, sorted by ``intelligenceIndex`` descending
    (missing index sorts last), and collapsed with ``groupby(...).first()``. Note that pandas'
    ``first()`` takes the first *non-null* value per column within each group, so where the
    top-ranked configuration lacks a benchmark score, the row carries that score from a lower-ranked
    configuration of the same base model. This is the archived computation and is reproduced
    verbatim; see ``docs/decisions.md`` (T2).
    """
    d = table.copy()
    d["base"] = d["name"].apply(base_model_key)
    d["_ii"] = d["intelligenceIndex"].fillna(-np.inf)
    out = (
        d.sort_values("_ii", ascending=False, kind="stable").groupby("base", as_index=False).first()
    )
    out = out.drop(columns=["_ii"])
    return out[[c for c in out.columns if c != "base"] + ["base"]]  # ``base`` last, as an extra


def _select_rows(table: pd.DataFrame, grid: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    primary = list(PRIMARY_BENCHMARKS)
    if grid == "complete_case":
        return table.dropna(subset=primary), PRIMARY_BENCHMARKS
    if grid == "dense9":
        return table.dropna(subset=list(DENSE9_BENCHMARKS)), DENSE9_BENCHMARKS
    if grid == "economic_dense":
        return table.dropna(subset=["gdpval_elo", "terminalbenchV21", "tauBanking"]), (
            PRIMARY_BENCHMARKS
        )
    if grid == "deduplicated":
        return deduplicate_by_base_model(table).dropna(subset=primary), PRIMARY_BENCHMARKS
    if grid == "compute_known":
        complete = table.dropna(subset=primary)
        return complete.dropna(subset=["totalParameters"]), PRIMARY_BENCHMARKS
    raise ValueError(f"unknown grid {grid!r}; known grids: {', '.join(GRIDS)}")


def build_grid(snapshot: Snapshot, grid: str) -> Grid:
    """Build one of the named grids (SPEC.md §3) from a loaded snapshot.

    ``z`` standardises the grid's benchmark columns over the grid's own rows with population SD,
    as the archive's ``zmat`` does; ``days`` counts from the earliest release date in the *full*
    snapshot, as the archive does. ``economic_dense`` is a coverage grid, not an analysis grid: its
    indicator matrix keeps NaN where a row lacks one of the other nine benchmarks and its ``days``
    may carry NaN where a release date is missing.
    """
    rows, benchmarks = _select_rows(snapshot.table, grid)
    rows = rows.copy().reset_index(drop=True)
    reference = snapshot.table[DATE_COLUMN].min()
    return Grid(
        name=grid,
        frame=rows,
        benchmarks=tuple(benchmarks),
        labels={b: LABELS[b] for b in benchmarks},
        blocks={b: BLOCKS[b] for b in benchmarks},
        z=zscore(rows, benchmarks),
        days=days_since_earliest_release(rows, reference, strict=grid != "economic_dense"),
    )
