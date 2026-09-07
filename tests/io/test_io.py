"""Unit tests for benchprobe.io (ticket T2). Fast; part of the smoke set."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchprobe.io as bio

pytestmark = pytest.mark.smoke

SNAPSHOT = "one_capability_2026-07-06"


@pytest.fixture(scope="module")
def snapshot():
    return bio.load_snapshot(SNAPSHOT)


@pytest.fixture
def snapshot_copy(tmp_path):
    """A writable copy of the vendored snapshot folder, under a fresh data directory."""
    src = Path(bio.__file__).parent / "data" / SNAPSHOT
    dst = tmp_path / "data" / SNAPSHOT
    shutil.copytree(src, dst)
    return tmp_path / "data"


# --------------------------------------------------------------------------------------------
# Snapshot loading and integrity
# --------------------------------------------------------------------------------------------


def test_list_snapshots_includes_the_vendored_one():
    assert SNAPSHOT in bio.list_snapshots()


def test_load_snapshot_verifies_against_manifest(snapshot):
    entry = snapshot.manifest["files"][snapshot.manifest["table"]]
    assert snapshot.sha256 == entry["sha256"]
    assert snapshot.origin == "vendored"
    assert snapshot.table.shape == (entry["rows"], entry["columns"])
    assert pd.api.types.is_datetime64_any_dtype(snapshot.table[bio.DATE_COLUMN])


def test_provenance_record_is_json_serialisable_and_complete(snapshot):
    record = snapshot.provenance()
    json.dumps(record)
    for key in (
        "snapshot",
        "table",
        "sha256",
        "bytes",
        "rows",
        "columns",
        "verified_against_manifest",
        "origin",
        "source",
        "upstream",
        "loaded_at",
        "benchprobe",
    ):
        assert key in record, key
    assert record["source"]["commit"].startswith("946ce845")
    assert {u.get("sha256") for u in record["upstream"]} >= {
        "6f19f8f08befaaf14cb2d9952e379404c39f246b52d7ae4d1c2aa44795404de6",
        "5e7e1550b5ba5a76efd35d2701a03250832514e81b6244de8e2d5fe5493b7671",
    }


def test_snapshot_file_returns_verified_paths_only(snapshot):
    path = snapshot.file("aa_analysis_models.csv")
    assert path.is_file() and path.parent == snapshot.folder
    with pytest.raises(KeyError):
        snapshot.file("not_listed.csv")


def test_multi_file_snapshot_verifies_every_listed_file():
    judge = bio.load_snapshot("judge_2026-08-31")
    assert set(judge.manifest["files"]) == {"judge_item_bank.csv", "sweep_grid.json"}
    assert judge.file("sweep_grid.json").is_file() and judge.table.shape == (210, 38)


def test_corrupted_table_raises(snapshot_copy):
    path = snapshot_copy / SNAPSHOT / "aa_analysis_models.csv"
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01  # flip one bit; size unchanged, hash changes
    path.write_bytes(bytes(data))
    with pytest.raises(bio.SnapshotIntegrityError, match="hashes to"):
        bio.load_snapshot(SNAPSHOT, data_dir=snapshot_copy)


def test_missing_listed_file_raises(snapshot_copy):
    (snapshot_copy / SNAPSHOT / "aa_analysis_models.csv").unlink()
    with pytest.raises(bio.SnapshotIntegrityError, match="missing"):
        bio.load_snapshot(SNAPSHOT, data_dir=snapshot_copy)


def test_manifest_name_mismatch_raises(snapshot_copy):
    manifest_path = snapshot_copy / SNAPSHOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "something_else"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(bio.SnapshotIntegrityError, match="does not match folder"):
        bio.load_snapshot(SNAPSHOT, data_dir=snapshot_copy)


def test_unknown_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bio.load_snapshot("no_such_snapshot", data_dir=tmp_path)


def test_env_var_redirects_the_search(snapshot_copy, monkeypatch):
    monkeypatch.setenv(bio.ENV_DATA_DIR, str(snapshot_copy))
    loaded = bio.load_snapshot(SNAPSHOT)
    assert loaded.origin == f"env:{bio.ENV_DATA_DIR}"
    assert loaded.sha256 == bio.load_snapshot(SNAPSHOT, data_dir=snapshot_copy).sha256


# --------------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------------


def test_unknown_grid_raises(snapshot):
    with pytest.raises(ValueError, match="known grids"):
        bio.build_grid(snapshot, "no_such_grid")


@pytest.mark.parametrize("name", bio.GRIDS)
def test_grid_is_well_formed(snapshot, name):
    g = bio.build_grid(snapshot, name)
    assert g.name == name
    assert g.n == len(g.frame) == len(g.z) == len(g.days)
    assert list(g.z.columns) == list(g.benchmarks)
    assert set(g.labels) == set(g.blocks) == set(g.benchmarks)
    assert all(b in bio.PRIMARY_BENCHMARKS for b in g.benchmarks)
    if name != "economic_dense":
        assert not g.z.isna().any().any()
        assert not np.isnan(g.days.astype(float)).any()
        np.testing.assert_allclose(g.z.mean().to_numpy(), 0, atol=1e-12)
        np.testing.assert_allclose(g.z.std(ddof=0).to_numpy(), 1, atol=1e-12)


def test_complete_case_grid_keeps_archive_benchmark_order(snapshot):
    g = bio.build_grid(snapshot, "complete_case")
    assert g.benchmarks == bio.PRIMARY_BENCHMARKS
    assert g.blocks["gdpval_elo"] == "Economic" and g.labels["tauBanking"] == "τ³-Banking"


def test_days_count_from_the_earliest_release_in_the_full_snapshot(snapshot):
    g = bio.build_grid(snapshot, "complete_case")
    reference = snapshot.table[bio.DATE_COLUMN].min()
    expected = (g.frame[bio.DATE_COLUMN] - reference).dt.days.to_numpy()
    np.testing.assert_array_equal(g.days, expected)
    assert g.days.min() > 0  # the earliest model in the snapshot is not on the complete-case grid


def test_dense9_excludes_the_three_sparse_economic_benchmarks(snapshot):
    g = bio.build_grid(snapshot, "dense9")
    assert set(g.benchmarks) == set(bio.PRIMARY_BENCHMARKS) - {
        "gdpval_elo",
        "terminalbenchV21",
        "tauBanking",
    }


def test_compute_known_is_the_complete_case_rows_with_parameters(snapshot):
    complete = bio.build_grid(snapshot, "complete_case").frame
    known = bio.build_grid(snapshot, "compute_known").frame
    assert set(known["slug"]) == set(complete.loc[complete["totalParameters"].notna(), "slug"])


def test_economic_dense_contains_the_complete_case_grid(snapshot):
    """Open item 1: the two grids are distinct objects; the smaller is nested in the larger."""
    complete = set(bio.build_grid(snapshot, "complete_case").frame["slug"])
    dense = set(bio.build_grid(snapshot, "economic_dense").frame["slug"])
    assert complete < dense


# --------------------------------------------------------------------------------------------
# Deduplication (archive R1 block, verbatim behaviour)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Model X (Reasoning)", "Model X"),
        ("Model X (Non-reasoning)", "Model X"),
        ("Model X high", "Model X"),
        ("Model X (Preview)", "Model X"),
        ("Model X (medium effort)", "Model X"),
        ("Model X", "Model X"),
    ],
)
def test_base_model_key(name, key):
    assert bio.base_model_key(name) == key


def test_deduplicated_rows_are_unique_by_base_key(snapshot):
    d = bio.deduplicate_by_base_model(snapshot.table)
    assert d["base"].is_unique
    assert len(d) == snapshot.table["name"].map(bio.base_model_key).nunique()


def test_deduplication_keeps_the_top_intelligence_index_per_base(snapshot):
    d = bio.deduplicate_by_base_model(snapshot.table).set_index("base")
    t = snapshot.table.assign(base=snapshot.table["name"].map(bio.base_model_key))
    top = t.sort_values("intelligenceIndex", ascending=False).groupby("base")["intelligenceIndex"]
    top_ii = top.first()
    pd.testing.assert_series_equal(
        d["intelligenceIndex"].sort_index(), top_ii.sort_index(), check_names=False
    )


def test_deduplication_is_first_non_null_per_column_as_archived(snapshot):
    """The archived ``groupby(...).first()`` fills a benchmark the top configuration lacks from a
    lower-ranked configuration of the same base model. Reproduced, and counted, so the paper's
    "one row per base model" wording can be checked against it (docs/decisions.md, T2)."""
    t = snapshot.table.assign(base=snapshot.table["name"].map(bio.base_model_key))
    t = t.sort_values("intelligenceIndex", ascending=False)
    strict_top = t.groupby("base", as_index=False).head(1)
    archived = bio.deduplicate_by_base_model(snapshot.table)
    primary = list(bio.PRIMARY_BENCHMARKS)
    n_strict = len(strict_top.dropna(subset=primary))
    n_archived = len(archived.dropna(subset=primary))
    assert n_archived >= n_strict
    assert (n_strict, n_archived) == (86, 89), (n_strict, n_archived)


# --------------------------------------------------------------------------------------------
# Coverage rules (archive Phase 1); the vendored table is already filtered, so it is a fixed point
# --------------------------------------------------------------------------------------------


def test_coverage_rules_are_a_fixed_point_on_the_vendored_table(snapshot):
    benchmarks = list(bio.PRIMARY_BENCHMARKS) + ["mmmuPro"]
    audit = bio.benchmark_coverage(snapshot.table, benchmarks, min_models=60)
    assert audit["kept"].all()
    kept = bio.model_coverage(snapshot.table, benchmarks, min_benchmarks=8)
    assert len(kept) == len(snapshot.table)
    assert (kept["n_bench"] == snapshot.table["n_bench"].to_numpy()).all()


def test_zscore_rejects_constant_columns():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match="zero-variance"):
        bio.zscore(frame, ["a", "b"])
    z = bio.zscore(frame, ["a"])
    np.testing.assert_allclose(z["a"].to_numpy(), [-1.2247449, 0.0, 1.2247449])


def test_deduplicated_frame_keeps_base_as_its_last_column(snapshot):
    frame = bio.build_grid(snapshot, "deduplicated").frame
    assert frame.columns[-1] == "base"
    assert list(frame.columns[:-1]) == [c for c in snapshot.table.columns]


def test_coverage_rules_drop_what_they_should():
    table = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan], "b": [1.0, np.nan, np.nan], "c": [np.nan, np.nan, np.nan]}
    )
    audit = bio.benchmark_coverage(table, ["a", "b", "c"], min_models=2)
    assert audit.set_index("benchmark")["kept"].to_dict() == {"a": True, "b": False, "c": False}
    kept = bio.model_coverage(table, ["a", "b", "c"], min_benchmarks=2)
    assert kept.index.tolist() == [0] and kept.loc[0, "n_bench"] == 2
