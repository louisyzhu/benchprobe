"""Unit tests for benchprobe.report (ticket T7). The quick path (small bootstraps) is in the smoke
set; the full comparison against the archive is a golden test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchprobe.io as bio
import benchprobe.report as r

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def snapshot():
    return bio.load_snapshot()


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory, snapshot):
    out = tmp_path_factory.mktemp("oc_quick")
    return r.reproduce_one_capability(out, quick=True, snapshot=snapshot)


def test_captions_state_the_tier(snapshot):
    for tier in (r.RECOMPUTED, r.STATISTICAL, r.REGENERATED):
        text = r.caption("x.csv", tier, study="s", snapshot=snapshot, deviation=0.1, tolerance=1)
        assert tier in text and snapshot.name in text
    with pytest.raises(ValueError, match="unknown tier"):
        r.caption("x.csv", "guessed", study="s", snapshot=snapshot)


def test_compare_joins_and_measures():
    a = pd.DataFrame({"key": ["p", "q"], "v": [1.0, 2.0], "w": [0.5, 0.5]})
    b = pd.DataFrame({"key": ["q", "p"], "v": [2.001, 1.0], "w": [0.5, 0.4]})
    worst, dev = r.compare(b, a, keys=["key"])
    assert worst == pytest.approx(0.1) and set(dev.columns) == {"key", "v", "w"}
    with pytest.raises(ValueError, match="one to one"):
        r.compare(b.iloc[:1], a, keys=["key"])


def test_interval_agreement_scales_by_monte_carlo_spread():
    reruns = np.array([[0.10, 0.50], [0.11, 0.52], [0.09, 0.48], [0.10, 0.50]])
    z, per = r.interval_agreement([0.10, 0.56], [0.10, 0.50], reruns, rounding=0.0)
    assert per[0] == 0 and z == pytest.approx(0.06 / reruns[:, 1].std(ddof=1))
    # NaN anywhere is a failure, never agreement
    assert r.interval_agreement([np.nan, 0.5], [0.1, 0.5], reruns, rounding=0.0)[0] == np.inf
    assert r.interval_agreement([0.1, 0.5], [0.1, np.nan], reruns, rounding=0.0)[0] == np.inf
    assert r.interval_agreement([0.1, 0.5], [0.1, 0.5], reruns * np.nan, rounding=0.0)[0] == np.inf


def test_compare_treats_nan_as_failure():
    a = pd.DataFrame({"key": ["p", "q"], "v": [1.0, 2.0]})
    b = pd.DataFrame({"key": ["p", "q"], "v": [1.0, np.nan]})
    worst, _ = r.compare(b, a, keys=["key"])
    assert np.isnan(worst) and not (worst <= 1.0)


def test_outside_tolerance_caption_says_so(snapshot):
    text = r.caption(
        "x.csv",
        r.RECOMPUTED,
        study="s",
        snapshot=snapshot,
        deviation=0.5,
        tolerance=0.1,
        within=False,
    )
    assert "NOT reproduced within tolerance" in text and "not adjusted" in text


def test_config_defaults_and_unknown_keys(tmp_path):
    assert r.load_config(None) == r.default_config()
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"k": 4}), encoding="utf-8")
    assert r.load_config(cfg)["k"] == 4
    cfg.write_text(json.dumps({"n_factors": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        r.load_config(cfg)


def test_quick_run_writes_twelve_tables_with_captions(quick_run):
    out = Path(quick_run.out_dir)
    names = [t.name for t in quick_run.tables]
    assert names == list(r.ONE_CAPABILITY_TABLES)
    for t in quick_run.tables:
        assert (out / t.name).is_file()
        assert t.tier in t.caption and t.name in t.caption
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert all(t.name in captions for t in quick_run.tables)
    assert (out / "comparison.csv").is_file() and (out / "provenance.json").is_file()
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["config"]["quick"] is True and prov["snapshot"]["verified_against_manifest"]


def test_quick_run_deterministic_tables_match_the_archive(quick_run):
    """Tables that do not depend on a bootstrap are exact even on the quick path."""
    by_name = {t.name: t for t in quick_run.tables}
    for name in (
        "loadings_raw.csv",
        "loadings_residualised.csv",
        "task2_error_analysis_gdpval.csv",
        "efa_factor_correlations.csv",
        "efa_uniquenesses.csv",
        "eda_distribution_stats.csv",
        "k_selection_evidence.csv",
    ):
        t = by_name[name]
        assert t.tier == r.RECOMPUTED and t.within_tolerance, (name, t.max_abs_deviation)
    assert by_name["cluster_validation.csv"].tier == r.REGENERATED
    assert by_name["lobo_rung_summary.csv"].tier == r.REGENERATED  # fast path
    for name in ("h4_bootstrap_dmse.csv", "ksweep_rung_iv.csv"):  # not judged on the quick path
        assert by_name[name].within_tolerance is None
        assert "not judged" in " ".join(by_name[name].notes)


def test_cli_wires_config_and_flags(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake(out_dir, **kwargs):
        calls["out_dir"] = str(out_dir)
        calls.update(kwargs)
        return r.RunRecord(
            study="s",
            out_dir=str(out_dir),
            tables=[],
            provenance={},
            config={},
            started_at="",
            finished_at="",
        )

    monkeypatch.setattr(r, "reproduce_one_capability", fake)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"bootstrap_B": 500, "out_dir": "ignored"}), encoding="utf-8")
    code = r.main(
        ["one-capability", "--out", str(tmp_path / "cli"), "--config", str(cfg), "--full-ladder"]
    )
    assert code == 0 and "written to" in capsys.readouterr().out
    assert calls["out_dir"] == str(tmp_path / "cli")
    assert calls["full_ladder"] is True and calls["bootstrap_B"] == 500 and calls["quick"] is False
    assert calls["snapshot"].name == bio.DEFAULT_SNAPSHOT
