"""Golden test for the M3 gate: the twelve One Capability tables reproduced by one command, each
recomputed or statistically reproduced within a tolerance fixed *here*, or regenerated with that
stated.

The tolerances and the SHA-256 of every archived table are locked in this file (rule 1), so the
gate cannot be loosened from inside ``src/``: a tolerance change in ``benchprobe.report`` does not
pass this test, and a change to a vendored archived table breaks the hash check below before any
comparison runs. Tolerances (T7, docs/decisions.md): 5e-4 on recomputed tables after the archive's
rounding; 1e-3 on the ladder (archived at three decimals); exact on ``selected_k``; bootstrap
intervals within 3 Monte-Carlo standard deviations (20 re-runs under other seeds). The four-learner
ladder needs ``full_ladder`` and is the ``slow`` variant (about forty minutes).
"""

import hashlib

import pytest

import benchprobe.io as bio
import benchprobe.report as r

pytestmark = pytest.mark.golden

ARCHIVED_SHA256 = {  # frontier-ai-economic-validity @ 946ce845, data/processed/
    "loadings_raw.csv": "a202c72b037844d5519c1d039c661c1f6f3fb56b73bacc3b1d199356a709684d",
    "loadings_residualised.csv": "419d7484cc38fecdd78bf0950c9039ed8204252d5ec9c856ef62bbd6be7d2d1e",
    "task1_structure_results.json": (
        "6012e1640af3eaf2419a73dfd2929d47c18530407d999ebe8d2d138b5b6646b8"
    ),
    "cluster_validation.csv": "e4048ffb9c0a913493802d6630afc65753167bdea97216516c2396dcd24192af",
    "lobo_rung_summary.csv": "fd6f99cc76bec597cb4b3ed22d3bdee1d90e2634772609d1345e974413a413ce",
    "h4_bootstrap_dmse.csv": "e9131fec7413cdf7e7adff0e9efb5dddf434274956b3c732d936fc24099690b8",
    "task2_error_analysis_gdpval.csv": (
        "0498e6190a4c32518166d512c5069963d96ec49bd66489b856bd35b1bc211b95"
    ),
    "k_selection_evidence.csv": "fab4a5e39d37500d19ad578a6df6c81bfcf5cf632295fae6cca35df6dfaa43d9",
    "ksweep_rung_iv.csv": "aab5cc70214a52367dac030b50b912935b93dc691c0d05ed63194aac89ce6cfd",
    "efa_factor_correlations.csv": (
        "17ec6ae2f354e7efa33c238540e26447c3b66f8741d37df484fd43294f689c2b"
    ),
    "efa_uniquenesses.csv": "f74ce3b98f054755900c8f5e6a84b7ee04e44aef677183061657caf593b1a6b7",
    "eda_distribution_stats.csv": (
        "f4eca5f900ac5ad9d8118d14097730b5cb9683eb41f62e5bcdd7f461f2983fed"
    ),
}

TOLERANCE = {  # per table; the deviation is what benchprobe.report measures, the bound is ours
    "loadings_raw.csv": 5e-4,
    "loadings_residualised.csv": 5e-4,
    "task1_structure_results.json": 5e-4,
    "lobo_rung_summary.csv": 1e-3,
    "h4_bootstrap_dmse.csv": 3.0,  # Monte-Carlo standard deviations
    "task2_error_analysis_gdpval.csv": 5e-4,
    "k_selection_evidence.csv": 0.0,
    "ksweep_rung_iv.csv": 3.0,  # Monte-Carlo standard deviations
    "efa_factor_correlations.csv": 5e-4,
    "efa_uniquenesses.csv": 5e-4,
    "eda_distribution_stats.csv": 5e-4,
}
EXPECTED_TIER = {
    "h4_bootstrap_dmse.csv": r.STATISTICAL,
    "ksweep_rung_iv.csv": r.STATISTICAL,
    "cluster_validation.csv": r.REGENERATED,
}
FAST_PATH_REGENERATED = {"cluster_validation.csv", "lobo_rung_summary.csv"}


def test_archived_tables_are_the_archive():
    snapshot = bio.load_snapshot()
    for name, digest in ARCHIVED_SHA256.items():
        path = snapshot.file(f"archived/{name}")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def _check(rec, regenerated):
    by_name = {t.name: t for t in rec.tables}
    assert set(by_name) == set(r.ONE_CAPABILITY_TABLES)
    for name, t in by_name.items():
        if name in regenerated:
            assert t.tier == r.REGENERATED, name
            continue
        assert t.tier == EXPECTED_TIER.get(name, r.RECOMPUTED), name
        assert t.max_abs_deviation is not None and t.max_abs_deviation <= TOLERANCE[name], (
            f"{name}: {t.tier}, deviation {t.max_abs_deviation}, tolerance {TOLERANCE[name]}"
        )
        assert t.within_tolerance is True, name
        assert "DIFFERS" not in " ".join(t.notes), (name, t.notes)


def test_twelve_tables_fast_path(tmp_path):
    rec = r.reproduce_one_capability(tmp_path / "fast")
    _check(rec, FAST_PATH_REGENERATED)


@pytest.mark.slow
def test_twelve_tables_full_ladder(tmp_path):
    rec = r.reproduce_one_capability(tmp_path / "full", full_ladder=True)
    _check(rec, {"cluster_validation.csv"})
    ladder = {t.name: t for t in rec.tables}["lobo_rung_summary.csv"]
    assert "matches the archive" in " ".join(ladder.notes)
