"""Smoke test for the repository skeleton (ticket T0).

Runs in seconds. CI runs exactly the ``smoke`` marker; nothing here touches a golden number.
"""

import hashlib
import importlib
import json
import re
import subprocess
from pathlib import Path

import pytest

import benchprobe

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(benchprobe.__file__).resolve().parent


def test_version_is_a_string():
    assert isinstance(benchprobe.__version__, str) and benchprobe.__version__


@pytest.mark.parametrize("name", benchprobe.MODULES)
def test_spec_module_imports_and_declares_public_names(name):
    module = importlib.import_module(f"benchprobe.{name}")
    assert isinstance(module.__all__, list)
    for public in module.__all__:
        assert hasattr(module, public), f"benchprobe.{name}.__all__ names missing {public!r}"


def test_spec_lists_every_module():
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    for name in benchprobe.MODULES:
        assert f"`benchprobe.{name}`" in spec


RULES = (  # handoff §5, verbatim; CLAUDE.md must carry exactly these, in this order
    "Never alter a golden number, a tolerance or a seed to make a test pass. "
    "A failing golden test is a finding; report it.",
    "Never claim a result, test outcome or file state not present in this session's own output.",
    "Edits move code, never numbers. Any numerical difference from an archived table is reported "
    "with the archive value, the recomputed value and the tolerance.",
    "No new statistical methods without a ticket. No scope beyond `SPEC.md`.",
    "No secrets, API keys, tokens or local paths in the repository. Config by environment "
    "variable, documented in README.",
    "Every function that produces a reported number has a test. Every module has a smoke test "
    "that runs in under a minute on a laptop.",
    "Reproducibility is stated per output as recomputed, statistically reproduced, or "
    'regenerated. "Fully reproducible" is used only for the first.',
    "The archived-vs-recomputed boundary is visible in every table and figure caption the "
    "package emits.",
    "The environment is pinned (uv, lockfile). A fresh clone runs the smoke test by one "
    "documented command.",
    "Nothing is tagged, published to PyPI, or made public by an agent. Louis does that, on the "
    "ship date.",
)


def test_claude_md_carries_the_ten_rules_verbatim():
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    found = re.findall(r"^(\d+)\. (.+)$", text, flags=re.MULTILINE)
    assert [n for n, _ in found] == [str(i) for i in range(1, 11)], "expected rules 1-10, once each"
    for (number, actual), expected in zip(found, RULES, strict=True):
        assert actual == expected, f"rule {number} differs from the handoff text:\n{actual}"


def _manifests():
    return sorted((PACKAGE / "data").glob("*/MANIFEST.json"))


def test_at_least_one_vendored_snapshot_exists():
    assert _manifests(), "no MANIFEST.json under benchprobe/data/*/"


@pytest.mark.parametrize("manifest_path", _manifests(), ids=lambda p: p.parent.name)
def test_vendored_snapshot_matches_its_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == manifest_path.parent.name
    for filename, expected in manifest["files"].items():
        path = manifest_path.parent / filename
        assert path.is_file(), f"{path} is listed in the manifest but missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected["sha256"], (
            f"{path.name}: manifest sha256 {expected['sha256']} but file hashes to {digest}"
        )
        assert path.stat().st_size == expected["bytes"]


def _tracked_files() -> list[Path]:
    """Files git tracks, or every file under the repository when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True, text=True
        ).stdout
        paths = [REPO / name for name in out.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", ".ruff_cache", ".pytest_cache", "__pycache__"}
        paths = [p for p in REPO.rglob("*") if p.is_file() and not (set(p.parts) & skip)]
    return [p for p in paths if p.is_file()]


def test_no_local_paths_or_secrets_in_tracked_files():
    """Rule 5: no local paths, keys or tokens anywhere in the repository (every tracked file)."""
    suspicious = re.compile(
        r"(/Users/|/home/[a-z]+/|C:\\\\Users|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    this_file = Path(__file__).resolve()  # carries the patterns themselves
    files = _tracked_files()
    assert files, "no files found to scan"
    for path in files:
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not suspicious.search(text), f"suspicious path or token in {path.relative_to(REPO)}"
