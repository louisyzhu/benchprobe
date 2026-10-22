"""Pre-release consistency check (ticket T9); the first step of the release workflow's test job.

Asserts, and exits non-zero naming every failure:

1. ``pyproject.toml``, the installed ``benchprobe.__version__`` and ``CITATION.cff`` carry the
   same version. ``__version__`` is read from the installed distribution's metadata, so run this
   inside the project environment (``uv run python scripts/release_check.py``).
2. The git working tree is clean (no modified, staged or untracked files).
3. ``CHANGELOG.md`` has a heading for the current version.

Standard library only; no network.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def pyproject_version() -> str:
    with (REPO / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def package_version() -> str:
    import benchprobe

    return benchprobe.__version__


def citation_version() -> str | None:
    text = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"""^version:\s*["']?([^"'\s]+)["']?\s*$""", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def dirty_paths() -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, check=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def changelog_has(version: str) -> bool:
    path = REPO / "CHANGELOG.md"
    if not path.is_file():
        return False
    pattern = rf"^##\s+\[?{re.escape(version)}\]?(\s|$)"
    return re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE) is not None


def main() -> int:
    failures = []
    expected = pyproject_version()

    for source, found in (
        ("benchprobe.__version__", package_version()),
        ("CITATION.cff version", citation_version()),
    ):
        if found != expected:
            failures.append(f"{source} is {found!r}; pyproject.toml says {expected!r}")

    dirty = dirty_paths()
    if dirty:
        failures.append("working tree is not clean:\n    " + "\n    ".join(dirty))

    if not changelog_has(expected):
        failures.append(f"CHANGELOG.md has no '## {expected}' heading")

    if failures:
        print(f"release_check: FAILED for version {expected}", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"release_check: OK — version {expected} consistent, tree clean, changelog entry present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
