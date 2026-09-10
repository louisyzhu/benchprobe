"""benchprobe — analysis code behind benchmark-validity studies.

Three families of function, one per programme layer: ``trust`` (reliability and grader
agreement), ``measure`` (factor structure and incremental validity; ``irt`` sits under it),
``predict`` (held-out designs and uncertainty for criterion studies), with ``io`` and ``report``
serving all three. Scope is fixed by ``SPEC.md``; the locked acceptance numbers live in
``tests/golden``. Extracted from archived code where the archive ships it; reconstructed, and
labelled as such, where it does not (``docs/decisions.md``).

v0.1: validated against three published studies; the door for other data is ``benchprobe.scores``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("benchprobe")
except PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

MODULES = ("trust", "measure", "predict", "irt", "io", "report")
"""The modules SPEC.md §1 defines; the smoke test imports each of them."""

__all__ = ["MODULES", "__version__"]
