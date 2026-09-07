"""Input layer: hash-pinned snapshot loading, provenance record, coverage rules, config-driven runs.

SPEC.md §1 scope. Snapshots vendored under ``benchprobe/data/<name>/`` carry a ``MANIFEST.json``
whose SHA-256 is verified on every load. Implemented under ticket T2; signatures in SPEC.md §2.

No functions yet (ticket T0 is the skeleton).
"""

__all__: list[str] = []
