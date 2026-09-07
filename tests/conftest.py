"""Shared pytest configuration.

Markers are declared in pyproject.toml (``smoke``, ``golden``, ``slow``) with ``--strict-markers``.
CI runs ``pytest -m smoke``. Golden tests are run by ticket and their output pasted verbatim.
"""
