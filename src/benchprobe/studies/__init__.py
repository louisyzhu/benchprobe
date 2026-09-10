"""The three studies benchprobe reproduces, each behind one function.

These are the *validation suite* of the library, kept apart from the general estimators: everything
here knows the name of a benchmark, a snapshot or an archived table, and nothing in ``trust``,
``measure``, ``predict``, ``irt`` or ``scores`` does. If you have your own data you never need this
package; if you want to check that the library reproduces a published number, start here.

* :func:`one_capability` — *One Capability or Many?* (arXiv:2608.29420): the twelve archived
  result tables, recomputed with a reproducibility tier in every caption.
* :func:`judge` — *Three Ways Classical Test Theory Misleads for LLM Judges*: the real-bank
  quantities and the simulation sweeps.
* :func:`price_of_intelligence` — *The Price of Intelligence*: the two-stage continuous-response
  fit, item parameters, abilities and convergence diagnostics.

Each returns plain objects; ``docs/reproducibility.md`` says what tier each output is at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["judge", "one_capability", "price_of_intelligence"]


def one_capability(out_dir: str | Path, *, full_ladder: bool = False, quick: bool = False):
    """Recompute the twelve archived One Capability tables into ``out_dir``.

    ``full_ladder`` refits the four-learner LOBO ladder (about 35 minutes; otherwise that table is
    copied from the archive and says so). Returns the :class:`benchprobe.report.RunRecord`.
    """
    from benchprobe.report import reproduce_one_capability

    return reproduce_one_capability(Path(out_dir), full_ladder=full_ladder, quick=quick)


def judge() -> dict[str, Any]:
    """The JUDGe real-bank summary and the archive's simulation sweeps (seeds preserved)."""
    from benchprobe.io import load_snapshot
    from benchprobe.trust import bank_summary, load_bank, sweeps

    bank = load_bank(load_snapshot("judge_2026-08-31"))
    return {"bank": bank_summary(bank), "sweeps": sweeps()}


def price_of_intelligence(*, repair_scale: bool = True):
    """The two-stage anchor-linked continuous-response fit on the frozen panel.

    ``repair_scale`` (default true) applies the documented correction to four benchmarks' scale
    metadata; with it false the published numbers do not reproduce (``docs/decisions.md``,
    2026-09-10). Returns the :class:`benchprobe.irt.TwoStageFit`; ``.summary()`` for the record,
    ``.item_parameters()`` and ``benchprobe.irt.ability_table(fit)`` for the tables.
    """
    from benchprobe.io import load_snapshot
    from benchprobe.irt import PANEL_SNAPSHOT, build_panel, estimation_spec, two_stage_link

    snap = load_snapshot(PANEL_SNAPSHOT)
    return two_stage_link(build_panel(snap, repair_scale=repair_scale), estimation_spec(snap))
