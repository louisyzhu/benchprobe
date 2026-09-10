"""The input schema: a model × item score matrix, from wide or long data.

This is the door for data that is not one of the vendored studies. Every estimator in
``benchprobe.trust``, ``benchprobe.measure`` and ``benchprobe.predict`` takes a wide table (rows =
models or subjects, columns = benchmarks or items) or an array; ``benchprobe.irt`` takes long data
(one row per model × benchmark cell). :class:`ScoreMatrix` holds both views of the same data and
does the bookkeeping — coverage, the two orientations, the long form ``irt`` wants — and computes
nothing statistical itself (rule 4).

Conventions, chosen to match the three studies the package reproduces:

* scores are proportions in [0, 1] unless you say otherwise (``scale="percent"`` divides by 100);
* a missing cell is ``NaN`` in the wide view and an absent row in the long view;
* a model's ``release_date`` is optional and only needed by the date-residualisation functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ScoreMatrix"]


@dataclass(frozen=True, eq=False)
class ScoreMatrix:
    """A model × item matrix of scores in [0, 1], with optional release dates."""

    wide: pd.DataFrame
    """Rows indexed by model, one column per item, NaN where unscored."""
    release_dates: pd.Series | None = None
    """Indexed like ``wide``; datetime; optional."""

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        *,
        model: str = "model",
        item: str = "item",
        score: str = "score",
        release_date: str | None = None,
        scale: str = "proportion",
        aggregate: str = "error",
    ) -> ScoreMatrix:
        """Build from long data: one row per (model, item) cell.

        ``scale`` is ``"proportion"`` (scores already in [0, 1]) or ``"percent"`` (divided by 100).
        ``aggregate`` says what to do when a (model, item) pair appears more than once: ``"error"``
        (default — duplicates are usually a data problem, and the Price of Intelligence archive
        shows why: 148 of its 782 models carry repeated cells), ``"mean"``, ``"first"`` or
        ``"last"``.
        """
        missing = {model, item, score} - set(frame.columns)
        if missing:
            raise KeyError(f"long frame is missing columns {sorted(missing)}")
        f = frame[[model, item, score]].copy()
        f[score] = _to_proportion(f[score].to_numpy(dtype=float), scale)
        dup = f.duplicated([model, item], keep=False)
        if dup.any():
            if aggregate == "error":
                ex = f.loc[dup, [model, item]].drop_duplicates().head(3).to_string(index=False)
                raise ValueError(
                    f"{int(dup.sum())} rows share a (model, item) pair; pass aggregate='mean', "
                    f"'first' or 'last' to say what a repeated cell means. First pairs:\n{ex}"
                )
            f = f.groupby([model, item], sort=False, as_index=False)[score].agg(aggregate)
        wide = f.pivot(index=model, columns=item, values=score).sort_index()
        wide.columns.name = None
        wide.index.name = "model"
        dates = None
        if release_date is not None:
            if release_date not in frame.columns:
                raise KeyError(f"no column {release_date!r} for release dates")
            d = frame.groupby(model)[release_date].first()
            dates = pd.to_datetime(d, errors="coerce").reindex(wide.index)
            dates.name = "release_date"
        return cls(wide=wide, release_dates=dates)

    @classmethod
    def from_wide(
        cls,
        frame: pd.DataFrame,
        *,
        items: list[str] | None = None,
        release_date: str | None = None,
        scale: str = "proportion",
    ) -> ScoreMatrix:
        """Build from wide data: one row per model, one column per item.

        ``items`` names the score columns (default: every numeric column except ``release_date``).
        The index, or a column named ``model``, identifies the model.
        """
        f = frame.copy()
        if "model" in f.columns:
            f = f.set_index("model")
        f.index.name = "model"
        if items is None:
            items = [
                c for c in f.columns if c != release_date and pd.api.types.is_numeric_dtype(f[c])
            ]
        if not items:
            raise ValueError("no score columns found; pass items=[...]")
        wide = f[list(items)].astype(float)
        wide = wide.apply(lambda col: _to_proportion(col.to_numpy(dtype=float), scale))
        wide = pd.DataFrame(wide, index=f.index, columns=list(items)).sort_index()
        dates = None
        if release_date is not None:
            dates = pd.to_datetime(f[release_date], errors="coerce").reindex(wide.index)
            dates.name = "release_date"
        return cls(wide=wide, release_dates=dates)

    # ---------------------------------------------------------------- views
    @property
    def models(self) -> list[str]:
        return list(self.wide.index)

    @property
    def items(self) -> list[str]:
        return list(self.wide.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.wide.shape

    def long(self) -> pd.DataFrame:
        """One row per scored cell: ``model, item, score`` — the form ``benchprobe.irt`` takes."""
        out = (
            self.wide.reset_index()
            .melt(id_vars="model", var_name="item", value_name="score")
            .dropna(subset=["score"])
            .reset_index(drop=True)
        )
        return out

    def complete(self, items: list[str] | None = None) -> pd.DataFrame:
        """Rows scored on every one of ``items`` (default: all), as a wide table — the
        complete-case grid the factor and prediction functions want."""
        cols = list(items) if items is not None else self.items
        return self.wide.loc[self.wide[cols].notna().all(axis=1), cols]

    def coverage(self) -> pd.DataFrame:
        """Per item: how many models score it and the mean score; per model on ``.T``."""
        return pd.DataFrame(
            {
                "n_models": self.wide.notna().sum(axis=0).astype(int),
                "mean_score": self.wide.mean(axis=0),
                "min_score": self.wide.min(axis=0),
                "max_score": self.wide.max(axis=0),
            }
        )

    def grid(self, items: list[str] | None = None, *, name: str = "complete"):
        """The complete-case rows as a :class:`benchprobe.io.Grid` — what ``predict.lobo`` and the
        rung builders take. ``z`` is the population-standardised matrix (``ddof=0``, as the studies
        use); ``days`` is days since the earliest release, or zeros if there are no release dates
        (then the date rungs are meaningless and ``lobo`` should be called without them)."""
        from benchprobe.io import Grid

        cols = list(items) if items is not None else self.items
        frame = self.complete(cols).copy()
        z = (frame - frame.mean()) / frame.std(ddof=0)
        if self.release_dates is not None and self.release_dates.reindex(frame.index).notna().all():
            d = self.release_dates.reindex(frame.index)
            days = (d - d.min()).dt.days.to_numpy(dtype=int)
            frame["releaseDate"] = d.to_numpy()
        else:
            days = np.zeros(len(frame), dtype=int)
        return Grid(
            name=name,
            frame=frame.reset_index(),
            benchmarks=tuple(cols),
            labels={c: c for c in cols},
            blocks={c: "" for c in cols},
            z=z,
            days=days,
        )

    def days_since_earliest_release(self) -> np.ndarray:
        """Integer days from the earliest release date, the covariate the date-residualisation
        functions take. Requires release dates."""
        if self.release_dates is None:
            raise ValueError("this ScoreMatrix has no release dates")
        if self.release_dates.isna().any():
            raise ValueError("release dates are missing for some models")
        return (self.release_dates - self.release_dates.min()).dt.days.to_numpy(dtype=int)


def _to_proportion(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "proportion":
        out = values
    elif scale == "percent":
        out = values / 100.0
    else:
        raise ValueError(f"scale must be 'proportion' or 'percent', not {scale!r}")
    finite = out[np.isfinite(out)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError(
            f"scores must lie in [0, 1] after scaling (got {finite.min():.4g} to "
            f"{finite.max():.4g}); pass scale='percent' if they are percentages"
        )
    return out
