"""Detect a refetch whose overlapping rows no longer match the stored rows.

Providers that serve adjusted prices restate every earlier price after a dividend or split.
Splicing such a refetch onto stored history puts the rows before the overlap on the old basis
and the rest on the new one, with a spurious return where they meet. Update paths compare the
overlap first and rewrite the whole stored range when it changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from ml4t.data.core.schemas import align_frames_for_concat

#: Relative change in an overlapping close above which the provider counts as having restated
#: the row. Repeated Yahoo retrievals of the same adjusted close differ by up to about 1.6e-6
#: relative; a dividend of 0.1% of the price changes every earlier close by 1e-3.
OVERLAP_RTOL = 1e-5


def covers_timestamps(candidate: pl.DataFrame, expected: pl.DataFrame) -> bool:
    """Whether every observed timestamp is present in the replacement response."""
    if expected.is_empty():
        return True
    if candidate.is_empty():
        return False
    left, right = align_frames_for_concat(
        expected.select("timestamp"), candidate.select("timestamp")
    )
    return left.join(right, on="timestamp", how="anti").is_empty()


@dataclass(frozen=True)
class OverlapRevision:
    """Stored rows that a refetch of the same timestamps no longer reproduces."""

    rows_compared: int
    rows_revised: int
    max_relative_change: float
    first_revised: datetime
    last_revised: datetime

    def log_fields(self) -> dict[str, Any]:
        """Return the revision as structured-log fields."""
        return {
            "rows_compared": self.rows_compared,
            "rows_revised": self.rows_revised,
            "max_relative_change": self.max_relative_change,
            "first_revised": self.first_revised.isoformat(),
            "last_revised": self.last_revised.isoformat(),
        }

    def describe(self) -> str:
        """Return a one-line description for error messages."""
        return (
            f"{self.rows_revised} of {self.rows_compared} overlapping rows changed, "
            f"by up to {self.max_relative_change:.3e} relative in close, "
            f"between {self.first_revised.isoformat()} and {self.last_revised.isoformat()}"
        )


def find_overlap_revision(
    stored: pl.DataFrame,
    refetched: pl.DataFrame,
    *,
    column: str = "close",
    rtol: float = OVERLAP_RTOL,
) -> OverlapRevision | None:
    """Compare a refetch with the stored rows that share its timestamps.

    Args:
        stored: Rows already in storage for one instrument.
        refetched: Rows the provider returned now for the same instrument.
        column: Price column to compare.
        rtol: Relative change above which a row counts as revised.

    Returns:
        The revision when any overlapping row changed by more than ``rtol``, else None.
        Rows where either value is null are not compared. A change confined to the last
        stored row is not a revision when earlier overlapping rows match: that row may have
        been a bar still forming when it was stored, and a restatement changes every row
        before the corporate action, not one.
    """
    if column not in stored.columns or column not in refetched.columns:
        return None
    if stored.is_empty() or refetched.is_empty():
        return None
    left, right = align_frames_for_concat(
        stored.select("timestamp", column), refetched.select("timestamp", column)
    )
    joined = (
        left.unique(subset=["timestamp"], keep="last")
        .join(
            right.unique(subset=["timestamp"], keep="last"),
            on="timestamp",
            suffix="_refetched",
        )
        .drop_nulls()
    )
    if joined.is_empty():
        return None
    before = pl.col(column).cast(pl.Float64)
    after = pl.col(f"{column}_refetched").cast(pl.Float64)
    change = (after - before).abs()
    relative = pl.when(before != 0).then(change / before.abs()).otherwise(change)
    compared = joined.with_columns(relative.alias("relative_change"))
    revised = compared.filter(pl.col("relative_change") > rtol).sort("timestamp")
    if revised.is_empty():
        return None
    last_stored = left["timestamp"].max()
    if compared.height > 1 and revised["timestamp"].to_list() == [last_stored]:
        return None
    return OverlapRevision(
        rows_compared=compared.height,
        rows_revised=revised.height,
        max_relative_change=float(revised["relative_change"].abs().max() or 0.0),
        first_revised=revised["timestamp"][0],
        last_revised=revised["timestamp"][-1],
    )
