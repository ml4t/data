"""Shared normalization for prediction-market providers.

Kalshi, ForecastEx and Polymarket US publish market outcomes and trades under different field
names and encodings. The helpers here give their adapters one resolution vocabulary and one
trade schema, so research code can stack markets from several venues.

Normalized resolution columns:

- ``result``: ``"yes"`` or ``"no"`` for a binary market resolved to that side, ``"void"`` for a
  market cancelled without a winning side, ``"other"`` for a resolved market whose payout is
  neither 1 nor 0 (scalar or split settlement), and null while the market is unresolved.
- ``settlement_value``: payout of one YES contract in dollars (1.0, 0.0 or a fraction).
- ``settlement_ts``: UTC time the venue settled the market, when the venue publishes it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import polars as pl

RESULT_YES = "yes"
RESULT_NO = "no"
RESULT_VOID = "void"
RESULT_OTHER = "other"
RESULT_VALUES = frozenset({RESULT_YES, RESULT_NO, RESULT_VOID, RESULT_OTHER})

UTC_DATETIME = pl.Datetime("us", "UTC")

RESOLUTION_SCHEMA: dict[str, pl.DataType] = {
    "result": pl.Utf8(),
    "settlement_value": pl.Float64(),
    "settlement_ts": UTC_DATETIME,
}

TRADE_SCHEMA: dict[str, pl.DataType] = {
    "trade_id": pl.Utf8(),
    "ticker": pl.Utf8(),
    "timestamp": UTC_DATETIME,
    "price": pl.Float64(),
    "count": pl.Float64(),
    "taker_side": pl.Utf8(),
    "is_block_trade": pl.Boolean(),
}

_RAW_RESULT_MAP = {
    "yes": RESULT_YES,
    "no": RESULT_NO,
    "void": RESULT_VOID,
    "voided": RESULT_VOID,
    "cancelled": RESULT_VOID,
    "canceled": RESULT_VOID,
    "scalar": RESULT_OTHER,
}


def result_from_label(raw: Any) -> str | None:
    """Map a venue's result label to the normalized vocabulary.

    Args:
        raw: Venue label such as ``"yes"``, ``"no"``, ``"scalar"`` or ``""``.

    Returns:
        A value in ``RESULT_VALUES``, or None for an empty or missing label (unresolved).
        An unrecognized non-empty label maps to ``"other"`` so a resolved market is never
        reported as unresolved.
    """
    if raw is None:
        return None
    label = str(raw).strip().lower()
    if not label:
        return None
    return _RAW_RESULT_MAP.get(label, RESULT_OTHER)


def result_from_payout(payout: float | None) -> str | None:
    """Map the final payout of one YES contract to the normalized vocabulary.

    Args:
        payout: Dollars paid per YES contract at settlement, or None if not settled.

    Returns:
        ``"yes"`` for 1.0, ``"no"`` for 0.0, ``"other"`` for any other payout, None for None.
    """
    if payout is None:
        return None
    if abs(payout - 1.0) < 1e-9:
        return RESULT_YES
    if abs(payout) < 1e-9:
        return RESULT_NO
    return RESULT_OTHER


def parse_utc_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp (``Z`` or offset suffix) into an aware UTC datetime.

    Returns None for None or an empty string.

    Raises:
        ValueError: The value is not an ISO-8601 timestamp.
    """
    if value is None or value == "":
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def to_unix_seconds(value: int | float | str | date | datetime | None) -> int | None:
    """Convert a timestamp argument to Unix seconds.

    Accepts Unix seconds, an ISO date or datetime string, or a date/datetime object. Naive
    datetimes and dates are interpreted as UTC.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Timestamp must not be a boolean")
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp())


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return an empty DataFrame with the given schema."""
    return pl.DataFrame(schema=schema)
