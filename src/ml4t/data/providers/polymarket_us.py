"""Polymarket US (QCX LLC) public market data provider.

Polymarket US is the CFTC-regulated exchange (DCM and DCO) operated by QCX LLC d/b/a Polymarket
US. It is a separate venue from the global, crypto-settled Polymarket served by
``PolymarketProvider``: separate markets, order books and identifiers (market slugs).

The public gateway ``https://gateway.polymarket.us`` needs no API key:

- ``GET /v1/markets``: offset-paginated market list with status, ``outcomes`` and final
  ``outcomePrices`` (page size capped at 500 by the server).
- ``GET /v1/markets/{slug}/settlement``: settlement price of a resolved market.
- ``GET /v1/price-history``: book-derived YES (``longPrice``) and NO (``shortPrice``) display
  prices, 20 requests/second/IP. These are not trades; no public trade history exists.

Example:
    >>> from ml4t.data.providers.polymarket_us import PolymarketUSProvider
    >>> provider = PolymarketUSProvider()
    >>> resolved = provider.fetch_markets(closed=True, categories=["macro"])
    >>> history = provider.fetch_price_history(resolved["ticker"][0])
    >>> provider.close()
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar

import polars as pl
import structlog

from ml4t.data.core.exceptions import DataValidationError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.prediction_markets import (
    RESOLUTION_SCHEMA,
    UTC_DATETIME,
    empty_frame,
    parse_utc_timestamp,
    result_from_payout,
    to_unix_seconds,
)

logger = structlog.get_logger()

TimestampArg = int | float | str | date | datetime | None

RESOLVED_STATUS = "MARKET_STATUS_RESOLVED"

MARKET_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.Utf8(),
    "market_id": pl.Utf8(),
    "title": pl.Utf8(),
    "question": pl.Utf8(),
    "category": pl.Utf8(),
    "market_type": pl.Utf8(),
    "status": pl.Utf8(),
    "start_date": UTC_DATETIME,
    "end_date": UTC_DATETIME,
    **RESOLUTION_SCHEMA,
    "yes_outcome": pl.Utf8(),
    "winning_outcome": pl.Utf8(),
    "outcomes": pl.List(pl.Utf8()),
    "outcome_prices": pl.List(pl.Float64()),
    "fee_coefficient": pl.Float64(),
}

PRICE_HISTORY_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": UTC_DATETIME,
    "ticker": pl.Utf8(),
    "long_price": pl.Float64(),
    "short_price": pl.Float64(),
}


class PolymarketUSProvider(BaseProvider):
    """Polymarket US markets, resolution outcomes and price history from the public gateway."""

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (20, 1.0)
    BASE_URL: ClassVar[str] = "https://gateway.polymarket.us"
    MAX_PAGE_SIZE: ClassVar[int] = 500
    MAX_WINDOW: ClassVar[timedelta] = timedelta(hours=24)

    def __init__(self, rate_limit: tuple[int, float] | None = None):
        """Initialize the provider.

        Args:
            rate_limit: Optional (calls, period_seconds) override.
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "polymarket_us"

    def iter_markets(
        self,
        closed: bool | None = None,
        active: bool | None = None,
        categories: Sequence[str] | None = None,
        slugs: Sequence[str] | None = None,
        end_date_min: TimestampArg = None,
        end_date_max: TimestampArg = None,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over every market matching the filters (offset pagination).

        Args:
            closed: Filter on the ``closed`` flag (True for finished markets).
            active: Filter on the ``active`` flag.
            categories: API category values (``"sports"``, ``"macro"``, ``"politics"``, ...).
            slugs: Restrict to these market slugs.
            end_date_min: Keep markets ending at or after this time.
            end_date_max: Keep markets ending at or before this time.
            page_size: Markets per request (1-500).
            max_pages: Optional cap on pages, for sampling.

        Yields:
            Raw market dictionaries from ``GET /v1/markets``.
        """
        limit = max(1, min(int(page_size), self.MAX_PAGE_SIZE))
        params: dict[str, Any] = {"limit": limit}
        if closed is not None:
            params["closed"] = str(closed).lower()
        if active is not None:
            params["active"] = str(active).lower()
        if categories:
            params["categories"] = list(categories)
        if slugs:
            params["slug"] = list(slugs)
        for key, value in (("endDateMin", end_date_min), ("endDateMax", end_date_max)):
            seconds = to_unix_seconds(value)
            if seconds is not None:
                params[key] = datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        offset = 0
        pages = 0
        while True:
            payload = self._request_json(
                f"{self.BASE_URL}/v1/markets",
                resource="markets",
                params={**params, "offset": offset},
            )
            markets = payload.get("markets")
            if not isinstance(markets, list):
                raise DataValidationError(
                    provider=self.name,
                    message="Markets response has no 'markets' list",
                    field="markets",
                )
            if not markets:
                return
            for market in markets:
                if not isinstance(market, dict):
                    raise DataValidationError(
                        provider=self.name,
                        message="Non-object entry in 'markets'",
                        field="markets",
                    )
                yield market
            pages += 1
            # Advance by what the server returned: it caps the page size silently.
            offset += len(markets)
            if max_pages is not None and pages >= max_pages:
                return

    def fetch_markets(
        self,
        closed: bool | None = None,
        active: bool | None = None,
        categories: Sequence[str] | None = None,
        slugs: Sequence[str] | None = None,
        end_date_min: TimestampArg = None,
        end_date_max: TimestampArg = None,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> pl.DataFrame:
        """Return every matching market with its normalized resolution outcome.

        Accepts the same filters as ``iter_markets``.

        Every market is one instrument: the long market side is the YES contract and the
        short side the NO contract. For a market with status ``MARKET_STATUS_RESOLVED`` the
        final long-side price in ``marketSides`` is the payout per YES contract. The
        ``outcomes`` labels are not always listed in the order of ``outcomePrices`` (e.g.
        ``["No","Yes"]`` with long-first prices), so they are never zipped to infer a winner.

        Returns:
            DataFrame with ``MARKET_SCHEMA`` columns, including ``ticker`` (slug), ``status``,
            ``result`` (``"yes"``/``"no"``, ``"other"`` for a split payout such as 50-50 on a
            cancelled game, null until resolved), ``settlement_value`` (YES payout),
            ``settlement_ts`` (always null: the gateway publishes no settlement time),
            ``yes_outcome`` (name of the long side, e.g. a team) and ``winning_outcome``.
        """
        rows = [
            self._normalize_market(market)
            for market in self.iter_markets(
                closed=closed,
                active=active,
                categories=categories,
                slugs=slugs,
                end_date_min=end_date_min,
                end_date_max=end_date_max,
                page_size=page_size,
                max_pages=max_pages,
            )
        ]
        if not rows:
            return empty_frame(MARKET_SCHEMA)
        return pl.DataFrame(rows, schema=MARKET_SCHEMA, orient="row")

    @staticmethod
    def _json_list(value: Any, field: str) -> list[Any]:
        """Decode a field that the gateway serializes as a JSON-encoded list."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list):
            raise ValueError(f"{field} is not a list")
        return value

    def _normalize_market(self, market: dict[str, Any]) -> tuple[Any, ...]:
        slug = market.get("slug")
        if not slug:
            raise DataValidationError(
                provider=self.name, message="Market entry without a slug", field="slug"
            )
        try:
            outcomes = [str(item) for item in self._json_list(market.get("outcomes"), "outcomes")]
            prices = [
                float(item)
                for item in self._json_list(market.get("outcomePrices"), "outcomePrices")
            ]
            sides = [side for side in market.get("marketSides") or [] if isinstance(side, dict)]
            long_side = next((side for side in sides if side.get("long") is True), None)
            short_side = next((side for side in sides if side.get("long") is False), None)
            yes_outcome = (
                str(long_side["description"])
                if long_side and long_side.get("description")
                else (outcomes[0] if outcomes else None)
            )
            status = market.get("status")
            payout: float | None = None
            if status == RESOLVED_STATUS:
                # outcomePrices is long-side first, but the outcomes labels are not always in
                # the same order, so the payout comes from the long market side when present.
                if long_side is not None and long_side.get("price") not in (None, ""):
                    payout = float(long_side["price"])
                elif prices:
                    payout = prices[0]
                else:
                    raise ValueError("resolved market without final prices")
            result = result_from_payout(payout)
            winning_outcome = None
            if result == "yes":
                winning_outcome = yes_outcome
            elif result == "no" and short_side is not None and short_side.get("description"):
                winning_outcome = str(short_side["description"])
            fee = market.get("feeCoefficient")
            row = (
                str(slug),
                None if market.get("id") is None else str(market.get("id")),
                market.get("title"),
                market.get("question"),
                market.get("category"),
                market.get("marketType"),
                status,
                parse_utc_timestamp(market.get("startDate")),
                parse_utc_timestamp(market.get("endDate")),
                result,
                payout,
                None,
                yes_outcome,
                winning_outcome,
                outcomes,
                prices,
                None if fee is None else float(fee),
            )
        except (TypeError, ValueError) as err:
            raise DataValidationError(
                provider=self.name,
                message=f"Malformed market record for {slug}: {err}",
                field="markets",
                value=slug,
            ) from err
        return row

    def get_settlement(self, slug: str) -> float | None:
        """Return the settlement price of a market from ``/v1/markets/{slug}/settlement``.

        Returns:
            The settlement price (the YES payout), or None when the gateway answers 404,
            which it does for both unknown and not-yet-settled markets.
        """
        try:
            payload = self._request_json(
                f"{self.BASE_URL}/v1/markets/{slug}/settlement",
                resource=f"settlement for {slug}",
            )
        except SymbolNotFoundError:
            return None
        value = payload.get("settlement")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as err:
            raise DataValidationError(
                provider=self.name,
                message=f"Non-numeric settlement for {slug}: {value!r}",
                field="settlement",
                value=value,
            ) from err

    def fetch_price_history(
        self,
        slug: str,
        start: TimestampArg = None,
        end: TimestampArg = None,
    ) -> pl.DataFrame:
        """Return the book-derived YES/NO display-price history of one market.

        Without ``start`` and ``end`` the full history is requested
        (``fixedInterval=INTERVAL_ALL``, three-hourly points for newer markets and daily
        points for longer ones). With both, the range is fetched in 24-hour windows at
        one-minute fidelity, the gateway's limit for custom ranges.

        Args:
            slug: Market slug.
            start: Range start (Unix seconds, ISO string, date or datetime; naive is UTC).
            end: Range end.

        Returns:
            DataFrame with ``PRICE_HISTORY_SCHEMA`` columns: ``timestamp`` (UTC), ``ticker``,
            ``long_price`` (YES display price, from the best ask) and ``short_price`` (NO
            display price, one minus the best bid). These are quotes, not trades.

        Raises:
            DataValidationError: Only one of ``start`` and ``end`` is given, or end < start.
        """
        start_seconds = to_unix_seconds(start)
        end_seconds = to_unix_seconds(end)
        url = f"{self.BASE_URL}/v1/price-history"
        resource = f"price history for {slug}"
        points: list[dict[str, Any]] = []
        if start_seconds is None and end_seconds is None:
            payload = self._request_json(
                url,
                resource=resource,
                params={"symbol": slug, "fixedInterval": "INTERVAL_ALL", "fidelity": 180},
            )
            points.extend(self._history_points(payload, slug))
        elif start_seconds is None or end_seconds is None or end_seconds < start_seconds:
            raise DataValidationError(
                provider=self.name,
                message="fetch_price_history needs both start and end, with end >= start",
                field="start",
            )
        else:
            window = int(self.MAX_WINDOW.total_seconds())
            window_start = start_seconds
            while window_start < end_seconds:
                window_end = min(window_start + window, end_seconds)
                payload = self._request_json(
                    url,
                    resource=resource,
                    params={
                        "symbol": slug,
                        "timestamp.startTimestamp": window_start,
                        "timestamp.endTimestamp": window_end,
                        "fidelity": 1,
                    },
                )
                points.extend(self._history_points(payload, slug))
                window_start = window_end

        if not points:
            return empty_frame(PRICE_HISTORY_SCHEMA)
        frame = pl.DataFrame(
            {
                "timestamp": [point["timestamp"] for point in points],
                "ticker": [slug] * len(points),
                "long_price": [point["long_price"] for point in points],
                "short_price": [point["short_price"] for point in points],
            },
            schema=PRICE_HISTORY_SCHEMA,
        )
        return frame.unique(subset=["timestamp"], keep="first").sort("timestamp")

    def _history_points(self, payload: dict[str, Any], slug: str) -> list[dict[str, Any]]:
        history = payload.get("history")
        if history is None:
            return []
        if not isinstance(history, list):
            raise DataValidationError(
                provider=self.name,
                message=f"Price history for {slug} is not a list",
                field="history",
            )
        points: list[dict[str, Any]] = []
        for point in history:
            try:
                points.append(
                    {
                        "timestamp": datetime.fromtimestamp(int(point["timestamp"]), UTC),
                        "long_price": _optional_float(point.get("longPrice")),
                        "short_price": _optional_float(point.get("shortPrice")),
                    }
                )
            except (KeyError, TypeError, ValueError) as err:
                raise DataValidationError(
                    provider=self.name,
                    message=f"Malformed price-history point for {slug}: {point!r}",
                    field="history",
                ) from err
        return points

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self._log_close_event("Closed Polymarket US client")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
