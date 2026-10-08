"""ForecastEx event-contract data provider.

ForecastEx LLC is a CFTC-registered exchange (DCM) and clearinghouse (DCO) affiliated with
Interactive Brokers. It publishes free daily CSV files from 2024-08-01 in the public S3 bucket
``forecastex-public-data`` (also served by ``https://forecastex.com/api/download``):

- ``prices/daily_prices_YYYYMMDD.csv``: one row per contract side (YES/NO) and trading date with
  daily open/high/low/close, settlement price (the daily mark, and the final payout on the last
  row), paired quantity, open interest and VWAP.
- ``pairs/pairs_YYYYMMDD.csv``: every matched YES/NO pair, i.e. the trade history. The current
  day's file is refreshed every 10 minutes.
- ``daily_summary/summary_YYYYMMDD.csv`` (from 2025-01-01): product names and categories.

No authentication is needed and no rate limit is documented. The files carry no explicit
outcome column: a contract has expired when it is absent from the next day's prices file, and its
YES ``settlement_price`` on its last row is the payout (1.00 yes, 0.00 no) when the file was cut
after resolution. See ``ForecastExProvider.fetch_markets`` for intraday contracts.

Example:
    >>> from ml4t.data.providers.forecastex import ForecastExProvider
    >>> provider = ForecastExProvider()
    >>> markets = provider.fetch_markets("2026-09-14", "2026-09-16")
    >>> trades = provider.fetch_trades("2026-09-16", "2026-09-16", product="FFDEC")
    >>> provider.close()
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import Any, ClassVar

import polars as pl
import structlog

from ml4t.data.core.exceptions import DataValidationError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.prediction_markets import (
    RESOLUTION_SCHEMA,
    RESULT_NO,
    RESULT_YES,
    TRADE_SCHEMA,
    UTC_DATETIME,
    empty_frame,
    result_from_payout,
)

logger = structlog.get_logger()

DateArg = str | date | datetime

PRICES_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "ticker": pl.Utf8(),
    "product": pl.Utf8(),
    "side": pl.Utf8(),
    "expiration": UTC_DATETIME,
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "settlement_price": pl.Float64(),
    "volume": pl.Float64(),
    "open_interest": pl.Float64(),
    "vwap": pl.Float64(),
}

FORECASTEX_TRADE_SCHEMA: dict[str, pl.DataType] = {
    **TRADE_SCHEMA,
    "no_price": pl.Float64(),
    "product": pl.Utf8(),
    "expiration": UTC_DATETIME,
}

MARKET_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.Utf8(),
    "product": pl.Utf8(),
    "expiration": UTC_DATETIME,
    "status": pl.Utf8(),
    "first_date": pl.Date(),
    "last_date": pl.Date(),
    **RESOLUTION_SCHEMA,
    "last_close": pl.Float64(),
    "volume": pl.Float64(),
}

PRODUCTS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "product": pl.Utf8(),
    "product_name": pl.Utf8(),
    "product_category": pl.Utf8(),
    "total_pairs": pl.Int64(),
}

_PRICES_COLUMNS = (
    "event_contract",
    "subtype",
    "expiration_date",
    "date",
    "start_price",
    "high_price",
    "low_price",
    "end_price",
    "settlement_price",
    "pair_quantity",
    "open_interest",
    "vwap",
)
_PAIRS_COLUMNS = (
    "pair_id",
    "event_contract",
    "expiration_date",
    "quantity",
    "yes_price",
    "no_price",
    "pair_time",
)
_SUMMARY_COLUMNS = ("product_id", "product_name", "product_category", "total_pairs")
_OFFSET_TS_FORMAT = "%Y-%m-%dT%H:%M:%S%.f%:z"


class ForecastExProvider(BaseProvider):
    """ForecastEx daily prices, matched pairs (trades) and resolution outcomes.

    All methods take an inclusive date range of trading dates (Central Time). Days without a
    published file (before 2024-08-01, or today's prices before publication) are skipped.
    """

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (10, 1.0)
    BASE_URL: ClassVar[str] = "https://forecastex-public-data.s3.amazonaws.com"
    FIRST_DATE: ClassVar[date] = date(2024, 8, 1)

    def __init__(self, rate_limit: tuple[int, float] | None = None):
        """Initialize the provider.

        Args:
            rate_limit: Optional (calls, period_seconds) override.
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "forecastex"

    @staticmethod
    def _to_date(value: DateArg) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    def _date_range(self, start: DateArg, end: DateArg) -> list[date]:
        first = self._to_date(start)
        last = self._to_date(end)
        if last < first:
            raise DataValidationError(
                provider=self.name,
                message=f"End date {last} is before start date {first}",
                field="end",
                value=last,
            )
        first = max(first, self.FIRST_DATE)
        return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]

    def _download_csv(self, path: str, columns: tuple[str, ...]) -> pl.DataFrame | None:
        """Download one CSV as all-string columns, or None when the file does not exist."""
        try:
            response = self._request(f"{self.BASE_URL}/{path}", resource=path)
        except SymbolNotFoundError:
            return None
        try:
            frame = pl.read_csv(io.BytesIO(response.content), infer_schema_length=0)
        except (pl.exceptions.ComputeError, pl.exceptions.NoDataError) as err:
            raise DataValidationError(
                provider=self.name, message=f"Malformed CSV in {path}: {err}"
            ) from err
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise DataValidationError(
                provider=self.name,
                message=f"{path} lacks columns {missing}; got {frame.columns}",
                field="columns",
                value=frame.columns,
            )
        return frame.select(columns)

    def _cast(self, frame: pl.DataFrame, exprs: list[pl.Expr], path: str) -> pl.DataFrame:
        try:
            return frame.select(exprs)
        except pl.exceptions.PolarsError as err:
            raise DataValidationError(
                provider=self.name, message=f"Unparseable values in {path}: {err}"
            ) from err

    @staticmethod
    def _float(column: str) -> pl.Expr:
        return pl.col(column).str.strip_chars().replace("", None).cast(pl.Float64)

    @staticmethod
    def _offset_timestamp(column: str) -> pl.Expr:
        return pl.col(column).str.to_datetime(_OFFSET_TS_FORMAT, time_zone="UTC", time_unit="us")

    @staticmethod
    def _product(column: str = "event_contract") -> pl.Expr:
        return pl.col(column).str.split("_").list.first()

    def _prices_for_day(self, day: date) -> pl.DataFrame | None:
        path = f"prices/daily_prices_{day:%Y%m%d}.csv"
        raw = self._download_csv(path, _PRICES_COLUMNS)
        if raw is None:
            return None
        return self._cast(
            raw,
            [
                pl.col("date").str.to_date("%Y-%m-%d"),
                pl.col("event_contract").alias("ticker"),
                self._product().alias("product"),
                pl.col("subtype").str.to_uppercase().alias("side"),
                self._offset_timestamp("expiration_date").alias("expiration"),
                self._float("start_price").alias("open"),
                self._float("high_price").alias("high"),
                self._float("low_price").alias("low"),
                self._float("end_price").alias("close"),
                self._float("settlement_price"),
                self._float("pair_quantity").alias("volume"),
                self._float("open_interest"),
                self._float("vwap"),
            ],
            path,
        )

    def fetch_prices(
        self,
        start: DateArg,
        end: DateArg,
        product: str | None = None,
    ) -> pl.DataFrame:
        """Return daily prices for every contract side over a date range.

        Args:
            start: First trading date (inclusive), ``YYYY-MM-DD`` or date.
            end: Last trading date (inclusive).
            product: Optional product code filter (e.g. ``"FFDEC"``), the ``event_contract``
                prefix before the first underscore.

        Returns:
            DataFrame with ``PRICES_SCHEMA`` columns: ``date``, ``ticker`` (event contract),
            ``product``, ``side`` (``YES``/``NO``), ``expiration`` (UTC), ``open``, ``high``,
            ``low``, ``close``, ``settlement_price`` (daily mark; final payout on the last
            row), ``volume`` (paired contracts), ``open_interest`` and ``vwap``. High and low
            are 0.0 on days without trades.
        """
        frames = [
            frame
            for day in self._date_range(start, end)
            if (frame := self._prices_for_day(day)) is not None
        ]
        if not frames:
            return empty_frame(PRICES_SCHEMA)
        prices = pl.concat(frames, how="vertical")
        if product:
            prices = prices.filter(pl.col("product") == product.upper())
        return prices.sort(["date", "ticker", "side"])

    def fetch_trades(
        self,
        start: DateArg,
        end: DateArg,
        product: str | None = None,
        ticker: str | None = None,
    ) -> pl.DataFrame:
        """Return every matched YES/NO pair (trade) over a date range.

        Args:
            start: First file date (inclusive).
            end: Last file date (inclusive). Today's file is partial (refreshed intraday).
            product: Optional product code filter.
            ticker: Optional event-contract filter.

        Returns:
            DataFrame with ``FORECASTEX_TRADE_SCHEMA`` columns: ``trade_id`` (pair id),
            ``ticker``, ``timestamp`` (UTC), ``price`` (YES price), ``count``, ``taker_side``
            (always null: ForecastEx pairs have no aggressor side), ``is_block_trade``
            (always False), ``no_price``, ``product`` and ``expiration``.
        """
        frames: list[pl.DataFrame] = []
        for day in self._date_range(start, end):
            path = f"pairs/pairs_{day:%Y%m%d}.csv"
            raw = self._download_csv(path, _PAIRS_COLUMNS)
            if raw is None:
                continue
            frames.append(
                self._cast(
                    raw,
                    [
                        pl.col("pair_id").alias("trade_id"),
                        pl.col("event_contract").alias("ticker"),
                        self._offset_timestamp("pair_time").alias("timestamp"),
                        self._float("yes_price").alias("price"),
                        self._float("quantity").alias("count"),
                        pl.lit(None, dtype=pl.Utf8).alias("taker_side"),
                        pl.lit(False).alias("is_block_trade"),
                        self._float("no_price"),
                        self._product().alias("product"),
                        self._offset_timestamp("expiration_date").alias("expiration"),
                    ],
                    path,
                )
            )
        if not frames:
            return empty_frame(FORECASTEX_TRADE_SCHEMA)
        trades = pl.concat(frames, how="vertical")
        if product:
            trades = trades.filter(pl.col("product") == product.upper())
        if ticker:
            trades = trades.filter(pl.col("ticker") == ticker)
        return trades.unique(subset=["trade_id"], keep="first").sort(["timestamp", "trade_id"])

    def fetch_markets(
        self,
        start: DateArg,
        end: DateArg,
        product: str | None = None,
    ) -> pl.DataFrame:
        """Return one row per contract listed in the date range, with its resolution outcome.

        The prices file for the day after ``end`` is also read: a contract absent from the
        next published file has expired. When the ``settlement_price`` of its last YES row is
        1.00 or 0.00, that is the payout and the contract is ``"resolved"``. Contracts that
        expire after the daily file is cut (intraday products such as hourly temperatures)
        drop out with their last daily mark instead of the payout; they are reported as
        ``"expired"`` with a null result, because the files do not publish their outcome.
        When the next file is not yet published, contracts trading on ``end`` are ``"open"``.

        Args:
            start: First trading date (inclusive).
            end: Last trading date (inclusive).
            product: Optional product code filter.

        Returns:
            DataFrame with ``MARKET_SCHEMA`` columns: ``ticker``, ``product``,
            ``expiration`` (UTC), ``status`` (``"resolved"``, ``"expired"`` or ``"open"``),
            ``first_date`` and ``last_date`` seen in the range, ``result`` (``"yes"`` or
            ``"no"`` when resolved, otherwise null), ``settlement_value`` (YES payout when
            resolved), ``settlement_ts`` (the contract's expiration time when resolved;
            ForecastEx publishes no separate settlement timestamp), ``last_close`` and total
            ``volume`` in the range.
        """
        days = self._date_range(start, end)
        if not days:
            return empty_frame(MARKET_SCHEMA)
        frames = [frame for day in days if (frame := self._prices_for_day(day)) is not None]
        if not frames:
            return empty_frame(MARKET_SCHEMA)
        next_day = self._prices_for_day(days[-1] + timedelta(days=1))
        prices = pl.concat(frames, how="vertical").filter(pl.col("side") == "YES")
        if product:
            prices = prices.filter(pl.col("product") == product.upper())
        if prices.is_empty():
            return empty_frame(MARKET_SCHEMA)

        # A contract has finished when a later file exists and it is missing from it. Within
        # the range that is any last_date before the final day; on the final day it needs the
        # next day's file.
        last_day = days[-1]
        listed_next = set(next_day["ticker"].to_list()) if next_day is not None else set[str]()
        summary = (
            prices.sort("date")
            .group_by("ticker", maintain_order=True)
            .agg(
                pl.col("product").last(),
                pl.col("expiration").last(),
                pl.col("date").first().alias("first_date"),
                pl.col("date").last().alias("last_date"),
                pl.col("settlement_price").last().alias("final_settlement"),
                pl.col("close").last().alias("last_close"),
                pl.col("volume").sum(),
            )
        )
        rows: list[dict[str, Any]] = []
        for record in summary.iter_rows(named=True):
            finished = record["last_date"] < last_day or (
                next_day is not None and record["ticker"] not in listed_next
            )
            final = record["final_settlement"]
            result = result_from_payout(final) if finished else None
            if result not in (RESULT_YES, RESULT_NO):
                result = None
            status = "resolved" if result else ("expired" if finished else "open")
            rows.append(
                {
                    "ticker": record["ticker"],
                    "product": record["product"],
                    "expiration": record["expiration"],
                    "status": status,
                    "first_date": record["first_date"],
                    "last_date": record["last_date"],
                    "result": result,
                    "settlement_value": final if result else None,
                    "settlement_ts": record["expiration"] if result else None,
                    "last_close": record["last_close"],
                    "volume": record["volume"],
                }
            )
        return pl.DataFrame(rows, schema=MARKET_SCHEMA).sort(["last_date", "ticker"])

    def fetch_products(self, day: DateArg) -> pl.DataFrame:
        """Return product names and categories from one daily summary file.

        Summary files exist from 2025-01-01 and list the products traded that day.

        Returns:
            DataFrame with ``PRODUCTS_SCHEMA`` columns, empty when no file exists.
        """
        target = self._to_date(day)
        path = f"daily_summary/summary_{target:%Y%m%d}.csv"
        raw = self._download_csv(path, _SUMMARY_COLUMNS)
        if raw is None:
            return empty_frame(PRODUCTS_SCHEMA)
        return self._cast(
            raw,
            [
                pl.lit(target).alias("date"),
                pl.col("product_id").alias("product"),
                pl.col("product_name"),
                pl.col("product_category"),
                pl.col("total_pairs").cast(pl.Int64),
            ],
            path,
        )

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self._log_close_event("Closed ForecastEx client")
