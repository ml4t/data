"""Alpaca Markets historical OHLCV bars via alpaca-py.

Equities and options require API keys. Crypto historical data does not (higher
rate limits if keys are provided). See
https://alpaca.markets/sdks/python/market_data.html

Authentication:
    For stocks/options, set ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` or pass
    ``api_key`` / ``secret_key`` to :class:`AlpacaProvider`.

    Optional: set ``ALPACA_STOCK_FEED`` (e.g. ``iex``, ``sip``) when ``feed`` is not
    passed to the constructor; useful for free-tier IEX access.

Date strings ``start`` / ``end`` may include stray whitespace (e.g. ``2024-12- 31``);
it is normalized before parsing.

This module requires ``alpaca-py`` (``pip install 'ml4t-data[alpaca]'``). If the
package is not installed, ``import ml4t.data.providers.alpaca`` raises
``ImportError``; :mod:`ml4t.data.providers` still loads and exposes
``AlpacaProvider`` as ``None`` when the extra is missing.

Example:
    >>> from ml4t.data.providers.alpaca import AlpacaProvider
    >>> p = AlpacaProvider()  # crypto only, no keys
    >>> # p.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-01-31", "daily")
    >>> p2 = AlpacaProvider(api_key="...", secret_key="...")
    >>> # p2.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31", "daily")
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from typing import ClassVar, Literal

import pandas as pd
import polars as pl
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    OptionHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.requests import CryptoBarsRequest, OptionBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataValidationError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider

__all__ = ["AlpacaHistoricalProvider", "AlpacaProvider"]


def _chunks(lst: list[str], n: int):
    """Yield successive n-sized chunks from lst (same pattern as Yahoo)."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


_OPTION_RE = re.compile(r"^[A-Z]{1,5}\d{6,7}[CP]\d{8}$")
Kind = Literal["crypto", "stock", "option"]


def _normalize_iso_date(s: str) -> str:
    """Collapse whitespace in YYYY-MM-DD strings so values like ``2024-12- 31`` parse."""
    return "".join(s.split())


def _optional_stock_feed_from_env() -> DataFeed | None:
    """Resolve ``ALPACA_STOCK_FEED`` to :class:`~alpaca.data.enums.DataFeed` if set."""
    raw = os.getenv("ALPACA_STOCK_FEED", "").strip()
    if not raw:
        return None
    try:
        return DataFeed(raw.lower())
    except ValueError:
        return None


def _infer_kind(symbol: str) -> Kind:
    if "/" in symbol:
        return "crypto"
    if _OPTION_RE.match(symbol):
        return "option"
    return "stock"


def _timeframe_for(frequency: str) -> TimeFrame:
    key = frequency.lower()
    match key:
        case "minute" | "1minute":
            return TimeFrame(1, TimeFrameUnit.Minute)
        case "5minute":
            return TimeFrame(5, TimeFrameUnit.Minute)
        case "15minute":
            return TimeFrame(15, TimeFrameUnit.Minute)
        case "30minute":
            return TimeFrame(30, TimeFrameUnit.Minute)
        case "hourly" | "1hour":
            return TimeFrame(1, TimeFrameUnit.Hour)
        case "daily" | "1day":
            return TimeFrame(1, TimeFrameUnit.Day)
        case "weekly" | "1week":
            return TimeFrame(1, TimeFrameUnit.Week)
        case "monthly" | "1month":
            return TimeFrame(1, TimeFrameUnit.Month)
        case _:
            raise DataValidationError(
                "alpaca",
                f"Unsupported frequency for Alpaca: {frequency!r}",
                field="frequency",
                value=frequency,
            )


def _utc_range(start: str, end: str) -> tuple[datetime, datetime, datetime]:
    """Parse YYYY-MM-DD range to Alpaca UTC window (end exclusive for API, inclusive filter)."""
    start_n = _normalize_iso_date(start)
    end_n = _normalize_iso_date(end)
    try:
        start_utc = datetime.combine(
            datetime.strptime(start_n, "%Y-%m-%d").date(), dt_time.min, tzinfo=UTC
        )
        end_day = datetime.strptime(end_n, "%Y-%m-%d").date()
    except ValueError as e:
        raise DataValidationError(
            "alpaca",
            f"Invalid start/end date (expected YYYY-MM-DD): {e}",
            field="start/end",
            value=f"{start!r} / {end!r}",
        ) from e
    end_excl = datetime.combine(end_day + timedelta(days=1), dt_time.min, tzinfo=UTC)
    end_inclusive = datetime.combine(end_day, dt_time(23, 59, 59, 999999), tzinfo=UTC)
    return start_utc, end_excl, end_inclusive


def _bars_pandas_to_polars(
    df_pd: pd.DataFrame,
    symbol: str,
    end_inclusive: datetime,
) -> pl.DataFrame:
    """Map Alpaca ``BarSet.df`` (pandas) to ml4t OHLCV Polars schema (single symbol)."""
    if isinstance(df_pd.index, pd.MultiIndex) and "symbol" in (df_pd.index.names or []):
        df_pd = df_pd.droplevel("symbol")
    df_pd = df_pd.reset_index()
    time_col = "timestamp" if "timestamp" in df_pd.columns else df_pd.columns[0]
    out = pl.from_pandas(df_pd[[time_col, "open", "high", "low", "close", "volume"]])
    out = out.rename({time_col: "timestamp"})
    return (
        out.with_columns(
            [
                pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        .filter(pl.col("timestamp") <= pl.lit(end_inclusive))
        .with_columns(pl.lit(symbol).alias("symbol"))
        .select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        .sort("timestamp")
    )


def _bars_pandas_to_polars_batch(df_pd: pd.DataFrame, end_inclusive: datetime) -> pl.DataFrame:
    """Map Alpaca multi-symbol ``BarSet.df`` to long-format canonical Polars."""
    if df_pd.empty:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime(time_zone="UTC"),
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    df_pd = df_pd.reset_index()
    if "symbol" not in df_pd.columns:
        raise DataValidationError(
            "alpaca",
            "Expected multi-symbol Alpaca bars to include a symbol column after reset_index",
            field="bars.df",
        )
    out = pl.from_pandas(df_pd[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])
    return (
        out.with_columns(
            [
                pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
                pl.col("symbol").cast(pl.Utf8),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        .filter(pl.col("timestamp") <= pl.lit(end_inclusive))
        .select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        .sort(["symbol", "timestamp"])
    )


class AlpacaProvider(BaseProvider):
    """Alpaca historical bars; output matches :class:`BaseProvider` OHLCV contract.

    Install: ``pip install 'ml4t-data[alpaca]'``
    """

    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "minute": "1minute",
        "1minute": "1minute",
        "5minute": "5minute",
        "15minute": "15minute",
        "30minute": "30minute",
        "hourly": "1hour",
        "1hour": "1hour",
        "daily": "1day",
        "1day": "1day",
        "weekly": "1week",
        "1week": "1week",
        "monthly": "1month",
        "1month": "1month",
    }

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        feed: DataFeed | None = None,
        adjustment: Adjustment | None = None,
    ) -> None:
        """Initialize Alpaca provider.

        Args:
            api_key: Alpaca API key; defaults to ``ALPACA_API_KEY``.
            secret_key: Alpaca secret key; defaults to ``ALPACA_SECRET_KEY``.
            feed: Stock bar feed (e.g. SIP); optional. If omitted, uses ``ALPACA_STOCK_FEED``
                when set, otherwise lets the SDK default apply.
            adjustment: Corporate-action adjustment for stock bars; optional.
        """
        super().__init__(rate_limit=None)
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self._feed = feed if feed is not None else _optional_stock_feed_from_env()
        self._adjustment = adjustment
        self._crypto: CryptoHistoricalDataClient | None = None
        self._stock: StockHistoricalDataClient | None = None
        self._option: OptionHistoricalDataClient | None = None

    @property
    def name(self) -> str:
        """Return provider name."""
        return "alpaca"

    def _client(self, kind: Kind):
        if kind == "crypto":
            if self._crypto is None:
                self._crypto = CryptoHistoricalDataClient()
            return self._crypto
        if self._api_key is None or self._secret_key is None:
            raise AuthenticationError(
                "alpaca",
                "api_key and secret_key are required for stock and option bars. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
                "or pass api_key and secret_key.",
            )
        if kind == "stock":
            if self._stock is None:
                self._stock = StockHistoricalDataClient(self._api_key, self._secret_key)
            return self._stock
        if self._option is None:
            self._option = OptionHistoricalDataClient(self._api_key, self._secret_key)
        return self._option

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        kind = _infer_kind(symbol)
        tf = _timeframe_for(self.FREQUENCY_MAP.get(frequency.lower(), frequency))
        client = self._client(kind)
        start_utc, end_excl, end_inclusive = _utc_range(start, end)

        self.logger.info(
            "Fetching Alpaca bars",
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
            kind=kind,
        )

        try:
            if kind == "crypto":
                req = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_utc,
                    end=end_excl,
                )
                bars = client.get_crypto_bars(req)
            elif kind == "stock":
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_utc,
                    end=end_excl,
                    feed=self._feed,
                    adjustment=self._adjustment,
                )
                bars = client.get_stock_bars(req)
            else:
                req = OptionBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_utc,
                    end=end_excl,
                )
                bars = client.get_option_bars(req)
        except Exception as e:
            raise DataValidationError(
                "alpaca",
                f"Failed to fetch {symbol}: {e}",
                details={"symbol": symbol, "error": str(e)},
            ) from e

        df_pd = bars.df
        if df_pd.empty:
            raise SymbolNotFoundError(
                "alpaca",
                symbol,
                details={"start": start, "end": end, "frequency": frequency},
            )

        out = _bars_pandas_to_polars(df_pd, symbol, end_inclusive)
        self.logger.info("Fetched Alpaca bars", symbol=symbol, rows=len(out))
        return out

    def fetch_batch_ohlcv(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        chunk_size: int = 50,
        delay_seconds: float = 0.0,
    ) -> pl.DataFrame:
        """Fetch OHLCV for multiple symbols using Alpaca multi-symbol bar requests.

        Symbols are grouped by asset class (crypto / stock / option). Each chunk
        calls the API with ``symbol_or_symbols=[...]`` (same idea as Yahoo batch).

        Args:
            symbols: Ticker list (e.g. ``[\"AAPL\", \"MSFT\"]`` or crypto ``BTC/USD``).
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD`` (inclusive).
            frequency: Same names as :meth:`fetch_ohlcv`.
            chunk_size: Max symbols per API request per asset class.
            delay_seconds: Pause between chunks (rate limiting).

        Returns:
            Long-format Polars frame, sorted by ``symbol``, ``timestamp``.
        """
        if not symbols:
            return self._create_empty_dataframe()

        tf = _timeframe_for(self.FREQUENCY_MAP.get(frequency.lower(), frequency))
        start_utc, end_excl, end_inclusive = _utc_range(start, end)

        by_kind: dict[Kind, list[str]] = defaultdict(list)
        for s in symbols:
            by_kind[_infer_kind(s)].append(s)

        if ("stock" in by_kind or "option" in by_kind) and (
            self._api_key is None or self._secret_key is None
        ):
            raise AuthenticationError(
                "alpaca",
                "api_key and secret_key are required when batch includes "
                "stock or option symbols. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
                "or pass keys to the constructor.",
            )

        self.logger.info(
            "Starting Alpaca batch download",
            total_symbols=len(symbols),
            chunk_size=chunk_size,
            start=start,
            end=end,
        )

        all_parts: list[pl.DataFrame] = []
        failed: list[str] = []

        for kind in ("crypto", "stock", "option"):
            syms = by_kind.get(kind)
            if not syms:
                continue
            client = self._client(kind)
            n_chunks = (len(syms) + chunk_size - 1) // chunk_size
            for i, chunk in enumerate(_chunks(syms, chunk_size), 1):
                try:
                    if kind == "crypto":
                        req = CryptoBarsRequest(
                            symbol_or_symbols=chunk,
                            timeframe=tf,
                            start=start_utc,
                            end=end_excl,
                        )
                        bars = client.get_crypto_bars(req)
                    elif kind == "stock":
                        req = StockBarsRequest(
                            symbol_or_symbols=chunk,
                            timeframe=tf,
                            start=start_utc,
                            end=end_excl,
                            feed=self._feed,
                            adjustment=self._adjustment,
                        )
                        bars = client.get_stock_bars(req)
                    else:
                        req = OptionBarsRequest(
                            symbol_or_symbols=chunk,
                            timeframe=tf,
                            start=start_utc,
                            end=end_excl,
                        )
                        bars = client.get_option_bars(req)

                    df_pd = bars.df
                    if df_pd.empty:
                        self.logger.warning("Empty Alpaca batch chunk", chunk=i, symbols=chunk)
                        failed.extend(chunk)
                        continue
                    all_parts.append(_bars_pandas_to_polars_batch(df_pd, end_inclusive))
                except Exception as e:
                    self.logger.error(
                        "Alpaca batch chunk failed", chunk=i, symbols=chunk, error=str(e)
                    )
                    failed.extend(chunk)

                if i < n_chunks and delay_seconds > 0:
                    time.sleep(delay_seconds)

        if not all_parts:
            self.logger.error("No Alpaca batch data", failed_symbols=failed)
            return self._create_empty_dataframe()

        result = pl.concat(all_parts).sort(["symbol", "timestamp"])
        self.logger.info(
            "Alpaca batch download complete",
            rows=len(result),
            failed_symbols=len(failed),
        )
        if failed:
            self.logger.warning(
                "Some Alpaca batch symbols failed", count=len(failed), sample=failed[:10]
            )
        return result

    async def fetch_batch_ohlcv_async(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        chunk_size: int = 50,
        delay_seconds: float = 0.0,
    ) -> pl.DataFrame:
        """Async batch fetch (sync Alpaca client via :func:`asyncio.to_thread`)."""
        return await asyncio.to_thread(
            self.fetch_batch_ohlcv,
            symbols,
            start,
            end,
            frequency,
            chunk_size,
            delay_seconds,
        )

    async def close_async(self) -> None:
        """No network session to close (SDK owns HTTP)."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()


AlpacaHistoricalProvider = AlpacaProvider
