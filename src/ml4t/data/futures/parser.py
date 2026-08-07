"""
Futures data parser for Quandl CHRIS and other formats.

This module handles parsing and cleaning raw futures data from various sources.
"""

import os
from pathlib import Path

import polars as pl

from ml4t.data.core.config import resolve_storage_path
from ml4t.data.futures.schema import (
    ContractSpec,
    get_contract_spec,
    price_conversion_factor,
)

DEFAULT_CHRIS_ENV_VAR = "ML4T_QUANDL_CHRIS_PATH"


def _resolve_chris_data_path(data_path: str | Path | None) -> Path:
    if data_path is None:
        env_path = os.getenv(DEFAULT_CHRIS_ENV_VAR)
        resolved = (
            Path(env_path).expanduser().resolve()
            if env_path
            else resolve_storage_path(None, "futures", "quandl", "chris_futures.parquet")
        )
    else:
        resolved = Path(data_path).expanduser().resolve()

    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Quandl CHRIS data file not found: {resolved}. "
        "The legacy CHRIS dataset is no longer available from NASDAQ Data Link. "
        f"Provide a local parquet via `data_path` or set {DEFAULT_CHRIS_ENV_VAR}."
    )


def parse_quandl_chris_raw(
    ticker: str,
    data_path: str | Path | None = None,
    contract_spec: ContractSpec | None = None,
) -> pl.DataFrame:
    """
    Parse Quandl CHRIS futures data without deduplication (keeps all contracts).

    Returns multi-contract data with duplicate dates, useful for roll analysis.

    Args:
        ticker: Contract ticker (e.g., "CL" for crude oil, "ES" for E-mini S&P 500)
        data_path: Path to Quandl CHRIS parquet file
        contract_spec: Contract specification. When omitted, the parser looks up
            ``ticker`` in ``MAJOR_CONTRACTS``.

    Returns:
        DataFrame with potentially multiple rows per date (one per contract month):
        - date: pl.Date
        - open, high, low, close: float
        - volume: float
        - open_interest: float (nullable)

        Prices are normalized to dollars, except index futures remain in index points.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the ticker, contract specification, or source price unit is invalid.

    Note:
        Use this function for roll detection. For continuous series, use parse_quandl_chris().
    """
    data_path = _resolve_chris_data_path(data_path)

    # Load data for ticker
    data = pl.read_parquet(data_path)
    ticker_data = data.filter(pl.col("ticker") == ticker)

    if len(ticker_data) == 0:
        raise ValueError(
            f"Ticker '{ticker}' not found in data. "
            f"Available tickers: {data.select('ticker').unique().sort('ticker').to_series().to_list()[:10]}..."
        )

    # Standardize price columns
    ticker_data = _standardize_price_columns(ticker_data)

    # Normalize price units (cents → dollars)
    ticker_data = _normalize_price_units(ticker_data, _resolve_contract_spec(ticker, contract_spec))

    columns = [
        pl.col("date").cast(pl.Date).alias("date"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("open_interest").cast(pl.Float64),
    ]
    if "symbol" in ticker_data.columns:
        columns.append(pl.col("symbol").cast(pl.String))
    result = ticker_data.select(columns).sort("date")

    return result


def parse_quandl_chris(
    ticker: str,
    data_path: str | Path | None = None,
    contract_spec: ContractSpec | None = None,
) -> pl.DataFrame:
    """
    Parse Quandl CHRIS futures data for a specific ticker.

    Handles:
    - Duplicate dates (multiple contracts per date) - selects front month by highest volume
    - Missing price data - uses fallback: settle → close → last → open
    - Price standardization (all OHLC columns use same price source)

    Args:
        ticker: Contract ticker (e.g., "CL" for crude oil, "ES" for E-mini S&P 500)
        data_path: Path to Quandl CHRIS parquet file
        contract_spec: Contract specification. When omitted, the parser looks up
            ``ticker`` in ``MAJOR_CONTRACTS``.

    Returns:
        Clean DataFrame with single row per date, columns:
        - date: pl.Date
        - open: float
        - high: float
        - low: float
        - close: float
        - volume: float
        - open_interest: float (nullable)

        Prices are normalized to dollars, except index futures remain in index points.

    Raises:
        ValueError: If the ticker, contract specification, or source price unit is invalid.
        FileNotFoundError: If data_path doesn't exist

    Examples:
        >>> # Parse ES (already continuous - no duplicates)
        >>> es_data = parse_quandl_chris("ES")
        >>> assert es_data.select(pl.col("date").unique().count()).item() == len(es_data)

        >>> # Parse CL (mixed contracts - has duplicates)
        >>> cl_data = parse_quandl_chris("CL")
        >>> # Returns front month only (highest volume on duplicate dates)
    """
    data_path = _resolve_chris_data_path(data_path)

    # Load data for ticker
    data = pl.read_parquet(data_path)
    ticker_data = data.filter(pl.col("ticker") == ticker)

    if len(ticker_data) == 0:
        raise ValueError(
            f"Ticker '{ticker}' not found in data. "
            f"Available tickers: {data.select('ticker').unique().sort('ticker').to_series().to_list()[:10]}..."
        )

    # Standardize price columns - use settle if available, fallback to close → last → open
    ticker_data = _standardize_price_columns(ticker_data)

    # Normalize price units (cents → dollars)
    ticker_data = _normalize_price_units(ticker_data, _resolve_contract_spec(ticker, contract_spec))

    # Handle duplicate dates - select front month (highest volume)
    ticker_data = _select_front_month_by_volume(ticker_data)

    # Select relevant columns and ensure proper types
    result = ticker_data.select(
        pl.col("date").cast(pl.Date).alias("date"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("open_interest").cast(pl.Float64),  # Nullable
    ).sort("date")

    return result


def _standardize_price_columns(data: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize OHLC price columns using best available source.

    Priority: settle > close > last > open
    All OHLC columns use the same source for consistency within a row.

    Args:
        data: Raw Quandl data with various price columns

    Returns:
        DataFrame with standardized open, high, low, close columns
    """
    # Determine best price column to use (prefer settle, then close, then last)
    # Note: Quandl CHRIS has inconsistent column usage across exchanges

    # Create standardized close column
    close = (
        pl.when(pl.col("settle").is_not_null())
        .then(pl.col("settle"))
        .when(pl.col("close").is_not_null())
        .then(pl.col("close"))
        .when(pl.col("last").is_not_null())
        .then(pl.col("last"))
        .otherwise(pl.col("open"))
        .alias("close")
    )

    # For other OHLC columns, use existing values if available, fallback to close
    open_col = (
        pl.when(pl.col("open").is_not_null()).then(pl.col("open")).otherwise(close).alias("open")
    )

    high_col = (
        pl.when(pl.col("high").is_not_null()).then(pl.col("high")).otherwise(close).alias("high")
    )

    low_col = pl.when(pl.col("low").is_not_null()).then(pl.col("low")).otherwise(close).alias("low")

    result = data.with_columns([open_col, high_col, low_col, close])

    return result


def _resolve_contract_spec(ticker: str, contract_spec: ContractSpec | None) -> ContractSpec:
    """Return explicit price-unit metadata for a CHRIS dataset."""
    resolved = contract_spec or get_contract_spec(ticker)
    if resolved is None:
        raise ValueError(
            f"Contract specification required for ticker '{ticker}'; price units cannot be guessed"
        )
    if resolved.ticker != ticker:
        raise ValueError(
            f"Contract specification ticker '{resolved.ticker}' does not match requested '{ticker}'"
        )
    return resolved


def _normalize_price_units(data: pl.DataFrame, contract_spec: ContractSpec) -> pl.DataFrame:
    """
    Normalize price units to consistent standard.

    The contract specification declares the source-specific unit. Magnitude is
    consulted only for sources explicitly declared as mixed cents and dollars.

    Args:
        data: DataFrame with OHLC columns
        contract_spec: Contract metadata declaring the source price unit

    Returns:
        DataFrame with normalized prices in standard units
    """
    source_unit = contract_spec.source_price_quote_units.get(
        "quandl_chris", contract_spec.price_quote_unit
    )
    target_unit = "index_points" if contract_spec.price_quote_unit == "index_points" else "dollars"
    columns = ("open", "high", "low", "close")

    if source_unit == "mixed_cents_dollars":
        cents_factor = price_conversion_factor("cents", "dollars")
        is_cents = pl.col("close").abs() >= 1000
        normalized = data.with_columns(
            pl.when(is_cents)
            .then(pl.col(column) * cents_factor)
            .otherwise(pl.col(column))
            .alias(column)
            for column in columns
        )
        _validate_mixed_price_consistency(normalized, contract_spec.ticker)
        return normalized

    try:
        factor = price_conversion_factor(source_unit, target_unit)
    except ValueError as error:
        raise ValueError(
            f"Unsupported price quote unit '{source_unit}' for ticker '{contract_spec.ticker}'"
        ) from error
    if factor == 1.0:
        return data
    return data.with_columns((pl.col(column) * factor).alias(column) for column in columns)


def _validate_mixed_price_consistency(data: pl.DataFrame, ticker: str) -> None:
    """Reject mixed-unit normalization that still contains an implausible scale split."""
    closes = data["close"].drop_nulls().abs()
    positive_closes = closes.filter(closes > 0)
    if len(positive_closes) < 2:
        return
    median = positive_closes.median()
    maximum = positive_closes.max()
    if not isinstance(median, int | float) or not isinstance(maximum, int | float):
        raise TypeError("'close' must contain non-null numeric values")
    if median > 0 and maximum / median > 10:
        raise ValueError(
            f"Normalized prices for ticker '{ticker}' remain inconsistent; "
            "provide corrected source price-unit metadata"
        )


def _select_front_month_by_volume(data: pl.DataFrame) -> pl.DataFrame:
    """
    Select front month contract for duplicate dates based on volume.

    When multiple contracts exist for the same date, the front month
    is identified as the contract with highest trading volume (most liquid).

    Args:
        data: DataFrame with potential duplicate dates

    Returns:
        DataFrame with single row per date (front month only)
    """
    # Group by date and select row with maximum volume
    # This handles both:
    # 1. Clean continuous data (ES) - no duplicates, returns as-is
    # 2. Mixed contract data (CL) - selects front month (highest volume)

    result = (
        data.sort("date", "volume", descending=[False, True])
        .group_by("date")
        .agg(pl.all().first())  # First row after sorting by volume desc = highest volume
        .sort("date")
    )

    return result
