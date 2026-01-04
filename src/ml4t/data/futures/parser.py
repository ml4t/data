"""
Futures data parser for Quandl CHRIS and other formats.

This module handles parsing and cleaning raw futures data from various sources.
"""

from pathlib import Path

import polars as pl

from ml4t.data.futures.schema import ContractSpec


def parse_quandl_chris_raw(
    ticker: str,
    data_path: str | Path = "/home/stefan/ml3t/data/futures/quandl/chris_futures.parquet",
    _contract_spec: ContractSpec | None = None,
) -> pl.DataFrame:
    """
    Parse Quandl CHRIS futures data without deduplication (keeps all contracts).

    Returns multi-contract data with duplicate dates, useful for roll analysis.

    Args:
        ticker: Contract ticker (e.g., "CL" for crude oil, "ES" for E-mini S&P 500)
        data_path: Path to Quandl CHRIS parquet file
        contract_spec: Optional contract specifications for price unit conversion

    Returns:
        DataFrame with potentially multiple rows per date (one per contract month):
        - date: pl.Date
        - open, high, low, close: float
        - volume: float
        - open_interest: float (nullable)

    Note:
        Use this function for roll detection. For continuous series, use parse_quandl_chris().
    """
    # Validate data path
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

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
    ticker_data = _normalize_price_units(ticker_data, ticker)

    # Select relevant columns (NO deduplication)
    result = ticker_data.select(
        pl.col("date").cast(pl.Date).alias("date"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("open_interest").cast(pl.Float64),
    ).sort("date")

    return result


def parse_quandl_chris(
    ticker: str,
    data_path: str | Path = "/home/stefan/ml3t/data/futures/quandl/chris_futures.parquet",
    _contract_spec: ContractSpec | None = None,
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
        contract_spec: Optional contract specifications for price unit conversion

    Returns:
        Clean DataFrame with single row per date, columns:
        - date: pl.Date
        - open: float
        - high: float
        - low: float
        - close: float
        - volume: float
        - open_interest: float (nullable)

    Raises:
        ValueError: If ticker not found in data
        FileNotFoundError: If data_path doesn't exist

    Examples:
        >>> # Parse ES (already continuous - no duplicates)
        >>> es_data = parse_quandl_chris("ES")
        >>> assert es_data.select(pl.col("date").unique().count()).item() == len(es_data)

        >>> # Parse CL (mixed contracts - has duplicates)
        >>> cl_data = parse_quandl_chris("CL")
        >>> # Returns front month only (highest volume on duplicate dates)
    """
    # Validate data path
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

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
    ticker_data = _normalize_price_units(ticker_data, ticker)

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


def _normalize_price_units(data: pl.DataFrame, _ticker: str) -> pl.DataFrame:
    """
    Normalize price units to consistent standard.

    Quandl CHRIS has inconsistent price units across rows:
    - Some rows have prices in cents (e.g., 6348¢ for CL)
    - Some rows have prices in dollars (e.g., $103 for CL)

    This function detects and converts to standard units:
    - CL (Crude Oil): $/barrel (divide cents by 100)
    - Energy contracts: Generally $/unit
    - Threshold: Prices > 1000 are assumed to be in cents

    Args:
        data: DataFrame with OHLC columns
        ticker: Contract ticker (for unit determination)

    Returns:
        DataFrame with normalized prices in standard units
    """
    # Heuristic: If open or close > 1000, likely in cents, divide by 100
    # This threshold works for most energy contracts (CL, NG, etc.)
    # Won't work for high-priced instruments, but Quandl CHRIS is mostly commodities

    # Use close price to detect unit (most reliable)
    is_cents = pl.col("close") > 1000

    # Convert cents to dollars for all OHLC columns
    result = data.with_columns(
        [
            pl.when(is_cents).then(pl.col("open") / 100).otherwise(pl.col("open")).alias("open"),
            pl.when(is_cents).then(pl.col("high") / 100).otherwise(pl.col("high")).alias("high"),
            pl.when(is_cents).then(pl.col("low") / 100).otherwise(pl.col("low")).alias("low"),
            pl.when(is_cents).then(pl.col("close") / 100).otherwise(pl.col("close")).alias("close"),
        ]
    )

    return result


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
