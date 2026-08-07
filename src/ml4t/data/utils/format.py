"""Format conversion utilities for multi-asset data.

This module provides utilities for converting between stacked (long) and wide
(pivoted) formats for multi-asset DataFrames.

**Canonical Format: Stacked/Long**

The recommended format for multi-asset data is the stacked (long) format with
a symbol column:

    ┌────────────┬────────┬───────┬──────┐
    │ timestamp  │ symbol │ close │ ...  │
    ├────────────┼────────┼───────┼──────┤
    │ 2024-01-01 │ AAPL   │ 180.0 │ ...  │
    │ 2024-01-01 │ MSFT   │ 370.0 │ ...  │
    │ 2024-01-02 │ AAPL   │ 182.0 │ ...  │
    │ 2024-01-02 │ MSFT   │ 372.0 │ ...  │
    └────────────┴────────┴───────┴──────┘

This format is efficient for storage, filtering, and feature engineering.

**Wide Format (Pivoted)**

Wide format pivots each symbol into separate columns:

    ┌────────────┬────────────┬────────────┐
    │ timestamp  │ close_AAPL │ close_MSFT │
    ├────────────┼────────────┼────────────┤
    │ 2024-01-01 │ 180.0      │ 370.0      │
    │ 2024-01-02 │ 182.0      │ 372.0      │
    └────────────┴────────────┴────────────┘

Wide format is useful for:
- Legacy tools expecting pivoted data
- Pandas-based analysis workflows
- Correlation matrices and cross-sectional analysis

**Performance Warning**

Wide format does NOT scale well beyond ~100 symbols:
- 100 symbols × 5 OHLCV columns = 500 columns
- Most tools struggle with >1000 columns
- Polars/Pandas performance degrades significantly
- Memory usage increases dramatically

For large symbol sets (>100), prefer the stacked format and use
tools that work natively with it.

Examples:
    Convert stacked format to wide for pandas analysis:

    >>> from ml4t.data.utils.format import pivot_to_wide, pivot_to_stacked
    >>>
    >>> # Load data in canonical stacked format
    >>> df_stacked = manager.batch_load(['AAPL', 'MSFT', 'GOOGL'])
    >>>
    >>> # Convert to wide for pandas/matplotlib
    >>> df_wide = pivot_to_wide(df_stacked)
    >>> df_pandas = df_wide.to_pandas()
    >>> df_pandas.plot()
    >>>
    >>> # Convert back to stacked for storage
    >>> df_stacked_again = pivot_to_stacked(df_wide)

    Specify custom columns to pivot:

    >>> # Only pivot close and volume
    >>> df_wide = pivot_to_wide(
    ...     df_stacked,
    ...     value_cols=['close', 'volume']
    ... )
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl

logger = logging.getLogger(__name__)


def pivot_to_wide(
    df: pl.DataFrame,
    value_cols: Sequence[str] | None = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Convert stacked multi-asset format to wide (pivoted) format.

    Pivots the DataFrame so each symbol becomes separate columns. For example,
    'close' becomes 'close_AAPL', 'close_MSFT', etc.

    Args:
        df: Multi-asset DataFrame in stacked format (must have symbol column)
        value_cols: Columns to pivot. If None, defaults to standard OHLCV columns.
                   Only columns that exist in df will be pivoted.
        symbol_col: Column containing symbol names (default: 'symbol')
        timestamp_col: Column containing timestamps (default: 'timestamp')

    Returns:
        Wide-format DataFrame with pivoted columns

    Raises:
        ValueError: If required columns missing or duplicate (timestamp, symbol) pairs exist

    Warnings:
        Wide format does not scale well beyond 100 symbols due to column count.
        Consider using stacked format for large symbol sets.

    Examples:
        >>> # Basic usage with default OHLCV columns
        >>> df_wide = pivot_to_wide(df_stacked)
        >>> df_wide.columns
        ['timestamp', 'close_AAPL', 'close_MSFT', 'volume_AAPL', 'volume_MSFT', ...]

        >>> # Pivot only specific columns
        >>> df_wide = pivot_to_wide(df_stacked, value_cols=['close', 'volume'])
        >>> df_wide.columns
        ['timestamp', 'close_AAPL', 'close_MSFT', 'volume_AAPL', 'volume_MSFT']

        >>> # Handle custom column names
        >>> df_wide = pivot_to_wide(
        ...     df,
        ...     symbol_col='ticker',
        ...     timestamp_col='date'
        ... )
    """
    # Validate required columns
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{timestamp_col}'")
    if symbol_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{symbol_col}'")

    # Default to standard OHLCV columns if not specified
    if value_cols is None:
        value_cols = ["open", "high", "low", "close", "volume"]

    # Filter to only columns that exist in the DataFrame
    available_value_cols = [col for col in value_cols if col in df.columns]

    if not available_value_cols:
        raise ValueError(
            f"None of the specified value columns {value_cols} exist in DataFrame. "
            f"Available columns: {df.columns}"
        )

    # Check for duplicate (timestamp, symbol) pairs
    duplicate_check = df.group_by([timestamp_col, symbol_col]).agg(pl.len().alias("count"))
    max_count = duplicate_check["count"].max()
    if max_count is None:
        max_count = 0
    elif not isinstance(max_count, int):
        raise RuntimeError("Could not determine duplicate row count")
    if max_count > 1:
        duplicates = duplicate_check.filter(pl.col("count") > 1)
        n_duplicates = len(duplicates)
        raise ValueError(
            f"Found {n_duplicates} duplicate (timestamp, symbol) pairs. "
            f"Wide format requires unique combinations. "
            f"First duplicate: {duplicates.head(1).to_dicts()[0]}"
        )

    # Get unique symbols for warning
    symbols = df[symbol_col].unique().to_list()
    n_symbols = len(symbols)
    n_output_cols = 1 + (n_symbols * len(available_value_cols))  # timestamp + pivoted cols

    # Warn about scalability
    if n_symbols > 100:
        logger.warning(
            f"Pivoting {n_symbols} symbols to wide format will create {n_output_cols} columns. "
            f"Wide format does not scale well beyond 100 symbols. "
            f"Consider using stacked format for large symbol sets."
        )
    elif n_symbols > 50:
        logger.info(
            f"Pivoting {n_symbols} symbols will create {n_output_cols} columns. "
            f"Performance may degrade with >100 symbols."
        )

    # Select only the columns we need for pivoting
    cols_to_keep = [timestamp_col, symbol_col] + available_value_cols
    df_subset = df.select(cols_to_keep)

    # Pivot each value column separately, then join
    # We do this because Polars doesn't support multi-column pivot natively
    pivoted_frames = []

    for value_col in available_value_cols:
        # Pivot this column
        pivoted = df_subset.select([timestamp_col, symbol_col, value_col]).pivot(
            on=symbol_col,
            index=timestamp_col,
            values=value_col,
            aggregate_function=None,  # Expect unique values
        )

        # Rename columns to include original column name
        # e.g., 'AAPL' -> 'close_AAPL'
        rename_map = {col: f"{value_col}_{col}" for col in pivoted.columns if col != timestamp_col}
        pivoted = pivoted.rename(rename_map)

        pivoted_frames.append(pivoted)

    # Join all pivoted frames on timestamp
    result = pivoted_frames[0]
    for pivoted_frame in pivoted_frames[1:]:
        result = result.join(pivoted_frame, on=timestamp_col, how="left")

    # Sort by timestamp for consistency
    result = result.sort(timestamp_col)

    return result


def pivot_to_stacked(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Convert wide (pivoted) format back to stacked multi-asset format.

    Unpivots a wide DataFrame back to the canonical stacked format with
    a symbol column. Automatically detects value columns and symbols based
    on column naming convention (e.g., 'close_AAPL', 'volume_MSFT').

    Expected column naming convention:
    - Timestamp column: unchanged
    - Pivoted columns: '{value_col}_{symbol}' (e.g., 'close_AAPL', 'volume_MSFT')

    Args:
        df: Wide-format DataFrame with pivoted columns
        timestamp_col: Column containing timestamps (default: 'timestamp')

    Returns:
        Stacked-format DataFrame with symbol column and standard columns
        [timestamp, symbol, {value_cols}]

    Raises:
        ValueError: If timestamp column missing or no pivoted columns detected

    Examples:
        >>> # Convert wide format back to stacked
        >>> df_stacked = pivot_to_stacked(df_wide)
        >>> df_stacked.columns
        ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']

        >>> # Handle custom timestamp column
        >>> df_stacked = pivot_to_stacked(df_wide, timestamp_col='date')

    Notes:
        This function expects column names in the format '{value}_{symbol}'.
        If columns don't follow this convention, they will be ignored.
    """
    # Validate timestamp column exists
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{timestamp_col}'")

    # Identify pivoted columns (exclude timestamp)
    pivoted_cols = [col for col in df.columns if col != timestamp_col]

    if not pivoted_cols:
        raise ValueError(f"No pivoted columns found. DataFrame only has '{timestamp_col}' column.")

    # Parse column names to extract value column and symbols
    # Expected format: {value_col}_{symbol}
    # e.g., 'close_AAPL', 'volume_MSFT'
    value_cols_set = set()
    symbols_set = set()
    col_mapping = {}  # {(value_col, symbol): original_col_name}

    for col in pivoted_cols:
        # Split on first underscore to separate value_col from symbol
        # This handles symbols with underscores (e.g., 'BRK_B')
        parts = col.split("_", 1)
        if len(parts) == 2:
            value_col, symbol = parts
            value_cols_set.add(value_col)
            symbols_set.add(symbol)
            col_mapping[(value_col, symbol)] = col

    if not col_mapping:
        raise ValueError(
            f"Could not parse any pivoted columns. Expected format: '{{value}}_{{symbol}}'. "
            f"Found columns: {pivoted_cols[:5]}..."
        )

    value_cols = sorted(value_cols_set)
    symbols = sorted(symbols_set)

    logger.info(
        f"Detected {len(value_cols)} value columns and {len(symbols)} symbols. "
        f"Value columns: {value_cols}. Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}"
    )

    # Unpivot each value column separately, then join
    unpivoted_frames = []

    for value_col in value_cols:
        # Get all columns for this value
        value_col_names = [
            col_mapping[(value_col, symbol)]
            for symbol in symbols
            if (value_col, symbol) in col_mapping
        ]

        if not value_col_names:
            continue

        # Unpivot (melt) this set of columns
        melted = df.select([timestamp_col] + value_col_names).unpivot(
            index=timestamp_col,
            on=value_col_names,
            variable_name="_pivot_col",
            value_name=value_col,
        )

        # Extract symbol from column name (strip value_col prefix)
        # e.g., 'close_AAPL' -> 'AAPL'
        melted = melted.with_columns(
            pl.col("_pivot_col").str.replace(f"^{value_col}_", "").alias("symbol")
        ).drop("_pivot_col")

        unpivoted_frames.append(melted)

    # Join all unpivoted frames on [timestamp, symbol]
    if not unpivoted_frames:
        raise ValueError("Failed to unpivot any columns")

    result = unpivoted_frames[0]
    for unpivoted_frame in unpivoted_frames[1:]:
        result = result.join(
            unpivoted_frame,
            on=[timestamp_col, "symbol"],
            how="left",
        )

    # Sort by timestamp, then symbol for consistency
    result = result.sort([timestamp_col, "symbol"])

    # Standardize column order: [timestamp, symbol, value_cols...]
    standard_cols = [timestamp_col, "symbol"] + value_cols
    result = result.select(standard_cols)

    return result
