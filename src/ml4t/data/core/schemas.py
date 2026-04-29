"""Multi-asset schema definitions and validation utilities.

This module provides schema definitions and utilities for working with multi-asset
data in a standardized format. The canonical multi-asset format uses a "stacked"
or "long" format with a symbol column to distinguish between different instruments.

Schema Design:
    The multi-asset schema uses the following columns:
    - timestamp: Datetime column with microsecond precision in UTC
    - symbol: String identifier for the instrument
    - open, high, low, close: OHLCV price data
    - volume: Trading volume

    Additional optional columns may be present depending on asset class:
    - equities: dividends, splits, adjusted_close
    - crypto: trades_count, taker_buy_volume, taker_buy_quote_volume
    - futures: open_interest
"""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl


def _build_fill_expression(name: str, value: Any, dtype: pl.DataType | None) -> pl.Expr:
    if dtype is not None:
        return pl.lit(value, dtype=dtype).alias(name)
    return pl.lit(value).alias(name)


def _align_frame_columns(
    df: pl.DataFrame,
    columns: list[str],
    schema: dict[str, pl.DataType],
    fill_values: dict[str, Any] | None = None,
) -> pl.DataFrame:
    fill_values = fill_values or {}
    missing = [col for col in columns if col not in df.columns]

    if missing:
        df = df.with_columns(
            [_build_fill_expression(col, fill_values.get(col), schema.get(col)) for col in missing]
        )

    return df.select(columns)


def align_frames_for_concat(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_fill_values: dict[str, Any] | None = None,
    right_fill_values: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align two DataFrames to a shared column set for safe vertical concatenation."""
    columns = [*left.columns, *[col for col in right.columns if col not in left.columns]]
    schema = {**right.schema, **left.schema}

    return (
        _align_frame_columns(left, columns, schema, left_fill_values),
        _align_frame_columns(right, columns, schema, right_fill_values),
    )


class MultiAssetSchema:
    """Schema definition and validation for multi-asset DataFrames.

    This class defines the canonical schema for multi-asset data and provides
    utilities for validation, creation, and standardization of multi-asset
    DataFrames.

    The multi-asset format uses a "stacked" or "long" layout where each row
    represents a single observation (timestamp + symbol combination) with
    OHLCV data. This format is efficient for:
    - Storage and querying (filter by symbol, time range)
    - Feature engineering across multiple assets
    - Portfolio-level analysis

    Examples:
        Validate a multi-asset DataFrame:

        >>> df = pl.DataFrame({
        ...     'timestamp': [datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)],
        ...     'symbol': ['AAPL'],
        ...     'open': [150.0],
        ...     'high': [152.0],
        ...     'low': [149.0],
        ...     'close': [151.0],
        ...     'volume': [1000000.0],
        ... })
        >>> MultiAssetSchema.validate(df)  # Returns True if valid, raises otherwise

        Convert single-asset data to multi-asset format:

        >>> single_asset_df = pl.DataFrame({
        ...     'timestamp': [...],
        ...     'open': [...],
        ...     # ... other OHLCV columns
        ... })
        >>> multi_asset_df = MultiAssetSchema.add_symbol_column(single_asset_df, 'AAPL')

        Create an empty multi-asset DataFrame:

        >>> empty_df = MultiAssetSchema.create_empty('equities')
    """

    # Required columns with their Polars data types
    SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "timestamp": pl.Datetime("us", "UTC"),
        "symbol": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    # Optional columns by asset class
    OPTIONAL_COLUMNS: ClassVar[dict[str, dict[str, pl.DataType]]] = {
        "equities": {
            "dividends": pl.Float64,
            "splits": pl.Float64,
            "adjusted_close": pl.Float64,
        },
        "equity": {  # Alias
            "dividends": pl.Float64,
            "splits": pl.Float64,
            "adjusted_close": pl.Float64,
        },
        "crypto": {
            "trades_count": pl.Int64,
            "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
        },
        "futures": {
            "open_interest": pl.Float64,
        },
        "future": {  # Alias
            "open_interest": pl.Float64,
        },
        "options": {
            "open_interest": pl.Float64,
            "implied_volatility": pl.Float64,
            "delta": pl.Float64,
            "gamma": pl.Float64,
            "theta": pl.Float64,
            "vega": pl.Float64,
        },
        "option": {  # Alias
            "open_interest": pl.Float64,
            "implied_volatility": pl.Float64,
            "delta": pl.Float64,
            "gamma": pl.Float64,
            "theta": pl.Float64,
            "vega": pl.Float64,
        },
    }

    # Column sort order for standardization
    COLUMN_ORDER: ClassVar[list[str]] = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    @classmethod
    def validate(cls, df: pl.DataFrame, strict: bool = True) -> bool:
        """Validate that a DataFrame conforms to the multi-asset schema.

        Checks that:
        1. All required columns are present
        2. Column data types match expected types (or are compatible)
        3. If strict=True, raises on validation failure

        Args:
            df: DataFrame to validate
            strict: If True, raise ValueError on validation errors.
                   If False, return False on validation errors.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails and strict=True

        Examples:
            >>> df = MultiAssetSchema.create_empty('equities')
            >>> MultiAssetSchema.validate(df)  # Returns True
            True

            >>> invalid_df = pl.DataFrame({'foo': [1, 2, 3]})
            >>> MultiAssetSchema.validate(invalid_df, strict=False)  # Returns False
            False

            >>> MultiAssetSchema.validate(invalid_df, strict=True)  # Raises
            Traceback (most recent call last):
                ...
            ValueError: Missing required column: timestamp
        """
        # Check for missing required columns
        missing_cols = set(cls.SCHEMA.keys()) - set(df.columns)
        if missing_cols:
            if strict:
                raise ValueError(f"Missing required column: {sorted(missing_cols)[0]}")
            return False

        # Check data types
        for col, _expected_dtype in cls.SCHEMA.items():
            actual_dtype = df[col].dtype

            # Special handling for timestamp column
            if col == "timestamp":
                if not isinstance(actual_dtype, pl.Datetime):
                    if strict:
                        raise ValueError(f"Column '{col}' must be Datetime, got {actual_dtype}")
                    return False
            # Numeric columns can be compatible types
            elif col in ("open", "high", "low", "close", "volume"):
                if actual_dtype not in (
                    pl.Float64,
                    pl.Float32,
                    pl.Int64,
                    pl.Int32,
                    pl.UInt64,
                    pl.UInt32,
                ):
                    if strict:
                        raise ValueError(f"Column '{col}' must be numeric, got {actual_dtype}")
                    return False
            # String column (symbol)
            elif col == "symbol":
                if actual_dtype not in (pl.Utf8, pl.Categorical):
                    if strict:
                        raise ValueError(f"Column '{col}' must be string type, got {actual_dtype}")
                    return False

        return True

    @classmethod
    def create_empty(cls, asset_class: str | None = None) -> pl.DataFrame:
        """Create an empty DataFrame with the multi-asset schema.

        Args:
            asset_class: If provided, include optional columns for this asset class.
                        Supported: 'equities', 'crypto', 'futures', 'options'

        Returns:
            Empty DataFrame with appropriate schema

        Examples:
            >>> df = MultiAssetSchema.create_empty()
            >>> df.columns
            ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']

            >>> df_eq = MultiAssetSchema.create_empty('equities')
            >>> 'dividends' in df_eq.columns
            True
            >>> 'splits' in df_eq.columns
            True
        """
        # Start with required schema
        schema = cls.SCHEMA.copy()

        # Add optional columns if asset class specified
        if asset_class and asset_class in cls.OPTIONAL_COLUMNS:
            schema.update(cls.OPTIONAL_COLUMNS[asset_class])

        return pl.DataFrame(schema=schema)

    @classmethod
    def add_symbol_column(cls, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Add a symbol column to a single-symbol DataFrame.

        This function is idempotent - if the DataFrame already has a symbol
        column with the correct value, it returns the DataFrame unchanged.

        If the symbol column exists but has different values, raises an error
        in strict mode or overwrites in non-strict mode.

        Args:
            df: Single-symbol DataFrame (must have timestamp and OHLCV columns)
            symbol: Symbol identifier to add

        Returns:
            DataFrame with symbol column added

        Raises:
            ValueError: If df already has 'symbol' column with different values

        Examples:
            >>> df = pl.DataFrame({
            ...     'timestamp': [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            ...     'open': [100.0],
            ...     'high': [101.0],
            ...     'low': [99.0],
            ...     'close': [100.5],
            ...     'volume': [1000.0],
            ... })
            >>> df_with_symbol = MultiAssetSchema.add_symbol_column(df, 'AAPL')
            >>> 'symbol' in df_with_symbol.columns
            True
            >>> df_with_symbol['symbol'][0]
            'AAPL'

            # Idempotent behavior
            >>> df2 = MultiAssetSchema.add_symbol_column(df_with_symbol, 'AAPL')
            >>> df2.frame_equal(df_with_symbol)
            True
        """
        # If symbol column already exists, check consistency
        if "symbol" in df.columns:
            # Check if all values match the requested symbol
            unique_symbols = df["symbol"].unique().to_list()
            if len(unique_symbols) == 1 and unique_symbols[0] == symbol:
                # Already has the correct symbol, return as-is (idempotent)
                return df
            else:
                # Has different symbol(s)
                raise ValueError(
                    f"DataFrame already has 'symbol' column with values {unique_symbols}, "
                    f"cannot add symbol '{symbol}'"
                )

        # Add symbol column with literal value
        return df.with_columns(pl.lit(symbol).alias("symbol"))

    @classmethod
    def standardize_order(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column order and sort by [timestamp, symbol].

        This ensures consistent ordering for:
        - Storage and caching (deterministic file layout)
        - Merging and joining operations
        - Time-series operations that require sorted data

        Args:
            df: Multi-asset DataFrame (must have timestamp and symbol columns)

        Returns:
            DataFrame sorted by [timestamp, symbol] with standardized column order

        Examples:
            >>> df = pl.DataFrame({
            ...     'volume': [1000.0, 2000.0],
            ...     'symbol': ['AAPL', 'AAPL'],
            ...     'close': [100.0, 101.0],
            ...     'timestamp': [
            ...         datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc),
            ...         datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc),
            ...     ],
            ...     'open': [99.0, 100.0],
            ...     'high': [101.0, 102.0],
            ...     'low': [98.0, 99.0],
            ... })
            >>> standardized = MultiAssetSchema.standardize_order(df)
            >>> standardized.columns[:3]
            ['timestamp', 'symbol', 'open']
        """
        # Sort by timestamp, then symbol
        df = df.sort(["timestamp", "symbol"])

        # Determine column order: standard columns + any extras
        columns = []
        for col in cls.COLUMN_ORDER:
            if col in df.columns:
                columns.append(col)

        # Add any additional columns not in COLUMN_ORDER
        for col in df.columns:
            if col not in columns:
                columns.append(col)

        return df.select(columns)

    @classmethod
    def cast_to_schema(cls, df: pl.DataFrame, asset_class: str | None = None) -> pl.DataFrame:
        """Cast DataFrame columns to match the multi-asset schema types.

        This is useful when ingesting data from external sources that may have
        slightly different types (e.g., Int64 instead of Float64 for OHLCV).

        Args:
            df: DataFrame to cast
            asset_class: If provided, also cast optional columns for this asset class

        Returns:
            DataFrame with columns cast to schema types

        Examples:
            >>> df = pl.DataFrame({
            ...     'timestamp': [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            ...     'symbol': ['AAPL'],
            ...     'open': [100],  # Int64
            ...     'high': [101],
            ...     'low': [99],
            ...     'close': [100],
            ...     'volume': [1000],
            ... })
            >>> df_cast = MultiAssetSchema.cast_to_schema(df)
            >>> df_cast['open'].dtype
            Float64
        """
        # Cast required columns
        cast_exprs = []
        for col, dtype in cls.SCHEMA.items():
            if col in df.columns:
                cast_exprs.append(pl.col(col).cast(dtype))

        # Cast optional columns if asset class specified
        if asset_class and asset_class in cls.OPTIONAL_COLUMNS:
            for col, dtype in cls.OPTIONAL_COLUMNS[asset_class].items():
                if col in df.columns:
                    cast_exprs.append(pl.col(col).cast(dtype))

        if cast_exprs:
            return df.with_columns(cast_exprs)
        return df
