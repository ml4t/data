"""Validation mixin for OHLCV data."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Literal

import polars as pl
import structlog

from ml4t.data.core.exceptions import DataValidationError

logger = structlog.get_logger()

# OHLC validation modes
OhlcMode = Literal["strict", "drop", "warn"]


class ValidationMixin:
    """Mixin providing OHLCV data validation.

    Validates that data conforms to the canonical OHLCV schema
    and enforces OHLC invariants (high >= low, etc.).

    The ``ohlc_mode`` attribute controls how OHLC violations are handled:

    - ``"strict"`` (default): raise ``DataValidationError``
    - ``"drop"``: log and drop invalid rows
    - ``"warn"``: log a warning but keep all rows

    Set ``ohlc_mode`` on the provider instance or override in subclass.

    Example:
        class MyProvider(ValidationMixin):
            def fetch_data(self, symbol):
                data = self._do_fetch(symbol)
                return self._validate_ohlcv(data, "my_provider")
    """

    CANONICAL_COLUMNS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    NUMERIC_COLUMNS: ClassVar[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = CANONICAL_COLUMNS
    CANONICAL_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "timestamp": pl.Datetime("us", "UTC"),
        "symbol": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    # OHLC validation mode: "strict" | "drop" | "warn"
    ohlc_mode: OhlcMode = "strict"

    def _validate_inputs(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str,  # noqa: ARG002
    ) -> None:
        """Validate input parameters.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Raises:
            ValueError: If inputs are invalid
        """
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {e}") from e

        if start_dt > end_dt:
            raise ValueError("Start date must be before or equal to end date")

    def _validate_ohlcv(
        self,
        df: pl.DataFrame,
        provider_name: str,
        symbol: str | None = None,
    ) -> pl.DataFrame:
        """Validate and normalize OHLCV data.

        Args:
            df: DataFrame to validate
            provider_name: Provider name for error messages
            symbol: Requested symbol, used to add or verify row identity

        Returns:
            Validated and normalized DataFrame

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(df, pl.DataFrame):
            raise DataValidationError(provider_name, "Provider returned a non-DataFrame response")

        expected_symbol = self._expected_ohlcv_symbol(symbol) if symbol is not None else None

        # Several provider APIs use a zero-column DataFrame as their no-data sentinel.
        if df.is_empty() and df.width == 0:
            logger.debug("Empty response normalized to the canonical OHLCV schema")
            return pl.DataFrame(schema=self.CANONICAL_SCHEMA)

        # Check required columns
        required_source_columns = set(self.CANONICAL_COLUMNS) - {"symbol"}
        missing = required_source_columns - set(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise DataValidationError(provider_name, f"Missing required columns: {missing_list}")

        if "symbol" not in df.columns:
            if expected_symbol is None:
                raise DataValidationError(provider_name, "Missing required column: symbol")
            df = df.with_columns(pl.lit(expected_symbol).alias("symbol"))

        self._validate_no_nulls(df, provider_name, list(self.CANONICAL_COLUMNS))
        df = self._normalize_schema(df, provider_name)
        self._validate_symbol_identity(df, provider_name, expected_symbol)
        self._validate_finite_values(df, provider_name)

        # Validate OHLC invariants (may drop rows depending on ohlc_mode)
        df = self._validate_ohlc_invariants(df, provider_name)

        duplicate_count = int(
            df.select(pl.struct(["timestamp", "symbol"]).is_duplicated().sum()).item()
        )
        if duplicate_count:
            raise DataValidationError(
                provider_name,
                f"Found {duplicate_count} duplicate timestamp and symbol rows",
            )

        extra_columns = [column for column in df.columns if column not in self.CANONICAL_COLUMNS]
        df = df.sort(["timestamp", "symbol"]).select([*self.CANONICAL_COLUMNS, *extra_columns])

        return df

    def _expected_ohlcv_symbol(self, symbol: str) -> str:
        """Return the public symbol identity expected in a single-symbol response."""
        return symbol.strip().upper()

    def _normalize_schema(self, df: pl.DataFrame, provider_name: str) -> pl.DataFrame:
        """Apply safe casts to the version 1 canonical OHLCV schema."""
        timestamp_type = df.schema["timestamp"]
        if not isinstance(timestamp_type, pl.Datetime):
            raise DataValidationError(
                provider_name,
                "Column 'timestamp' must be a timezone-aware Datetime",
                field="timestamp",
            )
        if timestamp_type.time_zone is None and not df.is_empty():
            raise DataValidationError(
                provider_name,
                "Column 'timestamp' must be timezone-aware",
                field="timestamp",
            )

        symbol_type = df.schema["symbol"]
        if symbol_type != pl.String:
            raise DataValidationError(
                provider_name,
                "Column 'symbol' must be a string",
                field="symbol",
            )

        for column in self.NUMERIC_COLUMNS:
            if not df.schema[column].is_numeric():
                raise DataValidationError(
                    provider_name,
                    f"Column '{column}' must be numeric",
                    field=column,
                )

        timestamp = pl.col("timestamp")
        if timestamp_type.time_zone is None:
            timestamp = timestamp.dt.replace_time_zone("UTC")
        else:
            timestamp = timestamp.dt.convert_time_zone("UTC")

        return df.with_columns(
            timestamp.cast(pl.Datetime("us", "UTC")),
            pl.col("symbol").str.to_uppercase(),
            *(pl.col(column).cast(pl.Float64) for column in self.NUMERIC_COLUMNS),
        )

    def _validate_symbol_identity(
        self,
        df: pl.DataFrame,
        provider_name: str,
        expected_symbol: str | None,
    ) -> None:
        """Ensure a single-symbol request cannot return another instrument."""
        if df.is_empty():
            return

        symbols = df["symbol"].unique().to_list()
        if len(symbols) != 1:
            raise DataValidationError(
                provider_name,
                f"Response contains {len(symbols)} symbols for a single-symbol request",
                field="symbol",
            )
        if expected_symbol is not None and symbols[0] != expected_symbol:
            raise DataValidationError(
                provider_name,
                f"Response symbol '{symbols[0]}' does not match requested symbol '{expected_symbol}'",
                field="symbol",
                value=symbols[0],
            )

    def _validate_finite_values(self, df: pl.DataFrame, provider_name: str) -> None:
        """Reject NaN and infinite OHLCV values."""
        for column in self.NUMERIC_COLUMNS:
            invalid_count = int((~df[column].is_finite()).sum())
            if invalid_count:
                raise DataValidationError(
                    provider_name,
                    f"Column '{column}' contains {invalid_count} non-finite values",
                    field=column,
                )

        negative_volume_count = int((df["volume"] < 0).sum())
        if negative_volume_count:
            raise DataValidationError(
                provider_name,
                f"Column 'volume' contains {negative_volume_count} negative values",
                field="volume",
            )

    def _validate_ohlc_invariants(
        self,
        df: pl.DataFrame,
        provider_name: str,
    ) -> pl.DataFrame:
        """Validate OHLC price invariants.

        Checks:
            - high >= low
            - high >= open
            - high >= close
            - low <= open
            - low <= close

        Behaviour depends on ``self.ohlc_mode``:

        - ``"strict"``: raise on any violation
        - ``"drop"``: remove violating rows, return cleaned frame
        - ``"warn"``: log but keep all rows

        Args:
            df: DataFrame to validate
            provider_name: Provider name for error messages

        Returns:
            DataFrame (potentially with rows removed in ``"drop"`` mode)

        Raises:
            DataValidationError: Only in ``"strict"`` mode
        """
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if not invalid_ohlc.any():
            return df

        n_invalid = int(invalid_ohlc.sum())
        mode: OhlcMode = getattr(self, "ohlc_mode", "strict")

        if mode == "strict":
            raise DataValidationError(
                provider_name, f"Found {n_invalid} rows with invalid OHLC relationships"
            )

        if mode == "drop":
            logger.info(
                "Dropped rows with invalid OHLC",
                provider=provider_name,
                n_dropped=n_invalid,
                n_total=len(df),
            )
            return df.filter(~invalid_ohlc)

        # mode == "warn"
        logger.warning(
            "Rows with invalid OHLC relationships (kept)",
            provider=provider_name,
            n_invalid=n_invalid,
            n_total=len(df),
        )
        return df

    def _validate_no_nulls(
        self,
        df: pl.DataFrame,
        provider_name: str,
        columns: list[str] | None = None,
    ) -> None:
        """Validate no null values in specified columns.

        Args:
            df: DataFrame to validate
            provider_name: Provider name for error messages
            columns: Columns to check (default: all required columns)

        Raises:
            DataValidationError: If nulls found
        """
        check_columns = columns or self.REQUIRED_COLUMNS

        for col in check_columns:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    raise DataValidationError(
                        provider_name,
                        f"Column '{col}' contains {null_count} null values",
                    )

    def _validate_positive_values(
        self,
        df: pl.DataFrame,
        provider_name: str,
    ) -> None:
        """Validate prices and volume are positive.

        Args:
            df: DataFrame to validate
            provider_name: Provider name for error messages

        Raises:
            DataValidationError: If negative values found
        """
        numeric_cols = ["open", "high", "low", "close", "volume"]

        for col in numeric_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    raise DataValidationError(
                        provider_name,
                        f"Column '{col}' contains {negative_count} negative values",
                    )
