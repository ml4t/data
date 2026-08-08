"""Asset-specific data schemas."""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl

from ml4t.data.assets.asset_class import AssetClass


class AssetSchema:
    """Define schemas for different asset classes."""

    # Core columns present in all asset classes
    CORE_COLUMNS: ClassVar[list[str]] = ["timestamp", "open", "high", "low", "close", "volume"]

    # Asset-specific schemas
    SCHEMAS: ClassVar[dict[AssetClass, dict[str, Any]]] = {
        AssetClass.EQUITY: {
            "required": ["timestamp", "open", "high", "low", "close", "volume"],
            "optional": ["adjusted_close", "dividends", "splits", "turnover"],
            "types": {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "adjusted_close": pl.Float64,
                "dividends": pl.Float64,
                "splits": pl.Float64,
                "turnover": pl.Float64,
            },
        },
        AssetClass.CRYPTO: {
            "required": ["timestamp", "open", "high", "low", "close", "volume"],
            "optional": [
                "volume_quote",  # Volume in quote currency
                "trades_count",  # Number of trades
                "taker_buy_volume",  # Taker buy volume
                "taker_buy_quote_volume",  # Taker buy volume in quote
            ],
            "types": {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "volume_quote": pl.Float64,
                "trades_count": pl.Int64,
                "taker_buy_volume": pl.Float64,
                "taker_buy_quote_volume": pl.Float64,
            },
        },
        AssetClass.FOREX: {
            "required": ["timestamp", "open", "high", "low", "close"],
            "optional": ["volume", "bid", "ask", "spread"],
            "types": {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "bid": pl.Float64,
                "ask": pl.Float64,
                "spread": pl.Float64,
            },
        },
        AssetClass.COMMODITY: {
            "required": ["timestamp", "open", "high", "low", "close"],
            "optional": ["volume", "open_interest", "settlement"],
            "types": {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "open_interest": pl.Float64,
                "settlement": pl.Float64,
            },
        },
        AssetClass.INDEX: {
            "required": ["timestamp", "open", "high", "low", "close"],
            "optional": ["volume", "adjusted_close"],
            "types": {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "adjusted_close": pl.Float64,
            },
        },
        AssetClass.OPTION: {
            "required": [
                "timestamp",
                "strike",
                "expiry",
                "option_type",
                "bid",
                "ask",
                "last",
                "volume",
            ],
            "optional": [
                "open_interest",
                "implied_volatility",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
            ],
            "types": {
                "timestamp": pl.Datetime,
                "strike": pl.Float64,
                "expiry": pl.Date,
                "option_type": pl.Utf8,
                "bid": pl.Float64,
                "ask": pl.Float64,
                "last": pl.Float64,
                "volume": pl.Float64,
                "open_interest": pl.Float64,
                "implied_volatility": pl.Float64,
                "delta": pl.Float64,
                "gamma": pl.Float64,
                "theta": pl.Float64,
                "vega": pl.Float64,
                "rho": pl.Float64,
            },
        },
    }

    @classmethod
    def get_schema(cls, asset_class: AssetClass) -> dict[str, Any]:
        """
        Get schema for asset class.

        Args:
            asset_class: Type of asset

        Returns:
            Schema dictionary
        """
        # Default to equity schema if not found
        return cls.SCHEMAS.get(asset_class, cls.SCHEMAS[AssetClass.EQUITY])

    @classmethod
    def get_required_columns(cls, asset_class: AssetClass) -> list[str]:
        """
        Get required columns for asset class.

        Args:
            asset_class: Type of asset

        Returns:
            List of required column names
        """
        schema = cls.get_schema(asset_class)
        return schema.get("required", cls.CORE_COLUMNS)

    @classmethod
    def get_optional_columns(cls, asset_class: AssetClass) -> list[str]:
        """
        Get optional columns for asset class.

        Args:
            asset_class: Type of asset

        Returns:
            List of optional column names
        """
        schema = cls.get_schema(asset_class)
        return schema.get("optional", [])

    @classmethod
    def get_column_types(cls, asset_class: AssetClass) -> dict[str, type]:
        """
        Get column types for asset class.

        Args:
            asset_class: Type of asset

        Returns:
            Dictionary of column name to Polars type
        """
        schema = cls.get_schema(asset_class)
        return schema.get("types", {})

    @classmethod
    def validate_dataframe(
        cls, df: pl.DataFrame, asset_class: AssetClass
    ) -> tuple[bool, list[str]]:
        """
        Validate DataFrame against asset schema.

        Args:
            df: DataFrame to validate
            asset_class: Expected asset class

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        schema = cls.get_schema(asset_class)

        # Check required columns
        required = schema.get("required", [])
        for col in required:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")

        # Check column types
        types = schema.get("types", {})
        for col in df.columns:
            if col in types:
                expected_type = types[col]
                actual_type = df[col].dtype

                # Allow compatible types
                if not cls._is_compatible_type(actual_type, expected_type):
                    issues.append(f"Column {col} has type {actual_type}, expected {expected_type}")

        return len(issues) == 0, issues

    @classmethod
    def _is_compatible_type(cls, actual: Any, expected: Any) -> bool:
        """
        Check if actual type is compatible with expected type.

        Args:
            actual: Actual Polars DataType
            expected: Expected Polars DataType

        Returns:
            True if types are compatible
        """
        # Direct match
        if actual == expected:
            return True

        # Allow numeric compatibility
        numeric_types = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        if actual in numeric_types and expected in numeric_types:
            return True

        # Allow string/categorical
        return bool(actual in [pl.Utf8, pl.Categorical] and expected == pl.Utf8)

    @classmethod
    def normalize_dataframe(cls, df: pl.DataFrame, asset_class: AssetClass) -> pl.DataFrame:
        """
        Normalize DataFrame to match asset schema.

        Args:
            df: Input DataFrame
            asset_class: Target asset class

        Returns:
            Normalized DataFrame
        """
        schema = cls.get_schema(asset_class)
        types = schema.get("types", {})

        # Cast columns to expected types
        for col, expected_type in types.items():
            if col in df.columns:
                try:
                    df = df.with_columns(df[col].cast(expected_type))
                except Exception:
                    # Skip if cast fails
                    pass

        # Ensure required columns exist (with nulls if needed)
        required = schema.get("required", [])
        for col in required:
            if col not in df.columns:
                # Add missing column with nulls
                df = df.with_columns(pl.lit(None).alias(col))

        return df


def get_asset_schema(asset_class: AssetClass) -> dict[str, Any]:
    """
    Get schema for asset class.

    Args:
        asset_class: Type of asset

    Returns:
        Schema dictionary
    """
    return AssetSchema.get_schema(asset_class)


def create_empty_dataframe(asset_class: AssetClass) -> pl.DataFrame:
    """
    Create empty DataFrame with correct schema for asset class.

    Args:
        asset_class: Type of asset

    Returns:
        Empty DataFrame with correct schema
    """
    schema = AssetSchema.get_schema(asset_class)
    types = schema.get("types", {})
    required = schema.get("required", [])

    # Create empty DataFrame with required columns
    data = {}
    for col in required:
        dtype = types.get(col, pl.Float64)
        data[col] = pl.Series([], dtype=dtype)

    return pl.DataFrame(data)
