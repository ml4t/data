"""Tests for asset schemas module."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ml4t.data.assets.asset_class import AssetClass
from ml4t.data.assets.schemas import (
    AssetSchema,
    create_empty_dataframe,
    get_asset_schema,
)


class TestAssetSchema:
    """Tests for AssetSchema class."""

    def test_core_columns(self):
        """Test CORE_COLUMNS are defined."""
        expected = ["timestamp", "open", "high", "low", "close", "volume"]
        assert expected == AssetSchema.CORE_COLUMNS

    def test_schemas_defined_for_asset_classes(self):
        """Test schemas are defined for main asset classes."""
        assert AssetClass.EQUITY in AssetSchema.SCHEMAS
        assert AssetClass.CRYPTO in AssetSchema.SCHEMAS
        assert AssetClass.FOREX in AssetSchema.SCHEMAS
        assert AssetClass.COMMODITY in AssetSchema.SCHEMAS
        assert AssetClass.INDEX in AssetSchema.SCHEMAS
        assert AssetClass.OPTION in AssetSchema.SCHEMAS


class TestGetSchema:
    """Tests for get_schema class method."""

    def test_get_equity_schema(self):
        """Test getting equity schema."""
        schema = AssetSchema.get_schema(AssetClass.EQUITY)
        assert "required" in schema
        assert "optional" in schema
        assert "types" in schema
        assert "timestamp" in schema["required"]

    def test_get_crypto_schema(self):
        """Test getting crypto schema."""
        schema = AssetSchema.get_schema(AssetClass.CRYPTO)
        assert "volume_quote" in schema["optional"]
        assert "trades_count" in schema["optional"]

    def test_get_forex_schema(self):
        """Test getting forex schema."""
        schema = AssetSchema.get_schema(AssetClass.FOREX)
        # Forex volume is optional
        assert "volume" in schema["optional"]
        assert "bid" in schema["optional"]
        assert "ask" in schema["optional"]

    def test_get_option_schema(self):
        """Test getting option schema."""
        schema = AssetSchema.get_schema(AssetClass.OPTION)
        assert "strike" in schema["required"]
        assert "expiry" in schema["required"]
        assert "option_type" in schema["required"]
        assert "implied_volatility" in schema["optional"]

    def test_get_unknown_returns_equity(self):
        """Test unknown asset class returns equity schema."""
        # Use a valid AssetClass that might not have a schema
        schema = AssetSchema.get_schema(AssetClass.ETF)
        equity_schema = AssetSchema.get_schema(AssetClass.EQUITY)
        assert schema == equity_schema


class TestGetRequiredColumns:
    """Tests for get_required_columns class method."""

    def test_equity_required_columns(self):
        """Test equity required columns."""
        required = AssetSchema.get_required_columns(AssetClass.EQUITY)
        assert "timestamp" in required
        assert "open" in required
        assert "high" in required
        assert "low" in required
        assert "close" in required
        assert "volume" in required

    def test_forex_required_columns(self):
        """Test forex required columns (no volume)."""
        required = AssetSchema.get_required_columns(AssetClass.FOREX)
        assert "timestamp" in required
        assert "close" in required
        # Volume is optional for forex
        assert "volume" not in required


class TestGetOptionalColumns:
    """Tests for get_optional_columns class method."""

    def test_equity_optional_columns(self):
        """Test equity optional columns."""
        optional = AssetSchema.get_optional_columns(AssetClass.EQUITY)
        assert "adjusted_close" in optional
        assert "dividends" in optional
        assert "splits" in optional

    def test_crypto_optional_columns(self):
        """Test crypto optional columns."""
        optional = AssetSchema.get_optional_columns(AssetClass.CRYPTO)
        assert "volume_quote" in optional
        assert "trades_count" in optional


class TestGetColumnTypes:
    """Tests for get_column_types class method."""

    def test_equity_column_types(self):
        """Test equity column types."""
        types = AssetSchema.get_column_types(AssetClass.EQUITY)
        assert types["timestamp"] == pl.Datetime
        assert types["close"] == pl.Float64
        assert types["volume"] == pl.Float64

    def test_option_column_types(self):
        """Test option column types."""
        types = AssetSchema.get_column_types(AssetClass.OPTION)
        assert types["strike"] == pl.Float64
        assert types["expiry"] == pl.Date
        assert types["option_type"] == pl.Utf8


class TestValidateDataframe:
    """Tests for validate_dataframe class method."""

    def test_valid_equity_dataframe(self):
        """Test valid equity DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )
        is_valid, issues = AssetSchema.validate_dataframe(df, AssetClass.EQUITY)
        assert is_valid
        assert issues == []

    def test_missing_required_column(self):
        """Test missing required column."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                # Missing 'close' and 'volume'
            }
        )
        is_valid, issues = AssetSchema.validate_dataframe(df, AssetClass.EQUITY)
        assert not is_valid
        assert any("close" in issue for issue in issues)

    def test_wrong_column_type(self):
        """Test wrong column type."""
        df = pl.DataFrame(
            {
                "timestamp": ["2023-01-01"],  # String instead of Datetime
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )
        is_valid, issues = AssetSchema.validate_dataframe(df, AssetClass.EQUITY)
        assert not is_valid
        assert any("timestamp" in issue for issue in issues)

    def test_compatible_numeric_types(self):
        """Test compatible numeric types are allowed."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": pl.Series([100], dtype=pl.Int64),  # Int instead of Float
                "high": pl.Series([101], dtype=pl.Int64),
                "low": pl.Series([99], dtype=pl.Int64),
                "close": pl.Series([100], dtype=pl.Int64),
                "volume": pl.Series([1000], dtype=pl.Int64),
            }
        )
        is_valid, issues = AssetSchema.validate_dataframe(df, AssetClass.EQUITY)
        assert is_valid


class TestIsCompatibleType:
    """Tests for _is_compatible_type class method."""

    def test_same_type_compatible(self):
        """Test same types are compatible."""
        assert AssetSchema._is_compatible_type(pl.Float64, pl.Float64)
        assert AssetSchema._is_compatible_type(pl.Int64, pl.Int64)

    def test_numeric_types_compatible(self):
        """Test numeric types are compatible."""
        assert AssetSchema._is_compatible_type(pl.Int64, pl.Float64)
        assert AssetSchema._is_compatible_type(pl.Float32, pl.Float64)
        assert AssetSchema._is_compatible_type(pl.Int32, pl.Int64)

    def test_string_categorical_compatible(self):
        """Test string and categorical are compatible."""
        assert AssetSchema._is_compatible_type(pl.Utf8, pl.Utf8)
        assert AssetSchema._is_compatible_type(pl.Categorical, pl.Utf8)

    def test_incompatible_types(self):
        """Test incompatible types."""
        assert not AssetSchema._is_compatible_type(pl.Utf8, pl.Float64)
        assert not AssetSchema._is_compatible_type(pl.Date, pl.Datetime)


class TestNormalizeDataframe:
    """Tests for normalize_dataframe class method."""

    def test_cast_columns_to_expected_types(self):
        """Test columns are cast to expected types."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": pl.Series([100], dtype=pl.Int64),
                "high": pl.Series([101], dtype=pl.Int64),
                "low": pl.Series([99], dtype=pl.Int64),
                "close": pl.Series([100], dtype=pl.Int64),
                "volume": pl.Series([1000], dtype=pl.Int64),
            }
        )

        result = AssetSchema.normalize_dataframe(df, AssetClass.EQUITY)

        # Check types are now Float64
        assert result["close"].dtype == pl.Float64
        assert result["volume"].dtype == pl.Float64

    def test_add_missing_required_columns(self):
        """Test missing required columns are added with nulls."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "close": [100.0],
            }
        )

        result = AssetSchema.normalize_dataframe(df, AssetClass.EQUITY)

        # Missing columns should be added
        assert "open" in result.columns
        assert "volume" in result.columns

    def test_handles_cast_failure_gracefully(self):
        """Test cast failures are handled gracefully."""
        df = pl.DataFrame(
            {
                "timestamp": ["not a datetime"],  # Can't cast to Datetime
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )

        # Should not raise, just skip the failed cast
        result = AssetSchema.normalize_dataframe(df, AssetClass.EQUITY)
        assert "timestamp" in result.columns


class TestGetAssetSchema:
    """Tests for get_asset_schema function."""

    def test_returns_schema_dict(self):
        """Test function returns schema dictionary."""
        schema = get_asset_schema(AssetClass.EQUITY)
        assert isinstance(schema, dict)
        assert "required" in schema


class TestCreateEmptyDataframe:
    """Tests for create_empty_dataframe function."""

    def test_create_empty_equity_df(self):
        """Test creating empty equity DataFrame."""
        df = create_empty_dataframe(AssetClass.EQUITY)
        assert df.is_empty()
        assert "timestamp" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_create_empty_option_df(self):
        """Test creating empty option DataFrame."""
        df = create_empty_dataframe(AssetClass.OPTION)
        assert df.is_empty()
        assert "strike" in df.columns
        assert "expiry" in df.columns
        assert "option_type" in df.columns

    def test_correct_dtypes(self):
        """Test empty DataFrame has correct dtypes."""
        df = create_empty_dataframe(AssetClass.EQUITY)
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64
