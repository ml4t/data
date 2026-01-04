"""Tests for multi-asset schema validation and utilities."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.data.core.schemas import MultiAssetSchema


class TestMultiAssetSchemaConstants:
    """Test schema constant definitions."""

    def test_schema_has_required_columns(self):
        """Verify SCHEMA contains all required columns."""
        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert set(MultiAssetSchema.SCHEMA.keys()) == set(required)

    def test_schema_data_types(self):
        """Verify SCHEMA has correct data types."""
        assert MultiAssetSchema.SCHEMA["timestamp"] == pl.Datetime("us", "UTC")
        assert MultiAssetSchema.SCHEMA["symbol"] == pl.Utf8
        assert MultiAssetSchema.SCHEMA["open"] == pl.Float64
        assert MultiAssetSchema.SCHEMA["high"] == pl.Float64
        assert MultiAssetSchema.SCHEMA["low"] == pl.Float64
        assert MultiAssetSchema.SCHEMA["close"] == pl.Float64
        assert MultiAssetSchema.SCHEMA["volume"] == pl.Float64

    def test_optional_columns_for_equities(self):
        """Verify optional columns for equities asset class."""
        assert "equities" in MultiAssetSchema.OPTIONAL_COLUMNS
        eq_cols = MultiAssetSchema.OPTIONAL_COLUMNS["equities"]
        assert "dividends" in eq_cols
        assert "splits" in eq_cols
        assert "adjusted_close" in eq_cols

    def test_optional_columns_for_crypto(self):
        """Verify optional columns for crypto asset class."""
        assert "crypto" in MultiAssetSchema.OPTIONAL_COLUMNS
        crypto_cols = MultiAssetSchema.OPTIONAL_COLUMNS["crypto"]
        assert "trades_count" in crypto_cols
        assert "taker_buy_volume" in crypto_cols
        assert "taker_buy_quote_volume" in crypto_cols

    def test_optional_columns_for_futures(self):
        """Verify optional columns for futures asset class."""
        assert "futures" in MultiAssetSchema.OPTIONAL_COLUMNS
        assert "open_interest" in MultiAssetSchema.OPTIONAL_COLUMNS["futures"]

    def test_optional_columns_for_options(self):
        """Verify optional columns for options asset class."""
        assert "options" in MultiAssetSchema.OPTIONAL_COLUMNS
        opt_cols = MultiAssetSchema.OPTIONAL_COLUMNS["options"]
        assert "open_interest" in opt_cols
        assert "implied_volatility" in opt_cols
        assert "delta" in opt_cols

    def test_column_order_definition(self):
        """Verify COLUMN_ORDER is defined correctly."""
        expected_order = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert expected_order == MultiAssetSchema.COLUMN_ORDER


class TestValidate:
    """Test schema validation."""

    @pytest.fixture
    def valid_df(self):
        """Create a valid multi-asset DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 31, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "AAPL"],
                "open": [150.0, 151.0],
                "high": [152.0, 153.0],
                "low": [149.0, 150.0],
                "close": [151.0, 152.0],
                "volume": [1000000.0, 1100000.0],
            }
        )

    def test_validate_valid_dataframe(self, valid_df):
        """Valid DataFrame passes validation."""
        assert MultiAssetSchema.validate(valid_df) is True

    def test_validate_missing_column_strict(self, valid_df):
        """Missing required column raises ValueError in strict mode."""
        df_missing = valid_df.drop("volume")
        with pytest.raises(ValueError, match="Missing required column: volume"):
            MultiAssetSchema.validate(df_missing, strict=True)

    def test_validate_missing_column_non_strict(self, valid_df):
        """Missing required column returns False in non-strict mode."""
        df_missing = valid_df.drop("symbol")
        assert MultiAssetSchema.validate(df_missing, strict=False) is False

    def test_validate_multiple_missing_columns(self, valid_df):
        """Multiple missing columns reports first alphabetically."""
        df_missing = valid_df.drop(["open", "close", "symbol"])
        with pytest.raises(ValueError, match="Missing required column: close"):
            MultiAssetSchema.validate(df_missing, strict=True)

    def test_validate_wrong_timestamp_type_strict(self, valid_df):
        """Wrong timestamp type raises ValueError in strict mode."""
        df_wrong_type = valid_df.with_columns(pl.col("timestamp").cast(pl.Int64).alias("timestamp"))
        with pytest.raises(ValueError, match="Column 'timestamp' must be Datetime"):
            MultiAssetSchema.validate(df_wrong_type, strict=True)

    def test_validate_wrong_timestamp_type_non_strict(self, valid_df):
        """Wrong timestamp type returns False in non-strict mode."""
        df_wrong_type = valid_df.with_columns(pl.col("timestamp").cast(pl.Int64).alias("timestamp"))
        assert MultiAssetSchema.validate(df_wrong_type, strict=False) is False

    def test_validate_wrong_numeric_type_strict(self, valid_df):
        """Wrong numeric type (non-numeric) raises ValueError in strict mode."""
        df_wrong_type = valid_df.with_columns(pl.col("open").cast(pl.Utf8).alias("open"))
        with pytest.raises(ValueError, match="Column 'open' must be numeric"):
            MultiAssetSchema.validate(df_wrong_type, strict=True)

    def test_validate_wrong_numeric_type_non_strict(self, valid_df):
        """Wrong numeric type returns False in non-strict mode."""
        df_wrong_type = valid_df.with_columns(pl.col("volume").cast(pl.Utf8).alias("volume"))
        assert MultiAssetSchema.validate(df_wrong_type, strict=False) is False

    def test_validate_accepts_compatible_numeric_types(self, valid_df):
        """Compatible numeric types (Float32, Int64, etc.) are accepted."""
        # Float32
        df_float32 = valid_df.with_columns(pl.col("open").cast(pl.Float32))
        assert MultiAssetSchema.validate(df_float32) is True

        # Int64
        df_int64 = valid_df.with_columns(pl.col("volume").cast(pl.Int64))
        assert MultiAssetSchema.validate(df_int64) is True

        # Int32
        df_int32 = valid_df.with_columns(pl.col("close").cast(pl.Int32))
        assert MultiAssetSchema.validate(df_int32) is True

        # UInt64
        df_uint64 = valid_df.with_columns(pl.col("high").cast(pl.UInt64))
        assert MultiAssetSchema.validate(df_uint64) is True

    def test_validate_accepts_categorical_symbol(self, valid_df):
        """Categorical type is accepted for symbol column."""
        df_categorical = valid_df.with_columns(pl.col("symbol").cast(pl.Categorical))
        assert MultiAssetSchema.validate(df_categorical) is True

    def test_validate_wrong_symbol_type_strict(self, valid_df):
        """Wrong symbol type raises ValueError in strict mode."""
        # Create DataFrame with integer symbol column directly
        df_wrong_type = pl.DataFrame(
            {
                "timestamp": valid_df["timestamp"],
                "symbol": [1, 2],  # Int64 instead of string
                "open": valid_df["open"],
                "high": valid_df["high"],
                "low": valid_df["low"],
                "close": valid_df["close"],
                "volume": valid_df["volume"],
            }
        )
        with pytest.raises(ValueError, match="Column 'symbol' must be string type"):
            MultiAssetSchema.validate(df_wrong_type, strict=True)

    def test_validate_wrong_symbol_type_non_strict(self, valid_df):
        """Wrong symbol type returns False in non-strict mode."""
        # Create DataFrame with integer symbol column directly
        df_wrong_type = pl.DataFrame(
            {
                "timestamp": valid_df["timestamp"],
                "symbol": [1, 2],  # Int64 instead of string
                "open": valid_df["open"],
                "high": valid_df["high"],
                "low": valid_df["low"],
                "close": valid_df["close"],
                "volume": valid_df["volume"],
            }
        )
        assert MultiAssetSchema.validate(df_wrong_type, strict=False) is False

    def test_validate_empty_dataframe(self):
        """Empty DataFrame with correct schema is valid."""
        df_empty = MultiAssetSchema.create_empty()
        assert MultiAssetSchema.validate(df_empty) is True

    def test_validate_with_extra_columns(self, valid_df):
        """DataFrame with extra columns is still valid."""
        df_extra = valid_df.with_columns(
            pl.lit(0.0).alias("dividends"),
            pl.lit(1.0).alias("splits"),
        )
        assert MultiAssetSchema.validate(df_extra) is True


class TestCreateEmpty:
    """Test empty DataFrame creation."""

    def test_create_empty_basic(self):
        """Create empty DataFrame with only required columns."""
        df = MultiAssetSchema.create_empty()

        # Check columns
        assert set(df.columns) == set(MultiAssetSchema.SCHEMA.keys())

        # Check types
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df["symbol"].dtype == pl.Utf8
        assert df["open"].dtype == pl.Float64
        assert df["high"].dtype == pl.Float64
        assert df["low"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Check empty
        assert len(df) == 0

    def test_create_empty_equities(self):
        """Create empty DataFrame with equities-specific columns."""
        df = MultiAssetSchema.create_empty("equities")

        # Should have required + optional equities columns
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "dividends" in df.columns
        assert "splits" in df.columns
        assert "adjusted_close" in df.columns

        # Check optional column types
        assert df["dividends"].dtype == pl.Float64
        assert df["splits"].dtype == pl.Float64
        assert df["adjusted_close"].dtype == pl.Float64

    def test_create_empty_equity_alias(self):
        """Create empty DataFrame using 'equity' alias."""
        df = MultiAssetSchema.create_empty("equity")

        # Should have same columns as 'equities'
        assert "dividends" in df.columns
        assert "splits" in df.columns

    def test_create_empty_crypto(self):
        """Create empty DataFrame with crypto-specific columns."""
        df = MultiAssetSchema.create_empty("crypto")

        # Should have required + optional crypto columns
        assert "trades_count" in df.columns
        assert "taker_buy_volume" in df.columns
        assert "taker_buy_quote_volume" in df.columns

        # Check types
        assert df["trades_count"].dtype == pl.Int64
        assert df["taker_buy_volume"].dtype == pl.Float64

    def test_create_empty_futures(self):
        """Create empty DataFrame with futures-specific columns."""
        df = MultiAssetSchema.create_empty("futures")

        assert "open_interest" in df.columns
        assert df["open_interest"].dtype == pl.Float64

    def test_create_empty_options(self):
        """Create empty DataFrame with options-specific columns."""
        df = MultiAssetSchema.create_empty("options")

        assert "open_interest" in df.columns
        assert "implied_volatility" in df.columns
        assert "delta" in df.columns
        assert "gamma" in df.columns
        assert "theta" in df.columns
        assert "vega" in df.columns

    def test_create_empty_unknown_asset_class(self):
        """Unknown asset class creates DataFrame with only required columns."""
        df = MultiAssetSchema.create_empty("unknown_asset_class")

        # Should only have required columns
        assert set(df.columns) == set(MultiAssetSchema.SCHEMA.keys())


class TestAddSymbolColumn:
    """Test adding symbol column to single-asset data."""

    @pytest.fixture
    def single_asset_df(self):
        """Create a single-asset DataFrame without symbol column."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 31, tzinfo=UTC),
                ],
                "open": [150.0, 151.0],
                "high": [152.0, 153.0],
                "low": [149.0, 150.0],
                "close": [151.0, 152.0],
                "volume": [1000000.0, 1100000.0],
            }
        )

    def test_add_symbol_column(self, single_asset_df):
        """Add symbol column to DataFrame without symbol."""
        df_with_symbol = MultiAssetSchema.add_symbol_column(single_asset_df, "AAPL")

        assert "symbol" in df_with_symbol.columns
        assert df_with_symbol["symbol"].to_list() == ["AAPL", "AAPL"]

        # Should have all original columns plus symbol
        assert len(df_with_symbol.columns) == len(single_asset_df.columns) + 1

    def test_add_symbol_column_preserves_data(self, single_asset_df):
        """Adding symbol column preserves original data."""
        df_with_symbol = MultiAssetSchema.add_symbol_column(single_asset_df, "AAPL")

        # Check OHLCV data is unchanged
        assert df_with_symbol["open"].to_list() == single_asset_df["open"].to_list()
        assert df_with_symbol["high"].to_list() == single_asset_df["high"].to_list()
        assert df_with_symbol["low"].to_list() == single_asset_df["low"].to_list()
        assert df_with_symbol["close"].to_list() == single_asset_df["close"].to_list()
        assert df_with_symbol["volume"].to_list() == single_asset_df["volume"].to_list()

    def test_add_symbol_column_idempotent(self, single_asset_df):
        """Adding symbol column is idempotent when symbol matches."""
        df1 = MultiAssetSchema.add_symbol_column(single_asset_df, "AAPL")
        df2 = MultiAssetSchema.add_symbol_column(df1, "AAPL")

        # Should be identical
        assert df1.equals(df2)

    def test_add_symbol_column_different_symbol_raises(self, single_asset_df):
        """Adding different symbol to DataFrame with symbol raises error."""
        df_with_symbol = MultiAssetSchema.add_symbol_column(single_asset_df, "AAPL")

        with pytest.raises(ValueError, match="already has 'symbol' column"):
            MultiAssetSchema.add_symbol_column(df_with_symbol, "MSFT")

    def test_add_symbol_column_mixed_symbols_raises(self):
        """DataFrame with mixed symbols raises error."""
        df_mixed = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 31, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "MSFT"],
                "open": [150.0, 250.0],
                "high": [152.0, 252.0],
                "low": [149.0, 249.0],
                "close": [151.0, 251.0],
                "volume": [1000000.0, 2000000.0],
            }
        )

        with pytest.raises(ValueError, match="already has 'symbol' column"):
            MultiAssetSchema.add_symbol_column(df_mixed, "GOOGL")

    def test_add_symbol_column_empty_dataframe(self):
        """Add symbol column to empty DataFrame."""
        df_empty = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).cast(
            {
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

        df_with_symbol = MultiAssetSchema.add_symbol_column(df_empty, "AAPL")

        assert "symbol" in df_with_symbol.columns
        assert len(df_with_symbol) == 0


class TestStandardizeOrder:
    """Test standardizing column order and sorting."""

    def test_standardize_order_sorts_by_timestamp_symbol(self):
        """DataFrame is sorted by [timestamp, symbol]."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 31, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "MSFT", "AAPL"],
                "open": [151.0, 250.0, 150.0],
                "high": [153.0, 252.0, 152.0],
                "low": [150.0, 249.0, 149.0],
                "close": [152.0, 251.0, 151.0],
                "volume": [1100000.0, 2000000.0, 1000000.0],
            }
        )

        standardized = MultiAssetSchema.standardize_order(df)

        # Should be sorted by timestamp then symbol
        expected_symbols = ["AAPL", "MSFT", "AAPL"]
        expected_opens = [150.0, 250.0, 151.0]

        assert standardized["symbol"].to_list() == expected_symbols
        assert standardized["open"].to_list() == expected_opens

    def test_standardize_order_column_order(self):
        """Columns are reordered to match COLUMN_ORDER."""
        df = pl.DataFrame(
            {
                "volume": [1000000.0],
                "symbol": ["AAPL"],
                "close": [151.0],
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
            }
        )

        standardized = MultiAssetSchema.standardize_order(df)

        # Should match COLUMN_ORDER
        assert standardized.columns == MultiAssetSchema.COLUMN_ORDER

    def test_standardize_order_preserves_extra_columns(self):
        """Extra columns are preserved after standard columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.0],
                "volume": [1000000.0],
                "dividends": [0.5],
                "splits": [1.0],
            }
        )

        standardized = MultiAssetSchema.standardize_order(df)

        # First 7 columns should be standard order
        assert standardized.columns[:7] == MultiAssetSchema.COLUMN_ORDER

        # Extra columns should be present
        assert "dividends" in standardized.columns
        assert "splits" in standardized.columns

    def test_standardize_order_empty_dataframe(self):
        """Standardize order works on empty DataFrame."""
        df_empty = MultiAssetSchema.create_empty()
        standardized = MultiAssetSchema.standardize_order(df_empty)

        assert standardized.columns == MultiAssetSchema.COLUMN_ORDER
        assert len(standardized) == 0

    def test_standardize_order_single_row(self):
        """Standardize order works on single-row DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.0],
                "volume": [1000000.0],
            }
        )

        standardized = MultiAssetSchema.standardize_order(df)
        assert len(standardized) == 1
        assert standardized.columns == MultiAssetSchema.COLUMN_ORDER


class TestCastToSchema:
    """Test casting DataFrame to schema types."""

    def test_cast_to_schema_int_to_float(self):
        """Cast integer OHLCV columns to Float64."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [150],  # Int64
                "high": [152],
                "low": [149],
                "close": [151],
                "volume": [1000000],
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df)

        assert df_cast["open"].dtype == pl.Float64
        assert df_cast["high"].dtype == pl.Float64
        assert df_cast["low"].dtype == pl.Float64
        assert df_cast["close"].dtype == pl.Float64
        assert df_cast["volume"].dtype == pl.Float64

    def test_cast_to_schema_preserves_correct_types(self):
        """Already-correct types are preserved."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.0],
                "volume": [1000000.0],
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df)

        # Should be unchanged
        assert df_cast.equals(df)

    def test_cast_to_schema_with_asset_class(self):
        """Cast optional columns when asset_class provided."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.0],
                "volume": [1000000.0],
                "dividends": [0],  # Int64
                "splits": [1],  # Int64
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df, asset_class="equities")

        assert df_cast["dividends"].dtype == pl.Float64
        assert df_cast["splits"].dtype == pl.Float64

    def test_cast_to_schema_crypto_columns(self):
        """Cast crypto-specific columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["BTCUSDT"],
                "open": [50000.0],
                "high": [51000.0],
                "low": [49000.0],
                "close": [50500.0],
                "volume": [100.0],
                "trades_count": [1000],  # Should stay Int64
                "taker_buy_volume": [50],  # Should cast to Float64
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df, asset_class="crypto")

        assert df_cast["trades_count"].dtype == pl.Int64
        assert df_cast["taker_buy_volume"].dtype == pl.Float64

    def test_cast_to_schema_empty_dataframe(self):
        """Cast empty DataFrame."""
        df_empty = pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).cast(
            {
                "timestamp": pl.Datetime("us", "UTC"),
                "symbol": pl.Utf8,
                "open": pl.Int64,
                "high": pl.Int64,
                "low": pl.Int64,
                "close": pl.Int64,
                "volume": pl.Int64,
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df_empty)

        assert df_cast["open"].dtype == pl.Float64
        assert len(df_cast) == 0

    def test_cast_to_schema_no_casting_needed(self):
        """DataFrame with no schema columns returns unchanged."""
        # DataFrame with only extra columns (no schema columns to cast)
        df = pl.DataFrame(
            {
                "extra_col1": [1, 2, 3],
                "extra_col2": ["a", "b", "c"],
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df)

        # Should be identical since no casting needed
        assert df_cast.equals(df)


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_workflow_single_to_multi_asset(self):
        """Complete workflow: single asset → add symbol → standardize → validate."""
        # Start with single-asset data
        single_asset = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 31, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                ],
                "open": [151.0, 150.0],
                "high": [153.0, 152.0],
                "low": [150.0, 149.0],
                "close": [152.0, 151.0],
                "volume": [1100000.0, 1000000.0],
            }
        )

        # Add symbol
        with_symbol = MultiAssetSchema.add_symbol_column(single_asset, "AAPL")

        # Standardize
        standardized = MultiAssetSchema.standardize_order(with_symbol)

        # Validate
        assert MultiAssetSchema.validate(standardized) is True

        # Check results
        assert standardized.columns == MultiAssetSchema.COLUMN_ORDER
        assert standardized["symbol"].to_list() == ["AAPL", "AAPL"]
        # Should be sorted by timestamp
        assert standardized["timestamp"][0] < standardized["timestamp"][1]

    def test_create_and_validate_equity_dataframe(self):
        """Create equity DataFrame and validate."""
        df = MultiAssetSchema.create_empty("equities")

        # Should validate successfully
        assert MultiAssetSchema.validate(df) is True

    def test_cast_and_validate(self):
        """Cast DataFrame then validate."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [150],  # Int64
                "high": [152],
                "low": [149],
                "close": [151],
                "volume": [1000000],
            }
        )

        df_cast = MultiAssetSchema.cast_to_schema(df)
        assert MultiAssetSchema.validate(df_cast) is True
