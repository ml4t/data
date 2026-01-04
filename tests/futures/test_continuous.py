"""Tests for continuous futures contract builder."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.futures.adjustment import BackAdjustment, RatioAdjustment
from ml4t.data.futures.continuous import (
    ContinuousContractBuilder,
    build_continuous_contract,
)
from ml4t.data.futures.roll import (
    OpenInterestBasedRoll,
    TimeBasedRoll,
    VolumeBasedRoll,
)
from ml4t.data.futures.schema import AssetClass, ContractSpec, SettlementType


# Test fixtures
@pytest.fixture
def sample_raw_data():
    """Create sample multi-contract raw data for roll detection."""
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 2, 1),
                date(2024, 2, 1),
                date(2024, 3, 1),
                date(2024, 3, 1),
            ],
            "symbol": [
                "ESH24",
                "ESM24",
                "ESH24",
                "ESM24",
                "ESH24",
                "ESM24",
                "ESH24",
                "ESM24",
            ],
            "open": [4000.0, 4010.0] * 4,
            "high": [4020.0, 4030.0] * 4,
            "low": [3980.0, 3990.0] * 4,
            "close": [4005.0, 4015.0] * 4,
            "volume": [
                10000.0,
                2000.0,  # Jan 1: ESH24 dominant
                9000.0,
                3000.0,  # Jan 2: ESH24 still dominant
                5000.0,
                8000.0,  # Feb 1: ESM24 becoming dominant
                2000.0,
                12000.0,  # Mar 1: ESM24 dominant (roll)
            ],
            "open_interest": [None] * 8,
        }
    )


@pytest.fixture
def sample_continuous_data():
    """Create sample continuous (front month) data."""
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 2, 1),
                date(2024, 3, 1),
            ],
            "open": [4000.0, 4005.0, 4010.0, 4020.0],
            "high": [4020.0, 4025.0, 4030.0, 4040.0],
            "low": [3980.0, 3985.0, 3990.0, 4000.0],
            "close": [4010.0, 4015.0, 4025.0, 4035.0],
            "volume": [10000.0, 9000.0, 8000.0, 12000.0],
            "open_interest": [None] * 4,
        }
    )


@pytest.fixture
def es_contract_spec():
    """Create E-mini S&P 500 contract spec."""
    return ContractSpec(
        ticker="ES",
        name="E-mini S&P 500",
        exchange="CME",
        asset_class=AssetClass.EQUITY_INDEX,
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.50,
        price_quote_unit="index_points",
        settlement_type=SettlementType.CASH,
        contract_months="HMUZ",
    )


class TestContinuousContractBuilderInit:
    """Tests for ContinuousContractBuilder initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        builder = ContinuousContractBuilder()

        assert builder.contract_spec is None
        assert isinstance(builder.roll_strategy, VolumeBasedRoll)
        assert isinstance(builder.adjustment_method, BackAdjustment)

    def test_init_with_contract_spec(self, es_contract_spec):
        """Test initialization with contract spec."""
        builder = ContinuousContractBuilder(contract_spec=es_contract_spec)

        assert builder.contract_spec is es_contract_spec

    def test_init_with_custom_roll_strategy(self):
        """Test initialization with custom roll strategy."""
        roll_strategy = TimeBasedRoll(days_before_expiration=5)
        builder = ContinuousContractBuilder(roll_strategy=roll_strategy)

        assert builder.roll_strategy is roll_strategy

    def test_init_with_custom_adjustment_method(self):
        """Test initialization with custom adjustment method."""
        adjustment_method = RatioAdjustment()
        builder = ContinuousContractBuilder(adjustment_method=adjustment_method)

        assert builder.adjustment_method is adjustment_method

    def test_init_with_all_parameters(self, es_contract_spec):
        """Test initialization with all parameters."""
        roll_strategy = OpenInterestBasedRoll()
        adjustment_method = RatioAdjustment()

        builder = ContinuousContractBuilder(
            contract_spec=es_contract_spec,
            roll_strategy=roll_strategy,
            adjustment_method=adjustment_method,
        )

        assert builder.contract_spec is es_contract_spec
        assert builder.roll_strategy is roll_strategy
        assert builder.adjustment_method is adjustment_method


class TestContinuousContractBuilderBuild:
    """Tests for ContinuousContractBuilder.build method."""

    def test_build_quandl_chris(self, sample_raw_data, sample_continuous_data):
        """Test building continuous contract from Quandl CHRIS data."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build("ES", data_source="quandl_chris")

            # Verify parsers were called
            mock_raw.assert_called_once_with("ES")
            mock_cont.assert_called_once_with("ES")

            # Verify result has expected columns
            assert "date" in result.columns
            assert "is_roll_date" in result.columns

    def test_build_databento(self, sample_raw_data, sample_continuous_data):
        """Test building continuous contract from Databento data."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.databento_parser.parse_databento_raw") as mock_raw,
            patch("ml4t.data.futures.databento_parser.parse_databento") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build("ES", data_source="databento")

            # Verify parsers were called with default path
            expected_path = Path("~/ml4t-data/futures").expanduser()
            mock_raw.assert_called_once_with("ES", expected_path)
            mock_cont.assert_called_once_with("ES", expected_path)

            # Verify result
            assert "date" in result.columns
            assert "is_roll_date" in result.columns

    def test_build_databento_custom_path(self, sample_raw_data, sample_continuous_data):
        """Test building from Databento with custom storage path."""
        builder = ContinuousContractBuilder()
        custom_path = "/custom/data/path"

        with (
            patch("ml4t.data.futures.databento_parser.parse_databento_raw") as mock_raw,
            patch("ml4t.data.futures.databento_parser.parse_databento") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            _result = builder.build("ES", data_source="databento", storage_path=custom_path)  # noqa: F841

            # Verify custom path was used
            mock_raw.assert_called_once_with("ES", Path(custom_path).expanduser())
            mock_cont.assert_called_once_with("ES", Path(custom_path).expanduser())

    def test_build_unknown_data_source_raises(self):
        """Test that unknown data source raises ValueError."""
        builder = ContinuousContractBuilder()

        with pytest.raises(ValueError, match="Unknown data source"):
            builder.build("ES", data_source="invalid_source")

    def test_build_applies_roll_strategy(self, sample_raw_data, sample_continuous_data):
        """Test that roll strategy is applied."""
        mock_roll_strategy = MagicMock()
        mock_roll_strategy.identify_rolls.return_value = [date(2024, 2, 1)]

        mock_adjustment = MagicMock()
        mock_adjustment.adjust.return_value = sample_continuous_data

        builder = ContinuousContractBuilder(
            roll_strategy=mock_roll_strategy,
            adjustment_method=mock_adjustment,
        )

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            _result = builder.build("ES")  # noqa: F841

            # Verify roll strategy was called with raw data
            mock_roll_strategy.identify_rolls.assert_called_once()
            call_args = mock_roll_strategy.identify_rolls.call_args
            assert call_args[0][0].equals(sample_raw_data)

    def test_build_applies_adjustment(self, sample_raw_data, sample_continuous_data):
        """Test that adjustment method is applied."""
        mock_adjustment = MagicMock()
        mock_adjustment.adjust.return_value = sample_continuous_data

        builder = ContinuousContractBuilder(adjustment_method=mock_adjustment)

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            _result = builder.build("ES")  # noqa: F841

            # Verify adjustment was called with continuous data and roll dates
            mock_adjustment.adjust.assert_called_once()

    def test_build_adds_is_roll_date_column(self, sample_raw_data, sample_continuous_data):
        """Test that is_roll_date column is added."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build("ES")

            assert "is_roll_date" in result.columns
            assert result["is_roll_date"].dtype == pl.Boolean

    def test_build_with_contract_spec(
        self, sample_raw_data, sample_continuous_data, es_contract_spec
    ):
        """Test that contract spec is passed to roll strategy."""
        mock_roll_strategy = MagicMock()
        mock_roll_strategy.identify_rolls.return_value = []

        mock_adjustment = MagicMock()
        mock_adjustment.adjust.return_value = sample_continuous_data

        builder = ContinuousContractBuilder(
            contract_spec=es_contract_spec,
            roll_strategy=mock_roll_strategy,
            adjustment_method=mock_adjustment,
        )

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            builder.build("ES")

            # Verify contract spec was passed to roll strategy
            call_args = mock_roll_strategy.identify_rolls.call_args
            assert call_args[0][1] is es_contract_spec


class TestContinuousContractBuilderBuildMultiple:
    """Tests for ContinuousContractBuilder.build_multiple method."""

    def test_build_multiple_tickers(self, sample_raw_data, sample_continuous_data):
        """Test building continuous contracts for multiple tickers."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build_multiple(["ES", "CL", "GC"])

            assert isinstance(result, dict)
            assert "ES" in result
            assert "CL" in result
            assert "GC" in result

            # Verify parsers were called for each ticker
            assert mock_raw.call_count == 3
            assert mock_cont.call_count == 3

    def test_build_multiple_empty_list(self):
        """Test building with empty ticker list."""
        builder = ContinuousContractBuilder()

        result = builder.build_multiple([])

        assert result == {}

    def test_build_multiple_databento(self, sample_raw_data, sample_continuous_data):
        """Test building multiple with Databento data source."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.databento_parser.parse_databento_raw") as mock_raw,
            patch("ml4t.data.futures.databento_parser.parse_databento") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build_multiple(
                ["ES", "CL"], data_source="databento", storage_path="/data/path"
            )

            assert len(result) == 2

    def test_build_multiple_returns_dataframes(self, sample_raw_data, sample_continuous_data):
        """Test that build_multiple returns DataFrames."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build_multiple(["ES"])

            assert isinstance(result["ES"], pl.DataFrame)


class TestBuildContinuousContractFunction:
    """Tests for build_continuous_contract convenience function."""

    def test_function_with_defaults(self, sample_raw_data, sample_continuous_data):
        """Test convenience function with default parameters."""
        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = build_continuous_contract("ES")

            assert isinstance(result, pl.DataFrame)
            assert "is_roll_date" in result.columns

    def test_function_with_custom_roll_strategy(self, sample_raw_data, sample_continuous_data):
        """Test convenience function with custom roll strategy."""
        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            roll_strategy = TimeBasedRoll(days_before_expiration=5)

            # Need to add expiration column for TimeBasedRoll
            raw_with_exp = sample_raw_data.with_columns(
                pl.lit(date(2024, 3, 15)).alias("expiration")
            )
            mock_raw.return_value = raw_with_exp

            result = build_continuous_contract("ES", roll_strategy=roll_strategy)

            assert isinstance(result, pl.DataFrame)

    def test_function_with_custom_adjustment(self, sample_raw_data, sample_continuous_data):
        """Test convenience function with custom adjustment method."""
        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = build_continuous_contract("ES", adjustment_method=RatioAdjustment())

            assert isinstance(result, pl.DataFrame)

    def test_function_with_contract_spec(
        self, sample_raw_data, sample_continuous_data, es_contract_spec
    ):
        """Test convenience function with contract spec."""
        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = build_continuous_contract("ES", contract_spec=es_contract_spec)

            assert isinstance(result, pl.DataFrame)

    def test_function_with_databento(self, sample_raw_data, sample_continuous_data):
        """Test convenience function with Databento data source."""
        with (
            patch("ml4t.data.futures.databento_parser.parse_databento_raw") as mock_raw,
            patch("ml4t.data.futures.databento_parser.parse_databento") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            result = build_continuous_contract(
                "ES", data_source="databento", storage_path="/data/path"
            )

            assert isinstance(result, pl.DataFrame)

    def test_function_invalid_data_source(self):
        """Test convenience function with invalid data source."""
        with pytest.raises(ValueError, match="Unknown data source"):
            build_continuous_contract("ES", data_source="invalid")


class TestContinuousContractEdgeCases:
    """Edge case tests for continuous contract building."""

    def test_empty_roll_dates(self, sample_continuous_data):
        """Test when no roll dates are detected."""
        builder = ContinuousContractBuilder()

        # Create raw data that won't trigger roll detection
        single_contract_data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "volume": [10000.0, 10000.0],
            }
        )

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = single_contract_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build("ES")

            # All is_roll_date should be False
            assert result["is_roll_date"].sum() == 0

    def test_all_dates_are_roll_dates(self, sample_continuous_data):
        """Test when all dates are roll dates."""
        # Mock adjustment that returns data with roll dates set
        mock_adjustment = MagicMock()
        mock_adjustment.adjust.return_value = sample_continuous_data

        mock_roll_strategy = MagicMock()
        # Return all dates as roll dates
        all_dates = sample_continuous_data["date"].to_list()
        mock_roll_strategy.identify_rolls.return_value = all_dates

        builder = ContinuousContractBuilder(
            roll_strategy=mock_roll_strategy,
            adjustment_method=mock_adjustment,
        )

        with (
            patch("ml4t.data.futures.continuous.parse_quandl_chris_raw") as mock_raw,
            patch("ml4t.data.futures.continuous.parse_quandl_chris") as mock_cont,
        ):
            mock_raw.return_value = sample_continuous_data
            mock_cont.return_value = sample_continuous_data

            result = builder.build("ES")

            # All is_roll_date should be True
            assert result["is_roll_date"].sum() == len(all_dates)

    def test_storage_path_as_string(self, sample_raw_data, sample_continuous_data):
        """Test storage path handling with string."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.databento_parser.parse_databento_raw") as mock_raw,
            patch("ml4t.data.futures.databento_parser.parse_databento") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            _result = builder.build("ES", data_source="databento", storage_path="~/my-data")  # noqa: F841

            # Verify path was expanded
            mock_raw.assert_called_once()
            call_path = mock_raw.call_args[0][1]
            assert "~" not in str(call_path)  # Should be expanded

    def test_storage_path_as_pathlib(self, sample_raw_data, sample_continuous_data):
        """Test storage path handling with Path object."""
        builder = ContinuousContractBuilder()

        with (
            patch("ml4t.data.futures.databento_parser.parse_databento_raw") as mock_raw,
            patch("ml4t.data.futures.databento_parser.parse_databento") as mock_cont,
        ):
            mock_raw.return_value = sample_raw_data
            mock_cont.return_value = sample_continuous_data

            _result = builder.build(  # noqa: F841
                "ES", data_source="databento", storage_path=Path("/data/futures")
            )

            mock_raw.assert_called_once()
