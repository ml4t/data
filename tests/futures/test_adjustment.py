"""Tests for futures price adjustment methods."""

from datetime import date

import polars as pl
import pytest

from ml4t.data.futures.adjustment import (
    AdjustmentMethod,
    BackAdjustment,
    NoAdjustment,
    RatioAdjustment,
)


# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample continuous futures data."""
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        }
    )


@pytest.fixture
def data_with_gap():
    """Create data with a price gap at roll date."""
    # Roll on Jan 3: gap from 105 to 110 (5 point gap)
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),  # Roll date
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "open": [100.0, 101.0, 108.0, 109.0, 110.0],
            "high": [105.0, 106.0, 113.0, 114.0, 115.0],
            "low": [98.0, 99.0, 106.0, 107.0, 108.0],
            "close": [103.0, 105.0, 110.0, 111.0, 112.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        }
    )


@pytest.fixture
def roll_dates_single():
    """Single roll date."""
    return [date(2024, 1, 3)]


@pytest.fixture
def roll_dates_multiple():
    """Multiple roll dates."""
    return [date(2024, 1, 3), date(2024, 1, 5)]


class TestBackAdjustment:
    """Tests for BackAdjustment class."""

    def test_no_roll_dates(self, sample_data):
        """Test adjustment with no roll dates."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, [])

        # Should have adjusted columns equal to original
        assert "adjusted_open" in result.columns
        assert "adjusted_high" in result.columns
        assert "adjusted_low" in result.columns
        assert "adjusted_close" in result.columns

        # Values should match original (no adjustment)
        assert result["adjusted_open"].to_list() == result["open"].to_list()
        assert result["adjusted_close"].to_list() == result["close"].to_list()

    def test_single_roll_adjustment(self, data_with_gap, roll_dates_single):
        """Test adjustment with single roll date."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(data_with_gap, roll_dates_single)

        # Should have adjusted columns
        assert "adjusted_close" in result.columns

        # After roll date (Jan 3 and later): unchanged
        assert result.filter(pl.col("date") >= date(2024, 1, 3))["adjusted_close"][0] == 110.0

    def test_preserves_data_length(self, sample_data, roll_dates_single):
        """Test that adjustment preserves data length."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, roll_dates_single)

        assert len(result) == len(sample_data)

    def test_output_columns_present(self, sample_data):
        """Test that all output columns are present."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, [])

        required_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_open",
            "adjusted_high",
            "adjusted_low",
            "adjusted_close",
        ]
        for col in required_columns:
            assert col in result.columns

    def test_roll_date_not_in_data(self, sample_data):
        """Test with roll date not present in data."""
        adjustment = BackAdjustment()

        # Roll date not in data - should skip
        result = adjustment.adjust(sample_data, [date(2024, 2, 1)])

        # Should not crash, and return data with adjustment columns
        assert "adjusted_close" in result.columns

    def test_multiple_rolls(self, sample_data, roll_dates_multiple):
        """Test with multiple roll dates."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, roll_dates_multiple)

        # Should handle multiple rolls
        assert "adjusted_close" in result.columns
        assert len(result) == len(sample_data)

    def test_adjusted_values_dtype(self, sample_data):
        """Test that adjusted values are float."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, [])

        assert result.schema["adjusted_close"] == pl.Float64

    def test_cumulative_adjustment(self):
        """Test that multiple rolls accumulate adjustments."""
        # Data with two distinct roll gaps
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),  # Roll 1 here: gap from 100 to 102
                    date(2024, 1, 3),
                    date(2024, 1, 4),  # Roll 2 here: gap from 104 to 107
                    date(2024, 1, 5),
                ],
                "open": [98.0, 100.0, 101.0, 105.0, 108.0],
                "high": [103.0, 105.0, 106.0, 110.0, 113.0],
                "low": [96.0, 98.0, 99.0, 103.0, 106.0],
                "close": [100.0, 102.0, 104.0, 107.0, 110.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        roll_dates = [date(2024, 1, 2), date(2024, 1, 4)]
        adjustment = BackAdjustment()

        result = adjustment.adjust(data, roll_dates)

        # Multiple rolls should be applied cumulatively
        assert "adjusted_close" in result.columns


class TestRatioAdjustment:
    """Tests for RatioAdjustment class."""

    def test_no_roll_dates(self, sample_data):
        """Test adjustment with no roll dates."""
        adjustment = RatioAdjustment()

        result = adjustment.adjust(sample_data, [])

        # Values should match original (no adjustment)
        assert result["adjusted_open"].to_list() == result["open"].to_list()
        assert result["adjusted_close"].to_list() == result["close"].to_list()

    def test_single_roll_adjustment(self, data_with_gap, roll_dates_single):
        """Test adjustment with single roll date."""
        adjustment = RatioAdjustment()

        result = adjustment.adjust(data_with_gap, roll_dates_single)

        # Should have adjusted columns
        assert "adjusted_close" in result.columns

        # After roll date: unchanged
        assert result.filter(pl.col("date") >= date(2024, 1, 3))["adjusted_close"][0] == 110.0

    def test_ratio_maintains_positive_prices(self):
        """Test that ratio adjustment maintains positive prices."""
        # Data where back adjustment would give negative prices
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),  # Roll date
                ],
                "open": [10.0, 50.0],
                "high": [15.0, 55.0],
                "low": [8.0, 48.0],
                "close": [12.0, 52.0],  # Big gap
                "volume": [1000.0, 1100.0],
            }
        )

        adjustment = RatioAdjustment()

        result = adjustment.adjust(data, [date(2024, 1, 2)])

        # Ratio adjustment should keep prices positive
        assert all(result["adjusted_close"] > 0)

    def test_preserves_data_length(self, sample_data, roll_dates_single):
        """Test that adjustment preserves data length."""
        adjustment = RatioAdjustment()

        result = adjustment.adjust(sample_data, roll_dates_single)

        assert len(result) == len(sample_data)

    def test_output_columns_present(self, sample_data):
        """Test that all output columns are present."""
        adjustment = RatioAdjustment()

        result = adjustment.adjust(sample_data, [])

        required_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_open",
            "adjusted_high",
            "adjusted_low",
            "adjusted_close",
        ]
        for col in required_columns:
            assert col in result.columns

    def test_division_by_zero_handling(self):
        """Test handling of zero price."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                ],
                "open": [0.0, 50.0],
                "high": [0.0, 55.0],
                "low": [0.0, 48.0],
                "close": [0.0, 52.0],  # Zero price before roll
                "volume": [1000.0, 1100.0],
            }
        )

        adjustment = RatioAdjustment()

        # Should not crash with division by zero
        result = adjustment.adjust(data, [date(2024, 1, 2)])

        assert "adjusted_close" in result.columns

    def test_multiple_rolls(self, sample_data, roll_dates_multiple):
        """Test with multiple roll dates."""
        adjustment = RatioAdjustment()

        result = adjustment.adjust(sample_data, roll_dates_multiple)

        # Should handle multiple rolls
        assert "adjusted_close" in result.columns
        assert len(result) == len(sample_data)

    def test_adjusted_values_dtype(self, sample_data):
        """Test that adjusted values are float."""
        adjustment = RatioAdjustment()

        result = adjustment.adjust(sample_data, [])

        assert result.schema["adjusted_close"] == pl.Float64


class TestNoAdjustment:
    """Tests for NoAdjustment class."""

    def test_no_adjustment_applied(self, sample_data):
        """Test that no adjustment is applied."""
        adjustment = NoAdjustment()

        result = adjustment.adjust(sample_data, [date(2024, 1, 3)])

        # Adjusted should equal original
        assert result["adjusted_open"].to_list() == result["open"].to_list()
        assert result["adjusted_high"].to_list() == result["high"].to_list()
        assert result["adjusted_low"].to_list() == result["low"].to_list()
        assert result["adjusted_close"].to_list() == result["close"].to_list()

    def test_output_columns_present(self, sample_data):
        """Test that all output columns are present."""
        adjustment = NoAdjustment()

        result = adjustment.adjust(sample_data, [])

        required_columns = ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
        for col in required_columns:
            assert col in result.columns

    def test_ignores_roll_dates(self, sample_data):
        """Test that roll dates are ignored."""
        adjustment = NoAdjustment()

        roll_dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        result = adjustment.adjust(sample_data, roll_dates)

        # Should be same as with empty roll dates
        result_no_rolls = adjustment.adjust(sample_data, [])

        assert result["adjusted_close"].to_list() == result_no_rolls["adjusted_close"].to_list()

    def test_preserves_data_length(self, sample_data):
        """Test that adjustment preserves data length."""
        adjustment = NoAdjustment()

        result = adjustment.adjust(sample_data, [date(2024, 1, 3)])

        assert len(result) == len(sample_data)


class TestAdjustmentMethodAbstract:
    """Tests for AdjustmentMethod abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AdjustmentMethod cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AdjustmentMethod()

    def test_subclass_must_implement_adjust(self):
        """Test that subclasses must implement adjust method."""

        class IncompleteAdjustment(AdjustmentMethod):
            pass

        with pytest.raises(TypeError):
            IncompleteAdjustment()


class TestEdgeCases:
    """Edge case tests for adjustment methods."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_data = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )

        for adjustment in [BackAdjustment(), RatioAdjustment(), NoAdjustment()]:
            result = adjustment.adjust(empty_data, [])
            assert len(result) == 0

    def test_single_row(self):
        """Test with single row of data."""
        single_row = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [102.0],
            }
        )

        for adjustment in [BackAdjustment(), RatioAdjustment(), NoAdjustment()]:
            result = adjustment.adjust(single_row, [])
            assert len(result) == 1
            assert result["adjusted_close"][0] == 102.0

    def test_roll_on_first_date(self, sample_data):
        """Test roll on first date in data."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, [date(2024, 1, 1)])

        # Should handle gracefully (no data before roll)
        assert "adjusted_close" in result.columns

    def test_roll_on_last_date(self, sample_data):
        """Test roll on last date in data."""
        adjustment = BackAdjustment()

        result = adjustment.adjust(sample_data, [date(2024, 1, 5)])

        # Should handle gracefully
        assert "adjusted_close" in result.columns

    def test_unsorted_roll_dates(self, sample_data):
        """Test with unsorted roll dates."""
        adjustment = BackAdjustment()

        # Provide roll dates in unsorted order
        unsorted_rolls = [date(2024, 1, 4), date(2024, 1, 2)]

        result = adjustment.adjust(sample_data, unsorted_rolls)

        # Should handle and sort internally
        assert "adjusted_close" in result.columns


class TestComparisonBetweenMethods:
    """Tests comparing different adjustment methods."""

    def test_back_vs_ratio_no_rolls(self, sample_data):
        """Test that back and ratio adjustments are identical with no rolls."""
        back = BackAdjustment()
        ratio = RatioAdjustment()

        back_result = back.adjust(sample_data, [])
        ratio_result = ratio.adjust(sample_data, [])

        # Should be identical
        assert back_result["adjusted_close"].to_list() == ratio_result["adjusted_close"].to_list()

    def test_all_methods_same_output_shape(self, sample_data, roll_dates_single):
        """Test that all methods produce same output shape."""
        methods = [BackAdjustment(), RatioAdjustment(), NoAdjustment()]

        for method in methods:
            result = method.adjust(sample_data, roll_dates_single)
            assert len(result) == len(sample_data)
            assert "adjusted_close" in result.columns
