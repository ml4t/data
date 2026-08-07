"""Tests for adjustments/core.py corporate actions module."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.data.adjustments.core import (
    apply_corporate_actions,
    apply_dividends,
    apply_splits,
)


class TestApplyCorporateActions:
    """Tests for apply_corporate_actions function."""

    def test_no_actions_returns_same_prices(self):
        """Test that data with no actions returns adjusted == unadjusted."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "split_ratio": [1.0, 1.0, 1.0],
                "ex-dividend": [0.0, 0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        # Adjusted prices should equal unadjusted (no adjustments needed)
        assert "adj_close" in result.columns
        assert "adj_open" in result.columns
        np.testing.assert_array_almost_equal(
            result["adj_close"].to_numpy(), result["close"].to_numpy(), decimal=6
        )

    def test_2_for_1_split(self):
        """Test 2-for-1 split adjustment."""
        # Before split: $100, after split: $50 (price halved)
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 50.0],
                "high": [105.0, 55.0],
                "low": [98.0, 48.0],
                "close": [102.0, 51.0],
                "volume": [1000.0, 2000.0],
                "split_ratio": [1.0, 2.0],  # 2-for-1 on day 2
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        np.testing.assert_allclose(result["adj_close"], [51.0, 51.0])
        np.testing.assert_allclose(result["adj_volume"], [2000.0, 2000.0])
        np.testing.assert_allclose(result["price_adjustment_factor"], [0.5, 1.0])
        np.testing.assert_allclose(result["volume_adjustment_factor"], [2.0, 1.0])

    def test_reverse_split(self):
        """Test reverse split (1-for-2) adjustment."""
        # Before reverse split: $50, after: $100 (price doubled)
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [50.0, 100.0],
                "high": [55.0, 105.0],
                "low": [48.0, 98.0],
                "close": [51.0, 102.0],
                "volume": [2000.0, 1000.0],
                "split_ratio": [1.0, 0.5],  # 1-for-2 reverse on day 2
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        np.testing.assert_allclose(result["adj_close"], [102.0, 102.0])
        np.testing.assert_allclose(result["adj_volume"], [1000.0, 1000.0])
        np.testing.assert_allclose(result["price_adjustment_factor"], [2.0, 1.0])
        np.testing.assert_allclose(result["volume_adjustment_factor"], [0.5, 1.0])

    def test_dividend_adjustment(self):
        """Test dividend adjustment."""
        # Day 2 has $1 dividend
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 101.0, 100.5],
                "high": [105.0, 106.0, 105.0],
                "low": [98.0, 99.0, 99.0],
                "close": [102.0, 103.0, 101.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "split_ratio": [1.0, 1.0, 1.0],
                "ex-dividend": [0.0, 1.0, 0.0],  # $1 dividend on day 2
            }
        )

        result = apply_corporate_actions(df)

        # Most recent day should be unchanged
        assert result["adj_close"][2] == 101.0

        # Historical days should be adjusted down for dividend
        # The adjustment accounts for total return

    def test_adjusted_return_includes_split_and_dividend(self):
        """Adjusted close return equals the shareholder wealth return."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "close": [200.0, 98.0],
                "volume": [1000.0, 2000.0],
                "split_ratio": [1.0, 2.0],
                "ex-dividend": [0.0, 3.0],
            }
        )

        result = apply_corporate_actions(df)

        adjusted_gross_return = result["adj_close"][1] / result["adj_close"][0]
        shareholder_gross_return = 2.0 * (98.0 + 3.0) / 200.0
        assert adjusted_gross_return == pytest.approx(shareholder_gross_return)

    def test_custom_price_columns(self):
        """Test with custom price columns."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "bid": [99.0, 100.0],
                "ask": [101.0, 102.0],
                "mid": [100.0, 101.0],
                "close": [100.0, 101.0],
                "split_ratio": [1.0, 1.0],
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df, price_cols=["bid", "ask", "mid"])

        assert "adj_bid" in result.columns
        assert "adj_ask" in result.columns
        assert "adj_mid" in result.columns

    def test_volume_adjustment(self):
        """Test volume is adjusted for splits."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 50.0],
                "high": [105.0, 55.0],
                "low": [98.0, 48.0],
                "close": [102.0, 51.0],
                "volume": [1000.0, 2000.0],
                "split_ratio": [1.0, 2.0],  # 2-for-1 split
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        assert "adj_volume" in result.columns

    def test_no_volume_column(self):
        """Test when volume column doesn't exist."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [98.0, 99.0],
                "close": [102.0, 103.0],
                "split_ratio": [1.0, 1.0],
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df, volume_col=None)

        assert "adj_volume" not in result.columns

    def test_data_sorted_by_date(self):
        """Test that data is sorted by date before processing."""
        # Input data is not sorted
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
                "open": [102.0, 100.0, 101.0],
                "high": [107.0, 105.0, 106.0],
                "low": [100.0, 98.0, 99.0],
                "close": [104.0, 102.0, 103.0],
                "volume": [1200.0, 1000.0, 1100.0],
                "split_ratio": [1.0, 1.0, 1.0],
                "ex-dividend": [0.0, 0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        # Result should be sorted by date
        dates = result["date"].to_list()
        assert dates == sorted(dates)

    def test_custom_column_names(self):
        """Test with custom split and dividend column names."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [98.0, 99.0],
                "close": [102.0, 103.0],
                "volume": [1000.0, 1100.0],
                "my_split": [1.0, 1.0],
                "my_div": [0.0, 0.5],
            }
        )

        result = apply_corporate_actions(
            df,
            split_col="my_split",
            dividend_col="my_div",
        )

        assert "adj_close" in result.columns


class TestApplySplits:
    """Tests for apply_splits function."""

    def test_no_splits(self):
        """Test with no splits (ratio = 1.0)."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [98.0, 99.0],
                "close": [102.0, 103.0],
                "volume": [1000.0, 1100.0],
                "split_ratio": [1.0, 1.0],
            }
        )

        result = apply_splits(df)

        np.testing.assert_array_almost_equal(
            result["adj_close"].to_numpy(), result["close"].to_numpy(), decimal=6
        )

    def test_2_for_1_split(self):
        """Test 2-for-1 split."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 50.0],
                "high": [105.0, 55.0],
                "low": [98.0, 48.0],
                "close": [102.0, 51.0],
                "volume": [1000.0, 2000.0],
                "split_ratio": [1.0, 2.0],
            }
        )

        result = apply_splits(df)

        assert "adj_close" in result.columns
        assert "adj_volume" in result.columns

    def test_matches_canonical_engine_for_forward_and_reverse_splits(self):
        """Both split entry points use the same event-date and share-basis convention."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 50.0, 100.0],
                "high": [100.0, 50.0, 100.0],
                "low": [100.0, 50.0, 100.0],
                "close": [100.0, 50.0, 100.0],
                "volume": [1000.0, 2000.0, 1000.0],
                "split_ratio": [1.0, 2.0, 0.5],
                "ex-dividend": [0.0, 0.0, 0.0],
            }
        )

        split_only = apply_splits(df)
        canonical = apply_corporate_actions(df)

        for column in (
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "adj_volume",
            "price_adjustment_factor",
            "volume_adjustment_factor",
        ):
            np.testing.assert_allclose(split_only[column], canonical[column])

    def test_custom_price_cols(self):
        """Test with custom price columns."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "price": [100.0, 101.0],
                "close": [102.0, 103.0],
                "split_ratio": [1.0, 1.0],
            }
        )

        result = apply_splits(df, price_cols=["price"])

        assert "adj_price" in result.columns
        assert "adj_close" not in result.columns  # Not in price_cols

    def test_custom_price_column_does_not_require_close(self):
        """Split-only adjustment is independent of a close column."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "price": [100.0, 50.0],
                "split_ratio": [1.0, 2.0],
            }
        )

        result = apply_splits(df, price_cols=["price"], volume_col=None)

        assert result["adj_price"].to_list() == [50.0, 50.0]

    def test_no_volume_adjustment(self):
        """Test without volume adjustment."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "close": [102.0, 103.0],
                "split_ratio": [1.0, 1.0],
            }
        )

        result = apply_splits(df, price_cols=["close"], volume_col=None)

        assert "adj_close" in result.columns
        assert "adj_volume" not in result.columns

    def test_removes_temp_column(self):
        """Test that temporary column is removed."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "close": [100.0],
                "split_ratio": [1.0],
            }
        )

        result = apply_splits(df, price_cols=["close"], volume_col=None)

        assert "_cumulative_split_factor" not in result.columns


class TestApplyDividends:
    """Tests for apply_dividends function."""

    def test_no_dividends(self):
        """Test with no dividends."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "adj_open": [100.0, 101.0],
                "adj_high": [105.0, 106.0],
                "adj_low": [98.0, 99.0],
                "adj_close": [102.0, 103.0],
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_dividends(df)

        np.testing.assert_array_almost_equal(
            result["adj_close"].to_numpy(), df["adj_close"].to_numpy(), decimal=6
        )

    def test_with_dividend(self):
        """Test dividend adjustment."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "adj_open": [100.0, 101.0],
                "adj_high": [105.0, 106.0],
                "adj_low": [98.0, 99.0],
                "adj_close": [102.0, 103.0],
                "ex-dividend": [0.0, 1.0],  # $1 dividend on day 2
            }
        )

        result = apply_dividends(df)

        adjusted_gross_return = result["adj_close"][1] / result["adj_close"][0]
        expected_gross_return = (103.0 + 1.0) / 102.0
        assert adjusted_gross_return == pytest.approx(expected_gross_return)
        assert result["adj_close"][1] == 103.0

    def test_composes_with_split_adjustment(self):
        """The two-step adapters equal the canonical combined adjustment."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 100.0, 50.0],
                "high": [100.0, 100.0, 50.0],
                "low": [100.0, 100.0, 50.0],
                "close": [100.0, 100.0, 50.0],
                "volume": [1000.0, 1000.0, 2000.0],
                "split_ratio": [1.0, 1.0, 2.0],
                "ex-dividend": [0.0, 1.0, 0.0],
            }
        )

        chained = apply_dividends(apply_splits(df))
        canonical = apply_corporate_actions(df)

        for column in (
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "price_adjustment_factor",
        ):
            np.testing.assert_allclose(chained[column], canonical[column])

    def test_custom_price_cols(self):
        """Test with custom price columns."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "price": [100.0, 101.0],
                "adj_close": [102.0, 103.0],
                "ex-dividend": [0.0, 0.0],
            }
        )

        result = apply_dividends(df, price_cols=["price"])

        # Only 'price' column should be adjusted (but output column has same name)
        assert "price" in result.columns

    def test_removes_temp_columns(self):
        """Test that temporary columns are removed."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "adj_close": [100.0],
                "ex-dividend": [0.0],
            }
        )

        result = apply_dividends(df, price_cols=["adj_close"])

        assert "_div_factor" not in result.columns
        assert "_cumulative_div_factor" not in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_row(self):
        """Test with single row of data."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [102.0],
                "volume": [1000.0],
                "split_ratio": [1.0],
                "ex-dividend": [0.0],
            }
        )

        result = apply_corporate_actions(df)

        assert len(result) == 1
        assert result["adj_close"][0] == 102.0

    def test_missing_date_uses_public_validation_error(self):
        df = pl.DataFrame(
            {
                "close": [100.0],
                "split_ratio": [1.0],
                "ex-dividend": [0.0],
            }
        )

        with pytest.raises(ValueError, match="Missing required adjustment columns: date"):
            apply_corporate_actions(df)

    @pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_split_ratios_are_rejected(self, ratio):
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "close": [100.0],
                "split_ratio": [ratio],
                "ex-dividend": [0.0],
            }
        )

        with pytest.raises(ValueError, match="Split ratios|finite"):
            apply_corporate_actions(df)

    def test_multiple_splits(self):
        """Test with multiple splits over time."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 50.0, 25.0],
                "high": [105.0, 55.0, 30.0],
                "low": [98.0, 48.0, 23.0],
                "close": [102.0, 51.0, 26.0],
                "volume": [1000.0, 2000.0, 4000.0],
                "split_ratio": [1.0, 2.0, 2.0],  # Two 2-for-1 splits
                "ex-dividend": [0.0, 0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        # All adjusted columns should exist
        assert "adj_close" in result.columns
        assert "adj_volume" in result.columns

    def test_mixed_actions(self):
        """Test with both splits and dividends."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 101.0, 50.5],
                "high": [105.0, 106.0, 55.0],
                "low": [98.0, 99.0, 48.0],
                "close": [102.0, 103.0, 51.5],
                "volume": [1000.0, 1100.0, 2200.0],
                "split_ratio": [1.0, 1.0, 2.0],  # Split on day 3
                "ex-dividend": [0.0, 0.5, 0.0],  # Dividend on day 2
            }
        )

        result = apply_corporate_actions(df)

        assert "adj_close" in result.columns
        assert "adj_volume" in result.columns
        assert len(result) == 3


class TestIntegration:
    """Integration tests for adjustment functions."""

    def test_no_action_factors_are_identity(self):
        """Rows without corporate actions retain raw values and unit factors."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "split_ratio": [1.0, 1.0, 1.0],
                "ex-dividend": [0.0, 0.0, 0.0],
            }
        )

        result = apply_corporate_actions(df)

        # With no actions, adjusted should equal unadjusted
        np.testing.assert_array_almost_equal(
            result["adj_close"].to_numpy(), result["close"].to_numpy(), decimal=6
        )
        np.testing.assert_allclose(result["price_adjustment_factor"], 1.0)
        np.testing.assert_allclose(result["volume_adjustment_factor"], 1.0)
