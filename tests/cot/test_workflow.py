"""Tests for COT workflow module."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from ml4t.data.cot.workflow import (
    combine_cot_ohlcv,
    combine_cot_ohlcv_pit,
    create_cot_features,
    load_combined_futures_data,
)


class TestCombineCOTOHLCV:
    """Tests for combine_cot_ohlcv function."""

    def test_basic_combine(self):
        """Test basic OHLCV and COT combination."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 2),
                    datetime(2023, 1, 3),
                    datetime(2023, 1, 4),
                    datetime(2023, 1, 5),
                    datetime(2023, 1, 6),
                ],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 3)],
                "open_interest": [50000],
                "lev_money_net": [1000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert "close" in result.columns
        assert "open_interest" in result.columns
        assert len(result) == 5

    def test_forward_fill_cot_data(self):
        """Test COT data is forward-filled to daily frequency."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 9),
                    datetime(2023, 1, 10),
                    datetime(2023, 1, 11),
                    datetime(2023, 1, 12),
                    datetime(2023, 1, 13),
                ],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        # COT data should be filled for dates after report_date
        assert (
            result.filter(pl.col("timestamp") == datetime(2023, 1, 11))["open_interest"][0] == 50000
        )
        assert (
            result.filter(pl.col("timestamp") == datetime(2023, 1, 12))["open_interest"][0] == 50000
        )

    def test_exclude_metadata_columns(self):
        """Test metadata columns are excluded from join."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 10)],
                "close": [100.0],
            }
        )

        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],
                "open_interest": [50000],
                "product": ["ES"],
                "report_type": ["traders_in_financial_futures_fut"],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert "product" not in result.columns
        assert "report_type" not in result.columns

    def test_custom_date_columns(self):
        """Test using custom date column names."""
        ohlcv = pl.DataFrame(
            {
                "date": [datetime(2023, 1, 10)],
                "close": [100.0],
            }
        )

        cot = pl.DataFrame(
            {
                "cot_date": [date(2023, 1, 10)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot, date_col="date", cot_date_col="cot_date")
        assert "open_interest" in result.columns

    def test_empty_ohlcv(self):
        """Test with empty OHLCV data."""
        ohlcv = pl.DataFrame({"timestamp": [], "close": []})
        cot = pl.DataFrame({"report_date": [date(2023, 1, 10)], "open_interest": [50000]})

        result = combine_cot_ohlcv(ohlcv, cot)
        assert result.is_empty()


class TestCombineCOTOHLCVPIT:
    """Tests for combine_cot_ohlcv_pit (point-in-time) function."""

    def test_publication_lag_applied(self):
        """Test publication lag is correctly applied."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 10),  # Tuesday (report date)
                    datetime(2023, 1, 11),  # Wednesday
                    datetime(2023, 1, 12),  # Thursday
                    datetime(2023, 1, 13),  # Friday (publication)
                    datetime(2023, 1, 16),  # Monday (conservative use)
                    datetime(2023, 1, 17),  # Tuesday
                ],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            }
        )

        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],  # Tuesday positions
                "open_interest": [50000],
            }
        )

        # Default 6-day lag (available Monday)
        result = combine_cot_ohlcv_pit(ohlcv, cot, publication_lag_days=6)

        # Before publication (first 4 rows) should have null COT data
        before_pub = result.filter(pl.col("timestamp") < datetime(2023, 1, 16))
        assert before_pub["open_interest"].null_count() == len(before_pub)

        # After publication should have COT data
        after_pub = result.filter(pl.col("timestamp") >= datetime(2023, 1, 16))
        assert after_pub["open_interest"].null_count() == 0

    def test_custom_publication_lag(self):
        """Test custom publication lag."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 10),
                    datetime(2023, 1, 13),
                    datetime(2023, 1, 14),
                ],
                "close": [100.0, 103.0, 104.0],
            }
        )

        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],
                "open_interest": [50000],
            }
        )

        # 3-day lag (Friday publication)
        result = combine_cot_ohlcv_pit(ohlcv, cot, publication_lag_days=3)

        # Jan 13 (Friday) should have COT data
        jan_13 = result.filter(pl.col("timestamp") == datetime(2023, 1, 13))
        assert jan_13["open_interest"][0] == 50000

    def test_excludes_metadata_columns(self):
        """Test metadata columns are excluded."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 16)],
                "close": [100.0],
            }
        )

        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],
                "open_interest": [50000],
                "product": ["ES"],
                "report_type": ["tff"],
            }
        )

        result = combine_cot_ohlcv_pit(ohlcv, cot)

        assert "product" not in result.columns
        assert "report_type" not in result.columns


class TestCreateCOTFeatures:
    """Tests for create_cot_features function."""

    def test_financial_futures_features(self):
        """Test feature creation for financial futures."""
        # Generate 59 weeks of data spanning multiple months
        timestamps = [datetime(2023, 1, 1) + timedelta(weeks=i) for i in range(59)]
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0] * 59,
                "open_interest": [100000.0] * 59,
                "lev_money_long": [50000.0] * 59,
                "lev_money_short": [40000.0] * 59,
                "lev_money_net": [10000.0] * 59,
                "asset_mgr_long": [30000.0] * 59,
                "asset_mgr_short": [20000.0] * 59,
                "asset_mgr_net": [10000.0] * 59,
                "dealer_long": [20000.0] * 59,
                "dealer_short": [15000.0] * 59,
                "dealer_net": [5000.0] * 59,
                "nonrept_long": [5000.0] * 59,
                "nonrept_short": [3000.0] * 59,
                "nonrept_net": [2000.0] * 59,
                "oi_change": [1000.0] * 59,
            }
        )

        result = create_cot_features(df)

        # Check financial futures features
        assert "cot_lev_money_pct_oi" in result.columns
        assert "cot_lev_money_zscore_52w" in result.columns
        assert "cot_lev_money_chg_4w" in result.columns
        assert "cot_asset_mgr_pct_oi" in result.columns
        assert "cot_dealer_pct_oi" in result.columns
        assert "cot_nonrept_pct_oi" in result.columns
        assert "cot_oi_change_pct" in result.columns

    def test_commodity_futures_features(self):
        """Test feature creation for commodity futures."""
        # Generate 59 weeks of data spanning multiple months
        timestamps = [datetime(2023, 1, 1) + timedelta(weeks=i) for i in range(59)]
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0] * 59,
                "open_interest": [100000.0] * 59,
                "commercial_long": [40000.0] * 59,
                "commercial_short": [30000.0] * 59,
                "commercial_net": [10000.0] * 59,
                "managed_money_long": [30000.0] * 59,
                "managed_money_short": [20000.0] * 59,
                "managed_money_net": [10000.0] * 59,
                "nonrept_long": [5000.0] * 59,
                "nonrept_short": [3000.0] * 59,
                "nonrept_net": [2000.0] * 59,
            }
        )

        result = create_cot_features(df)

        # Check commodity futures features
        assert "cot_commercial_pct_oi" in result.columns
        assert "cot_commercial_zscore_52w" in result.columns
        assert "cot_managed_money_pct_oi" in result.columns
        assert "cot_managed_money_zscore_52w" in result.columns
        assert "cot_managed_money_chg_4w" in result.columns

    def test_custom_prefix(self):
        """Test custom feature prefix."""
        df = pl.DataFrame(
            {
                "open_interest": [100000.0],
                "nonrept_long": [5000.0],
                "nonrept_short": [3000.0],
                "nonrept_net": [2000.0],
            }
        )

        result = create_cot_features(df, prefix="my_")
        assert "my_nonrept_pct_oi" in result.columns

    def test_no_matching_columns(self):
        """Test with no COT columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "close": [100.0],
            }
        )

        result = create_cot_features(df)
        # Should return original dataframe unchanged
        assert result.columns == df.columns

    def test_pct_oi_calculation(self):
        """Test percentage of open interest calculation."""
        df = pl.DataFrame(
            {
                "open_interest": [100000.0],
                "nonrept_long": [5000.0],
                "nonrept_short": [3000.0],
                "nonrept_net": [2000.0],
            }
        )

        result = create_cot_features(df)

        # 2000 / 100000 * 100 = 2.0%
        assert result["cot_nonrept_pct_oi"][0] == pytest.approx(2.0)


class TestLoadCombinedFuturesData:
    """Tests for load_combined_futures_data function."""

    def test_file_not_found_ohlcv(self, tmp_path):
        """Test error when OHLCV file not found."""
        with pytest.raises(FileNotFoundError, match="OHLCV data not found"):
            load_combined_futures_data(
                "ES",
                ohlcv_path=str(tmp_path / "ohlcv"),
                cot_path=str(tmp_path / "cot"),
            )

    def test_file_not_found_cot(self, tmp_path):
        """Test error when COT file not found."""
        # Create OHLCV file
        ohlcv_path = tmp_path / "ohlcv" / "product=ES"
        ohlcv_path.mkdir(parents=True)
        pl.DataFrame({"timestamp": [datetime(2023, 1, 1)], "close": [100.0]}).write_parquet(
            ohlcv_path / "data.parquet"
        )

        with pytest.raises(FileNotFoundError, match="COT data not found"):
            load_combined_futures_data(
                "ES",
                ohlcv_path=str(tmp_path / "ohlcv"),
                cot_path=str(tmp_path / "cot"),
            )

    def test_full_workflow(self, tmp_path):
        """Test full load and combine workflow."""
        # Create OHLCV file with 59 days spanning multiple months
        ohlcv_path = tmp_path / "ohlcv" / "product=ES"
        ohlcv_path.mkdir(parents=True)
        timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(59)]
        ohlcv_data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 59,
                "high": [101.0] * 59,
                "low": [99.0] * 59,
                "close": [100.5] * 59,
                "volume": [10000] * 59,
            }
        )
        ohlcv_data.write_parquet(ohlcv_path / "data.parquet")

        # Create COT file
        cot_path = tmp_path / "cot" / "product=ES"
        cot_path.mkdir(parents=True)
        cot_data = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10), date(2023, 1, 17)],
                "open_interest": [100000, 110000],
                "lev_money_net": [10000, 12000],
            }
        )
        cot_data.write_parquet(cot_path / "data.parquet")

        result = load_combined_futures_data(
            "ES",
            ohlcv_path=str(tmp_path / "ohlcv"),
            cot_path=str(tmp_path / "cot"),
        )

        assert "close" in result.columns
        assert "open_interest" in result.columns

    def test_date_filtering(self, tmp_path):
        """Test date filtering in load function."""
        # Create files
        ohlcv_path = tmp_path / "ohlcv" / "product=ES"
        ohlcv_path.mkdir(parents=True)
        # Use datetime for timestamps
        timestamps = [datetime(2023, 1, i) for i in range(1, 32)]
        ohlcv_data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0] * 31,
            }
        )
        ohlcv_data.write_parquet(ohlcv_path / "data.parquet")

        cot_path = tmp_path / "cot" / "product=ES"
        cot_path.mkdir(parents=True)
        cot_data = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],
                "open_interest": [100000],
            }
        )
        cot_data.write_parquet(cot_path / "data.parquet")

        # Use datetime objects for filtering instead of strings
        result = load_combined_futures_data(
            "ES",
            ohlcv_path=str(tmp_path / "ohlcv"),
            cot_path=str(tmp_path / "cot"),
            start_date=datetime(2023, 1, 15),
            end_date=datetime(2023, 1, 20),
        )

        assert len(result) == 6  # Jan 15-20
        assert result["timestamp"].min() >= datetime(2023, 1, 15)
        assert result["timestamp"].max() <= datetime(2023, 1, 20)
