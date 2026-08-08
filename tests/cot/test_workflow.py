"""Tests for COT workflow module."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from statistics import stdev

import polars as pl
import pytest

from ml4t.data.cot.workflow import (
    attach_cot_release_schedule,
    combine_cot_ohlcv,
    combine_cot_ohlcv_pit,
    create_cot_features,
    load_combined_futures_data,
)


class TestAttachCOTReleaseSchedule:
    """Tests for authoritative release schedule attachment."""

    def test_attaches_complete_timezone_aware_schedule(self):
        cot = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2), date(2024, 1, 9)],
                "open_interest": [100_000, 110_000],
            }
        )
        schedule = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2), date(2024, 1, 9)],
                "available_at": [
                    datetime(2024, 1, 5, 20, 30, tzinfo=UTC),
                    datetime(2024, 1, 12, 20, 30, tzinfo=UTC),
                ],
            }
        )

        result = attach_cot_release_schedule(cot, schedule)

        assert result["available_at"].dtype == pl.Datetime("us", "UTC")
        assert result["open_interest"].to_list() == [100_000, 110_000]

    def test_rejects_incomplete_schedule(self):
        cot = pl.DataFrame(
            {"report_date": [date(2024, 1, 2), date(2024, 1, 9)], "open_interest": [1, 2]}
        )
        schedule = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": [datetime(2024, 1, 5, 20, 30, tzinfo=UTC)],
            }
        )

        with pytest.raises(ValueError, match="2024-01-09"):
            attach_cot_release_schedule(cot, schedule)

    def test_rejects_naive_schedule_timestamps(self):
        cot = pl.DataFrame({"report_date": [date(2024, 1, 2)]})
        schedule = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": [datetime(2024, 1, 5, 15, 30)],
            }
        )

        with pytest.raises(ValueError, match="timezone-aware"):
            attach_cot_release_schedule(cot, schedule)

    def test_rejects_date_only_schedule_timestamps(self):
        cot = pl.DataFrame({"report_date": [date(2024, 1, 2)]})
        schedule = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": [date(2024, 1, 5)],
            }
        )

        with pytest.raises(ValueError, match="timezone-aware"):
            attach_cot_release_schedule(cot, schedule)

    def test_rejects_null_schedule_timestamps(self):
        cot = pl.DataFrame({"report_date": [date(2024, 1, 2)]})
        schedule = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": pl.Series("available_at", [None], dtype=pl.Datetime("us", "UTC")),
            }
        )

        with pytest.raises(ValueError, match="timestamps cannot contain null"):
            attach_cot_release_schedule(cot, schedule)


class TestCombineCOTOHLCV:
    """Tests for combine_cot_ohlcv function."""

    def test_report_is_hidden_until_exact_release_timestamp(self):
        """A report must not be visible before its official release timestamp."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 21, tzinfo=UTC),
                    datetime(2024, 1, 5, 20, 29, tzinfo=UTC),
                    datetime(2024, 1, 5, 20, 30, tzinfo=UTC),
                ],
                "close": [100.0, 101.0, 102.0],
            }
        )
        cot = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": [datetime(2024, 1, 5, 20, 30, tzinfo=UTC)],
                "open_interest": [100_000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert result["cot_open_interest"].to_list() == [None, None, 100_000]

    def test_explicit_schedule_handles_holiday_delayed_release(self):
        """The supplied release timestamp controls holiday-delayed availability."""
        ohlcv = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 3, 29, 20, 30, tzinfo=UTC),
                    datetime(2024, 4, 1, 19, 30, tzinfo=UTC),
                ],
                "close": [100.0, 101.0],
            }
        )
        cot = pl.DataFrame(
            {
                "report_date": [date(2024, 3, 26)],
                "available_at": [datetime(2024, 4, 1, 19, 30, tzinfo=UTC)],
                "open_interest": [100_000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert result["cot_open_interest"].to_list() == [None, 100_000]

    def test_missing_release_timestamp_is_rejected(self):
        """The safe default must not infer publication from a report date."""
        ohlcv = pl.DataFrame({"timestamp": [datetime(2024, 1, 8, tzinfo=UTC)], "close": [100.0]})
        cot = pl.DataFrame({"report_date": [date(2024, 1, 2)], "open_interest": [100_000]})

        with pytest.raises(ValueError, match="available_at"):
            combine_cot_ohlcv(ohlcv, cot)

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
                "available_at": [datetime(2023, 1, 6, tzinfo=UTC)],
                "open_interest": [50000],
                "lev_money_net": [1000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert "close" in result.columns
        assert "cot_open_interest" in result.columns
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
                "available_at": [datetime(2023, 1, 11, tzinfo=UTC)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        # COT data should be filled for dates after report_date
        assert (
            result.filter(pl.col("timestamp") == datetime(2023, 1, 11))["cot_open_interest"][0]
            == 50000
        )
        assert (
            result.filter(pl.col("timestamp") == datetime(2023, 1, 12))["cot_open_interest"][0]
            == 50000
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
                "available_at": [datetime(2023, 1, 10, tzinfo=UTC)],
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
                "available_at": [datetime(2023, 1, 10, tzinfo=UTC)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot, date_col="date", cot_date_col="cot_date")
        assert "cot_open_interest" in result.columns

    def test_empty_ohlcv(self):
        """Test with empty OHLCV data."""
        ohlcv = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "close": pl.Float64})
        cot = pl.DataFrame(
            {
                "report_date": [date(2023, 1, 10)],
                "available_at": [datetime(2023, 1, 13, 20, 30, tzinfo=UTC)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)
        assert result.is_empty()

    def test_normalizes_timestamp_units_before_asof_join(self):
        ohlcv = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    "timestamp",
                    [datetime(2024, 1, 5, 20, 30, tzinfo=UTC)],
                    dtype=pl.Datetime("ns", "UTC"),
                ),
                "close": [100.0],
            }
        )
        cot = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": pl.Series(
                    "available_at",
                    [datetime(2024, 1, 5, 20, 30, tzinfo=UTC)],
                    dtype=pl.Datetime("ms", "UTC"),
                ),
                "open_interest": [100_000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert result["cot_open_interest"].to_list() == [100_000]

    def test_latest_report_wins_when_backlog_is_released_together(self):
        released_at = datetime(2025, 11, 19, 20, 30, tzinfo=UTC)
        ohlcv = pl.DataFrame({"timestamp": [released_at], "close": [100.0]})
        cot = pl.DataFrame(
            {
                "report_date": [date(2025, 9, 23), date(2025, 9, 30)],
                "available_at": [released_at, released_at],
                "open_interest": [90_000, 100_000],
            }
        )

        result = combine_cot_ohlcv(ohlcv, cot)

        assert result["cot_report_date"].to_list() == [date(2025, 9, 30)]
        assert result["cot_open_interest"].to_list() == [100_000]


class TestCombineCOTOHLCVPIT:
    """Tests for combine_cot_ohlcv_pit (point-in-time) function."""

    def test_alias_uses_official_release_timestamp(self):
        """The compatibility alias has the same exact release semantics."""
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
                "available_at": [datetime(2023, 1, 16, tzinfo=UTC)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv_pit(ohlcv, cot)

        # Before publication (first 4 rows) should have null COT data
        before_pub = result.filter(pl.col("timestamp") < datetime(2023, 1, 16))
        assert before_pub["cot_open_interest"].null_count() == len(before_pub)

        # After publication should have COT data
        after_pub = result.filter(pl.col("timestamp") >= datetime(2023, 1, 16))
        assert after_pub["cot_open_interest"].null_count() == 0

    def test_custom_availability_column(self):
        """The official release timestamp column can be renamed explicitly."""
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
                "released_at": [datetime(2023, 1, 13, tzinfo=UTC)],
                "open_interest": [50000],
            }
        )

        result = combine_cot_ohlcv_pit(ohlcv, cot, available_at_col="released_at")

        # Jan 13 (Friday) should have COT data
        jan_13 = result.filter(pl.col("timestamp") == datetime(2023, 1, 13))
        assert jan_13["cot_open_interest"][0] == 50000

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
                "available_at": [datetime(2023, 1, 13, 20, 30, tzinfo=UTC)],
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

    def test_joined_features_use_cot_open_interest(self):
        """Position percentages use the report denominator, not OHLCV open interest."""
        combined = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 8, tzinfo=UTC)],
                "open_interest": [50.0],
                "cot_report_date": [date(2024, 1, 2)],
                "cot_open_interest": [100.0],
                "lev_money_net": [10.0],
            }
        )

        result = create_cot_features(combined)

        assert result["cot_lev_money_pct_oi"][0] == pytest.approx(10.0)

    def test_joined_features_do_not_fall_back_to_market_open_interest(self):
        combined = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 8, tzinfo=UTC)],
                "open_interest": [50.0],
                "cot_report_date": [date(2024, 1, 2)],
                "lev_money_net": [10.0],
            }
        )

        with pytest.raises(ValueError, match="cot_open_interest"):
            create_cot_features(combined)

    def test_raw_weekly_rows_without_report_date_remain_supported(self):
        weekly = pl.DataFrame(
            {
                "open_interest": [100.0, 200.0],
                "lev_money_net": [10.0, 40.0],
            }
        )

        result = create_cot_features(weekly)

        assert result["cot_lev_money_pct_oi"].to_list() == [10.0, 20.0]

    def test_four_week_change_counts_reports_not_daily_rows(self):
        """A four-week feature compares distinct weekly reports after daily expansion."""
        report_dates = [date(2024, 1, 2) + timedelta(weeks=week) for week in range(5)]
        rows: list[dict[str, object]] = []
        for report_index, report_date in enumerate(report_dates):
            for day in range(5):
                rows.append(
                    {
                        "timestamp": datetime(2024, 1, 8, tzinfo=UTC)
                        + timedelta(weeks=report_index, days=day),
                        "cot_report_date": report_date,
                        "cot_open_interest": 100.0,
                        "lev_money_net": float(report_index * 10),
                    }
                )
        combined = pl.DataFrame(rows)

        result = create_cot_features(combined)

        fifth_report = result.filter(pl.col("cot_report_date") == report_dates[-1])
        assert fifth_report["cot_lev_money_chg_4w"].unique().to_list() == [40.0]

    def test_52_week_zscore_counts_distinct_reports(self):
        """A 52-week feature uses 52 reports even after daily expansion."""
        rows: list[dict[str, object]] = []
        for report_index in range(52):
            report_date = date(2023, 1, 3) + timedelta(weeks=report_index)
            for day in range(5):
                rows.append(
                    {
                        "timestamp": datetime(2023, 1, 9, tzinfo=UTC)
                        + timedelta(weeks=report_index, days=day),
                        "cot_report_date": report_date,
                        "cot_open_interest": 100.0,
                        "lev_money_net": float(report_index),
                    }
                )

        result = create_cot_features(pl.DataFrame(rows))

        last_report = result.filter(
            pl.col("cot_report_date") == date(2023, 1, 3) + timedelta(weeks=51)
        )
        expected = (51 - 25.5) / stdev(range(52))
        assert last_report["cot_lev_money_zscore_52w"].unique().to_list() == [
            pytest.approx(expected)
        ]

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
                "available_at": [
                    datetime(2023, 1, 13, 20, 30, tzinfo=UTC),
                    datetime(2023, 1, 20, 20, 30, tzinfo=UTC),
                ],
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
        assert "cot_open_interest" in result.columns

    def test_attaches_schedule_to_raw_stored_fetcher_output(self, tmp_path):
        ohlcv_path = tmp_path / "ohlcv" / "product=ES"
        ohlcv_path.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 5, 20, 30, tzinfo=UTC)],
                "close": [100.0],
            }
        ).write_parquet(ohlcv_path / "data.parquet")

        cot_path = tmp_path / "cot" / "product=ES"
        cot_path.mkdir(parents=True)
        pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "open_interest": [100_000],
                "lev_money_net": [10_000],
            }
        ).write_parquet(cot_path / "data.parquet")
        schedule = pl.DataFrame(
            {
                "report_date": [date(2024, 1, 2)],
                "available_at": [datetime(2024, 1, 5, 20, 30, tzinfo=UTC)],
            }
        )

        result = load_combined_futures_data(
            "ES",
            ohlcv_path=str(tmp_path / "ohlcv"),
            cot_path=str(tmp_path / "cot"),
            release_schedule=schedule,
        )

        assert result["cot_open_interest"].to_list() == [100_000]

    def test_requires_schedule_for_raw_stored_fetcher_output(self, tmp_path):
        ohlcv_path = tmp_path / "ohlcv" / "product=ES"
        ohlcv_path.mkdir(parents=True)
        pl.DataFrame(
            {"timestamp": [datetime(2024, 1, 8, tzinfo=UTC)], "close": [100.0]}
        ).write_parquet(ohlcv_path / "data.parquet")
        cot_path = tmp_path / "cot" / "product=ES"
        cot_path.mkdir(parents=True)
        pl.DataFrame({"report_date": [date(2024, 1, 2)], "open_interest": [100_000]}).write_parquet(
            cot_path / "data.parquet"
        )

        with pytest.raises(ValueError, match="pass release_schedule"):
            load_combined_futures_data(
                "ES",
                ohlcv_path=str(tmp_path / "ohlcv"),
                cot_path=str(tmp_path / "cot"),
            )

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
                "available_at": [datetime(2023, 1, 13, 20, 30, tzinfo=UTC)],
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
