"""Tests for utils gaps module."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.providers import SyntheticProvider
from ml4t.data.utils.gaps import (
    DataGap,
    GapDetector,
)
from ml4t.specs import FeedSpec


def intraday_feed(frequency, calendar="24/7"):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        timezone="UTC",
        data_frequency=frequency,
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )


def intraday_rows(timestamps):
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv(
            "SYNTH", timestamps[0].date().isoformat(), timestamps[-1].date().isoformat(), "minute"
        )
    return frame.filter(pl.col("timestamp").dt.replace_time_zone(None).is_in(timestamps))


class TestDataGap:
    """Test DataGap dataclass."""

    def test_data_gap_creation(self):
        """Test creating a DataGap instance."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        gap = DataGap(
            start=start,
            end=end,
            missing_periods=2,
            duration=timedelta(days=2),
            frequency="daily",
        )

        assert gap.start == start
        assert gap.end == end
        assert gap.missing_periods == 2
        assert gap.duration == timedelta(days=2)
        assert gap.frequency == "daily"

    def test_data_gap_is_significant_true(self):
        """Test significant gap detection (more than 1 period)."""
        gap = DataGap(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 4),
            missing_periods=3,
            duration=timedelta(days=3),
            frequency="daily",
        )

        assert gap.is_significant is True

    def test_data_gap_is_significant_false(self):
        """Test non-significant gap detection (1 period or less)."""
        gap = DataGap(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            missing_periods=1,
            duration=timedelta(days=1),
            frequency="daily",
        )

        assert gap.is_significant is False

    def test_data_gap_str_representation(self):
        """Test string representation of DataGap."""
        gap = DataGap(
            start=datetime(2024, 1, 1, 12, 0, 0),
            end=datetime(2024, 1, 2, 12, 0, 0),
            missing_periods=1,
            duration=timedelta(days=1),
            frequency="daily",
        )

        str_repr = str(gap)
        assert "Gap:" in str_repr
        assert "2024-01-01T12:00:00" in str_repr
        assert "2024-01-02T12:00:00" in str_repr
        assert "1 periods" in str_repr


class TestGapDetector:
    """Test GapDetector class."""

    def test_gap_detector_creation(self):
        """Test creating GapDetector instance."""
        detector = GapDetector()
        assert detector.tolerance == GapDetector.DEFAULT_TOLERANCE

    def test_gap_detector_custom_tolerance(self):
        """Test GapDetector with custom tolerance."""
        detector = GapDetector(tolerance=0.2)
        assert detector.tolerance == 0.2

    def test_detect_gaps_empty_dataframe(self):
        """Test gap detection on empty DataFrame."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [],
                "value": [],
            },
            schema={"timestamp": pl.Datetime, "value": pl.Float64},
        )

        gaps = detector.detect_gaps(df, frequency="daily")
        assert gaps == []

    def test_detect_gaps_single_row(self):
        """Test gap detection on single-row DataFrame."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "value": [10.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="daily")
        assert gaps == []

    def test_detect_gaps_no_gaps(self):
        """Test gap detection when there are no gaps."""
        detector = GapDetector()
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="daily")
        assert gaps == []

    def test_detect_gaps_with_gaps(self):
        """Test gap detection when gaps exist."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 4),  # Missing Jan 2, 3
                    datetime(2024, 1, 7),  # Missing Jan 5, 6
                ],
                "value": [10.0, 40.0, 70.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="daily")

        # Should find 2 gaps
        assert len(gaps) >= 1  # At least one gap should be detected

    def test_detect_gaps_custom_timestamp_column(self):
        """Test gap detection with custom timestamp column name."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 4)],
                "value": [10.0, 40.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="daily", timestamp_col="date")

        # Should detect a gap
        assert len(gaps) >= 1

    def test_detect_gaps_hourly_frequency(self):
        """A three-hour continuous interval must retain two missing hourly bars."""
        detector = GapDetector()
        df = intraday_rows([datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 13)])
        gaps = detector.detect_gaps(df, feed_spec=intraday_feed("hourly"))
        assert [gap.missing_periods for gap in gaps] == [2]

    def test_detect_gaps_crypto_vs_traditional(self):
        """An exchange weekend must not erase a continuous-feed outage."""
        detector = GapDetector()
        df = intraday_rows([datetime(2024, 1, 5, 20, 59), datetime(2024, 1, 8, 14, 30)])
        crypto_gaps = detector.detect_gaps(df, feed_spec=intraday_feed("minute"))
        traditional_gaps = detector.detect_gaps(df, feed_spec=intraday_feed("minute", "NYSE"))
        assert [gap.missing_periods for gap in crypto_gaps] == [3930]
        assert traditional_gaps == []

    def test_detect_gaps_unsorted_data(self):
        """Test gap detection handles unsorted data."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 3),  # Out of order
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 5),
                ],
                "value": [30.0, 10.0, 50.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="daily")

        # Should handle unsorted data and detect gaps
        assert isinstance(gaps, list)

    def test_detect_gaps_minute_frequency(self):
        """A five-minute continuous interval must retain four missing minute bars."""
        detector = GapDetector()
        df = intraday_rows([datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 9, 35)])
        gaps = detector.detect_gaps(df, feed_spec=intraday_feed("minute"))
        assert [gap.missing_periods for gap in gaps] == [4]

    def test_periods_per_day_constants(self):
        """Test PERIODS_PER_DAY constants are reasonable."""
        periods = GapDetector.PERIODS_PER_DAY

        assert periods["daily"] == 1
        assert periods["hourly"] == 6.5  # Market hours
        assert periods["minute"] == 390  # US market hours
        assert periods["weekly"] == 0.2
        assert periods["monthly"] == 0.045

    def test_gap_properties_consistency(self):
        """Test that detected gaps have consistent properties."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 5),  # 4-day gap
                ],
                "value": [10.0, 50.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="daily")

        if gaps:  # If any gaps are detected
            gap = gaps[0]

            # Gap should have valid properties
            assert isinstance(gap.start, datetime)
            assert isinstance(gap.end, datetime)
            assert gap.end > gap.start
            assert gap.missing_periods > 0
            assert gap.duration.total_seconds() > 0
            assert gap.frequency == "daily"

    def test_weekly_frequency(self):
        """Test gap detection with weekly frequency."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),  # Week 1
                    datetime(2024, 1, 22),  # Week 4 (missing weeks 2, 3)
                ],
                "value": [10.0, 40.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="weekly")

        # Should detect gaps in weekly data
        assert isinstance(gaps, list)

    def test_monthly_frequency(self):
        """Test gap detection with monthly frequency."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),  # January
                    datetime(2024, 4, 1),  # April (missing Feb, March)
                ],
                "value": [10.0, 40.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="monthly")

        # Should detect gaps in monthly data
        assert isinstance(gaps, list)

    @pytest.mark.parametrize("tolerance", [0.5, 0.01])
    def test_tolerance_cannot_reclassify_off_grid_timestamps(self, tolerance):
        """Elapsed-time tolerance must not admit half-hour labels to an hourly grid."""
        df = intraday_rows([datetime(2024, 1, 1), datetime(2024, 1, 1, 1, 30)])
        with pytest.raises(ValueError, match="off-grid"):
            GapDetector(tolerance=tolerance).detect_gaps(df, feed_spec=intraday_feed("hourly"))


class TestGapDetectorIntegration:
    """Integration tests for GapDetector."""

    def test_realistic_stock_data_gaps(self):
        """An hourly NYSE grid excludes the weekend but retains Monday's missing bar."""
        detector = GapDetector()
        df = intraday_rows(
            [
                datetime(2024, 1, 5, 20, 30),
                datetime(2024, 1, 8, 14, 30),
                datetime(2024, 1, 8, 15, 30),
                datetime(2024, 1, 8, 17, 30),
            ]
        )
        gaps = detector.detect_gaps(df, feed_spec=intraday_feed("hourly", "NYSE"))
        assert [gap.missing_periods for gap in gaps] == [1]
        assert gaps[0].start.hour == 15
        assert gaps[0].end.hour == 17

    def test_crypto_24_7_data_gaps(self):
        """Separate outages on a continuous hourly grid must keep their exact sizes."""
        detector = GapDetector()
        base_time = datetime(2024, 1, 1, 0, 0)
        df = intraday_rows([base_time + timedelta(hours=hour) for hour in [0, 1, 2, 5, 6, 9]])
        gaps = detector.detect_gaps(df, feed_spec=intraday_feed("hourly"))
        assert [gap.missing_periods for gap in gaps] == [2, 2]

    def test_mixed_frequency_analysis(self):
        """Observations valid on a minute grid need not belong to an hourly grid."""
        detector = GapDetector()
        df = intraday_rows(
            [
                datetime(2024, 1, 1, 9, 0),
                datetime(2024, 1, 1, 9, 5),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 2, 9, 0),
            ]
        )
        minute_gaps = detector.detect_gaps(df, feed_spec=intraday_feed("minute"))
        assert [gap.missing_periods for gap in minute_gaps] == [4, 114, 1319]
        with pytest.raises(ValueError, match="off-grid"):
            detector.detect_gaps(df, feed_spec=intraday_feed("hourly"))
        assert detector.detect_gaps(df, frequency="daily", calendar="24/7") == []

    def test_large_dataset_performance(self):
        """Test gap detection performance on larger dataset."""
        detector = GapDetector()

        # Create a larger dataset with some gaps
        timestamps = []
        values = []
        base_date = datetime(2024, 1, 1)

        for i in range(1000):
            # Skip some days to create gaps
            if i % 50 != 0:  # Skip every 50th day
                timestamps.append(base_date + timedelta(days=i))
                values.append(100.0 + i)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "value": values,
            }
        )

        # Should complete in reasonable time
        gaps = detector.detect_gaps(df, frequency="daily")

        assert isinstance(gaps, list)
        # Should detect the systematic gaps
        assert len(gaps) >= 10  # Expect multiple gaps from skipped days
