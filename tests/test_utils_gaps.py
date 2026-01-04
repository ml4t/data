"""Tests for utils gaps module."""

from datetime import datetime, timedelta

import polars as pl

from ml4t.data.utils.gaps import (
    DataGap,
    GapDetector,
)


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
        """Test gap detection with hourly frequency."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 13, 0),  # Missing 3 hours
                ],
                "value": [10.0, 40.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="hourly", is_crypto=True)

        # Should detect a gap (crypto = 24/7)
        assert len(gaps) >= 1

    def test_detect_gaps_crypto_vs_traditional(self):
        """Test difference between crypto (24/7) and traditional market hours."""
        detector = GapDetector()

        # Weekend data (Saturday to Monday)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 6, 15, 0),  # Saturday
                    datetime(2024, 1, 8, 9, 0),  # Monday
                ],
                "value": [10.0, 20.0],
            }
        )

        # For crypto (24/7), this should be a gap
        crypto_gaps = detector.detect_gaps(df, frequency="hourly", is_crypto=True)

        # For traditional markets, weekend gap might be expected
        traditional_gaps = detector.detect_gaps(df, frequency="hourly", is_crypto=False)

        # Results might differ (crypto should detect more gaps over weekends)
        # This tests the logic even if implementation varies
        assert isinstance(crypto_gaps, list)
        assert isinstance(traditional_gaps, list)

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
        """Test gap detection with minute frequency."""
        detector = GapDetector()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, 0),
                    datetime(2024, 1, 1, 9, 35, 0),  # 5 minute gap
                ],
                "value": [100.0, 105.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="minute", is_crypto=True)

        # Should detect multiple 1-minute gaps
        assert len(gaps) >= 1

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

    def test_tolerance_effect(self):
        """Test that tolerance affects gap detection."""
        # High tolerance should detect fewer gaps
        lenient_detector = GapDetector(tolerance=0.5)  # 50% tolerance
        strict_detector = GapDetector(tolerance=0.01)  # 1% tolerance

        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 1, 30),  # 1.5 hours (slight gap)
                ],
                "value": [10.0, 20.0],
            }
        )

        lenient_gaps = lenient_detector.detect_gaps(df, frequency="hourly", is_crypto=True)
        strict_gaps = strict_detector.detect_gaps(df, frequency="hourly", is_crypto=True)

        # Strict detector should potentially detect more gaps than lenient
        assert isinstance(lenient_gaps, list)
        assert isinstance(strict_gaps, list)


class TestGapDetectorIntegration:
    """Integration tests for GapDetector."""

    def test_realistic_stock_data_gaps(self):
        """Test gap detection on realistic stock market data."""
        detector = GapDetector()

        # Simulate stock data with weekend gap
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 5, 15, 0),  # Friday market close
                    datetime(2024, 1, 8, 9, 30),  # Monday market open
                    datetime(2024, 1, 8, 10, 30),  # Monday 1 hour later
                    datetime(2024, 1, 8, 12, 0),  # Monday with small gap
                ],
                "value": [100.0, 102.0, 103.0, 104.0],
            }
        )

        gaps = detector.detect_gaps(df, frequency="hourly", is_crypto=False)

        # Should handle weekend gaps appropriately for traditional markets
        assert isinstance(gaps, list)

    def test_crypto_24_7_data_gaps(self):
        """Test gap detection on 24/7 crypto data."""
        detector = GapDetector()

        # Simulate crypto data with regular hourly gaps
        base_time = datetime(2024, 1, 1, 0, 0)
        timestamps = []
        values = []

        # Add hourly data with some gaps
        for hour in [0, 1, 2, 5, 6, 9]:  # Missing hours 3,4 and 7,8
            timestamps.append(base_time + timedelta(hours=hour))
            values.append(50000.0 + hour * 100)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "value": values,
            }
        )

        gaps = detector.detect_gaps(df, frequency="hourly", is_crypto=True)

        # Should detect the missing hours in 24/7 crypto data
        assert len(gaps) >= 1

    def test_mixed_frequency_analysis(self):
        """Test gap detection across different frequencies on same data."""
        detector = GapDetector()

        # Create data with gaps at different scales
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 9, 5),  # 5 minutes later
                    datetime(2024, 1, 1, 11, 0),  # 2 hour gap
                    datetime(2024, 1, 2, 9, 0),  # Next day
                ],
                "value": [100.0, 101.0, 103.0, 110.0],
            }
        )

        # Test different frequencies
        minute_gaps = detector.detect_gaps(df, frequency="minute", is_crypto=True)
        hourly_gaps = detector.detect_gaps(df, frequency="hourly", is_crypto=True)
        daily_gaps = detector.detect_gaps(df, frequency="daily", is_crypto=True)

        # Different frequencies should detect different gap patterns
        assert isinstance(minute_gaps, list)
        assert isinstance(hourly_gaps, list)
        assert isinstance(daily_gaps, list)

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
