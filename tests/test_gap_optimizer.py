"""Tests for gap optimizer functionality."""

from datetime import datetime, timedelta

import pytest

from ml4t.data.utils.gap_optimizer import GapOptimizer
from ml4t.data.utils.gaps import DataGap


class TestGapOptimizer:
    """Test gap optimizer functionality."""

    @pytest.fixture
    def sample_gaps(self) -> list[DataGap]:
        """Create sample gaps for testing."""
        return [
            DataGap(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
            DataGap(
                start=datetime(2024, 1, 3),
                end=datetime(2024, 1, 3),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
            DataGap(
                start=datetime(2024, 1, 5),
                end=datetime(2024, 1, 6),
                missing_periods=2,
                duration=timedelta(days=2),
                frequency="daily",
            ),
            DataGap(
                start=datetime(2024, 1, 10),
                end=datetime(2024, 1, 12),
                missing_periods=3,
                duration=timedelta(days=3),
                frequency="daily",
            ),
        ]

    def test_optimizer_init(self):
        """Test gap optimizer initialization."""
        optimizer = GapOptimizer()
        assert optimizer.max_batch_days == 30

        optimizer = GapOptimizer(max_batch_days=60)
        assert optimizer.max_batch_days == 60

    def test_consolidate_empty_gaps(self):
        """Test consolidation with no gaps."""
        optimizer = GapOptimizer()
        result = optimizer.consolidate_gaps([])
        assert result == []

    def test_consolidate_single_gap(self, sample_gaps):
        """Test consolidation with single gap."""
        optimizer = GapOptimizer()
        single_gap = sample_gaps[:1]

        result = optimizer.consolidate_gaps(single_gap)

        assert len(result) == 1
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 1))

    def test_consolidate_adjacent_gaps(self):
        """Test consolidation of adjacent gaps."""
        optimizer = GapOptimizer()
        gaps = [
            DataGap(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
            DataGap(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 2),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
            DataGap(
                start=datetime(2024, 1, 3),
                end=datetime(2024, 1, 3),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
        ]

        result = optimizer.consolidate_gaps(gaps, merge_threshold_days=1)

        # Should merge into single range
        assert len(result) == 1
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 3))

    def test_consolidate_distant_gaps(self, sample_gaps):
        """Test consolidation with distant gaps."""
        optimizer = GapOptimizer()

        result = optimizer.consolidate_gaps(sample_gaps, merge_threshold_days=1)

        # Gaps are too far apart to merge with threshold=1
        assert len(result) >= 3

    def test_consolidate_with_merge_threshold(self, sample_gaps):
        """Test consolidation with different merge thresholds."""
        optimizer = GapOptimizer()

        # With large threshold, more gaps should be merged
        result_large = optimizer.consolidate_gaps(sample_gaps, merge_threshold_days=10)
        result_small = optimizer.consolidate_gaps(sample_gaps, merge_threshold_days=1)

        assert len(result_large) <= len(result_small)

    def test_estimate_api_calls_empty(self):
        """Test API call estimation with empty gaps."""
        optimizer = GapOptimizer()

        result = optimizer.estimate_api_calls([])

        expected = {"individual_calls": 0, "consolidated_calls": 0, "savings": 0, "savings_pct": 0}
        assert result == expected

    def test_estimate_api_calls_multiple_gaps(self, sample_gaps):
        """Test API call estimation with multiple gaps."""
        optimizer = GapOptimizer()

        result = optimizer.estimate_api_calls(sample_gaps)

        assert result["individual_calls"] == len(sample_gaps)
        assert result["consolidated_calls"] <= len(sample_gaps)
        assert result["savings"] >= 0
        assert result["savings_pct"] >= 0
        assert "total_days" in result

    def test_should_consolidate_few_gaps(self):
        """Test consolidation recommendation with few gaps."""
        optimizer = GapOptimizer()

        # With only 2 gaps, should not recommend consolidation
        few_gaps = [
            DataGap(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
            DataGap(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 2),
                missing_periods=1,
                duration=timedelta(days=1),
                frequency="daily",
            ),
        ]

        result = optimizer.should_consolidate(few_gaps, threshold_gaps=3)
        assert result is False
