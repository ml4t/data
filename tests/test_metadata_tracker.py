"""Tests for metadata tracking functionality."""

from datetime import datetime, timedelta
from pathlib import Path

from ml4t.data.storage.metadata_tracker import (
    DatasetMetadata,
    MetadataTracker,
    UpdateRecord,
)


class TestUpdateRecord:
    """Test UpdateRecord class."""

    def test_create_update_record(self) -> None:
        """Test creating an update record."""
        record = UpdateRecord(
            timestamp=datetime(2024, 1, 1, 12, 0),
            update_type="incremental",
            rows_before=100,
            rows_after=150,
            rows_added=50,
            rows_updated=10,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            provider="yahoo",
            duration_seconds=5.2,
            gaps_filled=2,
            errors=["Warning: some data missing"],
        )

        assert record.update_type == "incremental"
        assert record.rows_added == 50
        assert record.gaps_filled == 2
        assert len(record.errors) == 1

    def test_update_record_serialization(self) -> None:
        """Test serialization of update record."""
        record = UpdateRecord(
            timestamp=datetime(2024, 1, 1, 12, 0),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            provider="yahoo",
            duration_seconds=3.5,
        )

        # Convert to dict
        data = record.to_dict()
        assert data["update_type"] == "full"
        assert data["rows_added"] == 100

        # Recreate from dict
        record2 = UpdateRecord.from_dict(data)
        assert record2.update_type == record.update_type
        assert record2.rows_added == record.rows_added
        assert record2.timestamp == record.timestamp


class TestDatasetMetadata:
    """Test DatasetMetadata class."""

    def test_create_metadata(self) -> None:
        """Test creating dataset metadata."""
        metadata = DatasetMetadata(
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            provider="yahoo",
            first_update=datetime(2024, 1, 1),
            last_update=datetime(2024, 1, 15),
            total_rows=1000,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 1, 15),
            update_count=5,
            health_status="healthy",
        )

        assert metadata.symbol == "AAPL"
        assert metadata.update_count == 5
        assert metadata.health_status == "healthy"

    def test_metadata_serialization(self) -> None:
        """Test serialization of metadata."""
        metadata = DatasetMetadata(
            symbol="GOOGL",
            asset_class="equities",
            frequency="daily",
            provider="yahoo",
            first_update=datetime(2024, 1, 1),
            last_update=datetime(2024, 1, 15),
            total_rows=500,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 1, 15),
            update_count=3,
        )

        # Convert to dict
        data = metadata.to_dict()
        assert data["symbol"] == "GOOGL"
        assert data["total_rows"] == 500

        # Recreate from dict
        metadata2 = DatasetMetadata.from_dict(data)
        assert metadata2.symbol == metadata.symbol
        assert metadata2.total_rows == metadata.total_rows


class TestMetadataTracker:
    """Test MetadataTracker functionality."""

    def test_basic_metadata_operations(self, tmp_path: Path) -> None:
        """Test basic metadata tracking operations."""
        tracker = MetadataTracker(base_path=tmp_path)

        # Create update record
        record = UpdateRecord(
            timestamp=datetime.now(),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            provider="yahoo",
            duration_seconds=2.5,
        )

        # Update metadata
        key = "equities/daily/AAPL"
        metadata = tracker.update_metadata(
            key=key,
            update_record=record,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 31),
        )

        assert metadata.symbol == "AAPL"
        assert metadata.total_rows == 100
        assert metadata.update_count == 1

        # Retrieve metadata
        retrieved = tracker.get_metadata(key)
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.total_rows == 100

    def test_incremental_updates(self, tmp_path: Path) -> None:
        """Test tracking incremental updates."""
        tracker = MetadataTracker(base_path=tmp_path)
        key = "equities/daily/MSFT"

        # Initial update
        record1 = UpdateRecord(
            timestamp=datetime(2024, 1, 1, 10, 0),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            provider="yahoo",
            duration_seconds=3.0,
        )

        metadata1 = tracker.update_metadata(
            key=key,
            update_record=record1,
            total_rows=100,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2023, 12, 31),
        )

        assert metadata1.update_count == 1

        # Incremental update
        record2 = UpdateRecord(
            timestamp=datetime(2024, 1, 15, 10, 0),
            update_type="incremental",
            rows_before=100,
            rows_after=110,
            rows_added=10,
            rows_updated=5,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 14),
            provider="yahoo",
            duration_seconds=1.5,
        )

        metadata2 = tracker.update_metadata(
            key=key,
            update_record=record2,
            total_rows=110,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 1, 14),
        )

        assert metadata2.update_count == 2
        assert metadata2.total_rows == 110

    def test_update_history(self, tmp_path: Path) -> None:
        """Test tracking update history."""
        tracker = MetadataTracker(base_path=tmp_path)
        key = "equities/daily/TSLA"

        # Add multiple updates
        for i in range(5):
            record = UpdateRecord(
                timestamp=datetime(2024, 1, i + 1, 10, 0),
                update_type="incremental" if i > 0 else "full",
                rows_before=100 * i,
                rows_after=100 * (i + 1),
                rows_added=100,
                rows_updated=10 if i > 0 else 0,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, i + 1),
                provider="yahoo",
                duration_seconds=2.0 + i,
            )

            tracker.update_metadata(
                key=key,
                update_record=record,
                total_rows=100 * (i + 1),
                date_range_start=datetime(2024, 1, 1),
                date_range_end=datetime(2024, 1, i + 1),
            )

        # Get history
        history = tracker.get_update_history(key, limit=3)

        assert len(history) == 3
        # Most recent first
        assert history[0].timestamp == datetime(2024, 1, 5, 10, 0)
        assert history[1].timestamp == datetime(2024, 1, 4, 10, 0)
        assert history[2].timestamp == datetime(2024, 1, 3, 10, 0)

    def test_health_check(self, tmp_path: Path) -> None:
        """Test dataset health checking."""
        tracker = MetadataTracker(base_path=tmp_path)
        key = "equities/daily/NVDA"

        # Create recent update
        record = UpdateRecord(
            timestamp=datetime.now() - timedelta(days=1),
            update_type="incremental",
            rows_before=100,
            rows_after=101,
            rows_added=1,
            rows_updated=0,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=1),
            provider="yahoo",
            duration_seconds=1.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=record,
            total_rows=101,
            date_range_start=datetime.now() - timedelta(days=365),
            date_range_end=datetime.now() - timedelta(days=1),
        )

        # Check health - should be healthy
        status, message = tracker.check_health(key, stale_days=7)
        assert status == "healthy"

        # Create stale update
        key2 = "equities/daily/AMD"
        old_record = UpdateRecord(
            timestamp=datetime.now() - timedelta(days=10),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2023, 1, 1),
            end_date=datetime.now() - timedelta(days=10),
            provider="yahoo",
            duration_seconds=3.0,
        )

        tracker.update_metadata(
            key=key2,
            update_record=old_record,
            total_rows=100,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime.now() - timedelta(days=10),
        )

        # Check health - should be stale
        status2, message2 = tracker.check_health(key2, stale_days=7)
        assert status2 == "stale"
        assert "days old" in message2

    def test_summary(self, tmp_path: Path) -> None:
        """Test getting summary of all datasets."""
        tracker = MetadataTracker(base_path=tmp_path)

        # Add multiple datasets
        datasets = [
            ("equities/daily/AAPL", "yahoo", 100),
            ("equities/daily/GOOGL", "yahoo", 200),
            ("crypto/hourly/BTC", "cryptocompare", 500),
            ("crypto/hourly/ETH", "cryptocompare", 400),
        ]

        for key, provider, rows in datasets:
            record = UpdateRecord(
                timestamp=datetime.now(),
                update_type="full",
                rows_before=0,
                rows_after=rows,
                rows_added=rows,
                rows_updated=0,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                provider=provider,
                duration_seconds=2.0,
            )

            tracker.update_metadata(
                key=key,
                update_record=record,
                total_rows=rows,
                date_range_start=datetime(2024, 1, 1),
                date_range_end=datetime(2024, 1, 31),
            )

        # Get summary
        summary = tracker.get_summary()

        assert summary["total_datasets"] == 4
        assert summary["total_rows"] == 1200  # 100 + 200 + 500 + 400
        assert summary["by_asset_class"]["equities"] == 2
        assert summary["by_asset_class"]["crypto"] == 2

    def test_error_tracking(self, tmp_path: Path) -> None:
        """Test tracking errors in updates."""
        tracker = MetadataTracker(base_path=tmp_path)
        key = "equities/daily/SPY"

        # First create the metadata
        initial_record = UpdateRecord(
            timestamp=datetime.now() - timedelta(hours=10),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=1),
            provider="yahoo",
            duration_seconds=1.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=initial_record,
            total_rows=100,
            date_range_start=datetime.now() - timedelta(days=30),
            date_range_end=datetime.now() - timedelta(days=1),
        )

        # Add updates with errors (4 with errors, 1 without)
        for i in range(5):
            record = UpdateRecord(
                timestamp=datetime.now() - timedelta(hours=5 - i),
                update_type="incremental",
                rows_before=100,
                rows_after=100,
                rows_added=0,
                rows_updated=0,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now() - timedelta(days=1),
                provider="yahoo",
                duration_seconds=1.0,
                errors=["Connection timeout"] if i < 4 else [],  # 4 errors, 1 success
            )

            tracker.add_update_record(key, record)

        # Check health - should be error due to recent failures (4 out of 5)
        status, message = tracker.check_health(key)
        assert status == "error"
        assert "errors in last 5 updates" in message
