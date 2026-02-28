"""Metadata tracking for update status and history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from ml4t.data.storage.key_codec import decode_storage_key, encode_storage_key
from ml4t.data.storage.manifest import normalize_manifest_payload
from ml4t.data.utils.locking import file_lock

logger = structlog.get_logger()


@dataclass
class UpdateRecord:
    """Record of a data update operation."""

    timestamp: datetime
    update_type: str  # "full", "incremental", "gap_fill"
    rows_before: int
    rows_after: int
    rows_added: int
    rows_updated: int
    start_date: datetime
    end_date: datetime
    provider: str
    duration_seconds: float
    gaps_filled: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "update_type": self.update_type,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_added": self.rows_added,
            "rows_updated": self.rows_updated,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "provider": self.provider,
            "duration_seconds": self.duration_seconds,
            "gaps_filled": self.gaps_filled,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UpdateRecord:
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            update_type=data["update_type"],
            rows_before=data["rows_before"],
            rows_after=data["rows_after"],
            rows_added=data["rows_added"],
            rows_updated=data["rows_updated"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            provider=data["provider"],
            duration_seconds=data["duration_seconds"],
            gaps_filled=data.get("gaps_filled", 0),
            errors=data.get("errors", []),
        )


@dataclass
class DatasetMetadata:
    """Metadata about a dataset."""

    symbol: str
    asset_class: str
    frequency: str
    provider: str
    first_update: datetime
    last_update: datetime
    total_rows: int
    date_range_start: datetime
    date_range_end: datetime
    update_count: int
    last_check: datetime | None = None
    health_status: str = "healthy"  # "healthy", "stale", "error"
    health_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "frequency": self.frequency,
            "provider": self.provider,
            "first_update": self.first_update.isoformat(),
            "last_update": self.last_update.isoformat(),
            "total_rows": self.total_rows,
            "date_range_start": self.date_range_start.isoformat(),
            "date_range_end": self.date_range_end.isoformat(),
            "update_count": self.update_count,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "health_status": self.health_status,
            "health_message": self.health_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            asset_class=data["asset_class"],
            frequency=data["frequency"],
            provider=data["provider"],
            first_update=datetime.fromisoformat(data["first_update"]),
            last_update=datetime.fromisoformat(data["last_update"]),
            total_rows=data["total_rows"],
            date_range_start=datetime.fromisoformat(data["date_range_start"]),
            date_range_end=datetime.fromisoformat(data["date_range_end"]),
            update_count=data["update_count"],
            last_check=datetime.fromisoformat(data["last_check"])
            if data.get("last_check")
            else None,
            health_status=data.get("health_status", "healthy"),
            health_message=data.get("health_message", ""),
        )


@dataclass
class DatasetUpdate:
    """Latest update snapshot for CLI health/info views."""

    symbol: str
    provider: str
    frequency: str
    timestamp: datetime
    asset_class: str = ""


class MetadataTracker:
    """Track metadata and update history for datasets."""

    def __init__(self, base_path: Path) -> None:
        """
        Initialize metadata tracker.

        Args:
            base_path: Base directory for metadata storage
        """
        self.base_path = Path(base_path)
        self.metadata_dir = self.base_path / ".metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_metadata_path(self, key: str) -> Path:
        """Get path to metadata file for a dataset."""
        safe_key = key.replace("/", "_")
        return self.metadata_dir / f"{safe_key}_metadata.json"

    def _get_history_path(self, key: str) -> Path:
        """Get path to update history file for a dataset."""
        safe_key = key.replace("/", "_")
        return self.metadata_dir / f"{safe_key}_history.json"

    def _manifest_metadata_paths(self, key: str) -> list[Path]:
        """Get possible storage manifest paths for a dataset key."""
        return [
            self.metadata_dir / f"{encode_storage_key(key)}.json",
            self.metadata_dir / f"{key.replace('/', '_')}.json",
        ]

    def _parse_datetime(self, value: Any) -> datetime | None:
        """Parse datetime values from metadata payloads."""
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _metadata_from_manifest(
        self,
        key: str,
        payload: dict[str, Any],
    ) -> DatasetMetadata | None:
        """Convert storage manifest metadata into DatasetMetadata shape."""
        normalized = normalize_manifest_payload(key, payload)
        last_update = self._parse_datetime(normalized.get("last_updated"))
        first_update = self._parse_datetime(normalized.get("first_update")) or last_update

        data_range = normalized.get("data_range")
        if not isinstance(data_range, dict):
            data_range = {}

        range_start = self._parse_datetime(data_range.get("start")) or first_update
        range_end = self._parse_datetime(data_range.get("end")) or last_update

        if not last_update or not first_update or not range_start or not range_end:
            return None

        update_count = normalized.get("update_count")
        if not isinstance(update_count, int):
            update_count = 1

        row_count = normalized.get("total_rows", normalized.get("row_count", 0))
        if not isinstance(row_count, int):
            row_count = 0

        return DatasetMetadata(
            symbol=str(normalized.get("symbol", "")),
            asset_class=str(normalized.get("asset_class", "")),
            frequency=str(normalized.get("frequency", "")),
            provider=str(normalized.get("provider", "")),
            first_update=first_update,
            last_update=last_update,
            total_rows=row_count,
            date_range_start=range_start,
            date_range_end=range_end,
            update_count=update_count,
            last_check=self._parse_datetime(normalized.get("last_check")),
            health_status=str(normalized.get("health_status", "healthy")),
            health_message=str(normalized.get("health_message", "")),
        )

    def _load_metadata_file(self, metadata_file: Path) -> tuple[str, DatasetMetadata] | None:
        """Load one metadata file and return canonical key plus metadata."""
        if metadata_file.name.endswith("_history.json"):
            return None

        with open(metadata_file) as f:
            payload = json.load(f)

        if metadata_file.name.endswith("_metadata.json"):
            metadata = DatasetMetadata.from_dict(payload)
            if metadata.asset_class and metadata.frequency and metadata.symbol:
                key = f"{metadata.asset_class}/{metadata.frequency}/{metadata.symbol}"
            else:
                key = metadata.symbol
            return key, metadata

        decoded_key = decode_storage_key(metadata_file.stem)
        metadata = self._metadata_from_manifest(decoded_key, payload)
        if metadata is None:
            return None

        if metadata.asset_class and metadata.frequency and metadata.symbol:
            key = f"{metadata.asset_class}/{metadata.frequency}/{metadata.symbol}"
        else:
            key = decoded_key
        return key, metadata

    def _collect_all_metadata(self) -> list[DatasetMetadata]:
        """Collect metadata from tracker and storage manifest files."""
        by_key: dict[str, tuple[int, DatasetMetadata]] = {}

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                loaded = self._load_metadata_file(metadata_file)
                if loaded is None:
                    continue
                key, metadata = loaded
                priority = 2 if metadata_file.name.endswith("_metadata.json") else 1
                existing = by_key.get(key)
                if existing is None or priority >= existing[0]:
                    by_key[key] = (priority, metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")

        return [value[1] for _, value in sorted(by_key.items(), key=lambda item: item[0])]

    def get_metadata(self, key: str) -> DatasetMetadata | None:
        """
        Get metadata for a dataset.

        Args:
            key: Dataset key (e.g., "equities/daily/AAPL")

        Returns:
            DatasetMetadata if exists, None otherwise
        """
        metadata_path = self._get_metadata_path(key)
        if metadata_path.exists():
            with file_lock(metadata_path), open(metadata_path) as f:
                data = json.load(f)
            return DatasetMetadata.from_dict(data)

        for manifest_path in self._manifest_metadata_paths(key):
            if not manifest_path.exists():
                continue
            with file_lock(manifest_path), open(manifest_path) as f:
                manifest = json.load(f)
            metadata = self._metadata_from_manifest(key, manifest)
            if metadata is not None:
                return metadata

        return None

    def update_metadata(
        self,
        key: str,
        update_record: UpdateRecord,
        total_rows: int,
        date_range_start: datetime,
        date_range_end: datetime,
    ) -> DatasetMetadata:
        """
        Update metadata for a dataset.

        Args:
            key: Dataset key
            update_record: Record of the update operation
            total_rows: Total rows after update
            date_range_start: Start of data range
            date_range_end: End of data range

        Returns:
            Updated DatasetMetadata
        """
        metadata_path = self._get_metadata_path(key)

        # Get existing metadata or create new
        existing = self.get_metadata(key)

        if existing:
            # Update existing metadata
            metadata = DatasetMetadata(
                symbol=existing.symbol,
                asset_class=existing.asset_class,
                frequency=existing.frequency,
                provider=update_record.provider,
                first_update=existing.first_update,
                last_update=update_record.timestamp,
                total_rows=total_rows,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                update_count=existing.update_count + 1,
                last_check=datetime.now(),
                health_status="healthy",
                health_message="",
            )
        else:
            # Create new metadata
            parts = key.split("/")
            if len(parts) == 3:
                asset_class, frequency, symbol = parts
            else:
                asset_class, frequency, symbol = "", "", key

            metadata = DatasetMetadata(
                symbol=symbol,
                asset_class=asset_class,
                frequency=frequency,
                provider=update_record.provider,
                first_update=update_record.timestamp,
                last_update=update_record.timestamp,
                total_rows=total_rows,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                update_count=1,
                last_check=datetime.now(),
                health_status="healthy",
                health_message="",
            )

        # Save metadata
        with file_lock(metadata_path), open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Add to update history
        self.add_update_record(key, update_record)

        logger.info(
            f"Updated metadata for {key}",
            update_count=metadata.update_count,
            total_rows=metadata.total_rows,
        )

        return metadata

    def add_update_record(self, key: str, record: UpdateRecord) -> None:
        """
        Add an update record to the history.

        Args:
            key: Dataset key
            record: Update record to add
        """
        history_path = self._get_history_path(key)

        # Load existing history
        if history_path.exists():
            with file_lock(history_path), open(history_path) as f:
                history = json.load(f)
        else:
            history = []

        # Add new record
        history.append(record.to_dict())

        # Keep only last 100 records
        if len(history) > 100:
            history = history[-100:]

        # Save updated history
        with file_lock(history_path), open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    def get_update_history(
        self,
        key: str,
        limit: int = 10,
    ) -> list[UpdateRecord]:
        """
        Get update history for a dataset.

        Args:
            key: Dataset key
            limit: Maximum number of records to return

        Returns:
            List of UpdateRecord objects (most recent first)
        """
        history_path = self._get_history_path(key)

        if not history_path.exists():
            return []

        with file_lock(history_path), open(history_path) as f:
            history = json.load(f)

        # Convert to UpdateRecord objects
        records = [UpdateRecord.from_dict(data) for data in history]

        # Return most recent first, limited
        return records[-limit:][::-1]

    def check_health(self, key: str, stale_days: int = 7) -> tuple[str, str]:
        """
        Check health status of a dataset.

        Args:
            key: Dataset key
            stale_days: Number of days before data is considered stale

        Returns:
            Tuple of (health_status, health_message)
        """
        metadata = self.get_metadata(key)

        if not metadata:
            return "error", "No metadata found"

        # Check if data is stale
        days_since_update = (datetime.now() - metadata.last_update).days

        if days_since_update > stale_days:
            return "stale", f"Data is {days_since_update} days old"

        # Check if data range is current
        if metadata.frequency == "daily":
            expected_end = datetime.now() - timedelta(days=1)  # Yesterday
            days_behind = (expected_end - metadata.date_range_end).days

            if days_behind > 3:
                return "stale", f"Data is {days_behind} days behind"

        # Check for recent errors
        history = self.get_update_history(key, limit=5)
        recent_errors = sum(1 for r in history if r.errors)

        if recent_errors >= 3:
            return "error", f"{recent_errors} errors in last 5 updates"

        return "healthy", "Dataset is up to date"

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of all tracked datasets.

        Returns:
            Dictionary with summary statistics
        """
        all_metadata = self._collect_all_metadata()

        # Calculate summary
        total_datasets = len(all_metadata)
        healthy = sum(1 for m in all_metadata if m.health_status == "healthy")
        stale = sum(1 for m in all_metadata if m.health_status == "stale")
        error = sum(1 for m in all_metadata if m.health_status == "error")
        unique_providers = len({m.provider for m in all_metadata if m.provider})
        unique_symbols = len({m.symbol for m in all_metadata if m.symbol})

        total_rows = sum(m.total_rows for m in all_metadata)
        total_updates = sum(m.update_count for m in all_metadata)

        # Group by asset class
        by_asset_class = {}
        for m in all_metadata:
            if m.asset_class not in by_asset_class:
                by_asset_class[m.asset_class] = 0
            by_asset_class[m.asset_class] += 1

        return {
            "total_datasets": total_datasets,
            "healthy": healthy,
            "stale": stale,
            "error": error,
            "total_rows": total_rows,
            "total_updates": total_updates,
            "unique_providers": unique_providers,
            "unique_symbols": unique_symbols,
            "by_asset_class": by_asset_class,
            "last_updated": datetime.now().isoformat(),
        }

    def list_updates(self) -> list[DatasetUpdate]:
        """List latest updates for all tracked datasets (most recent first)."""
        all_metadata = self._collect_all_metadata()
        updates = [
            DatasetUpdate(
                symbol=metadata.symbol,
                provider=metadata.provider,
                frequency=metadata.frequency,
                timestamp=metadata.last_update,
                asset_class=metadata.asset_class,
            )
            for metadata in all_metadata
        ]
        return sorted(updates, key=lambda x: x.timestamp, reverse=True)
