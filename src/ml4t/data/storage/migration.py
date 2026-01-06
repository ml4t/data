"""Data migration framework for QLDM."""

from __future__ import annotations

import json
import os
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl
import structlog

logger = structlog.get_logger()


class MigrationStatus(str, Enum):
    """Status of a migration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Represents a data migration."""

    version: str
    description: str
    upgrade_fn: Callable[[pl.DataFrame], pl.DataFrame]
    downgrade_fn: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    schema_changes: dict[str, Any] | None = None


@dataclass
class MigrationRecord:
    """Record of a migration execution."""

    version: str
    status: MigrationStatus
    started_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    rows_affected: int = 0


class MigrationManager:
    """Manages data migrations and schema updates."""

    def __init__(self, data_root: Path):
        """
        Initialize migration manager.

        Args:
            data_root: Root directory for data storage
        """
        self.data_root = Path(data_root)
        self.migrations_dir = self.data_root / ".migrations"
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.migrations_dir / "migration_state.json"
        self.migrations: dict[str, Migration] = {}
        self._load_state()

    def register_migration(self, migration: Migration) -> None:
        """
        Register a new migration.

        Args:
            migration: Migration to register
        """
        self.migrations[migration.version] = migration
        logger.info(f"Registered migration {migration.version}")

    def migrate(
        self,
        target_version: str | None = None,
        dry_run: bool = False,
    ) -> list[MigrationRecord]:
        """
        Run migrations up to target version.

        Args:
            target_version: Target version to migrate to (latest if None)
            dry_run: If True, don't actually apply migrations

        Returns:
            List of migration records
        """
        current_version = self._get_current_version()
        pending_migrations = self._get_pending_migrations(current_version, target_version)

        if not pending_migrations:
            logger.info("No pending migrations")
            return []

        logger.info(
            f"Running {len(pending_migrations)} migrations",
            from_version=current_version,
            to_version=target_version or "latest",
        )

        records = []

        for version in pending_migrations:
            migration = self.migrations[version]
            record = MigrationRecord(
                version=version,
                status=MigrationStatus.PENDING,
                started_at=datetime.now(),
            )

            if dry_run:
                logger.info(f"[DRY RUN] Would migrate to {version}: {migration.description}")
                record.status = MigrationStatus.COMPLETED
                record.completed_at = datetime.now()
            else:
                try:
                    record = self._run_migration(migration)
                except Exception as e:
                    logger.error(f"Migration {version} failed: {e}")
                    record.status = MigrationStatus.FAILED
                    record.error = str(e)
                    record.completed_at = datetime.now()
                    records.append(record)

                    # Stop on first failure
                    break

            records.append(record)

        # Save state
        if not dry_run:
            self._save_state(records)

        return records

    def rollback(self, target_version: str) -> list[MigrationRecord]:
        """
        Rollback migrations to target version.

        Args:
            target_version: Version to rollback to

        Returns:
            List of rollback records
        """
        current_version = self._get_current_version()

        if current_version <= target_version:
            logger.info("Already at or before target version")
            return []

        rollback_versions = self._get_rollback_versions(current_version, target_version)

        logger.info(
            f"Rolling back {len(rollback_versions)} migrations",
            from_version=current_version,
            to_version=target_version,
        )

        records = []

        for version in rollback_versions:
            migration = self.migrations.get(version)

            if not migration or not migration.downgrade_fn:
                logger.error(f"Cannot rollback {version}: No downgrade function")
                break

            record = MigrationRecord(
                version=version,
                status=MigrationStatus.PENDING,
                started_at=datetime.now(),
            )

            try:
                record = self._run_rollback(migration)
            except Exception as e:
                logger.error(f"Rollback {version} failed: {e}")
                record.status = MigrationStatus.FAILED
                record.error = str(e)
                record.completed_at = datetime.now()
                records.append(record)
                break

            records.append(record)

        self._save_state(records)
        return records

    def _run_migration(self, migration: Migration) -> MigrationRecord:
        """Run a single migration."""
        logger.info(f"Running migration {migration.version}: {migration.description}")

        record = MigrationRecord(
            version=migration.version,
            status=MigrationStatus.IN_PROGRESS,
            started_at=datetime.now(),
        )

        rows_affected = 0

        # Find all data files to migrate
        data_files = list(self.data_root.rglob("*.parquet"))

        for file_path in data_files:
            if ".migrations" in str(file_path):
                continue  # Skip migration directory

            try:
                # Read data
                df = pl.read_parquet(file_path)
                original_rows = len(df)

                # Apply migration
                migrated_df = migration.upgrade_fn(df)

                # Write back
                migrated_df.write_parquet(file_path)

                rows_affected += original_rows
                logger.debug(f"Migrated {file_path}: {original_rows} rows")

            except Exception as e:
                logger.error(f"Failed to migrate {file_path}: {e}")
                raise

        record.status = MigrationStatus.COMPLETED
        record.completed_at = datetime.now()
        record.rows_affected = rows_affected

        logger.info(
            f"Migration {migration.version} completed",
            rows_affected=rows_affected,
            duration=(record.completed_at - record.started_at).total_seconds(),
        )

        return record

    def _run_rollback(self, migration: Migration) -> MigrationRecord:
        """Run a single rollback."""
        logger.info(f"Rolling back migration {migration.version}")

        record = MigrationRecord(
            version=migration.version,
            status=MigrationStatus.IN_PROGRESS,
            started_at=datetime.now(),
        )

        rows_affected = 0

        # Find all data files to rollback
        data_files = list(self.data_root.rglob("*.parquet"))

        for file_path in data_files:
            if ".migrations" in str(file_path):
                continue

            try:
                # Read data
                df = pl.read_parquet(file_path)
                original_rows = len(df)

                # Apply rollback
                if migration.downgrade_fn:
                    rolled_back_df = migration.downgrade_fn(df)
                else:
                    raise ValueError(f"No downgrade function for {migration.version}")

                # Write back
                rolled_back_df.write_parquet(file_path)

                rows_affected += original_rows

            except Exception as e:
                logger.error(f"Failed to rollback {file_path}: {e}")
                raise

        record.status = MigrationStatus.ROLLED_BACK
        record.completed_at = datetime.now()
        record.rows_affected = rows_affected

        logger.info(
            f"Rollback {migration.version} completed",
            rows_affected=rows_affected,
        )

        return record

    def _get_current_version(self) -> str:
        """Get current migration version."""
        state = self._load_state()

        if not state or "records" not in state:
            return "0.0.0"

        # Find last successful migration
        for record in reversed(state["records"]):
            if record["status"] == MigrationStatus.COMPLETED.value:
                return record["version"]

        return "0.0.0"

    def _get_pending_migrations(
        self,
        current_version: str,
        target_version: str | None,
    ) -> list[str]:
        """Get list of pending migrations."""
        all_versions = sorted(self.migrations.keys())

        if not target_version:
            target_version = all_versions[-1] if all_versions else current_version

        pending = []
        for version in all_versions:
            if version > current_version and version <= target_version:
                pending.append(version)

        return pending

    def _get_rollback_versions(
        self,
        current_version: str,
        target_version: str,
    ) -> list[str]:
        """Get list of versions to rollback."""
        all_versions = sorted(self.migrations.keys(), reverse=True)

        rollback = []
        for version in all_versions:
            if version <= current_version and version > target_version:
                rollback.append(version)

        return rollback

    def _load_state(self) -> dict[str, Any]:
        """Load migration state from file."""
        if not self.state_file.exists():
            return {}

        with open(self.state_file) as f:
            return json.load(f)

    def _save_state(self, records: list[MigrationRecord]) -> None:
        """Save migration state to file."""
        state = self._load_state()

        if "records" not in state:
            state["records"] = []

        for record in records:
            state["records"].append(
                {
                    "version": record.version,
                    "status": record.status.value,
                    "started_at": record.started_at.isoformat(),
                    "completed_at": record.completed_at.isoformat()
                    if record.completed_at
                    else None,
                    "error": record.error,
                    "rows_affected": record.rows_affected,
                }
            )

        state["current_version"] = self._get_current_version()
        state["last_updated"] = datetime.now().isoformat()

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)


class BackupManager:
    """Manages data backups and restoration."""

    def __init__(self, data_root: Path):
        """
        Initialize backup manager.

        Args:
            data_root: Root directory for data storage
        """
        self.data_root = Path(data_root)
        self.backups_dir = self.data_root / ".backups"
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self,
        backup_name: str | None = None,
        compression: bool = True,
    ) -> Path:
        """
        Create a backup of all data.

        Args:
            backup_name: Name for backup (auto-generated if None)
            compression: Whether to compress backup

        Returns:
            Path to backup file
        """
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_file = self.backups_dir / f"{backup_name}.tar"
        if compression:
            backup_file = backup_file.with_suffix(".tar.gz")

        logger.info(f"Creating backup: {backup_file}")

        # Create tar archive
        mode = "w:gz" if compression else "w"
        with tarfile.open(backup_file, mode) as tar:
            # Add all data files
            for file_path in self.data_root.rglob("*.parquet"):
                if ".backups" not in str(file_path):
                    arcname = file_path.relative_to(self.data_root)
                    tar.add(file_path, arcname=arcname)

            # Add metadata files
            for pattern in ["*.json", "*.manifest"]:
                for file_path in self.data_root.rglob(pattern):
                    if ".backups" not in str(file_path):
                        arcname = file_path.relative_to(self.data_root)
                        tar.add(file_path, arcname=arcname)

        file_size = backup_file.stat().st_size
        logger.info(
            "Backup created successfully",
            path=str(backup_file),
            size_mb=file_size / (1024 * 1024),
        )

        # Save backup metadata
        self._save_backup_metadata(backup_name, backup_file, file_size)

        return backup_file

    def restore_backup(
        self,
        backup_path: Path,
        target_dir: Path | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Restore data from backup.

        Args:
            backup_path: Path to backup file
            target_dir: Target directory (data_root if None)
            overwrite: Whether to overwrite existing data
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        if target_dir is None:
            target_dir = self.data_root

        target_dir = Path(target_dir)

        # Check for existing data
        if not overwrite and any(target_dir.rglob("*.parquet")):
            raise ValueError("Target directory contains data. Use overwrite=True to replace.")

        logger.info(f"Restoring backup from {backup_path} to {target_dir}")

        # Extract backup
        mode = "r:gz" if backup_path.suffix == ".gz" else "r"
        with tarfile.open(str(backup_path), mode) as tar:
            # Use data filter for safety (Python 3.12+)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(target_dir, filter="data")
            else:
                # For older Python versions, validate members before extraction
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                safe_members = []
                for member in tar.getmembers():
                    member_path = os.path.join(target_dir, member.name)
                    if is_within_directory(target_dir, member_path):
                        safe_members.append(member)
                    else:
                        logger.warning(f"Skipping potentially unsafe path: {member.name}")

                tar.extractall(target_dir, members=safe_members)  # nosec B202 - members validated above

        logger.info("Backup restored successfully")

    def list_backups(self) -> list[dict[str, Any]]:
        """
        List available backups.

        Returns:
            List of backup information
        """
        metadata_file = self.backups_dir / "backup_metadata.json"

        if not metadata_file.exists():
            return []

        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata.get("backups", [])

    def _save_backup_metadata(
        self,
        name: str,
        path: Path,
        size: int,
    ) -> None:
        """Save backup metadata."""
        metadata_file = self.backups_dir / "backup_metadata.json"

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {"backups": []}

        metadata["backups"].append(
            {
                "name": name,
                "path": str(path),
                "size": size,
                "created_at": datetime.now().isoformat(),
            }
        )

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


# Example migrations
def create_standard_migrations() -> list[Migration]:
    """Create standard migrations for QLDM."""
    migrations = []

    # Migration to add volume column if missing
    def add_volume_column(df: pl.DataFrame) -> pl.DataFrame:
        if "volume" not in df.columns:
            return df.with_columns(pl.lit(0).alias("volume"))
        return df

    def remove_volume_column(df: pl.DataFrame) -> pl.DataFrame:
        if "volume" in df.columns:
            return df.drop("volume")
        return df

    migrations.append(
        Migration(
            version="1.0.0",
            description="Add volume column",
            upgrade_fn=add_volume_column,
            downgrade_fn=remove_volume_column,
        )
    )

    # Migration to rename columns
    def rename_to_standard(df: pl.DataFrame) -> pl.DataFrame:
        renames = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        for old, new in renames.items():
            if old in df.columns and new not in df.columns:
                df = df.rename({old: new})

        return df

    migrations.append(
        Migration(
            version="1.1.0",
            description="Standardize column names to lowercase",
            upgrade_fn=rename_to_standard,
        )
    )

    return migrations
