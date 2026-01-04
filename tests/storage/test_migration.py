"""Tests for storage migration module."""

from __future__ import annotations

import json
import tarfile
from datetime import datetime

import polars as pl
import pytest

from ml4t.data.storage.migration import (
    BackupManager,
    Migration,
    MigrationManager,
    MigrationRecord,
    MigrationStatus,
    create_standard_migrations,
)


class TestMigrationStatus:
    """Tests for MigrationStatus enum."""

    def test_status_values(self):
        """Test all status values are defined."""
        assert MigrationStatus.PENDING.value == "pending"
        assert MigrationStatus.IN_PROGRESS.value == "in_progress"
        assert MigrationStatus.COMPLETED.value == "completed"
        assert MigrationStatus.FAILED.value == "failed"
        assert MigrationStatus.ROLLED_BACK.value == "rolled_back"


class TestMigration:
    """Tests for Migration dataclass."""

    def test_create_migration(self):
        """Test creating a Migration."""

        def upgrade(df: pl.DataFrame) -> pl.DataFrame:
            return df

        migration = Migration(
            version="1.0.0",
            description="Test migration",
            upgrade_fn=upgrade,
        )

        assert migration.version == "1.0.0"
        assert migration.description == "Test migration"
        assert migration.upgrade_fn == upgrade
        assert migration.downgrade_fn is None

    def test_migration_with_downgrade(self):
        """Test Migration with downgrade function."""

        def upgrade(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit(1).alias("new_col"))

        def downgrade(df: pl.DataFrame) -> pl.DataFrame:
            return df.drop("new_col")

        migration = Migration(
            version="1.0.0",
            description="Reversible migration",
            upgrade_fn=upgrade,
            downgrade_fn=downgrade,
        )

        assert migration.downgrade_fn is not None


class TestMigrationRecord:
    """Tests for MigrationRecord dataclass."""

    def test_create_record(self):
        """Test creating a MigrationRecord."""
        record = MigrationRecord(
            version="1.0.0",
            status=MigrationStatus.PENDING,
            started_at=datetime.now(),
        )

        assert record.version == "1.0.0"
        assert record.status == MigrationStatus.PENDING
        assert record.completed_at is None
        assert record.error is None
        assert record.rows_affected == 0


class TestMigrationManager:
    """Tests for MigrationManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MigrationManager for testing."""
        return MigrationManager(tmp_path)

    @pytest.fixture
    def sample_migration(self):
        """Create a sample migration."""

        def add_column(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit("test").alias("new_column"))

        def remove_column(df: pl.DataFrame) -> pl.DataFrame:
            return df.drop("new_column")

        return Migration(
            version="1.0.0",
            description="Add new column",
            upgrade_fn=add_column,
            downgrade_fn=remove_column,
        )

    def test_init_creates_migrations_dir(self, tmp_path):
        """Test initialization creates migrations directory."""
        _manager = MigrationManager(tmp_path)  # noqa: F841
        assert (tmp_path / ".migrations").exists()

    def test_register_migration(self, manager, sample_migration):
        """Test registering a migration."""
        manager.register_migration(sample_migration)
        assert "1.0.0" in manager.migrations

    def test_migrate_no_pending(self, manager):
        """Test migrate with no pending migrations."""
        records = manager.migrate()
        assert records == []

    def test_migrate_dry_run(self, manager, sample_migration, tmp_path):
        """Test dry run migration."""
        manager.register_migration(sample_migration)

        # Create a data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(data_dir / "test.parquet")

        records = manager.migrate(dry_run=True)

        assert len(records) == 1
        assert records[0].status == MigrationStatus.COMPLETED

        # Data should not be modified
        loaded = pl.read_parquet(data_dir / "test.parquet")
        assert "new_column" not in loaded.columns

    def test_migrate_applies_changes(self, manager, sample_migration, tmp_path):
        """Test migration applies changes to data."""
        manager.register_migration(sample_migration)

        # Create a data file
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        records = manager.migrate()

        assert len(records) == 1
        assert records[0].status == MigrationStatus.COMPLETED

        # Data should be modified
        loaded = pl.read_parquet(tmp_path / "test.parquet")
        assert "new_column" in loaded.columns

    def test_migrate_handles_failure(self, manager, tmp_path):
        """Test migration handles failures gracefully."""

        def failing_upgrade(df: pl.DataFrame) -> pl.DataFrame:
            raise ValueError("Migration failed!")

        migration = Migration(
            version="1.0.0",
            description="Failing migration",
            upgrade_fn=failing_upgrade,
        )
        manager.register_migration(migration)

        # Create a data file
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        records = manager.migrate()

        assert len(records) == 1
        assert records[0].status == MigrationStatus.FAILED
        assert records[0].error is not None

    def test_migrate_target_version(self, manager, tmp_path):
        """Test migrating to specific version."""

        def upgrade1(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit("v1").alias("version"))

        def upgrade2(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit("v2").alias("version"))

        migration1 = Migration(version="1.0.0", description="V1", upgrade_fn=upgrade1)
        migration2 = Migration(version="2.0.0", description="V2", upgrade_fn=upgrade2)

        manager.register_migration(migration1)
        manager.register_migration(migration2)

        # Create a data file
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        # Migrate only to 1.0.0
        records = manager.migrate(target_version="1.0.0")

        assert len(records) == 1
        assert records[0].version == "1.0.0"

    def test_rollback_no_downgrade(self, manager, sample_migration, tmp_path):
        """Test rollback without downgrade function."""

        def upgrade(df: pl.DataFrame) -> pl.DataFrame:
            return df

        migration = Migration(
            version="1.0.0",
            description="No downgrade",
            upgrade_fn=upgrade,
            downgrade_fn=None,  # No downgrade
        )

        manager.register_migration(migration)

        # Create a data file and migrate
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")
        manager.migrate()

        # Try to rollback
        records = manager.rollback("0.0.0")

        # Should not rollback without downgrade function
        assert len(records) == 0

    def test_rollback_applies_downgrade(self, manager, sample_migration, tmp_path):
        """Test rollback applies downgrade function."""
        manager.register_migration(sample_migration)

        # Create a data file and migrate
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")
        manager.migrate()

        # Verify migration was applied
        loaded = pl.read_parquet(tmp_path / "test.parquet")
        assert "new_column" in loaded.columns

        # Rollback
        records = manager.rollback("0.0.0")

        assert len(records) == 1
        assert records[0].status == MigrationStatus.ROLLED_BACK

        # Verify rollback was applied
        loaded = pl.read_parquet(tmp_path / "test.parquet")
        assert "new_column" not in loaded.columns

    def test_get_current_version_empty(self, manager):
        """Test getting current version when no migrations."""
        version = manager._get_current_version()
        assert version == "0.0.0"

    def test_get_pending_migrations(self, manager, sample_migration):
        """Test getting pending migrations."""
        manager.register_migration(sample_migration)
        pending = manager._get_pending_migrations("0.0.0", "1.0.0")
        assert pending == ["1.0.0"]

    def test_get_pending_migrations_already_applied(self, manager, sample_migration):
        """Test getting pending migrations when already at target."""
        manager.register_migration(sample_migration)
        pending = manager._get_pending_migrations("1.0.0", "1.0.0")
        assert pending == []

    def test_get_rollback_versions(self, manager):
        """Test getting versions to rollback."""

        def noop(df: pl.DataFrame) -> pl.DataFrame:
            return df

        manager.register_migration(Migration("1.0.0", "V1", noop))
        manager.register_migration(Migration("2.0.0", "V2", noop))

        rollback = manager._get_rollback_versions("2.0.0", "0.0.0")
        assert "2.0.0" in rollback
        assert "1.0.0" in rollback

    def test_save_and_load_state(self, manager, sample_migration, tmp_path):
        """Test saving and loading migration state."""
        manager.register_migration(sample_migration)

        # Create a data file and migrate
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")
        manager.migrate()

        # Create new manager and check state
        new_manager = MigrationManager(tmp_path)
        version = new_manager._get_current_version()
        assert version == "1.0.0"

    def test_skips_migrations_dir_files(self, manager, sample_migration, tmp_path):
        """Test migration skips files in .migrations directory."""
        manager.register_migration(sample_migration)

        # Create a parquet file in migrations dir
        migration_data = tmp_path / ".migrations" / "internal.parquet"
        migration_data.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"internal": [1]}).write_parquet(migration_data)

        # Create a data file
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        manager.migrate()

        # Internal file should not be modified
        internal = pl.read_parquet(migration_data)
        assert "new_column" not in internal.columns


class TestBackupManager:
    """Tests for BackupManager class."""

    @pytest.fixture
    def backup_manager(self, tmp_path):
        """Create a BackupManager for testing."""
        return BackupManager(tmp_path)

    def test_init_creates_backups_dir(self, tmp_path):
        """Test initialization creates backups directory."""
        _manager = BackupManager(tmp_path)  # noqa: F841
        assert (tmp_path / ".backups").exists()

    def test_create_backup_compressed(self, backup_manager, tmp_path):
        """Test creating compressed backup."""
        # Create some data
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        backup_path = backup_manager.create_backup(compression=True)

        assert backup_path.exists()
        assert backup_path.suffix == ".gz"

    def test_create_backup_uncompressed(self, backup_manager, tmp_path):
        """Test creating uncompressed backup."""
        # Create some data
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        backup_path = backup_manager.create_backup(compression=False)

        assert backup_path.exists()
        assert backup_path.suffix == ".tar"

    def test_create_backup_custom_name(self, backup_manager, tmp_path):
        """Test creating backup with custom name."""
        # Create some data
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        backup_path = backup_manager.create_backup(backup_name="my_backup")

        assert "my_backup" in str(backup_path)

    def test_create_backup_auto_name(self, backup_manager, tmp_path):
        """Test backup auto-generates timestamp name."""
        # Create some data
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        backup_path = backup_manager.create_backup()

        assert "backup_" in str(backup_path)

    def test_create_backup_includes_parquet(self, backup_manager, tmp_path):
        """Test backup includes parquet files."""
        # Create some data
        df = pl.DataFrame({"col": [1, 2, 3]})
        (tmp_path / "data").mkdir()
        df.write_parquet(tmp_path / "data" / "test.parquet")

        backup_path = backup_manager.create_backup()

        # Verify file is in backup
        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
            assert any("test.parquet" in name for name in names)

    def test_create_backup_includes_json(self, backup_manager, tmp_path):
        """Test backup includes JSON files."""
        # Create a JSON file
        with open(tmp_path / "config.json", "w") as f:
            json.dump({"key": "value"}, f)

        backup_path = backup_manager.create_backup()

        # Verify file is in backup
        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
            assert any("config.json" in name for name in names)

    def test_create_backup_excludes_backups_dir(self, backup_manager, tmp_path):
        """Test backup excludes .backups directory."""
        # Create some data
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        # Create first backup
        backup_manager.create_backup()

        # Create second backup (should not include first)
        backup_path = backup_manager.create_backup(backup_name="second")

        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
            assert not any(".backups" in name for name in names)

    def test_restore_backup_file_not_found(self, backup_manager, tmp_path):
        """Test restore with non-existent file."""
        with pytest.raises(FileNotFoundError):
            backup_manager.restore_backup(tmp_path / "nonexistent.tar.gz")

    def test_restore_backup_without_overwrite(self, backup_manager, tmp_path):
        """Test restore fails without overwrite when data exists."""
        # Create data and backup
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")
        backup_path = backup_manager.create_backup()

        # Try to restore without overwrite
        with pytest.raises(ValueError, match="overwrite"):
            backup_manager.restore_backup(backup_path)

    def test_restore_backup_with_overwrite(self, backup_manager, tmp_path):
        """Test restore with overwrite."""
        # Create data and backup
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")
        backup_path = backup_manager.create_backup()

        # Modify data
        df2 = pl.DataFrame({"col": [4, 5, 6]})
        df2.write_parquet(tmp_path / "test.parquet")

        # Restore
        backup_manager.restore_backup(backup_path, overwrite=True)

        # Verify restored data
        loaded = pl.read_parquet(tmp_path / "test.parquet")
        assert loaded["col"].to_list() == [1, 2, 3]

    def test_restore_backup_to_target_dir(self, backup_manager, tmp_path):
        """Test restore to different target directory."""
        # Create data and backup
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")
        backup_path = backup_manager.create_backup()

        # Restore to new directory
        target = tmp_path / "restored"
        target.mkdir()
        backup_manager.restore_backup(backup_path, target_dir=target)

        # Verify restored data
        loaded = pl.read_parquet(target / "test.parquet")
        assert loaded["col"].to_list() == [1, 2, 3]

    def test_list_backups_empty(self, backup_manager):
        """Test listing backups when none exist."""
        backups = backup_manager.list_backups()
        assert backups == []

    def test_list_backups(self, backup_manager, tmp_path):
        """Test listing backups."""
        # Create data and multiple backups
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        backup_manager.create_backup(backup_name="backup1")
        backup_manager.create_backup(backup_name="backup2")

        backups = backup_manager.list_backups()

        assert len(backups) == 2
        assert backups[0]["name"] == "backup1"
        assert backups[1]["name"] == "backup2"

    def test_backup_metadata_saved(self, backup_manager, tmp_path):
        """Test backup metadata is saved."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_parquet(tmp_path / "test.parquet")

        backup_manager.create_backup(backup_name="test_backup")

        metadata_file = tmp_path / ".backups" / "backup_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "backups" in metadata
        assert len(metadata["backups"]) == 1
        assert metadata["backups"][0]["name"] == "test_backup"


class TestCreateStandardMigrations:
    """Tests for create_standard_migrations function."""

    def test_creates_migrations(self):
        """Test function creates migrations."""
        migrations = create_standard_migrations()
        assert len(migrations) >= 2

    def test_add_volume_migration(self):
        """Test add volume column migration."""
        migrations = create_standard_migrations()
        volume_migration = migrations[0]

        assert volume_migration.version == "1.0.0"
        assert "volume" in volume_migration.description.lower()

        # Test upgrade
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = volume_migration.upgrade_fn(df)
        assert "volume" in result.columns

        # Test upgrade when volume exists
        df_with_volume = pl.DataFrame({"col": [1, 2, 3], "volume": [100, 200, 300]})
        result = volume_migration.upgrade_fn(df_with_volume)
        assert result["volume"].to_list() == [100, 200, 300]

        # Test downgrade
        df_with_volume = pl.DataFrame({"col": [1, 2, 3], "volume": [0, 0, 0]})
        result = volume_migration.downgrade_fn(df_with_volume)
        assert "volume" not in result.columns

    def test_rename_columns_migration(self):
        """Test rename columns migration."""
        migrations = create_standard_migrations()
        rename_migration = migrations[1]

        assert rename_migration.version == "1.1.0"
        assert "lowercase" in rename_migration.description.lower()

        # Test upgrade
        df = pl.DataFrame({"Open": [1], "High": [2], "Low": [3], "Close": [4]})
        result = rename_migration.upgrade_fn(df)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "Open" not in result.columns

    def test_rename_skips_existing_lowercase(self):
        """Test rename migration skips already lowercase columns."""
        migrations = create_standard_migrations()
        rename_migration = migrations[1]

        # Already lowercase
        df = pl.DataFrame({"open": [1], "high": [2], "low": [3], "close": [4]})
        result = rename_migration.upgrade_fn(df)
        assert result.columns == df.columns
