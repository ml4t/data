"""Tests for storage migration module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.storage.migration import (
    BackupManager,
    Migration,
    MigrationManager,
    MigrationRecord,
    MigrationStatus,
)


class TestMigrationStatus:
    """Tests for MigrationStatus enum."""

    def test_status_values(self):
        """Test enum values."""
        assert MigrationStatus.PENDING.value == "pending"
        assert MigrationStatus.IN_PROGRESS.value == "in_progress"
        assert MigrationStatus.COMPLETED.value == "completed"
        assert MigrationStatus.FAILED.value == "failed"
        assert MigrationStatus.ROLLED_BACK.value == "rolled_back"


class TestMigration:
    """Tests for Migration dataclass."""

    def test_migration_basic(self):
        """Test basic migration creation."""

        def upgrade_fn(df):
            return df

        migration = Migration(
            version="1.0.0",
            description="Add column",
            upgrade_fn=upgrade_fn,
        )

        assert migration.version == "1.0.0"
        assert migration.description == "Add column"
        assert migration.upgrade_fn is upgrade_fn
        assert migration.downgrade_fn is None
        assert migration.schema_changes is None

    def test_migration_with_downgrade(self):
        """Test migration with downgrade function."""

        def upgrade_fn(df):
            return df.with_columns(pl.lit("new").alias("new_col"))

        def downgrade_fn(df):
            return df.drop("new_col")

        migration = Migration(
            version="1.0.0",
            description="Add column",
            upgrade_fn=upgrade_fn,
            downgrade_fn=downgrade_fn,
            schema_changes={"new_col": "string"},
        )

        assert migration.downgrade_fn is downgrade_fn
        assert migration.schema_changes == {"new_col": "string"}


class TestMigrationRecord:
    """Tests for MigrationRecord dataclass."""

    def test_record_basic(self):
        """Test basic record creation."""
        now = datetime.now()

        record = MigrationRecord(
            version="1.0.0",
            status=MigrationStatus.PENDING,
            started_at=now,
        )

        assert record.version == "1.0.0"
        assert record.status == MigrationStatus.PENDING
        assert record.started_at == now
        assert record.completed_at is None
        assert record.error is None
        assert record.rows_affected == 0

    def test_record_complete(self):
        """Test record with all fields."""
        now = datetime.now()

        record = MigrationRecord(
            version="1.0.0",
            status=MigrationStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            error=None,
            rows_affected=100,
        )

        assert record.rows_affected == 100


class TestMigrationManager:
    """Tests for MigrationManager class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def migration_manager(self, temp_data_dir):
        """Create migration manager."""
        return MigrationManager(temp_data_dir)

    def test_init(self, migration_manager, temp_data_dir):
        """Test initialization."""
        assert migration_manager.data_root == temp_data_dir
        assert migration_manager.migrations_dir.exists()
        assert migration_manager.migrations == {}

    def test_register_migration(self, migration_manager):
        """Test registering a migration."""
        migration = Migration(
            version="1.0.0",
            description="Test migration",
            upgrade_fn=lambda df: df,
        )

        migration_manager.register_migration(migration)

        assert "1.0.0" in migration_manager.migrations
        assert migration_manager.migrations["1.0.0"] is migration

    def test_get_current_version_empty(self, migration_manager):
        """Test getting current version when no migrations."""
        version = migration_manager._get_current_version()
        assert version == "0.0.0"

    def test_get_pending_migrations_none(self, migration_manager):
        """Test getting pending migrations when none registered."""
        pending = migration_manager._get_pending_migrations("0.0.0", None)
        assert pending == []

    def test_get_pending_migrations_with_registrations(self, migration_manager):
        """Test getting pending migrations with registered migrations."""
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            migration_manager.register_migration(
                Migration(version=v, description=f"v{v}", upgrade_fn=lambda df: df)
            )

        pending = migration_manager._get_pending_migrations("0.0.0", "2.0.0")
        assert pending == ["1.0.0", "1.1.0", "2.0.0"]

        pending = migration_manager._get_pending_migrations("1.0.0", "2.0.0")
        assert pending == ["1.1.0", "2.0.0"]

        pending = migration_manager._get_pending_migrations("1.0.0", "1.1.0")
        assert pending == ["1.1.0"]

    def test_migrate_no_pending(self, migration_manager):
        """Test migration with no pending migrations."""
        records = migration_manager.migrate()
        assert records == []

    def test_migrate_dry_run(self, migration_manager):
        """Test dry run migration."""
        migration_manager.register_migration(
            Migration(version="1.0.0", description="Test", upgrade_fn=lambda df: df)
        )

        records = migration_manager.migrate(dry_run=True)

        assert len(records) == 1
        assert records[0].status == MigrationStatus.COMPLETED
        assert records[0].version == "1.0.0"

    def test_migrate_with_data(self, migration_manager, temp_data_dir):
        """Test migration with actual data files."""
        # Create test data file
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()
        test_file = data_dir / "test.parquet"

        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.write_parquet(test_file)

        # Register migration that adds a column
        def upgrade(df):
            return df.with_columns(pl.lit("new").alias("new_col"))

        migration_manager.register_migration(
            Migration(version="1.0.0", description="Add column", upgrade_fn=upgrade)
        )

        records = migration_manager.migrate()

        assert len(records) == 1
        assert records[0].status == MigrationStatus.COMPLETED
        assert records[0].rows_affected == 3

        # Verify data was migrated
        migrated_df = pl.read_parquet(test_file)
        assert "new_col" in migrated_df.columns

    def test_migrate_failure(self, migration_manager, temp_data_dir):
        """Test migration failure handling."""
        # Create test data file
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()
        test_file = data_dir / "test.parquet"

        df = pl.DataFrame({"col1": [1, 2, 3]})
        df.write_parquet(test_file)

        # Register migration that fails
        def failing_upgrade(df):
            raise ValueError("Migration failed!")

        migration_manager.register_migration(
            Migration(version="1.0.0", description="Failing", upgrade_fn=failing_upgrade)
        )

        records = migration_manager.migrate()

        assert len(records) == 1
        assert records[0].status == MigrationStatus.FAILED
        assert "Migration failed!" in records[0].error

    def test_get_rollback_versions(self, migration_manager):
        """Test getting rollback versions."""
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            migration_manager.register_migration(
                Migration(version=v, description=f"v{v}", upgrade_fn=lambda df: df)
            )

        rollback = migration_manager._get_rollback_versions("2.0.0", "1.0.0")
        assert rollback == ["2.0.0", "1.1.0"]

        rollback = migration_manager._get_rollback_versions("2.0.0", "1.1.0")
        assert rollback == ["2.0.0"]

    def test_rollback_already_at_version(self, migration_manager):
        """Test rollback when already at target version."""
        records = migration_manager.rollback("0.0.0")
        assert records == []

    def test_load_save_state(self, migration_manager):
        """Test state persistence."""
        # Save some state
        record = MigrationRecord(
            version="1.0.0",
            status=MigrationStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            rows_affected=100,
        )
        migration_manager._save_state([record])

        # Verify state file exists
        assert migration_manager.state_file.exists()

        # Load state
        state = migration_manager._load_state()
        assert "records" in state
        assert len(state["records"]) == 1
        assert state["records"][0]["version"] == "1.0.0"

    def test_get_current_version_after_migration(self, migration_manager):
        """Test current version tracking after successful migration."""
        # Save completed migration record
        record = MigrationRecord(
            version="1.0.0",
            status=MigrationStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        migration_manager._save_state([record])

        version = migration_manager._get_current_version()
        assert version == "1.0.0"

    def test_multiple_migrations(self, migration_manager, temp_data_dir):
        """Test running multiple migrations in sequence."""
        # Create test data file
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()
        test_file = data_dir / "test.parquet"

        df = pl.DataFrame({"col1": [1, 2, 3]})
        df.write_parquet(test_file)

        # Register multiple migrations
        migration_manager.register_migration(
            Migration(
                version="1.0.0",
                description="Add col2",
                upgrade_fn=lambda df: df.with_columns(pl.lit("a").alias("col2")),
            )
        )
        migration_manager.register_migration(
            Migration(
                version="1.1.0",
                description="Add col3",
                upgrade_fn=lambda df: df.with_columns(pl.lit("b").alias("col3")),
            )
        )

        records = migration_manager.migrate()

        assert len(records) == 2
        assert all(r.status == MigrationStatus.COMPLETED for r in records)

        # Verify all columns added
        migrated_df = pl.read_parquet(test_file)
        assert "col2" in migrated_df.columns
        assert "col3" in migrated_df.columns


class TestBackupManager:
    """Tests for BackupManager class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def backup_manager(self, temp_data_dir):
        """Create backup manager."""
        return BackupManager(temp_data_dir)

    @pytest.fixture
    def backup_manager_with_data(self, temp_data_dir):
        """Create backup manager with test data."""
        # Create some test data
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()

        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.write_parquet(data_dir / "test1.parquet")
        df.write_parquet(data_dir / "test2.parquet")

        # Create metadata file
        meta_file = temp_data_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({"version": "1.0.0"}, f)

        return BackupManager(temp_data_dir)

    def test_init(self, backup_manager, temp_data_dir):
        """Test initialization."""
        assert backup_manager.data_root == temp_data_dir
        assert backup_manager.backups_dir.exists()

    def test_backups_dir_created(self, temp_data_dir):
        """Test that backups directory is created."""
        _manager = BackupManager(temp_data_dir)  # noqa: F841
        assert (temp_data_dir / ".backups").is_dir()

    def test_create_backup_compressed(self, backup_manager_with_data):
        """Test creating a compressed backup."""
        backup_path = backup_manager_with_data.create_backup(
            backup_name="test_backup",
            compression=True,
        )

        assert backup_path.exists()
        assert backup_path.suffix == ".gz"
        assert "test_backup" in backup_path.name

    def test_create_backup_uncompressed(self, backup_manager_with_data):
        """Test creating an uncompressed backup."""
        backup_path = backup_manager_with_data.create_backup(
            backup_name="test_backup_raw",
            compression=False,
        )

        assert backup_path.exists()
        assert backup_path.suffix == ".tar"
        assert "test_backup_raw" in backup_path.name

    def test_create_backup_auto_name(self, backup_manager_with_data):
        """Test creating a backup with auto-generated name."""
        backup_path = backup_manager_with_data.create_backup()

        assert backup_path.exists()
        assert "backup_" in backup_path.name

    def test_list_backups_empty(self, backup_manager):
        """Test listing backups when none exist."""
        backups = backup_manager.list_backups()
        assert backups == []

    def test_list_backups_with_data(self, backup_manager_with_data):
        """Test listing backups after creating one."""
        backup_manager_with_data.create_backup(backup_name="backup1")
        backup_manager_with_data.create_backup(backup_name="backup2")

        backups = backup_manager_with_data.list_backups()

        assert len(backups) == 2
        assert any(b["name"] == "backup1" for b in backups)
        assert any(b["name"] == "backup2" for b in backups)

    def test_restore_backup_success(self, backup_manager_with_data, temp_data_dir):
        """Test restoring a backup."""
        # Create backup
        backup_path = backup_manager_with_data.create_backup(backup_name="restore_test")

        # Create new target directory
        restore_dir = temp_data_dir / "restored"
        restore_dir.mkdir()

        # Restore
        backup_manager_with_data.restore_backup(
            backup_path=backup_path,
            target_dir=restore_dir,
        )

        # Verify restored data
        assert (restore_dir / "data" / "test1.parquet").exists()
        assert (restore_dir / "data" / "test2.parquet").exists()

    def test_restore_backup_not_found(self, backup_manager):
        """Test restoring non-existent backup."""
        fake_path = Path("/nonexistent/backup.tar.gz")

        with pytest.raises(FileNotFoundError):
            backup_manager.restore_backup(backup_path=fake_path)

    def test_restore_backup_overwrite_protection(self, backup_manager_with_data, temp_data_dir):
        """Test restore refuses to overwrite without flag."""
        backup_path = backup_manager_with_data.create_backup(backup_name="overwrite_test")

        # Create target with existing data
        target_dir = temp_data_dir / "target"
        target_dir.mkdir()
        (target_dir / "existing.parquet").write_bytes(b"data")

        # Should raise without overwrite=True
        with pytest.raises(ValueError, match="contains data"):
            backup_manager_with_data.restore_backup(
                backup_path=backup_path,
                target_dir=target_dir,
            )

    def test_restore_backup_with_overwrite(self, backup_manager_with_data, temp_data_dir):
        """Test restore with overwrite flag."""
        backup_path = backup_manager_with_data.create_backup(backup_name="overwrite_ok")

        # Create target with existing data
        target_dir = temp_data_dir / "target_overwrite"
        target_dir.mkdir()
        data_subdir = target_dir / "data"
        data_subdir.mkdir()
        pl.DataFrame({"x": [1]}).write_parquet(data_subdir / "existing.parquet")

        # Should succeed with overwrite=True
        backup_manager_with_data.restore_backup(
            backup_path=backup_path,
            target_dir=target_dir,
            overwrite=True,
        )

        # Verify restoration
        assert (target_dir / "data" / "test1.parquet").exists()

    def test_backup_metadata_saved(self, backup_manager_with_data):
        """Test that backup metadata is saved correctly."""
        backup_manager_with_data.create_backup(backup_name="meta_test")

        metadata_file = backup_manager_with_data.backups_dir / "backup_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "backups" in metadata
        assert len(metadata["backups"]) == 1
        assert metadata["backups"][0]["name"] == "meta_test"
        assert "path" in metadata["backups"][0]
        assert "size" in metadata["backups"][0]
        assert "created_at" in metadata["backups"][0]


class TestMigrationIntegration:
    """Integration tests for migration workflow."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_migration_workflow(self, temp_data_dir):
        """Test complete migration workflow."""
        # Setup
        manager = MigrationManager(temp_data_dir)

        # Create test data
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()

        for i in range(3):
            df = pl.DataFrame({"value": [i * 10 + j for j in range(5)]})
            df.write_parquet(data_dir / f"file_{i}.parquet")

        # Register migrations
        manager.register_migration(
            Migration(
                version="1.0.0",
                description="Multiply values by 2",
                upgrade_fn=lambda df: df.with_columns((pl.col("value") * 2).alias("value")),
                downgrade_fn=lambda df: df.with_columns((pl.col("value") / 2).alias("value")),
            )
        )

        # Dry run first
        dry_records = manager.migrate(dry_run=True)
        assert len(dry_records) == 1
        assert dry_records[0].status == MigrationStatus.COMPLETED

        # Verify data unchanged after dry run
        original_df = pl.read_parquet(data_dir / "file_0.parquet")
        assert original_df["value"][0] == 0

        # Run actual migration
        records = manager.migrate()
        assert len(records) == 1
        assert records[0].status == MigrationStatus.COMPLETED
        assert records[0].rows_affected == 15  # 3 files × 5 rows

        # Verify data changed
        migrated_df = pl.read_parquet(data_dir / "file_0.parquet")
        assert migrated_df["value"][0] == 0  # 0 * 2 = 0

    def test_migration_state_persistence(self, temp_data_dir):
        """Test that migration state persists across manager instances."""
        # First manager runs migration
        manager1 = MigrationManager(temp_data_dir)
        manager1.register_migration(
            Migration(version="1.0.0", description="Test", upgrade_fn=lambda df: df)
        )

        # Create minimal data
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()
        pl.DataFrame({"x": [1]}).write_parquet(data_dir / "test.parquet")

        manager1.migrate()

        # Second manager should see migration was run
        manager2 = MigrationManager(temp_data_dir)
        assert manager2._get_current_version() == "1.0.0"

    def test_skip_already_migrated(self, temp_data_dir):
        """Test that already-applied migrations are skipped."""
        manager = MigrationManager(temp_data_dir)

        # Create test data
        data_dir = temp_data_dir / "data"
        data_dir.mkdir()
        pl.DataFrame({"x": [1]}).write_parquet(data_dir / "test.parquet")

        manager.register_migration(
            Migration(version="1.0.0", description="First", upgrade_fn=lambda df: df)
        )

        # Run first migration
        records1 = manager.migrate()
        assert len(records1) == 1

        # Run again - should be empty
        records2 = manager.migrate()
        assert len(records2) == 0
