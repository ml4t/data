"""Tests for transaction rollback functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.storage.filesystem import FileSystemBackend
from ml4t.data.storage.transaction import (
    Transaction,
    TransactionalStorage,
    TransactionError,
    TransactionState,
)


@pytest.fixture
def temp_storage():
    """Create a temporary storage backend for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FileSystemBackend(data_root=Path(tmpdir))


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000000, 1100000, 1200000],
        }
    )

    metadata = Metadata(
        provider="test",
        symbol="TEST",
        asset_class="equities",
        bar_type="time",
        bar_params={"frequency": "daily"},
        schema_version="1.0",
    )

    return DataObject(data=df, metadata=metadata)


@pytest.fixture
def updated_data():
    """Create updated sample data for testing."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),  # New day
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [104.0, 105.0, 106.0, 107.0],
            "volume": [1000000, 1100000, 1200000, 1300000],
        }
    )

    metadata = Metadata(
        provider="test",
        symbol="TEST",
        asset_class="equities",
        bar_type="time",
        bar_params={"frequency": "daily"},
        schema_version="1.0",
    )

    return DataObject(data=df, metadata=metadata)


class TestTransaction:
    """Test the Transaction class."""

    def test_transaction_write_commit(self, temp_storage, sample_data):
        """Test successful transaction write and commit."""
        txn = Transaction(temp_storage)

        # Write data
        key = txn.write(sample_data)
        assert key == "equities/daily/TEST"

        # Verify data exists
        assert temp_storage.exists(key)
        stored_data = temp_storage.read(key)
        assert len(stored_data.data) == 3

        # Commit transaction
        txn.commit()
        assert txn.state == TransactionState.COMMITTED

        # Data should still exist after commit
        assert temp_storage.exists(key)

    def test_transaction_write_rollback(self, temp_storage, sample_data):
        """Test transaction write followed by rollback."""
        txn = Transaction(temp_storage)

        # Write data
        key = txn.write(sample_data)
        assert temp_storage.exists(key)

        # Rollback transaction
        txn.rollback()
        assert txn.state == TransactionState.ROLLED_BACK

        # Data should be removed after rollback
        assert not temp_storage.exists(key)

    def test_transaction_update_rollback(self, temp_storage, sample_data, updated_data):
        """Test transaction update followed by rollback."""
        # First write initial data outside transaction
        key = temp_storage.write(sample_data)
        original_data = temp_storage.read(key)
        assert len(original_data.data) == 3

        # Start transaction and update
        txn = Transaction(temp_storage)
        txn.update(key, updated_data)

        # Verify update happened
        current_data = temp_storage.read(key)
        assert len(current_data.data) == 4  # Updated data has 4 rows

        # Rollback transaction
        txn.rollback()

        # Data should be restored to original
        restored_data = temp_storage.read(key)
        assert len(restored_data.data) == 3
        assert restored_data.data.equals(original_data.data)

    def test_transaction_delete_rollback(self, temp_storage, sample_data):
        """Test transaction delete followed by rollback."""
        # First write data outside transaction
        key = temp_storage.write(sample_data)
        original_data = temp_storage.read(key)

        # Start transaction and delete
        txn = Transaction(temp_storage)
        txn.delete(key)

        # Verify deletion happened
        assert not temp_storage.exists(key)

        # Rollback transaction
        txn.rollback()

        # Data should be restored
        assert temp_storage.exists(key)
        restored_data = temp_storage.read(key)
        assert restored_data.data.equals(original_data.data)

    def test_transaction_context_manager_success(self, temp_storage, sample_data):
        """Test transaction context manager with successful operation."""
        with Transaction(temp_storage) as txn:
            key = txn.write(sample_data)
            assert temp_storage.exists(key)

        # Should auto-commit on success
        assert temp_storage.exists(key)

    def test_transaction_context_manager_failure(self, temp_storage, sample_data):
        """Test transaction context manager with exception."""
        key = None

        try:
            with Transaction(temp_storage) as txn:
                key = txn.write(sample_data)
                assert temp_storage.exists(key)
                raise ValueError("Simulated failure")
        except ValueError:
            pass

        # Should auto-rollback on exception
        assert not temp_storage.exists(key)

    def test_transaction_multiple_operations(self, temp_storage, sample_data, updated_data):
        """Test transaction with multiple operations."""
        # Create initial data
        key1 = temp_storage.write(sample_data)

        # Start transaction with multiple operations
        txn = Transaction(temp_storage)

        # Update existing data
        txn.update(key1, updated_data)

        # Write new data
        from copy import deepcopy

        new_data = deepcopy(sample_data)
        new_data.metadata.symbol = "TEST2"
        key2 = txn.write(new_data)

        # Delete third piece of data (create it first)
        delete_data = deepcopy(sample_data)
        delete_data.metadata.symbol = "TEST3"
        key3 = temp_storage.write(delete_data)
        txn.delete(key3)

        # Verify operations happened
        assert len(temp_storage.read(key1).data) == 4  # Updated
        assert temp_storage.exists(key2)  # New data
        assert not temp_storage.exists(key3)  # Deleted

        # Rollback all operations
        txn.rollback()

        # Verify rollback
        assert len(temp_storage.read(key1).data) == 3  # Restored
        assert not temp_storage.exists(key2)  # New data removed
        assert temp_storage.exists(key3)  # Deleted data restored

    def test_transaction_invalid_state_operations(self, temp_storage, sample_data):
        """Test that operations fail on committed/rolled back transactions."""
        txn = Transaction(temp_storage)

        # Write and commit
        txn.write(sample_data)
        txn.commit()

        # Should not allow further operations
        with pytest.raises(
            TransactionError, match="Cannot write in TransactionState.COMMITTED transaction"
        ):
            txn.write(sample_data)

        # Test with rolled back transaction
        txn2 = Transaction(temp_storage)
        txn2.rollback()

        with pytest.raises(
            TransactionError, match="Cannot write in TransactionState.ROLLED_BACK transaction"
        ):
            txn2.write(sample_data)


class TestTransactionalStorage:
    """Test the TransactionalStorage wrapper."""

    def test_transactional_storage_basic_operations(self, temp_storage, sample_data):
        """Test basic operations through transactional storage."""
        txn_storage = TransactionalStorage(temp_storage)

        # Test write
        key = txn_storage.write(sample_data)
        assert txn_storage.exists(key)

        # Test read
        data = txn_storage.read(key)
        assert len(data.data) == 3

        # Test delete
        txn_storage.delete(key)
        assert not txn_storage.exists(key)

    def test_transactional_storage_context_manager(self, temp_storage, sample_data, updated_data):
        """Test transactional storage with context manager."""
        txn_storage = TransactionalStorage(temp_storage)

        # Successful transaction
        with txn_storage.transaction() as txn:
            key = txn.write(sample_data)
            assert temp_storage.exists(key)

        # Should be committed
        assert temp_storage.exists(key)

        # Failed transaction
        try:
            with txn_storage.transaction() as txn:
                txn.update(key, updated_data)
                assert len(temp_storage.read(key).data) == 4
                raise ValueError("Simulated failure")
        except ValueError:
            pass

        # Should be rolled back
        data = temp_storage.read(key)
        assert len(data.data) == 3  # Original data restored

    def test_nested_transactions_not_allowed(self, temp_storage, sample_data):
        """Test that nested transactions are not allowed."""
        txn_storage = TransactionalStorage(temp_storage)

        with pytest.raises(TransactionError, match="Nested transactions are not supported"):
            with txn_storage.transaction():
                with txn_storage.transaction():
                    pass

    def test_operations_outside_transaction(self, temp_storage, sample_data):
        """Test that operations work outside of transaction context."""
        txn_storage = TransactionalStorage(temp_storage)

        # Operations should work normally outside transaction
        key = txn_storage.write(sample_data)
        assert txn_storage.exists(key)

        data = txn_storage.read(key)
        assert len(data.data) == 3

        txn_storage.delete(key)
        assert not txn_storage.exists(key)


class TestTransactionFailureScenarios:
    """Test various failure scenarios with transactions."""

    def test_storage_write_failure_rollback(self, temp_storage, sample_data):
        """Test rollback when storage write fails."""

        # Create a storage that will fail on write
        class FailingStorage:
            def __init__(self, real_storage):
                self.real_storage = real_storage
                self.fail_next_write = False

            def write(self, data):
                if self.fail_next_write:
                    raise OSError("Simulated write failure")
                return self.real_storage.write(data)

            def __getattr__(self, name):
                return getattr(self.real_storage, name)

        failing_storage = FailingStorage(temp_storage)

        # First write some data
        key1 = failing_storage.write(sample_data)
        original_data = failing_storage.read(key1)

        # Start transaction
        txn = Transaction(failing_storage)

        # Update existing data successfully
        # Create a copy of sample data with different symbol
        from copy import deepcopy

        update_data = deepcopy(sample_data)
        update_data.metadata.symbol = "TEST2"
        txn.update(key1, update_data)

        # Prepare to fail the next write
        failing_storage.fail_next_write = True

        # Try to write new data (should fail)
        write_data = deepcopy(sample_data)
        write_data.metadata.symbol = "TEST3"
        with pytest.raises(TransactionError):
            txn.write(write_data)

        # Transaction should be in failed state
        assert txn.state == TransactionState.FAILED

        # Reset the failure flag so rollback can succeed
        failing_storage.fail_next_write = False

        # Rollback should restore original data
        txn.rollback()
        restored_data = failing_storage.read(key1)
        assert restored_data.data.equals(original_data.data)

    def test_rollback_partial_failure(self, temp_storage, sample_data):
        """Test rollback when some rollback operations fail."""

        # Create storage that fails on specific operations
        class PartiallyFailingStorage:
            def __init__(self, real_storage):
                self.real_storage = real_storage
                self.fail_operations = set()

            def write(self, data):
                if "write" in self.fail_operations:
                    raise OSError("Write failed")
                return self.real_storage.write(data)

            def delete(self, key):
                if "delete" in self.fail_operations:
                    raise OSError("Delete failed")
                return self.real_storage.delete(key)

            def __getattr__(self, name):
                return getattr(self.real_storage, name)

        failing_storage = PartiallyFailingStorage(temp_storage)

        # First, write some data outside the transaction
        key = failing_storage.write(sample_data)

        # Create transaction and update the data (this will create a backup)
        txn = Transaction(failing_storage)
        from copy import deepcopy

        updated_data = deepcopy(sample_data)
        updated_data.metadata.symbol = "UPDATED"
        txn.update(key, updated_data)

        # Make rollback fail on write (when trying to restore backup)
        failing_storage.fail_operations.add("write")

        # Rollback should fail
        with pytest.raises(TransactionError, match="Rollback failed with errors"):
            txn.rollback()

        assert txn.state == TransactionState.FAILED
