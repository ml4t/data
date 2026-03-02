"""Tests for transaction rollback functionality."""

from copy import deepcopy
from datetime import datetime

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.storage.transaction import (
    Transaction,
    TransactionalStorage,
    TransactionError,
    TransactionState,
)


class _InMemoryStorage:
    """Minimal in-memory storage implementing the legacy DataObject interface.

    Used only in tests — NOT a production backend.
    """

    def __init__(self):
        self._store: dict[str, DataObject] = {}

    def write(self, data: DataObject) -> str:
        key = f"{data.metadata.asset_class}/{data.metadata.frequency}/{data.metadata.symbol}"
        self._store[key] = deepcopy(data)
        return key

    def read(self, key: str) -> DataObject:
        if key not in self._store:
            raise KeyError(key)
        return deepcopy(self._store[key])

    def exists(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._store if k.startswith(prefix))


@pytest.fixture
def temp_storage():
    """Create an in-memory storage backend for testing."""
    return _InMemoryStorage()


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
                datetime(2024, 1, 4),
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
        txn = Transaction(temp_storage)
        key = txn.write(sample_data)
        assert key == "equities/daily/TEST"
        assert temp_storage.exists(key)
        stored_data = temp_storage.read(key)
        assert len(stored_data.data) == 3
        txn.commit()
        assert txn.state == TransactionState.COMMITTED
        assert temp_storage.exists(key)

    def test_transaction_write_rollback(self, temp_storage, sample_data):
        txn = Transaction(temp_storage)
        key = txn.write(sample_data)
        assert temp_storage.exists(key)
        txn.rollback()
        assert txn.state == TransactionState.ROLLED_BACK
        assert not temp_storage.exists(key)

    def test_transaction_update_rollback(self, temp_storage, sample_data, updated_data):
        key = temp_storage.write(sample_data)
        original_data = temp_storage.read(key)
        assert len(original_data.data) == 3

        txn = Transaction(temp_storage)
        txn.update(key, updated_data)
        current_data = temp_storage.read(key)
        assert len(current_data.data) == 4

        txn.rollback()
        restored_data = temp_storage.read(key)
        assert len(restored_data.data) == 3
        assert restored_data.data.equals(original_data.data)

    def test_transaction_delete_rollback(self, temp_storage, sample_data):
        key = temp_storage.write(sample_data)
        original_data = temp_storage.read(key)

        txn = Transaction(temp_storage)
        txn.delete(key)
        assert not temp_storage.exists(key)

        txn.rollback()
        assert temp_storage.exists(key)
        restored_data = temp_storage.read(key)
        assert restored_data.data.equals(original_data.data)

    def test_transaction_context_manager_success(self, temp_storage, sample_data):
        with Transaction(temp_storage) as txn:
            key = txn.write(sample_data)
            assert temp_storage.exists(key)
        assert temp_storage.exists(key)

    def test_transaction_context_manager_failure(self, temp_storage, sample_data):
        key = None
        try:
            with Transaction(temp_storage) as txn:
                key = txn.write(sample_data)
                assert temp_storage.exists(key)
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        assert not temp_storage.exists(key)

    def test_transaction_multiple_operations(self, temp_storage, sample_data, updated_data):
        key1 = temp_storage.write(sample_data)

        txn = Transaction(temp_storage)
        txn.update(key1, updated_data)

        new_data = deepcopy(sample_data)
        new_data.metadata.symbol = "TEST2"
        key2 = txn.write(new_data)

        delete_data = deepcopy(sample_data)
        delete_data.metadata.symbol = "TEST3"
        key3 = temp_storage.write(delete_data)
        txn.delete(key3)

        assert len(temp_storage.read(key1).data) == 4
        assert temp_storage.exists(key2)
        assert not temp_storage.exists(key3)

        txn.rollback()

        assert len(temp_storage.read(key1).data) == 3
        assert not temp_storage.exists(key2)
        assert temp_storage.exists(key3)

    def test_transaction_invalid_state_operations(self, temp_storage, sample_data):
        txn = Transaction(temp_storage)
        txn.write(sample_data)
        txn.commit()

        with pytest.raises(
            TransactionError, match="Cannot write in TransactionState.COMMITTED transaction"
        ):
            txn.write(sample_data)

        txn2 = Transaction(temp_storage)
        txn2.rollback()
        with pytest.raises(
            TransactionError, match="Cannot write in TransactionState.ROLLED_BACK transaction"
        ):
            txn2.write(sample_data)


class TestTransactionalStorage:
    """Test the TransactionalStorage wrapper."""

    def test_transactional_storage_basic_operations(self, temp_storage, sample_data):
        txn_storage = TransactionalStorage(temp_storage)
        key = txn_storage.write(sample_data)
        assert txn_storage.exists(key)
        data = txn_storage.read(key)
        assert len(data.data) == 3
        txn_storage.delete(key)
        assert not txn_storage.exists(key)

    def test_transactional_storage_context_manager(self, temp_storage, sample_data, updated_data):
        txn_storage = TransactionalStorage(temp_storage)
        with txn_storage.transaction() as txn:
            key = txn.write(sample_data)
            assert temp_storage.exists(key)
        assert temp_storage.exists(key)

        try:
            with txn_storage.transaction() as txn:
                txn.update(key, updated_data)
                assert len(temp_storage.read(key).data) == 4
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        data = temp_storage.read(key)
        assert len(data.data) == 3

    def test_nested_transactions_not_allowed(self, temp_storage, sample_data):
        txn_storage = TransactionalStorage(temp_storage)
        with pytest.raises(TransactionError, match="Nested transactions are not supported"):
            with txn_storage.transaction():
                with txn_storage.transaction():
                    pass

    def test_operations_outside_transaction(self, temp_storage, sample_data):
        txn_storage = TransactionalStorage(temp_storage)
        key = txn_storage.write(sample_data)
        assert txn_storage.exists(key)
        data = txn_storage.read(key)
        assert len(data.data) == 3
        txn_storage.delete(key)
        assert not txn_storage.exists(key)


class TestTransactionFailureScenarios:
    """Test various failure scenarios with transactions."""

    def test_storage_write_failure_rollback(self, temp_storage, sample_data):
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
        key1 = failing_storage.write(sample_data)
        original_data = failing_storage.read(key1)

        txn = Transaction(failing_storage)
        update_data = deepcopy(sample_data)
        update_data.metadata.symbol = "TEST2"
        txn.update(key1, update_data)

        failing_storage.fail_next_write = True
        write_data = deepcopy(sample_data)
        write_data.metadata.symbol = "TEST3"
        with pytest.raises(TransactionError):
            txn.write(write_data)
        assert txn.state == TransactionState.FAILED

        failing_storage.fail_next_write = False
        txn.rollback()
        restored_data = failing_storage.read(key1)
        assert restored_data.data.equals(original_data.data)

    def test_rollback_partial_failure(self, temp_storage, sample_data):
        class PartiallyFailingStorage:
            def __init__(self, real_storage):
                self.real_storage = real_storage
                self.fail_operations: set[str] = set()

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
        key = failing_storage.write(sample_data)

        txn = Transaction(failing_storage)
        updated_data = deepcopy(sample_data)
        updated_data.metadata.symbol = "UPDATED"
        txn.update(key, updated_data)

        failing_storage.fail_operations.add("write")
        with pytest.raises(TransactionError, match="Rollback failed with errors"):
            txn.rollback()
        assert txn.state == TransactionState.FAILED
