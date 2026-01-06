"""Transaction support for storage operations with rollback capability."""

from __future__ import annotations

import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import structlog

from ml4t.data.core.models import DataObject
from ml4t.data.storage.base import StorageBackend

logger = structlog.get_logger()


class TransactionState(Enum):
    """Transaction state."""

    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionOperation:
    """Represents a single operation within a transaction."""

    operation_type: str  # "write", "update", "delete"
    key: str
    data: DataObject | None = None
    backup_data: DataObject | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class TransactionError(Exception):
    """Raised when a transaction operation fails."""


class Transaction:
    """
    Manages atomic storage operations with rollback capability.

    Ensures data consistency by creating backups before modifications
    and providing rollback on failure.
    """

    def __init__(
        self,
        storage: StorageBackend,
        transaction_id: str | None = None,
        backup_dir: Path | None = None,
    ):
        """
        Initialize a transaction.

        Args:
            storage: Storage backend to use
            transaction_id: Optional transaction ID (generated if not provided)
            backup_dir: Optional directory for backups (temp dir if not provided)
        """
        self.storage = storage
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.state = TransactionState.PENDING
        self.operations: list[TransactionOperation] = []
        self.backup_dir = backup_dir or Path(tempfile.gettempdir()) / "ml4t_data_transactions"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Transaction started", transaction_id=self.transaction_id)

    def _create_backup(self, key: str) -> DataObject | None:
        """
        Create a backup of existing data.

        Args:
            key: Storage key to backup

        Returns:
            Backup of the data if it exists, None otherwise
        """
        try:
            if self.storage.exists(key):
                data = self.storage.read(key)
                logger.debug("Created backup", key=key, transaction_id=self.transaction_id)
                return data
        except Exception as e:
            logger.warning(
                "Failed to create backup",
                key=key,
                error=str(e),
                transaction_id=self.transaction_id,
            )
        return None

    def write(self, data: DataObject) -> str:
        """
        Write data within the transaction.

        Args:
            data: Data to write

        Returns:
            Storage key

        Raises:
            TransactionError: If transaction is not in pending state
        """
        if self.state != TransactionState.PENDING:
            raise TransactionError(f"Cannot write in {self.state} transaction")

        # Generate key from metadata
        key = f"{data.metadata.asset_class}/{data.metadata.frequency}/{data.metadata.symbol}"

        # Create backup if data exists
        backup = self._create_backup(key)

        # Record the operation
        operation = TransactionOperation(
            operation_type="write",
            key=key,
            data=data,
            backup_data=backup,
        )
        self.operations.append(operation)

        # Perform the write
        try:
            # Write the DataObject directly (new API)
            result_key = self.storage.write(data)
            logger.info(
                "Transaction write completed",
                key=result_key,
                transaction_id=self.transaction_id,
            )
            return result_key
        except Exception as e:
            logger.error(
                "Transaction write failed",
                key=key,
                error=str(e),
                transaction_id=self.transaction_id,
            )
            self.state = TransactionState.FAILED
            raise TransactionError(f"Write failed for {key}: {e}") from e

    def update(self, key: str, data: DataObject) -> str:
        """
        Update data within the transaction.

        Args:
            key: Storage key to update
            data: New data

        Returns:
            Storage key

        Raises:
            TransactionError: If transaction is not in pending state
        """
        if self.state != TransactionState.PENDING:
            raise TransactionError(f"Cannot update in {self.state} transaction")

        # Create backup of existing data
        backup = self._create_backup(key)
        if backup is None:
            raise TransactionError(f"Cannot update non-existent key: {key}")

        # Record the operation
        operation = TransactionOperation(
            operation_type="update",
            key=key,
            data=data,
            backup_data=backup,
        )
        self.operations.append(operation)

        # Perform the update (delete + write)
        try:
            self.storage.delete(key)
            # Write the DataObject directly (new API)
            result_key = self.storage.write(data)
            logger.info(
                "Transaction update completed",
                key=result_key,
                transaction_id=self.transaction_id,
            )
            return result_key
        except Exception as e:
            logger.error(
                "Transaction update failed",
                key=key,
                error=str(e),
                transaction_id=self.transaction_id,
            )
            self.state = TransactionState.FAILED
            raise TransactionError(f"Update failed for {key}: {e}") from e

    def delete(self, key: str) -> None:
        """
        Delete data within the transaction.

        Args:
            key: Storage key to delete

        Raises:
            TransactionError: If transaction is not in pending state
        """
        if self.state != TransactionState.PENDING:
            raise TransactionError(f"Cannot delete in {self.state} transaction")

        # Create backup before deletion
        backup = self._create_backup(key)
        if backup is None:
            logger.warning(
                "Deleting non-existent key",
                key=key,
                transaction_id=self.transaction_id,
            )
            return

        # Record the operation
        operation = TransactionOperation(
            operation_type="delete",
            key=key,
            backup_data=backup,
        )
        self.operations.append(operation)

        # Perform the deletion
        try:
            self.storage.delete(key)
            logger.info(
                "Transaction delete completed",
                key=key,
                transaction_id=self.transaction_id,
            )
        except Exception as e:
            logger.error(
                "Transaction delete failed",
                key=key,
                error=str(e),
                transaction_id=self.transaction_id,
            )
            self.state = TransactionState.FAILED
            raise TransactionError(f"Delete failed for {key}: {e}") from e

    def commit(self) -> None:
        """
        Commit the transaction.

        All operations are already applied, this just marks the transaction as committed.

        Raises:
            TransactionError: If transaction is not in pending state
        """
        if self.state != TransactionState.PENDING:
            raise TransactionError(f"Cannot commit {self.state} transaction")

        self.state = TransactionState.COMMITTED
        logger.info(
            "Transaction committed",
            transaction_id=self.transaction_id,
            operations=len(self.operations),
        )

    def rollback(self) -> None:
        """
        Rollback all operations in the transaction.

        Restores data to the state before the transaction started.

        Raises:
            TransactionError: If rollback fails
        """
        if self.state == TransactionState.ROLLED_BACK:
            logger.warning(
                "Transaction already rolled back",
                transaction_id=self.transaction_id,
            )
            return

        logger.info(
            "Starting transaction rollback",
            transaction_id=self.transaction_id,
            operations=len(self.operations),
        )

        # Rollback operations in reverse order
        errors = []
        for operation in reversed(self.operations):
            try:
                if operation.operation_type in ("write", "update"):
                    # Restore backup or delete if no backup
                    if operation.backup_data:
                        # Restore previous data
                        self.storage.write(operation.backup_data)
                        logger.debug(
                            "Restored backup",
                            key=operation.key,
                            transaction_id=self.transaction_id,
                        )
                    else:
                        # No backup means it was new, so delete it
                        if self.storage.exists(operation.key):
                            self.storage.delete(operation.key)
                            logger.debug(
                                "Deleted new data",
                                key=operation.key,
                                transaction_id=self.transaction_id,
                            )

                elif operation.operation_type == "delete" and operation.backup_data:
                    # Restore deleted data
                    self.storage.write(operation.backup_data)
                    logger.debug(
                        "Restored deleted data",
                        key=operation.key,
                        transaction_id=self.transaction_id,
                    )

            except Exception as e:
                error_msg = f"Failed to rollback {operation.operation_type} on {operation.key}: {e}"
                logger.error(
                    error_msg,
                    transaction_id=self.transaction_id,
                )
                errors.append(error_msg)

        if errors:
            self.state = TransactionState.FAILED
            raise TransactionError(f"Rollback failed with errors: {'; '.join(errors)}")

        self.state = TransactionState.ROLLED_BACK
        logger.info(
            "Transaction rolled back successfully",
            transaction_id=self.transaction_id,
        )

    def __enter__(self):
        """Enter the transaction context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the transaction context.

        Commits if no exception, rolls back on exception.
        """
        if exc_type is None:
            # No exception, commit the transaction
            try:
                self.commit()
            except Exception as e:
                logger.error(
                    "Failed to commit transaction",
                    transaction_id=self.transaction_id,
                    error=str(e),
                )
                self.rollback()
                raise
        else:
            # Exception occurred, rollback
            logger.info(
                "Exception in transaction, rolling back",
                transaction_id=self.transaction_id,
                exception=str(exc_val),
            )
            try:
                self.rollback()
            except Exception as e:
                logger.error(
                    "Rollback failed",
                    transaction_id=self.transaction_id,
                    error=str(e),
                )
            # Re-raise the original exception
            return False


class TransactionalStorage:
    """
    Wrapper for storage backend that provides transaction support.
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize transactional storage.

        Args:
            storage: Underlying storage backend
        """
        self.storage = storage
        self._current_transaction: Transaction | None = None

    @contextmanager
    def transaction(self, transaction_id: str | None = None):
        """
        Create a transaction context.

        Args:
            transaction_id: Optional transaction ID

        Yields:
            Transaction object

        Example:
            ```python
            with storage.transaction() as txn:
                txn.write(data1)
                txn.update(key2, data2)
                txn.delete(key3)
                # Automatically commits on success, rolls back on exception
            ```
        """
        if self._current_transaction is not None:
            raise TransactionError("Nested transactions are not supported")

        transaction = Transaction(self.storage, transaction_id)
        self._current_transaction = transaction

        try:
            with transaction:
                yield transaction
        finally:
            self._current_transaction = None

    def write(self, data: DataObject) -> str:
        """
        Write data, using transaction if active.

        Args:
            data: Data to write

        Returns:
            Storage key
        """
        if self._current_transaction:
            return self._current_transaction.write(data)
        return self.storage.write(data)

    def update(self, key: str, data: DataObject) -> str:
        """
        Update data, using transaction if active.

        Args:
            key: Storage key
            data: New data

        Returns:
            Storage key
        """
        if self._current_transaction:
            return self._current_transaction.update(key, data)
        # Without transaction, update is delete + write
        if self.storage.exists(key):
            self.storage.delete(key)
        return self.storage.write(data)

    def delete(self, key: str) -> None:
        """
        Delete data, using transaction if active.

        Args:
            key: Storage key
        """
        if self._current_transaction:
            return self._current_transaction.delete(key)
        return self.storage.delete(key)

    def read(self, key: str) -> DataObject:
        """Read data from storage."""
        return self.storage.read(key)

    def exists(self, key: str) -> bool:
        """Check if data exists."""
        return self.storage.exists(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """List storage keys."""
        return self.storage.list_keys(prefix)
