"""Base storage abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.data.core.models import DataObject


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write(self, data: DataObject) -> str:
        """
        Write data to storage.

        Args:
            data: DataObject to write

        Returns:
            Storage key for the written data
        """

    @abstractmethod
    def read(self, key: str) -> DataObject:
        """
        Read data from storage.

        Args:
            key: Storage key

        Returns:
            DataObject from storage
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if data exists in storage.

        Args:
            key: Storage key

        Returns:
            True if data exists, False otherwise
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete data from storage.

        Args:
            key: Storage key
        """

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys in storage.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of storage keys
        """
