# Async Storage Backend Guide

ML4T Data now includes a high-performance async storage backend that provides non-blocking I/O operations for improved concurrency and scalability.

## Overview

The async storage backend is built on top of `aiofiles` and provides the same interface as the synchronous storage backend, but with async/await support for better performance in high-concurrency scenarios.

## Features

- **Non-blocking I/O** - Uses `aiofiles` for async file operations
- **Shared Lock Coordination** - Prevents race conditions with coordinated locking
- **Migration Utilities** - Easy conversion between sync and async storage
- **ACID Transactions** - Full transaction support with rollback capability
- **Exponential Backoff** - Automatic retry with intelligent backoff
- **Path Security** - Built-in protection against path traversal attacks

## Installation

```bash
# Install async storage dependencies
pip install -e ".[async]"
```

## Basic Usage

### Async Storage Backend

```python
import asyncio
from pathlib import Path
from ml4t.data.storage.async_filesystem import AsyncFileSystemBackend
from ml4t.data.core.models import DataObject, Metadata
import polars as pl
from datetime import datetime

async def main():
    # Create async backend
    storage = AsyncFileSystemBackend(
        data_root=Path("./data"),
        lock_timeout=30.0,
        max_retries=3
    )

    # Create sample data
    df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [99.0, 100.0],
        "close": [104.0, 105.0],
        "volume": [1000000, 1100000]
    })

    metadata = Metadata(
        provider="test",
        symbol="AAPL",
        asset_class="equities",
        frequency="daily",
        schema_version="1.0"
    )

    data = DataObject(data=df, metadata=metadata)

    # Write data
    key = await storage.write(data)
    print(f"Data written with key: {key}")

    # Read data
    loaded_data = await storage.read(key)
    print(f"Loaded data for {loaded_data.metadata.symbol}")

    # Check existence
    exists = await storage.exists(key)
    print(f"Data exists: {exists}")

    # List all keys
    keys = await storage.list_keys()
    print(f"All keys: {keys}")

    # Delete data
    await storage.delete(key)
    print("Data deleted")

# Run the async function
asyncio.run(main())
```

### Using Sync Adapter

If you need to use the async storage backend in synchronous code:

```python
from ml4t.data.storage.async_filesystem import AsyncFileSystemBackend
from ml4t.data.storage.async_migration import AsyncStorageAdapter

# Create async backend
async_backend = AsyncFileSystemBackend(data_root=Path("./data"))

# Wrap with sync adapter
sync_storage = AsyncStorageAdapter(async_backend)

# Use like a regular sync storage backend
key = sync_storage.write(data)
loaded_data = sync_storage.read(key)
```

## Migration Between Sync and Async

### Sync to Async Migration

```python
import asyncio
from ml4t.data.storage.filesystem import FileSystemBackend
from ml4t.data.storage.async_filesystem import AsyncFileSystemBackend
from ml4t.data.storage.async_migration import StorageMigrator

async def migrate_to_async():
    # Create backends
    sync_backend = FileSystemBackend(data_root=Path("./sync_data"))
    async_backend = AsyncFileSystemBackend(data_root=Path("./async_data"))

    # Migrate all data
    successful, failed = await StorageMigrator.sync_to_async(
        sync_backend=sync_backend,
        async_backend=async_backend,
        batch_size=10  # Process 10 items at once
    )

    print(f"Migration complete: {successful} successful, {failed} failed")

asyncio.run(migrate_to_async())
```

### Async to Sync Migration

```python
async def migrate_to_sync():
    # Create backends
    async_backend = AsyncFileSystemBackend(data_root=Path("./async_data"))
    sync_backend = FileSystemBackend(data_root=Path("./sync_data"))

    # Migrate all data
    successful, failed = await StorageMigrator.async_to_sync(
        async_backend=async_backend,
        sync_backend=sync_backend,
        prefix="equities/daily/",  # Only migrate specific prefix
        batch_size=5
    )

    print(f"Migration complete: {successful} successful, {failed} failed")

asyncio.run(migrate_to_sync())
```

## Advanced Configuration

### Lock Configuration

```python
storage = AsyncFileSystemBackend(
    data_root=Path("./data"),
    lock_timeout=60.0,        # Wait up to 60 seconds for lock
    max_retries=5,            # Retry up to 5 times
    retry_base_delay=0.1,     # Start with 100ms delay
)
```

### Concurrent Operations

The async backend is designed for high concurrency:

```python
async def concurrent_writes():
    storage = AsyncFileSystemBackend(data_root=Path("./data"))

    # Create multiple data objects
    tasks = []
    for i in range(10):
        data = create_sample_data(f"STOCK{i}")
        task = storage.write(data)
        tasks.append(task)

    # Execute all writes concurrently
    keys = await asyncio.gather(*tasks)
    print(f"Wrote {len(keys)} files concurrently")

asyncio.run(concurrent_writes())
```

### Concurrent Reads

```python
async def concurrent_reads():
    storage = AsyncFileSystemBackend(data_root=Path("./data"))

    # List all available keys
    keys = await storage.list_keys()

    # Read all data concurrently
    tasks = [storage.read(key) for key in keys]
    data_objects = await asyncio.gather(*tasks)

    print(f"Read {len(data_objects)} files concurrently")

asyncio.run(concurrent_reads())
```

## Performance Considerations

### When to Use Async Storage

- **High Concurrency**: When you need to handle many simultaneous requests
- **I/O Bound Operations**: When storage operations are the bottleneck
- **API Servers**: When building async web applications with FastAPI/aiohttp
- **Batch Processing**: When processing large datasets with many files

### When to Use Sync Storage

- **Simple Scripts**: For simple data loading scripts
- **Interactive Use**: For interactive analysis in Jupyter notebooks
- **Legacy Code**: When integrating with existing synchronous codebases

## Error Handling

```python
from ml4t.data.storage.async_filesystem import LockAcquisitionError, StorageError

async def robust_operation():
    storage = AsyncFileSystemBackend(data_root=Path("./data"))

    try:
        data = await storage.read("equities/daily/AAPL")
    except StorageError as e:
        print(f"Storage error: {e}")
    except LockAcquisitionError as e:
        print(f"Could not acquire lock: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Testing

The async storage backend includes comprehensive tests:

```bash
# Run async storage tests
pytest tests/test_async_storage.py -v

# Run with coverage
pytest tests/test_async_storage.py --cov=src/ml4t-data/storage/async_filesystem
```

## Implementation Details

### Lock Management

- Each storage key gets its own `asyncio.Lock` for fine-grained locking
- Locks are shared across `AsyncFileLock` instances via a backend-level registry
- File locks serve as indicators while actual coordination happens via asyncio
- Automatic cleanup prevents lock leakage

### File Format Compatibility

The async storage backend uses the same file formats as the sync backend:

- **Data files**: `.parquet` format using Polars
- **Metadata files**: `.json` format with proper datetime serialization
- **Directory structure**: Same hierarchical organization

### Migration Safety

- Migration utilities preserve data integrity
- Batch processing prevents memory issues with large datasets
- Progress callbacks for monitoring long-running migrations
- Error recovery and reporting for failed items

## Future Enhancements

- Database backend support (PostgreSQL, MongoDB)
- S3/cloud storage integration
- Distributed locking for multi-node deployments
- Advanced caching with Redis
- Compression during async I/O operations
