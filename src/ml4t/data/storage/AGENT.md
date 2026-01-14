# storage/ - 4.4k Lines

Hive-partitioned Parquet storage (7x faster).

## Modules

| File | Purpose |
|------|---------|
| hive.py | HiveStorage main class |
| backend.py | Storage backend interface |
| flat.py | Simple flat file storage |
| chunked.py | Chunked file storage |
| filesystem.py | Filesystem operations |
| metadata_tracker.py | Metadata management |
| protocols.py | Storage protocols |
| async_base.py | Async storage base |

## Key

`HiveStorage`, `get_storage()`, `read_symbol()`, `write_symbol()`
