# storage/ - 4.6k Lines

Hive-partitioned Parquet storage (7x faster) and data profiling.

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
| data_profile.py | Column-level statistics and ProfileMixin |

## Key Classes

- `HiveStorage` - Main storage class with Hive partitioning
- `ProfileMixin` - Mixin for on-demand data profiling (generate_profile, load_profile)
- `DatasetProfile` - Column statistics container
- `ColumnProfile` - Per-column statistics (dtype, nulls, min/max, mean/std)

## Data Profiling

```python
from ml4t.data.storage import ProfileMixin, generate_profile

# Any data manager can inherit ProfileMixin for profiling
class MyManager(ProfileMixin):
    def _get_profile_data(self) -> pl.DataFrame:
        return self.load_data()
    def _get_profile_data_path(self) -> Path:
        return self.data_path
    def _get_profile_source_name(self) -> str:
        return "MyManager"

# Now has generate_profile() and load_profile() methods
```

## Key Functions

`HiveStorage`, `get_storage()`, `generate_profile()`, `save_profile()`, `load_profile()`
