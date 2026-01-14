# ml4t-data

Market data acquisition and storage library.

## Structure

| Directory | Purpose |
|-----------|---------|
| src/ml4t/data/ | Package root |
| tests/ | Test suite |
| examples/ | Usage examples |

## Key Modules

| Module | Purpose |
|--------|---------|
| data_manager.py | Orchestration API |
| update_manager.py | Incremental updates |
| universe.py | Asset universe management |
| providers/ | 22 data source integrations |
| storage/ | Hive-partitioned storage |

## Entry Point

```python
from ml4t.data import DataManager
```
