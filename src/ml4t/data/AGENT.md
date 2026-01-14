# ml4t.data - Package Index

## Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| data_manager.py | 564 | Main orchestration API |
| update_manager.py | 829 | Incremental update system |
| universe.py | 987 | Asset universe management |
| provider_updater.py | 449 | Provider coordination |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| providers/ | 14k | 22 data source integrations |
| storage/ | 4.4k | Hive-partitioned backends |
| futures/ | 4.6k | Databento futures downloader |
| core/ | 1k | Models, schemas, config |
| utils/ | 1.6k | Rate limiting, gaps, retry |
| assets/ | 1.3k | Asset class definitions |
| anomaly/ | 1k | Data quality detection |
| cot/ | 1k | CFTC COT data |
| validation/ | 1.2k | OHLC validation |
| sessions/ | 462 | Session assignment |
| macro/ | 455 | Macro data downloader |
| calendar/ | 359 | Trading calendars |
| export/ | 231 | CSV/JSON/Excel export |

## Key

`DataManager`, `UpdateManager`, `HiveStorage`, `get_provider()`
