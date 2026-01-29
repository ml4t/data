# ml4t-data Library

Market data acquisition and storage library for ML4T 3rd Edition.

## Structure

| Directory          | Purpose                          |
| ------------------ | -------------------------------- |
| `src/ml4t/data/`   | Package root                     |
| `tests/`           | Test suite                       |
| `examples/`        | Usage examples                   |
| `docs/`            | MkDocs documentation             |

## Key Modules

| Module        | Purpose                                        |
| ------------- | ---------------------------------------------- |
| `futures/`    | Databento futures downloaders and roll logic   |
| `providers/`  | 22 data source integrations                    |
| `storage/`    | Hive-partitioned Parquet storage               |
| `managers/`   | Data orchestration and updates                 |
| `assets/`     | Asset universe definitions                     |
| `cot/`        | Commitment of Traders data                     |

## Futures Module Classes

| Class                  | Purpose                                            |
| ---------------------- | -------------------------------------------------- |
| `ContinuousDownloader` | Pre-rolled continuous contracts (.v.0, .v.1, .v.2) |
| `IndividualDownloader` | Specific contract symbols (ESH24, CLF24, etc.)     |
| `FuturesDownloader`    | All contracts via parent symbology ({PRODUCT}.FUT) |
| `FuturesDataManager`   | High-level interface for book readers              |

## Entry Points

```python
# Continuous contracts (pre-rolled)
from ml4t.data.futures import ContinuousDownloader, ContinuousDownloadConfig

# Individual contracts (for roll demonstration)
from ml4t.data.futures import IndividualDownloader, IndividualDownloadConfig

# All contracts (parent symbology)
from ml4t.data.futures import FuturesDownloader, FuturesDownloadConfig

# High-level interface
from ml4t.data.futures import FuturesDataManager
```

## Commands

```bash
# Development
uv sync
uv run pytest tests/ -q
uv run ty check
pre-commit run --all-files

# Build & publish
uv build
uv publish
```

## Version

Check `pyproject.toml` for current version.
PyPI: https://pypi.org/project/ml4t-data/
