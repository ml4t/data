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
| `etfs/`       | ETFDataManager for Yahoo Finance ETF data      |
| `crypto/`     | CryptoDataManager for Binance premium index    |
| `providers/`  | 22 data source integrations                    |
| `storage/`    | Hive-partitioned Parquet + ProfileMixin        |
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

## Book Data Managers (with Profiling)

| Manager | Asset Class | Source | Profiling |
|---------|-------------|--------|-----------|
| `ETFDataManager` | 50 ETFs | Yahoo Finance | `generate_profile()` |
| `CryptoDataManager` | Crypto | Binance Public | `generate_profile()` |
| `FuturesDataManager` | CME Futures | Databento | `generate_profile(product)` |

All managers support on-demand data profiling via `ProfileMixin`.

## Entry Points

```python
# ETF data (Yahoo Finance)
from ml4t.data.etfs import ETFDataManager

# Crypto premium index (Binance Public - no API key needed)
from ml4t.data.crypto import CryptoDataManager

# Futures data (Databento)
from ml4t.data.futures import FuturesDataManager

# Data profiling
from ml4t.data.storage import ProfileMixin, generate_profile
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
