# ML4T Book Integration

This page maps ml4t-data features to their usage in
**Machine Learning for Trading** (Third Edition).

## How the Book Uses ml4t-data

The book uses ml4t-data at two levels:

1. **Download scripts** (`data/download_all.py`): Config-driven managers
   (`ETFDataManager`, `CryptoDataManager`, `MacroDataManager`, `FuturesDataManager`)
   that fetch and store the book's canonical datasets from YAML configs.

2. **Notebooks**: Provider-level access for analysis and demonstration,
   plus high-level APIs (`DataManager`, `Universe`, `HiveStorage`) for
   production data management patterns.

## Chapter-to-Provider Mapping

| Chapter | Topic | Providers Used | Notebooks |
|---------|-------|----------------|-----------|
| **2** | Financial Data Universe | Yahoo, Wiki Prices, EODHD, FRED, Binance Public, CoinGecko | 02-18 (11 notebooks) |
| **4** | Fundamental & Alternative | FRED, CoinGecko, Kalshi, Polymarket, COT | 08, 10, 11, 13, 14 |
| **5** | Synthetic Data | SyntheticProvider | 00 |
| **16** | Strategy Simulation | Binance Public | 02, 03, 05 |
| **17** | Portfolio Construction | FRED, AQR, Fama-French, Binance Public | 02, 07, 09 |
| **19** | Risk Management | Fama-French | 04 |

## Chapter 2: Deep Integration

Chapter 2 is the primary data chapter and uses almost every ml4t-data feature:

### Data Management (Notebooks 17-18)

| Feature | Class | Notebook | Section |
|---------|-------|----------|---------|
| Unified fetch | `DataManager.fetch()` | 17 | 1. DataManager |
| Batch loading | `DataManager.batch_load()` | 17 | 1. DataManager |
| Symbol universes | `Universe.SP500`, `.get()`, `.add_custom()` | 17 | 2. Universe |
| Hive storage | `HiveStorage`, `StorageConfig` | 17 | 3. HiveStorage |
| Incremental updates | `DataManager.update()` | 17, 18 | 4. Updates |
| Gap detection | `GapDetector.detect_gaps()` | 17, 18 | Gap Detection |
| Update strategies | `UpdateStrategy` enum | 18 | 2. Strategies |
| Health monitoring | `OHLCVValidator` + `GapDetector` | 18 | 4. Health Dashboard |
| CLI commands | `ml4t-data fetch`, `update`, `validate` | 17 | 5. CLI |
| Config-driven downloads | `download_all.py --update` | 18 | 5. Update Workflow |

### Data Quality (Notebook 12)

| Feature | Class | Purpose |
|---------|-------|---------|
| OHLCV validation | `OHLCVValidator` | 8 configurable checks |
| Return outliers | `ReturnOutlierDetector` | Extreme return detection |
| Volume spikes | `VolumeSpikeDetector` | Volume anomaly detection |
| Price staleness | `PriceStalenessDetector` | Stale price detection |
| Anomaly management | `AnomalyManager`, `AnomalyConfig` | Combined anomaly pipeline |

### Provider Comparison (Notebook 15)

| Provider | Asset Class | API Key Required |
|----------|-------------|------------------|
| Yahoo Finance | Equities, ETFs | No |
| Wiki Prices | US Equities (1962-2018) | NASDAQ Data Link (free) |
| EODHD | Global equities | Yes (paid) |
| FRED | Macro indicators | Yes (free) |

### Complete Pipeline (Notebook 16)

Notebook 16 demonstrates end-to-end pipelines for both case studies,
then shows how `DataManager` + `Universe` simplify the manual approach.

## Provider Usage Across the Book

### Free Providers (No API Key)

| Provider | Chapters | Primary Use |
|----------|----------|-------------|
| **Yahoo Finance** | 2 | ETF universe, equity prices |
| **Binance Public** | 2, 16, 17 | Crypto OHLCV, premium index |
| **Fama-French** | 17, 19 | Factor returns |
| **AQR** | 17 | Quality-minus-junk, BAB |
| **SyntheticProvider** | 5 | Simulated market data |

### Free Providers (API Key Required)

| Provider | Chapters | Key Source |
|----------|----------|------------|
| **FRED** | 2, 4, 17 | Treasury yields, macro indicators |
| **CoinGecko** | 4 | On-chain fundamentals |
| **Wiki Prices** | 2 | Historical US equities (frozen 2018) |

### Paid Providers

| Provider | Chapters | Use Case |
|----------|----------|----------|
| **EODHD** | 2 | Provider comparison demo |
| **Kalshi** | 4 | Prediction markets |
| **Polymarket** | 4 | Prediction markets |
| **DataBento** | (data scripts) | CME futures |

## Feature Coverage

### Used in Book

| Feature | Status |
|---------|--------|
| DataManager (fetch, batch, load, update) | Notebook 17 |
| Universe (SP500, NASDAQ100, custom) | Notebook 17 |
| HiveStorage + StorageConfig | Notebooks 17, 18 |
| GapDetector | Notebooks 17, 18 |
| IncrementalUpdater + UpdateStrategy | Notebook 18 |
| OHLCVValidator | Notebooks 12, 18 |
| Anomaly detection (3 detectors) | Notebook 12 |
| Corporate actions | Notebook 02 |
| Config-driven managers (ETF, Crypto, Macro, Futures) | `data/download_all.py` |
| CLI (`ml4t-data fetch/update/validate`) | Notebook 17 (docs) |
| 14 of 20 providers | Various chapters |

### Library-Only (Not in Book)

| Feature | Documentation |
|---------|---------------|
| Export module (CSV, JSON, Excel) | [CLI Reference](../user-guide/cli-reference.md) |
| Session management | [Storage Guide](../user-guide/storage.md) |
| Trading calendar | [Incremental Updates](../user-guide/incremental-updates.md) |
| Async providers | [Architecture](../contributing/architecture.md) |
| Transaction support | [Storage Guide](../user-guide/storage.md) |
| Remaining 6 providers | [Provider docs](../providers/index.md) |

## Download Scripts

The book's `data/` directory contains download scripts that use ml4t-data:

```bash
# Initial download (run once)
python data/download_all.py

# Update all datasets to present
python data/download_all.py --update
```

| Script | ml4t-data Class | Config |
|--------|-----------------|--------|
| `data/etfs/download.py` | `YahooFinanceProvider` | `etfs/config.yaml` |
| `data/crypto/download.py` | `BinancePublicProvider` | `crypto/config.yaml` |
| `data/macro/download.py` | `FREDProvider` | `macro/config.yaml` |
| `data/futures/download.py` | `databento` (direct) | `futures/config.yaml` |
| `data/fx/download.py` | `OandaProvider` | `fx/config.yaml` |
| `data/factors/ff_download.py` | `FamaFrenchProvider` | Built-in |
| `data/factors/aqr_download.py` | `AQRFactorProvider` | Built-in |

The orchestrator (`download_all.py`) wraps these with higher-level managers:

```python
from ml4t.data.etfs import ETFDataManager
from ml4t.data.crypto import CryptoDataManager
from ml4t.data.macro import MacroDataManager
from ml4t.data.futures import FuturesDataManager
```

## Cross-References

- **User Guide**: [Incremental Updates](../user-guide/incremental-updates.md),
  [Storage](../user-guide/storage.md), [Data Quality](../user-guide/data-quality.md)
- **Tutorials**: [Understanding OHLCV](../tutorials/01_understanding_ohlcv.md),
  [Multi-Provider](../tutorials/05_multi_provider.md)
- **API Reference**: [DataManager](../api/index.md), [HiveStorage](../api/index.md)
