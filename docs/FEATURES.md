# ML4T-Data: Quick Reference Guide

## At a Glance

| Metric | Value |
|--------|-------|
| **Status** | Pre-release (0.1.0) - breaking changes acceptable |
| **Data Providers** | 13 active (12 live + 1 historical) + 1 mock |
| **Test Files** | 73 (50+ unit, 10 integration, 5+ acceptance) |
| **Lines of Code** | ~18,900 in src/ |
| **Python Version** | 3.9+ |
| **Performance** | 7x faster queries (Hive storage) |
| **Maturity** | 70-75% complete |

---

## Core Features Checklist

### Providers (13/13 Complete)
- [x] Yahoo Finance (unlimited, free)
- [x] EODHD (500 calls/day)
- [x] CoinGecko (free, no key)
- [x] Binance (crypto spot/futures)
- [x] DataBento (institutional, trial)
- [x] Wiki Prices (historical 1962-2018)
- [x] Tiingo, Finnhub, Polygon, Twelve Data, CryptoCompare, Oanda
- [x] MockProvider (testing)

### Storage (Complete)
- [x] Hive-partitioned Parquet (7x faster)
- [x] Metadata tracking (last updates, row counts)
- [x] Atomic writes with file locking
- [x] Transaction support with rollback
- [x] Migration system for schema evolution

### Data Quality (Complete)
- [x] OHLCV validator (8 rules)
- [x] Anomaly detection (3 detectors)
- [x] Cross-provider validation
- [x] Deduplication & gap detection
- [x] Validation reports with JSON export

### Updates (Complete)
- [x] Incremental updates (only fetch new)
- [x] Gap detection & filling
- [x] Resume capability on failure
- [x] Multiple strategies (INCREMENTAL, APPEND_ONLY, FULL_REFRESH, BACKFILL)

### CLI (Complete)
- [x] `fetch` - Get data from provider
- [x] `update-all` - Automated incremental updates
- [x] `list` - Show stored data
- [x] `validate` - Data quality checks
- [x] `export` - CSV, JSON, Excel, Parquet

### Utilities (Complete)
- [x] Rate limiting (global + per-provider)
- [x] Circuit breaker (5 failures → 5min timeout)
- [x] Retry with exponential backoff
- [x] File locking (concurrent access)
- [x] Gap optimization

### Configuration (Complete)
- [x] YAML/JSON support
- [x] Environment variable override
- [x] Type validation (Pydantic)
- [x] Provider availability checks

### Asset Management (Complete)
- [x] Asset class registry (8 classes)
- [x] Contract specifications
- [x] Symbol validation
- [x] Schema per asset type

### Futures (Partial ⚠️)
- [x] Parser (Quandl CHRIS format)
- [x] Roll strategies (3 types)
- [x] Adjustment methods (3 types)
- [x] Continuous contract builder
- [ ] Time-based rolling with expiration calendar (TODO)

---

## File Organization

```
src/ml4t/data/
├── providers/           # 13 provider implementations
│   ├── base.py         # Template Method pattern
│   ├── yahoo.py, coingecko.py, binance.py, ... (12 live)
│   └── wiki_prices.py  # Historical fallback
├── storage/            # Data persistence
│   ├── hive.py        # Hive-partitioned (primary)
│   ├── metadata_tracker.py  # Update tracking
│   ├── migration.py    # Schema evolution
│   └── transaction.py  # ACID-like operations
├── validation/         # Data quality
│   ├── ohlcv.py       # OHLC invariant checks
│   ├── anomaly.py     # 3 anomaly detectors
│   └── report.py      # Quality reports
├── update_manager.py   # Incremental updates
├── data_manager.py     # Unified interface
├── cli_interface.py    # CLI commands
├── config/            # Configuration system
├── utils/             # Utilities (rate limit, gaps, locking, etc.)
├── futures/           # Futures handling (partial)
├── assets/            # Asset management
├── core/              # Models, schemas, exceptions
├── export/            # Export formats (CSV, JSON, Excel)
├── sessions/          # Session management
├── calendar/          # Trading calendars
└── security/          # Path validation
```

---

## Quick Start Examples

### Fetch Data
```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
print(data.head())
```

### Store & Retrieve
```python
from ml4t.data.storage.hive import HiveStorage, StorageConfig
from ml4t.data.data_manager import DataManager

storage = HiveStorage(config=StorageConfig(base_path="~/ml4t-data"))
manager = DataManager(storage=storage)

# Store data
manager.store("AAPL", data)

# Retrieve later
retrieved = manager.fetch("AAPL")
```

### Validate Data
```python
from ml4t.data.validation import OHLCVValidator

validator = OHLCVValidator()
report = validator.validate(data)

if report.has_errors:
    print(f"Found {len(report.violations)} issues")
    fixed = validator.fix_auto_fixable(data)
```

### Detect Anomalies
```python
from ml4t.data.anomaly import AnomalyManager

manager = AnomalyManager()
report = manager.analyze(data, symbol="AAPL")

critical = report.get_critical_anomalies()
if critical:
    print(f"Alert: {len(critical)} critical anomalies found")
```

### CLI Usage
```bash
# Fetch single symbol
ml4t-data fetch -s AAPL --start 2024-01-01 --end 2024-12-31

# Fetch from file
ml4t-data fetch -f symbols.txt --start 2024-01-01 --end 2024-12-31

# Automated incremental updates
ml4t-data update-all -c ml4t-data.yaml

# Validate data
ml4t-data validate --symbol AAPL

# Export to CSV
ml4t-data export --symbol AAPL --output data.csv
```

---

## Known Gaps

### Critical Issues
None identified

### Important Issues
1. **Binance Integration Tests** - Missing (requires VPN in US)
2. **Async Storage** - Implemented but under-tested
3. **Type Hints** - Not strict (mypy strict mode disabled)

### Nice-to-Have
1. WebSocket streaming (Phase 2)
2. Parallel symbol updates
3. Time-based futures rolling with expiration calendar

### Won't Fix
1. Alpha Vantage - Free tier too limited (25 calls/day)
2. IEX Cloud - Company closed
3. Python <3.9 - Using modern syntax

---

## Performance Notes

### Best Practices
```python
# Good ✅ - Reuse provider (5ms overhead per call)
provider = YahooFinanceProvider()
for symbol in symbols:
    data = provider.fetch_ohlcv(symbol, start, end)

# Bad ❌ - New instance per call (35ms overhead per call)
for symbol in symbols:
    provider = YahooFinanceProvider()  # Don't do this!
    data = provider.fetch_ohlcv(symbol, start, end)
```

### Benchmark Results
- **Provider Init**: 30-35ms (httpx.Client setup)
- **Per-call Overhead**: ~5ms (negligible)
- **Storage Query**: 7x faster with Hive partitioning
- **Metadata Lookup**: O(1) efficiency

---

## Integration Points

### Uses ml4t-data
- `ml4t-features` - Loads OHLCV for feature engineering
- `ml4t-eval` - Validates backtest data
- `ml4t-backtest` - Gets historical prices
- `ml4t-book` (third edition) - 6 example notebooks

### Coordinate Breaking Changes With
- `/home/stefan/ml4t/software/.claude/`
- Parent-level context for multi-library workflows

---

## Testing

**Run Tests**:
```bash
# Unit tests only (fast, ~1 min)
pytest

# With coverage
pytest --cov=src/ml4t/data --cov-report=html

# Integration tests (requires API keys, slow, ~15 min)
pytest tests/integration/ -v -s

# Specific provider
pytest tests/integration/test_yahoo.py -v

# Specific test
pytest tests/test_config.py::test_storage_config -v
```

**Test Markers**:
- `@pytest.mark.slow` - Slow tests (excluded by default)
- `@pytest.mark.paid_tier` - Requires paid API tier
- `@pytest.mark.integration` - External API calls
- `@pytest.mark.requires_api_key` - API key needed

---

## Configuration Example

```yaml
# ml4t-data.yaml
storage:
  path: ~/ml4t-data
  backend: hive

datasets:
  # Free tier (unlimited)
  sp500_daily:
    provider: yahoo
    symbols_file: config/symbols/sp500.txt
    frequency: daily
    update_strategy: incremental

  # Free tier (500/day)
  nasdaq100_daily:
    provider: eodhd
    symbols_file: config/symbols/nasdaq100.txt
    frequency: daily

  # Crypto
  major_crypto:
    provider: binance
    symbols:
      - BTC
      - ETH
      - SOL
    frequency: daily
```

---

## Common Commands

### Development
```bash
# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy src/

# Pre-commit hooks
pre-commit run --all-files

# Build
uv build

# Install locally
pip install -e .
```

### Data Operations
```bash
# Fetch and save
ml4t-data fetch -s BTC -s ETH --start 2024-01-01 --end 2024-12-31 -o crypto.parquet

# Update daily from config
ml4t-data update-all -c ml4t-data.yaml --dry-run
ml4t-data update-all -c ml4t-data.yaml

# List what's stored
ml4t-data list -c ml4t-data.yaml

# Export to different format
ml4t-data export -s AAPL --output data.csv
ml4t-data export -s AAPL --output data.xlsx
```

---

## Documentation Links

- **Full Inventory**: See comprehensive feature list above
- **README**: `/home/stefan/ml4t/software/data/README.md` (8000+ lines)
- **Project Map**: `/home/stefan/ml4t/software/data/.claude/PROJECT_MAP.md`
- **Book Integration**: 6 example notebooks in `/home/stefan/ml4t/third_edition/`
- **Performance Analysis**: `PERFORMANCE_BENCHMARKS.md` and `PERFORMANCE_ANALYSIS.md`

---

**Generated**: 2025-11-27
**Library Status**: Pre-release (0.1.0)
**Confidence Level**: HIGH (comprehensive codebase analysis)
