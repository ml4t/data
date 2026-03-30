# Quickstart

Get from install to first dataset quickly, then move to storage and automated
updates when you need a reusable workflow.

## Fetch a Single Dataset

```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily",
)

print(df.select("timestamp", "close").head())
```

If you want provider routing, storage, and updates behind one interface, use
`DataManager` instead:

```python
from ml4t.data import DataManager

manager = DataManager()
df = manager.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")
```

## Fetch Multiple Symbols Faster

```python
import asyncio
from ml4t.data.managers.async_batch import async_batch_load
from ml4t.data.providers import YahooFinanceProvider


async def main():
    async with YahooFinanceProvider() as provider:
        return await async_batch_load(
            provider,
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            start="2024-01-01",
            end="2024-12-31",
        )


df = asyncio.run(main())
print(f"Fetched {len(df)} rows for {df['symbol'].n_unique()} symbols")
```

## CLI Usage

```bash
# Fetch data to stdout or an output file
ml4t-data fetch -s AAPL -s MSFT -s GOOGL \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --provider yahoo \
    --output prices.parquet

# Validate stored data
ml4t-data validate -s equities/daily/AAPL --storage-path ./data

# Update all datasets from a YAML config
ml4t-data update-all -c ml4t-data.yaml
```

## Configuration for `update-all`

Create `ml4t-data.yaml` for recurring downloads:

```yaml
storage:
  path: ~/ml4t-data

datasets:
  etf_core:
    provider: yahoo
    symbols: [SPY, QQQ, IWM, TLT, GLD]
    frequency: daily

  sp500:
    provider: yahoo
    symbols_file: sp500.txt
    frequency: daily
```

Then run:

```bash
ml4t-data update-all -c ml4t-data.yaml
```

## For Book Readers

If you are working through *Machine Learning for Trading, Third Edition*, start
with these companion scripts:

- [ETF download workflow](https://github.com/ml4t/third-edition/blob/main/code/data/etfs/download.py)
- [Chapter 2 data management notebook script](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/18_data_management.py)
- [Chapter 2 incremental updates notebook script](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/19_incremental_updates.py)

## Next Steps

- [Provider Selection Guide](provider-selection.md) for source-specific tradeoffs
- [User Guide](../user-guide/index.md) for storage, validation, and automation
- [Book Guide](../book-guide/index.md) for chapter-to-API mapping
- [API Reference](../api/index.md) for the current public surface
