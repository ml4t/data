# Quickstart

Get market data in under 5 minutes.

## Basic Usage

```python
from ml4t.data.providers import YahooFinanceProvider

# Create a provider
provider = YahooFinanceProvider()

# Fetch OHLCV data
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)

print(df.head())
```

## Multiple Symbols (Async)

For faster multi-symbol fetches, use async batch loading:

```python
import asyncio
from ml4t.data.managers.async_batch import async_batch_load
from ml4t.data.providers import YahooFinanceProvider

async def main():
    async with YahooFinanceProvider() as provider:
        df = await async_batch_load(
            provider,
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            start="2024-01-01",
            end="2024-12-31",
        )
    return df

df = asyncio.run(main())
print(f"Fetched {len(df)} rows for {df['symbol'].n_unique()} symbols")
```

## CLI Usage

```bash
# Fetch data via command line
ml4t-data fetch AAPL MSFT GOOGL \
    --start 2024-01-01 \
    --provider yahoo \
    --output ~/data

# Update all datasets from config
ml4t-data update-all -c ml4t-data.yaml
```

## Configuration File

Create `ml4t-data.yaml` for automated updates:

```yaml
storage:
  path: ~/ml4t-data

datasets:
  sp500:
    provider: yahoo
    symbols_file: sp500.txt
    frequency: daily
    start_date: 2020-01-01
```

Then run:

```bash
ml4t-data update-all -c ml4t-data.yaml
```

## Next Steps

- [Provider Selection Guide](provider-selection.md) - Choose the right data source
- [User Guide](../user-guide/index.md) - Complete documentation
- [API Reference](../api/index.md) - Detailed API docs
