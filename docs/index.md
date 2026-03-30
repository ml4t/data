---
hide:
  - navigation
---

# ML4T Data

**High-performance market data acquisition for quantitative finance.**

ML4T Data is the data layer for the ML4T stack: provider adapters, storage,
incremental updates, and validation for research and production workflows.
Start with [Getting Started](getting-started/quickstart.md) if you want a first
result, move to the [User Guide](user-guide/index.md) for recurring workflows,
use the [API Reference](api/index.md) for exact interfaces, and follow the
[Book Guide](book-guide/index.md) if you are working from *Machine Learning for
Trading, Third Edition*.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __5-Minute Setup__

    ---

    Get market data in 3 lines of code. No API keys required for basic usage.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-database:{ .lg .middle } __19 Data Providers__

    ---

    Crypto, equities, forex, futures, and factor data from trusted sources.

    [:octicons-arrow-right-24: Provider Guide](getting-started/provider-selection.md)

-   :material-lightning-bolt:{ .lg .middle } __10-100x Faster__

    ---

    Polars-based processing with async batch loading for maximum throughput.

    [:octicons-arrow-right-24: Performance](user-guide/index.md)

-   :material-shield-check:{ .lg .middle } __Production Ready__

    ---

    Circuit breakers, rate limiting, OHLC validation, incremental updates.

    [:octicons-arrow-right-24: Features](user-guide/data-quality.md)

</div>

## Quick Example

```python
from ml4t.data.providers import YahooFinanceProvider

# Fetch OHLCV data (no API key needed)
provider = YahooFinanceProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")

print(df.head())
# shape: (252, 7)
# ┌─────────────────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
# │ timestamp           ┆ symbol ┆ open   ┆ high   ┆ low    ┆ close  ┆ volume   │
# │ ---                 ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---      │
# │ datetime[μs, UTC]   ┆ str    ┆ f64    ┆ f64    ┆ f64    ┆ f64    ┆ f64      │
# ╞═════════════════════╪════════╪════════╪════════╪════════╪════════╪══════════╡
# │ 2024-01-02 00:00:00 ┆ AAPL   ┆ 187.15 ┆ 188.44 ┆ 183.89 ┆ 185.64 ┆ 82488700 │
# └─────────────────────┴────────┴────────┴────────┴────────┴────────┴──────────┘
```

## Async Batch Loading (3-10x Faster)

```python
import asyncio
from ml4t.data.managers.async_batch import async_batch_load
from ml4t.data.providers import YahooFinanceProvider

async def fetch_portfolio():
    async with YahooFinanceProvider() as provider:
        return await async_batch_load(
            provider,
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            start="2024-01-01",
            end="2024-12-31",
            max_concurrent=10,
        )

df = asyncio.run(fetch_portfolio())
print(f"Fetched {len(df)} rows for {df['symbol'].n_unique()} symbols")
```

## Installation

=== "pip"

    ```bash
    pip install ml4t-data
    ```

=== "uv"

    ```bash
    uv add ml4t-data
    ```

=== "With providers"

    ```bash
    pip install "ml4t-data[yahoo,databento]"
    ```

## Provider Comparison

| Provider | Asset Class | Free Tier | Async | Best For |
|----------|-------------|-----------|-------|----------|
| **Yahoo** | Stocks, ETFs, Crypto | Unlimited | Thread | Learning, backtesting |
| **CoinGecko** | Crypto | 10K+ coins | Native | Crypto historical |
| **EODHD** | Global Stocks | 500/day | Native | Global coverage |
| **DataBento** | Futures, Options | $10 credits | Thread | Institutional data |
| **Fama-French** | Factors | Unlimited | Thread | Academic research |

[:octicons-arrow-right-24: Full provider comparison](getting-started/provider-selection.md)

## For ML4T Book Readers

This library is the reference implementation for **Machine Learning for Trading (Third Edition)**.
The book uses it both for chapter-scale demonstrations and for the reusable
dataset download scripts under the `ml4t/third-edition` repository.

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } __Chapter-Feature Mapping__

    ---

    - **Chapter 2**: `DataManager`, `Universe`, `HiveStorage`, validation, updates
    - **Chapter 4**: `FREDProvider`, `CoinGeckoProvider`, `KalshiProvider`, `PolymarketProvider`
    - **Chapters 17-19**: `FamaFrenchProvider`, `AQRFactorProvider`, factor and risk workflows

    [:octicons-arrow-right-24: Full book guide](book-guide/index.md)

-   :material-rocket-launch:{ .lg .middle } __Production Ready__

    ---

    Graduate from notebook examples to reusable pipelines with
    [`code/data/download_all.py`](https://github.com/ml4t/third-edition/blob/main/code/data/download_all.py),
    incremental updates, and CLI automation.

    [:octicons-arrow-right-24: Incremental updates](user-guide/incremental-updates.md)

</div>

Start from the exact book entry points if you want to match the code in print:

- [Chapter 2 complete pipeline](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/17_complete_pipeline.py)
- [Chapter 2 data management](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/18_data_management.py)
- [Chapter 2 incremental updates](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/19_incremental_updates.py)
- [Canonical download orchestrator](https://github.com/ml4t/third-edition/blob/main/code/data/download_all.py)

## Next Steps

<div class="grid cards" markdown>

-   [:octicons-download-24: __Installation__](getting-started/installation.md)

    Detailed installation guide with all optional dependencies.

-   [:octicons-book-24: __User Guide__](user-guide/index.md)

    Complete documentation for all features.

-   [:octicons-code-24: __API Reference__](api/index.md)

    Auto-generated documentation from source code.

-   [:octicons-book-24: __Book Guide__](book-guide/index.md)

    Chapter, notebook, and production API mapping for ML4T.

</div>
