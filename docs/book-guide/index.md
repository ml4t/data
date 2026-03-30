# ML4T Book Guide

This guide maps `ml4t-data` to the current
[*Machine Learning for Trading, Third Edition*](https://github.com/ml4t/third-edition)
codebase so you can move cleanly between the book's pedagogical examples and the
library's reusable APIs.

## How to Read This Guide

The book uses `ml4t-data` in three distinct ways:

1. provider-level exploration inside chapter notebooks and companion scripts
2. library-level workflows built around `DataManager`, storage, validation, and
   updates
3. dataset download scripts under `code/data/` that package canonical book data
   into repeatable pipelines

If you are starting from the book, the fastest path is:

1. follow the chapter script that introduces the concept
2. identify the corresponding `ml4t-data` class or provider
3. reuse that API in your own config-driven workflow

## Book Entry Points

| Goal | Library surface | Book path |
|---|---|---|
| Download the canonical datasets used across the book | `ETFDataManager`, `CryptoDataManager`, `MacroDataManager`, `FuturesDataManager` | [code/data/download_all.py](https://github.com/ml4t/third-edition/blob/main/code/data/download_all.py) |
| Learn provider-first data exploration | provider classes such as `YahooFinanceProvider`, `BinanceBulkProvider`, `FREDProvider` | [code/data/etfs/notebook.py](https://github.com/ml4t/third-edition/blob/main/code/data/etfs/notebook.py), [code/data/crypto/notebook.py](https://github.com/ml4t/third-edition/blob/main/code/data/crypto/notebook.py), [code/data/macro/notebook.py](https://github.com/ml4t/third-edition/blob/main/code/data/macro/notebook.py) |
| Build a reusable storage-backed data workflow | `DataManager`, `HiveStorage`, `Universe` | [code/02_financial_data_universe/18_data_management.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/18_data_management.py) |
| Understand updates, validation, and health checks | `DataManager.update()`, `GapDetector`, `OHLCVValidator` | [code/02_financial_data_universe/19_incremental_updates.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/19_incremental_updates.py) |
| See the end-to-end Chapter 2 workflow | providers plus storage and validation APIs | [code/02_financial_data_universe/17_complete_pipeline.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/17_complete_pipeline.py) |

## Chapter-to-Feature Mapping

### Chapter 2: Financial Data Universe

This is the deepest `ml4t-data` integration point in the book.

| Book path | What it teaches | `ml4t-data` surface |
|---|---|---|
| [02_corporate_actions.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/02_corporate_actions.py) | price adjustment workflows | `apply_corporate_actions`, `YahooFinanceProvider` |
| [16_provider_comparison.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/16_provider_comparison.py) | source tradeoffs across free and paid providers | `YahooFinanceProvider`, `WikiPricesProvider`, `EODHDProvider`, `FREDProvider` |
| [17_complete_pipeline.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/17_complete_pipeline.py) | complete data pipeline | `DataManager`, `HiveStorage`, `Universe`, `GapDetector`, `OHLCVValidator` |
| [18_data_management.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/18_data_management.py) | storage-backed data management | `DataManager`, `HiveStorage`, `Universe` |
| [19_incremental_updates.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/19_incremental_updates.py) | refreshing existing datasets safely | `DataManager.update()`, `GapDetector`, `OHLCVValidator` |
| [13_data_quality_framework.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/13_data_quality_framework.py) | structural validation and anomaly detection | `OHLCVValidator`, `AnomalyManager` |

### Chapter 4: Fundamental and Alternative Data

This chapter leans on provider adapters more than on `DataManager`.

| Book path | Provider(s) | Notes |
|---|---|---|
| [08_fred_macro_eda.py](https://github.com/ml4t/third-edition/blob/main/code/04_fundamental_alternative_data/08_fred_macro_eda.py) | `FREDProvider` | Macro and rate series |
| [09_macro_data_alignment.py](https://github.com/ml4t/third-edition/blob/main/code/04_fundamental_alternative_data/09_macro_data_alignment.py) | `FREDProvider` | Vintage-aware macro discussion |
| [11_onchain_fundamentals.py](https://github.com/ml4t/third-edition/blob/main/code/04_fundamental_alternative_data/11_onchain_fundamentals.py) | `CoinGeckoProvider` | Crypto prices paired with on-chain metrics |
| [13_kalshi_prediction_markets.py](https://github.com/ml4t/third-edition/blob/main/code/04_fundamental_alternative_data/13_kalshi_prediction_markets.py) | `KalshiProvider` | Event-contract OHLCV |
| [14_polymarket_prediction_markets.py](https://github.com/ml4t/third-edition/blob/main/code/04_fundamental_alternative_data/14_polymarket_prediction_markets.py) | `PolymarketProvider` | Prediction market history and snapshots |

### Chapters 17 and 19: Factors, Allocation, and Risk

These chapters use `ml4t-data` for factor datasets and risk-free inputs.

| Book path | Provider(s) | Notes |
|---|---|---|
| [17_portfolio_construction/02_mean_variance_optimization.py](https://github.com/ml4t/third-edition/blob/main/code/17_portfolio_construction/02_mean_variance_optimization.py) | `FREDProvider` | Risk-free rate loading |
| [17_portfolio_construction/05_factor_allocation_evidence.py](https://github.com/ml4t/third-edition/blob/main/code/17_portfolio_construction/05_factor_allocation_evidence.py) | `FamaFrenchProvider`, `AQRFactorProvider` | Long-history factor evidence |
| [17_portfolio_construction/08_library_comparison.py](https://github.com/ml4t/third-edition/blob/main/code/17_portfolio_construction/08_library_comparison.py) | `FREDProvider` | Comparison workflows with real rates |
| [19_risk_management/04_factor_exposure.py](https://github.com/ml4t/third-edition/blob/main/code/19_risk_management/04_factor_exposure.py) | `FamaFrenchProvider` | Factor loading and exposure analysis |

## Canonical Download Scripts

The `code/data/` directory is where book examples become reusable download
pipelines. These are the most direct examples of how the library is intended to
be used outside a notebook.

| Script | `ml4t-data` surface | Dataset role |
|---|---|---|
| [code/data/etfs/download.py](https://github.com/ml4t/third-edition/blob/main/code/data/etfs/download.py) | `ETFDataManager` | Core ETF and benchmark history |
| [code/data/crypto/download.py](https://github.com/ml4t/third-edition/blob/main/code/data/crypto/download.py) | `BinanceBulkProvider`, `CryptoDataManager` | Spot and futures crypto data |
| [code/data/macro/download.py](https://github.com/ml4t/third-edition/blob/main/code/data/macro/download.py) | `FREDProvider`, `MacroDataManager` | Treasury and macro series |
| [code/data/factors/ff_download.py](https://github.com/ml4t/third-edition/blob/main/code/data/factors/ff_download.py) | `FamaFrenchProvider` | Factor benchmarks |
| [code/data/factors/aqr_download.py](https://github.com/ml4t/third-edition/blob/main/code/data/factors/aqr_download.py) | `AQRFactorProvider` | Extended factor datasets |
| [code/data/prediction_markets/download.py](https://github.com/ml4t/third-edition/blob/main/code/data/prediction_markets/download.py) | `KalshiProvider`, `PolymarketProvider` | Alternative market data |

## From Notebook Code to Reusable Workflows

The book often introduces a concept manually before the library wraps it in a
more reusable API. A few common transitions:

| In the book | In the library | Why it matters |
|---|---|---|
| provider calls inside one notebook | `DataManager.fetch()` | consistent routing and output handling |
| ad hoc lists of symbols | `Universe` | reusable universes and custom symbol sets |
| one-off Parquet writes | `HiveStorage` | partition pruning, metadata, and update support |
| manual gap checks | `GapDetector` and `DataManager.update()` | safer recurring refreshes |
| exploratory validation logic | `OHLCVValidator` and `AnomalyManager` | repeatable data-quality gates |
| chapter download helpers | `update-all` plus manager classes | automation outside the notebook |

## Suggested Reader Journey

If you want the shortest path from the book to production-style code:

1. Read [17_complete_pipeline.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/17_complete_pipeline.py).
2. Move to [18_data_management.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/18_data_management.py) for `DataManager`, `Universe`, and storage.
3. Read [19_incremental_updates.py](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/19_incremental_updates.py) before setting up recurring refreshes.
4. Use [code/data/download_all.py](https://github.com/ml4t/third-edition/blob/main/code/data/download_all.py) as the reference shape for your own dataset automation.

## Related Docs

- [Getting Started](../getting-started/quickstart.md)
- [Incremental Updates](../user-guide/incremental-updates.md)
- [Data Quality](../user-guide/data-quality.md)
- [API Reference](../api/index.md)
