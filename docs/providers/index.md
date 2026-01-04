# Providers

ML4T Data supports 15 data providers with a unified API.

## Provider Comparison

| Provider | Asset Class | Free Tier | Async | API Key |
|----------|-------------|-----------|-------|---------|
| [Yahoo Finance](yahoo.md) | Stocks, ETFs, Crypto | Unlimited | Thread | No |
| [CoinGecko](coingecko.md) | Crypto | 10K+ coins | Native | No |
| [EODHD](eodhd.md) | Global Stocks | 500/day | Native | Yes |
| [Binance](binance.md) | Crypto | Unlimited | Native | No |
| [TwelveData](twelve_data.md) | Multi-asset | 800/day | Native | Yes |
| [CryptoCompare](cryptocompare.md) | Crypto | Good | Native | Optional |
| [DataBento](databento.md) | Futures, Options | $10 credits | Thread | Yes |
| [Tiingo](tiingo.md) | US Stocks | 1000/day | Thread | Yes |
| [Finnhub](finnhub.md) | Global Stocks | 30/day OHLCV | Thread | Yes |
| [Polygon](polygon.md) | Multi-asset | Paid only | Thread | Yes |
| [Oanda](oanda.md) | Forex | Demo only | Thread | Yes |
| [Fama-French](fama-french.md) | Factors | Unlimited | Thread | No |
| [AQR](aqr.md) | Factors | Unlimited | Thread | No |
| [Wiki Prices](wiki-prices.md) | Historical | Static | Thread | No |

## Async Support

All providers support async via `async_batch_load()`:

- **Native async**: Uses `httpx.AsyncClient` for true non-blocking I/O
- **Thread-wrapped**: Uses `asyncio.to_thread()` for sync SDKs

```python
from ml4t.data.managers.async_batch import async_batch_load

async with YahooFinanceProvider() as provider:
    df = await async_batch_load(
        provider,
        symbols=["AAPL", "MSFT", "GOOGL"],
        start="2024-01-01",
        end="2024-12-31",
    )
```

## Unified API

All providers implement the same interface:

```python
provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

Returns a Polars DataFrame with columns:
`timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`
