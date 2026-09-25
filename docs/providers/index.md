# Providers

ML4T Data supports 20+ live and specialized data providers, plus synthetic and testing providers.

For the wider vendor landscape, including sources the library does not wrap, start with
[Market Data Sources](market_data.md) or go directly to the [equity](equities.md),
[ETF](etfs.md), [futures](futures.md), [options](options.md), [foreign exchange](fx.md), or
[cryptocurrency](crypto.md) reference. Separate references cover [fixed income](fixed_income.md),
[macroeconomic data](macro.md), [fundamentals](fundamentals.md),
[alternative data](alternative_data.md), [research factors](factors.md), and
[prediction markets](prediction_markets.md).

## Provider Comparison

| Provider | Asset Class | Access | Async | API Key |
|----------|-------------|--------|-------|---------|
| [Yahoo Finance](yahoo.md) | Stocks, ETFs, Crypto | Unlimited | Thread | No |
| [CoinGecko](coingecko.md) | Crypto | 10K+ coins | Native | No |
| [FRED](fred.md) | Economic Data | 120/min | Thread | Yes |
| FXMacroData | FX Macro, Forex Context | Public USD/free endpoints | Thread | Optional |
| [Fama-French](fama_french.md) | Factors | Unlimited | Thread | No |
| [AQR](aqr.md) | Factors | Unlimited | Thread | No |
| [Wiki Prices](wiki_prices.md) | Historical | Static | Thread | No |
| [Kalshi](kalshi.md) | Prediction Markets | Public data | Thread | No |
| [Polymarket](polymarket.md) | Prediction Markets | Public data | Thread | No |
| [Binance Public](binance_public.md) | Crypto | Bulk downloads | Thread | No |
| [NASDAQ ITCH](nasdaq_itch.md) | Tick Data | Sample data | Thread | No |
| [EODHD](eodhd.md) | Global Stocks | 20 calls/day | Native | Yes |
| [Tiingo](tiingo.md) | US Stocks | 1000/day | Thread | Yes |
| [TwelveData](twelve_data.md) | Multi-asset | 800/day | Native | Yes |
| [DataBento](databento.md) | Futures, Options | Free metadata; metered history | Thread | Yes |
| [Massive](massive.md) | Multi-asset | Account-dependent | Thread | Yes |
| [Finnhub](finnhub.md) | US quotes; premium OHLCV | 60 requests/minute | Thread | Yes |
| [Binance](binance.md) | Crypto | Unlimited | Native | No |
| [OKX](okx.md) | Crypto Perpetuals | No geo-limits | Native | No |
| [CryptoCompare](cryptocompare.md) | Crypto | Unverified for 0.1.0 | Native | Required |
| [Oanda](oanda.md) | Forex | Demo only | Thread | Yes |

## Async Support

OHLCV providers that implement `fetch_ohlcv_async()` can use `async_batch_load()`:

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

## Provider APIs

OHLCV providers use the following interface:

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

Other providers expose capability-specific methods. FRED uses `fetch_ohlcv()` for one series,
`fetch_multiple()` for batches, and `fetch_series_metadata()` for descriptors. Factor providers
use `fetch()`. Specialized providers document their contracts on their provider pages. Check the
provider's capabilities before passing it to `DataManager` or `async_batch_load()`.

## Selection by Requirement

| Requirement | Providers to Evaluate |
|-------------|-----------------------|
| No credential | Yahoo Finance, CoinGecko, Fama-French, AQR, Kalshi, Polymarket, Binance Public, NASDAQ ITCH |
| US equities | Yahoo Finance, Alpaca, Tiingo, Massive |
| Global equities | EODHD, Finnhub, Twelve Data |
| Futures and options | Databento, Massive |
| Cryptocurrency | Binance, Binance Public, OKX, CoinGecko, CryptoCompare |
| Foreign exchange | Oanda, Twelve Data, FXMacroData |
| Economic series | FRED, FXMacroData; external official sources are compared in [Macroeconomic Data Sources](macro.md) |
| Academic factors | Fama-French, AQR; external libraries are compared in [Research Factor Data Sources](factors.md) |
| Prediction markets | Kalshi, Polymarket; external markets are compared in [Prediction-Market Data Sources](prediction_markets.md) |
| Equity fundamentals | Yahoo Finance, EODHD, Finnhub, Massive; other vendors and the SEC routes are compared in [Fundamental Data Sources](fundamentals.md) |

Provider access, coverage, retention, and redistribution terms can differ by account tier. Confirm
the provider page and the provider's current terms before selecting it for a production dataset.
CryptoCompare is included for evaluation but is not release-qualified for 0.1.0 because account
registration was unavailable during release validation and no live contract evidence was obtained.

## Authentication

The library reads credentials from constructor arguments, typed configuration, or process
environment variables. It does not automatically load a project `.env` file. Applications that
use `.env` files must load them before constructing a provider.

| Provider | Environment Variables |
|----------|-----------------------|
| Alpaca | `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`, or `ALPACA_API_KEY` and `ALPACA_API_SECRET` |
| Tiingo | `TIINGO_API_KEY` |
| Finnhub | `FINNHUB_API_KEY` |
| EODHD | `EODHD_API_KEY` |
| FRED | `FRED_API_KEY` |
| FXMacroData | `FXMACRODATA_API_KEY` or `FXMD_API_KEY` |
| CoinGecko | `COINGECKO_API_KEY` (optional) |
| CryptoCompare | `CRYPTOCOMPARE_API_KEY` |
| Oanda | `OANDA_API_KEY` |
| Massive | `MASSIVE_API_KEY` or legacy `POLYGON_API_KEY` |
| Twelve Data | `TWELVE_DATA_API_KEY` |
| Databento | `DATABENTO_API_KEY` |

## Error Handling

Provider failures use the package exception hierarchy:

```python
from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    RateLimitError,
    SymbolNotFoundError,
)

try:
    frame = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-06-01", "daily")
except AuthenticationError:
    # Credential is missing, invalid, or unauthorized for the endpoint.
    ...
except RateLimitError:
    # Retry policy was exhausted or the provider requested a later retry.
    ...
except (SymbolNotFoundError, DataNotAvailableError):
    # The symbol or requested range is unavailable.
    ...
except DataValidationError:
    # The response did not satisfy the provider data contract.
    ...
```

Reuse provider instances across requests so their HTTP connection pools and rate-limit state remain
effective. Close them explicitly or use their context-manager interface when the provider supports
it.
