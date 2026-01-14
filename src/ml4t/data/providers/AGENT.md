# providers/ - 22 Data Sources

## Base Classes

| File | Lines | Purpose |
|------|-------|---------|
| base.py | 290 | BaseDataProvider abstract |
| async_base.py | 256 | AsyncBaseProvider |
| protocols.py | 249 | Provider protocols |

## Market Data Providers

| File | Lines | Purpose |
|------|-------|---------|
| yahoo.py | 603 | Yahoo Finance (free) |
| binance.py | 410 | Binance authenticated |
| binance_public.py | 1430 | Binance public API |
| eodhd.py | 464 | EOD Historical Data |
| databento.py | 430 | Databento (futures) |
| tiingo.py | 259 | Tiingo API |
| twelve_data.py | 288 | Twelve Data API |
| polygon.py | 242 | Polygon.io |
| oanda.py | 373 | OANDA forex |
| okx.py | 577 | OKX exchange |

## Crypto Providers

| File | Lines | Purpose |
|------|-------|---------|
| coingecko.py | 497 | CoinGecko (free) |
| cryptocompare.py | 535 | CryptoCompare |

## Alternative Data

| File | Lines | Purpose |
|------|-------|---------|
| fred.py | 598 | Federal Reserve |
| fama_french.py | 987 | FF factor data |
| aqr.py | 970 | AQR factor data |
| wiki_prices.py | 747 | Quandl Wiki |
| kalshi.py | 828 | Kalshi prediction |
| polymarket.py | 913 | Polymarket |

## Synthetic

| File | Lines | Purpose |
|------|-------|---------|
| synthetic.py | 528 | Random price gen |
| learned_synthetic.py | 544 | ML-based synthetic |
| mock.py | 355 | Testing mock |

## Key

`get_provider()`, `BaseDataProvider`, `fetch_ohlcv()`
