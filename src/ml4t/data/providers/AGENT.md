# providers/ - Provider Adapters

## Base Classes

| File | Lines | Purpose |
|------|-------|---------|
| base.py | 290 | `BaseProvider` abstract base |
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

`BaseProvider`, `Provider`, `fetch_ohlcv()`, `fetch_series()`, `fetch()`

## Prediction Market API References

- Polymarket docs: `https://docs.polymarket.com/market-data/overview`
- Polymarket live slug lookup uses `GET /markets/slug/{slug}` on Gamma. The older `GET /markets?slug=...` query has returned empty results for current markets during March 2026 verification.
- Polymarket current-market discovery should prefer `active=true` with `closed=false`. `active=true` alone still returns many closed 2020-2021 markets.
- Kalshi docs: `https://docs.kalshi.com/welcome`
- For Kalshi OHLCV, verify the live candlestick schema before changing transforms. Recent live responses have used nested structs like `price`, `yes_bid`, `yes_ask`, plus `volume_fp`.
