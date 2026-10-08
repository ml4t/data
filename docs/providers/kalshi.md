# Kalshi Provider

**Provider**: `KalshiProvider`
**Website**: [kalshi.com](https://kalshi.com)
**API Key**: Not required
**Free Tier**: Free

---

## Overview

Kalshi is a US-regulated prediction market (CFTC-regulated) offering event contracts on economics, politics, and other events.

**Best For**: US-regulated event probabilities, economic predictions

---

## Quick Start

```python
from ml4t.data.providers import KalshiProvider

provider = KalshiProvider()

# Fetch event contract data
df = provider.fetch_ohlcv("INXD-24DEC31", "2024-01-01", "2024-12-01")

provider.close()
```

---

## Contract Types

- **Economic**: Fed rate decisions, CPI, unemployment
- **Political**: Election outcomes
- **Climate**: Temperature records
- **Other**: Various binary event contracts

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | ✅ |
| `1h` | ✅ |
| `daily` | ✅ |

---

## Resolved Markets and Outcomes

`fetch_markets()` enumerates every market matching the filters, following Kalshi's cursor
pagination across two tiers: the live `GET /markets` endpoint and the archive
`GET /historical/markets`, which alone holds markets settled before the cutoff reported by
`GET /historical/cutoff` (`get_historical_cutoff()`). The archive accepts only one of
`event_ticker`, `series_ticker` or the multivariate filter, so the remaining filters (status,
close-time window, multivariate exclusion) are applied on the client. Archived markets are all
settled, so the archive is skipped for other statuses.

```python
from ml4t.data.providers import KalshiProvider

provider = KalshiProvider()

fed = provider.fetch_markets(status="settled", series_ticker="KXFEDDECISION")
print(fed.select("ticker", "result", "settlement_value", "settlement_ts", "source"))

old = provider.fetch_markets(event_ticker="FED-23DEC")
window = provider.fetch_markets(
    status="settled", min_close_ts="2026-01-01", max_close_ts="2026-06-30",
    exclude_multivariate=True,
)
provider.close()
```

| Column | Meaning |
|--------|---------|
| `ticker`, `event_ticker`, `title`, `status`, `market_type` | Market identity and lifecycle status |
| `open_time`, `close_time` | Trading window (UTC) |
| `result` | `yes`, `no`, `other` (scalar settlement) or null while unresolved |
| `result_raw` | Kalshi's own `result` label |
| `settlement_value` | YES payout in dollars |
| `settlement_ts` | Settlement time (UTC) |
| `expiration_value`, `last_price`, `volume`, `open_interest` | Settlement source value and last market statistics |
| `source` | `live` or `historical` tier |

`iter_markets()` yields the raw market dictionaries instead. `list_markets()` keeps returning a
single page of raw dictionaries.

---

## Trade History

`fetch_trades(ticker, min_ts=None, max_ts=None)` pages through `GET /markets/trades` and
`GET /historical/trades`, skipping a tier the time window cannot reach, and returns one row per
trade sorted by time:

| Column | Meaning |
|--------|---------|
| `trade_id`, `ticker` | Trade and market identifiers |
| `timestamp` | Trade time (UTC) |
| `price` | YES price in dollars (0-1) |
| `count` | Contracts; fractional counts are allowed |
| `taker_side` | `yes` or `no` |
| `is_block_trade` | Block-trade flag |
| `source` | `live` or `historical` tier |

Prices and counts are read from the fixed-point `*_dollars` and `*_fp` fields; legacy unsuffixed
fields are also accepted (integers as cents, strings as dollars).

---

## Regulatory Status

Kalshi is regulated by the CFTC (Commodity Futures Trading Commission) as a Designated Contract Market (DCM).

---

## See Also

- [Prediction-market data sources](prediction_markets.md)
- [Kalshi Developer Docs](https://kalshi.com/developer)
- [Polymarket Provider](polymarket.md)
- [Polymarket US Provider](polymarket_us.md)
- [ForecastEx Provider](forecastex.md)
- [Provider reference](index.md)
