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
`event_ticker`, `series_ticker` or `mve_filter=exclude`, so the remaining filters (status,
close-time window, the other multivariate selections) are applied on the client. Archived
markets are all settled, so the archive is skipped for other statuses.

`mve_filter` selects multivariate combo (parlay) markets, whose tickers start with `KXMVE`:
None includes them (Kalshi's default), `"exclude"` drops them and `"only"` keeps only them. The
archive rejects `mve_filter=only`, so selecting combos there pages through the whole archive and
filters on the client; use `max_pages` to sample.

```python
from ml4t.data.providers import KalshiProvider

provider = KalshiProvider()

fed = provider.fetch_markets(status="settled", series_ticker="KXFEDDECISION")
print(fed.select("ticker", "result", "settlement_value", "settlement_ts", "source"))

old = provider.fetch_markets(event_ticker="FED-23DEC")
window = provider.fetch_markets(
    status="settled", min_close_ts="2026-01-01", max_close_ts="2026-06-30",
    mve_filter="exclude",
)
combos = provider.fetch_markets(status="settled", mve_filter="only", max_pages=1)
provider.close()
```

| Column | Meaning |
|--------|---------|
| `ticker`, `event_ticker`, `title`, `status`, `market_type` | Market identity and lifecycle status |
| `series` | The `series_ticker` filter when given, otherwise the event-ticker prefix (legacy events such as `FED-23DEC` belong to a differently named series, `KXFED`) |
| `open_time`, `close_time` | Trading window (UTC) |
| `result` | `yes`, `no`, `other` (scalar settlement) or null while unresolved |
| `result_raw` | Kalshi's own `result` label |
| `settlement_value` | YES payout in dollars |
| `settlement_ts` | Settlement time (UTC) |
| `expiration_value`, `last_price`, `previous_price`, `volume`, `open_interest` | Settlement source value and last market statistics (prices in dollars, volume in contracts) |
| `source` | `live` or `historical` tier |

`iter_market_frames(chunk_size=100_000, ...)` streams the same normalized rows as frames of at
most `chunk_size` rows, for listings too large to hold in memory (the full settled listing runs to
millions of markets). `iter_markets()` yields the raw market dictionaries instead. `list_markets()` keeps returning a
single page of raw dictionaries.

---

## Price History Across Tiers

`fetch_candles(ticker, start_ts, end_ts, period="1h", series_ticker=None, tier=None)` returns
candlesticks for one market from whichever tier holds it. Markets settled before the historical
cutoff are served only by `GET /historical/markets/{ticker}/candlesticks`; the others by
`GET /series/{series}/markets/{ticker}/candlesticks`, which needs the market's series.

- Pass `tier` (the `source` column of `fetch_markets()`) to go straight to the right endpoint.
- Pass `series_ticker` (the `series` column, or the event's series) for live markets. Without
  it, the ticker prefix is tried first; if both tiers answer 404, the series is looked up from
  the market's event and the live endpoint is retried.
- Without `tier`, the live endpoint is tried first and the archive on a 404.
- Ranges longer than Kalshi's 5000-candle limit are split into several requests.

```python
candles = provider.fetch_candles(
    "KXFEDDECISION-26JUL-H0", "2026-07-26", "2026-07-29", period="1h",
    series_ticker="KXFEDDECISION", tier="historical",
)
```

| Column | Meaning |
|--------|---------|
| `timestamp` | End of the candle period (UTC) |
| `open`, `high`, `low`, `close`, `mean`, `previous` | Trade prices in dollars; null when nothing traded (except `previous`) |
| `yes_bid_close`, `yes_ask_close` | Closing YES bid and ask in dollars |
| `volume`, `open_interest` | Contracts |
| `source` | `live` or `historical` tier |

Live candles carry `*_dollars`/`*_fp` fields and archive candles unsuffixed dollar strings;
both are returned in dollars (0-1). `fetch_ohlcv()` uses the same tier routing, so it also
works for archived markets such as `FED-23DEC-T5.25`, and it keeps its OHLCV schema.

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
| `taker_side` | `yes` or `no`; falls back to `taker_outcome_side` |
| `is_block_trade` | Block-trade flag |
| `taker_outcome_side`, `taker_book_side` | Outcome side and book side (`bid`/`ask`) of the taker, null when absent |
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
