# Polymarket Provider

Polymarket is the largest prediction market by volume, operating on the Polygon blockchain. This provider fetches historical probability data for prediction markets.

## Overview

| Feature | Value |
|---------|-------|
| **Asset Class** | Prediction Markets |
| **Data Type** | Probability prices (0.00 - 1.00) |
| **Authentication** | None required (public data) |
| **Rate Limit** | ~30 requests/minute |
| **Free Tier** | Unlimited |

## Quick Start

```python
from ml4t.data.providers.polymarket import PolymarketProvider

provider = PolymarketProvider()

# Fetch daily probability data
df = provider.fetch_ohlcv(
    "us-recession-in-2025",
    "2025-12-01",
    "2025-12-07",
    outcome="yes"
)

print(df)
# shape: (1, 7)
# ┌─────────────────────┬──────────────────────────┬───────┬────────┬───────┬────────┬────────┐
# │ timestamp           ┆ symbol                   ┆ open  ┆ high   ┆ low   ┆ close  ┆ volume │
# │ datetime[μs]        ┆ str                      ┆ f64   ┆ f64    ┆ f64   ┆ f64    ┆ f64    │
# ╞═════════════════════╪══════════════════════════╪═══════╪════════╪═══════╪════════╪════════╡
# │ 2025-12-07 00:00:00 ┆ US-RECESSION-IN-2025:YES ┆ 0.015 ┆ 0.0155 ┆ 0.015 ┆ 0.0155 ┆ 61.0   │
# └─────────────────────┴──────────────────────────┴───────┴────────┴───────┴────────┴────────┘

provider.close()
```

## Price Interpretation

Prices represent **implied probabilities** (0.00 to 1.00):
- `0.65` = 65% implied probability of event occurring
- YES + NO prices should sum to ~1.00 (minus spread)

**Note**: Volume is a proxy based on number of price updates, not actual trading volume.

## Symbol Formats

The provider accepts multiple symbol formats:

| Format | Example | Description |
|--------|---------|-------------|
| **Slug** | `us-recession-in-2025` | Human-readable URL slug (recommended) |
| **Condition ID** | `0xabcd1234...` | Blockchain condition identifier |
| **Token ID** | `104173557214744...` | Direct CLOB token ID |

```python
# All of these work:
df = provider.fetch_ohlcv("us-recession-in-2025", start, end)
df = provider.fetch_ohlcv("0x1234...", start, end)
df = provider.fetch_ohlcv("104173557214744...", start, end)
```

## API Methods

### fetch_ohlcv()

Fetch OHLCV data for a single outcome.

```python
df = provider.fetch_ohlcv(
    symbol="us-recession-in-2025",  # slug, condition_id, or token_id
    start="2025-01-01",
    end="2025-12-31",
    frequency="daily",              # minute, hourly, daily, weekly
    outcome="yes"                   # "yes" or "no"
)
```

### fetch_both_outcomes()

Fetch data for both YES and NO outcomes in a single call.

```python
df = provider.fetch_both_outcomes(
    "us-recession-in-2025",
    "2025-12-01",
    "2025-12-07",
    frequency="daily"
)

# Returns long-format DataFrame with both outcomes
# symbol column contains: "US-RECESSION-IN-2025:YES" and "US-RECESSION-IN-2025:NO"
```

### get_token_prices()

Get current prices for both outcomes.

```python
prices = provider.get_token_prices("us-recession-in-2025")
print(f"YES: {prices['yes']:.2%}")  # YES: 1.55%
print(f"NO: {prices['no']:.2%}")    # NO: 98.45%
print(f"Sum: {prices['yes'] + prices['no']:.2%}")  # Sum: 100.00%
```

### list_markets()

List available markets with filtering.

```python
# List active, non-closed markets
markets = provider.list_markets(
    active=True,
    closed=False,  # Important: closed=True returns resolved markets
    limit=20
)

for m in markets:
    print(f"{m['slug']}: {m['question'][:50]}")
```

### search_markets()

Search markets by question text.

```python
markets = provider.search_markets("bitcoin", limit=10)
for m in markets:
    print(m['question'])
```

### resolve_symbol()

Resolve a slug or condition ID to a token ID.

```python
yes_token = provider.resolve_symbol("us-recession-in-2025", "yes")
no_token = provider.resolve_symbol("us-recession-in-2025", "no")
```

### get_market_metadata()

Get detailed metadata for a market.

```python
meta = provider.get_market_metadata("us-recession-in-2025")
print(f"Question: {meta['question']}")
print(f"Volume: ${meta.get('volume', 0):,.2f}")
print(f"End Date: {meta.get('endDate')}")
```

## Resolved Markets and Outcomes

`fetch_markets()` enumerates every market matching the filters through Gamma's
`GET /markets/keyset`, which pages by cursor (`GET /markets` refuses offsets above 2000). Pages
hold at most 100 markets, in ascending Gamma id order. The server applies `closed`, the minimum
lifetime volume and the scheduled end-date window; the provider applies them again on the client.

```python
from ml4t.data.providers import PolymarketProvider

provider = PolymarketProvider()

resolved = provider.fetch_markets(
    min_volume=100_000, end_date_min="2025-01-01", end_date_max="2025-12-31T23:59:59Z"
)
print(resolved.select("ticker", "question", "yes_outcome", "result", "settlement_ts"))

for frame in provider.iter_market_frames(chunk_size=50_000, min_volume=10_000):
    ...  # write each frame out before the next pages are requested
provider.close()
```

Gamma markets are binary, and their outcome labels, CLOB token ids and final prices share one
order. The first outcome is treated as the YES side whatever its label: in a "Trump vs. Boden"
market `yes_outcome` is "Trump", and `result == "yes"` means the first outcome won.

| Column | Meaning |
|--------|---------|
| `ticker` | Condition id (the identifier `fetch_trades` takes) |
| `market_id`, `slug`, `question` | Gamma id, URL slug and question text |
| `event_slug`, `event_title` | The market's event |
| `category`, `tags` | Gamma category (mostly null on recent markets) and tag labels |
| `yes_outcome`, `no_outcome`, `yes_token_id`, `no_token_id` | First and second outcome labels and their CLOB tokens (the identifiers `fetch_candles` takes) |
| `closed`, `open_time`, `close_time` | Closed flag, `startDate` and scheduled `endDate` (UTC) |
| `result` | `yes` for final prices `[1, 0]`, `no` for `[0, 1]`, `void` for a 50/50 resolution, `other` for a UMA-resolved market with other prices, null otherwise |
| `settlement_value` | Final payout of the first outcome token |
| `settlement_ts` | `closedTime`, else `umaEndDate` (UTC); null while unresolved |
| `uma_resolution_status`, `outcome_prices` | Raw UMA status and final or current prices |
| `volume` | Lifetime volume in USDC |
| `neg_risk`, `neg_risk_market_id` | Negative-risk (mutually exclusive outcomes) grouping |
| `fees_enabled`, `fee_type`, `maker_base_fee`, `taker_base_fee`, `fee_schedule` | Gamma's raw fee fields; `fee_schedule` as JSON text |

Closed markets carry final prices even while their UMA status is still `proposed`; keep rows with
`uma_resolution_status == "resolved"` to exclude resolutions that could still be disputed.
`settlement_ts` often precedes `close_time`, because many markets resolve before their scheduled
end. Gamma timestamps that cannot be parsed (`umaEndDate` is `"NOW*()"` on some 2024 markets) are
null. `iter_markets()` yields the raw Gamma dictionaries instead.

## Price History of Resolved Markets

`fetch_candles(token_id, start, end, fidelity_minutes=60)` returns `timestamp` (UTC), `token_id`
and `price`, one row per CLOB price sample; these are samples, not OHLC bars. The CLOB's
`GET /prices-history` answers `interval=max` with an empty history for resolved markets at any
fidelity finer than one day, and rejects explicit `startTs`/`endTs` spans longer than 15 days with
HTTP 400. The provider sends explicit bounds in windows of at most 15 days, which serves resolved
markets down to one-minute fidelity. An unknown token returns an empty frame, not an error.

```python
market = resolved.row(0, named=True)
end = market["settlement_ts"]
prices = provider.fetch_candles(
    market["yes_token_id"], end.timestamp() - 30 * 86_400, end, fidelity_minutes=1
)
```

## Trade History

`fetch_trades(condition_id, start=None, end=None)` reads the data API's
`GET /trades?market=<condition id>`, which returns taker trades newest first, takes inclusive
`start`/`end` bounds in Unix seconds, serves at most 10,000 trades per request and rejects offsets
above 10,000. Beyond that depth the provider moves `end` to the oldest second seen and continues,
skipping the trades of that second it already returned.

| Column | Meaning |
|--------|---------|
| `trade_id` | Hash of transaction, token, wallet, side, size, price and time (the API has no trade id) |
| `ticker`, `timestamp` | Condition id and trade time (UTC, whole seconds) |
| `price` | YES price: the traded token's price, or one minus it for the second outcome |
| `count` | Shares |
| `taker_side` | `yes` when the taker bought the first outcome or sold the second, else `no` |
| `is_block_trade` | Always false |
| `outcome`, `outcome_index`, `side`, `outcome_price` | Traded outcome label and index, taker `BUY`/`SELL`, and the traded token's price |
| `asset`, `proxy_wallet`, `transaction_hash` | Token id, taker wallet and on-chain transaction |

## Supported Frequencies

| Frequency | API Interval | Notes |
|-----------|--------------|-------|
| `minute` / `1m` | 1m | High resolution |
| `hourly` / `1h` | 1h | Standard |
| `6h` | 6h | Medium resolution |
| `daily` / `1d` | 1d | Recommended |
| `weekly` / `1w` | 1w | Low resolution |

## API Limitations

### Date Range Limits

The CLOB rejects price-history spans longer than 15 days. `fetch_ohlcv()` requests 14-day chunks
and `fetch_candles()` 15-day windows, so any range works; long ranges at fine frequencies take
one request per chunk.

### Closed Markets

`list_markets()` returns one page of the offset listing, which stops at offset 2000. Use
`fetch_markets()` or `iter_market_frames()` to enumerate closed markets.

## Example Use Cases

### Event-Driven Strategy Signals

```python
from ml4t.data.providers.polymarket import PolymarketProvider

provider = PolymarketProvider()

# Track recession probability as a macro indicator
df = provider.fetch_ohlcv("us-recession-in-2025", "2025-01-01", "2025-12-07")

# Use as a regime indicator
recession_prob = df['close'].to_list()[-1]
if recession_prob > 0.30:
    print("High recession probability - reduce risk exposure")
```

### Multi-Market Analysis

```python
markets_of_interest = [
    "fed-rate-hike-in-2025",
    "us-recession-in-2025",
    "tether-insolvent-in-2025",
]

import polars as pl

all_data = []
for slug in markets_of_interest:
    try:
        df = provider.fetch_ohlcv(slug, "2025-12-01", "2025-12-07", outcome="yes")
        all_data.append(df)
    except Exception as e:
        print(f"Skipping {slug}: {e}")

combined = pl.concat(all_data)
print(combined.group_by("symbol").agg(pl.col("close").last()))
```

## API Documentation Links

- **CLOB Timeseries**: https://docs.polymarket.com/developers/clob-api/price-history
- **Gamma Markets API**: https://docs.polymarket.com/developers/gamma-markets-api/get-markets
- **Data API Trades**: https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets
- **py-clob-client**: https://github.com/Polymarket/py-clob-client

## Technical Notes

### API Response Format

The Gamma API returns token IDs in a JSON string format:

```json
{
  "clobTokenIds": "[\"token1\", \"token2\"]",
  "outcomes": "[\"Yes\", \"No\"]",
  "outcomePrices": "[\"0.015\", \"0.985\"]"
}
```

The provider automatically parses these JSON strings and maps:
- Index 0 → YES outcome
- Index 1 → NO outcome

### Rate Limiting

The provider includes built-in rate limiting (30 req/min). For bulk operations, reuse the provider instance:

```python
provider = PolymarketProvider()  # Create once

for market in markets:
    df = provider.fetch_ohlcv(market['slug'], start, end)  # Rate-limited automatically

provider.close()  # Close when done
```

## Changelog

See [Prediction-Market Data Sources](prediction_markets.md) for source and market-mechanism
comparisons.

### 2026-10-08
- Added `fetch_markets()`, `iter_markets()` and `iter_market_frames()` for resolved markets with normalized outcomes
- Added `fetch_candles()` for windowed price history of resolved markets and `fetch_trades()`
- Fixed `fetch_ohlcv()` returning no rows for past ranges and reading dates in the local time zone

### 2025-12-07
- Fixed token resolution for new API format (`clobTokenIds` JSON string)
- Fixed `get_token_prices()` to use `outcomes`/`outcomePrices` arrays
- Updated tests for better market selection (filter `closed=False`)
