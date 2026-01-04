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

The CLOB API has date range limits, especially for high-frequency data:
- **7 days** is safe for all frequencies
- Longer ranges may return `HTTP 400: interval is too long` error

```python
# Safe: 7-day range
df = provider.fetch_ohlcv("market-slug", "2025-12-01", "2025-12-07")

# May fail: 30-day range with hourly data
df = provider.fetch_ohlcv("market-slug", "2025-11-01", "2025-12-01", frequency="hourly")
```

### Closed Markets

Markets that have resolved (closed) may have limited or no recent price data:

```python
# Get only active, non-resolved markets
markets = provider.list_markets(active=True, closed=False)
```

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

### 2025-12-07
- Fixed token resolution for new API format (`clobTokenIds` JSON string)
- Fixed `get_token_prices()` to use `outcomes`/`outcomePrices` arrays
- Updated tests for better market selection (filter `closed=False`)
