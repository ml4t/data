# Alpaca Provider

**Provider**: `AlpacaDataProvider`
**Website**: [alpaca.markets](https://alpaca.markets)
**API Key**: Required (key + secret pair)
**Free Tier**: 200 requests/min, real-time IEX feed

---

## Overview

Alpaca provides long-history, high-frequency US market data across equities and
crypto over a single historical REST API. One provider serves both asset
classes: plain tickers route to the stock bars endpoint, `BASE/QUOTE` symbols
route to the crypto bars endpoint.

**Best For**: Free US intraday equities, US crypto bars

**Pricing**:
| Tier | Price | Features |
|------|-------|----------|
| Basic | $0/mo | 200 req/min, real-time IEX feed, no recent-15-min SIP access |
| Algo Trader Plus | $99/mo | 10,000 req/min, full SIP (consolidated tape) |

---

## Quick Start

```python
import os
os.environ["ALPACA_API_KEY"] = "your_key_here"
os.environ["ALPACA_API_SECRET"] = "your_secret_here"

from ml4t.data.providers import AlpacaDataProvider

provider = AlpacaDataProvider()

# US stocks
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01")

# Crypto (BASE/QUOTE symbol routes to the crypto endpoint)
df = provider.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-01-31")

# Intraday with RFC-3339 datetime bounds
df = provider.fetch_ohlcv(
    "BTC/USD", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", frequency="minute"
)

# Multi-year intraday backfills with minute multiples (5m / 15m / 30m)
df = provider.fetch_ohlcv("AAPL", "2021-01-01", "2024-12-31", frequency="15m")

provider.close()
```

Async usage:

```python
async with AlpacaDataProvider() as provider:
    df = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-12-01")
```

---

## Symbol Format

| Asset class | Format | Examples |
|-------------|--------|----------|
| US stocks | Plain ticker | AAPL, MSFT |
| Crypto | BASE/QUOTE | BTC/USD, ETH/USD |

Symbols are uppercased into requests and into the output `symbol` column; the
crypto slash is preserved (e.g. `BTC/USD`).

---

## Supported Frequencies

| Frequency | Availability |
|-----------|--------------|
| `daily` / `1d` | ✅ |
| `hourly` / `1h` | ✅ |
| `minute` / `1m` | ✅ |
| `5m` / `5minute` | ✅ |
| `15m` / `15minute` | ✅ |
| `30m` / `30minute` | ✅ |

`start`/`end` accept `YYYY-MM-DD` dates or RFC-3339 datetimes (both inclusive);
datetime bounds are the natural shape for sub-day minute/hour windows.

---

## Feeds and Adjustment

- `feed="iex"` (default): the free feed, real-time but served from a single
  exchange (IEX, roughly 2-3% of US volume). The Basic plan additionally cannot
  query the most recent 15 minutes of SIP data.
- `feed="sip"`: the consolidated tape (paid plans).
- `adjustment="raw"` (default, Alpaca's own default): stock bars are **not**
  adjusted for splits or dividends. Pass `adjustment="split"`, `"dividend"`, or
  `"all"` for adjusted bars. Crypto has no adjustment concept.

```python
provider = AlpacaDataProvider(feed="sip", adjustment="all")
```

---

## API Key Setup

```bash
# .env file (project convention)
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

Alpaca's own SDK/CLI names `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` are also
accepted. Get a free key at [alpaca.markets](https://alpaca.markets/).

---

## Rate Limits

| Tier | Limit |
|------|-------|
| Basic (free) | 200 req/min |
| Algo Trader Plus | 10,000 req/min |

The provider throttles client-side to 200 req/min by default (override with the
`rate_limit` constructor argument), honors 429 `Retry-After`/rate-limit-reset
headers, and retries transient failures per pagination page.

---

## Not Yet Implemented

| Feature | Priority |
|---------|----------|
| Quotes / trades (tick) endpoints | MEDIUM |
| Options bars | LOW |
| News API | LOW |

---

## See Also

- [Alpaca Market Data docs](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Provider README](README.md)
- [Provider Audit](PROVIDER_AUDIT.md)
