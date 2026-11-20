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
| `weekly` / `1w` | ✅ |
| `monthly` / `1mo` | ✅ |

`start`/`end` accept `YYYY-MM-DD` dates or RFC-3339 datetimes (both inclusive);
datetime bounds are the natural shape for sub-day minute/hour windows.

---

## Many Stocks at Once

`fetch_batch_ohlcv` sends up to `chunk_size` (default 1,000) stock tickers per
request to the multi-symbol endpoint. The symbols of a request share each page
of 10,000 bars, so a bulk load takes about one request per 10,000 bars rather
than one per symbol. A ticker without bars in the range is simply absent, which
makes monthly bars a cheap probe of which tickers traded, delisted ones
included. One invalid ticker string fails its whole request with HTTP 400.

```python
provider = AlpacaDataProvider(feed="sip", adjustment="raw")
months = provider.fetch_batch_ohlcv(
    ["AAPL", "TWTR", "SIVB"], "2016-01-01", "2024-12-31", frequency="monthly"
)
```

The result is one long frame in the canonical schema. Store it per symbol with
`DataManager.import_data`, or keep using `fetch_ohlcv` through `update` for
incremental refreshes.

---

## Renamed and Reused Tickers

Alpaca maps a stock ticker to the company that holds it on an `asof` date,
today by default, and returns that company's whole history. Measured on daily
and monthly bars since 2016:

| Request | Default (`asof` today) | `asof="-"` | `asof="2020-01-01"` |
|---|---|---|---|
| `META` | Meta's whole history, FB years included | a fund from 2021-06, then Meta from 2022-06 | as `"-"` |
| `FB` | absent from the multi-symbol endpoint; the single-symbol endpoint returns the `"-"` bars | Meta until 2022-06, then the next holder of FB | Meta's whole history, META years included |
| `BBBY` | as `"-"` | Bed Bath & Beyond to 2023-05, then a later holder from 2025 | Bed Bath & Beyond only |

With the default, a renamed company's history appears twice, under both its
tickers, and the multi-symbol endpoint drops the old ticker. With `asof="-"`
each bar appears once, under the ticker it traded as, whoever traded it; joining
a company's tickers across a rename, and splitting a reused ticker, is then the
job of an `InstrumentRegistry`.

---

## Feeds and Adjustment

- `feed="iex"` (default): the free feed, real-time but served from a single
  exchange (IEX, roughly 2-3% of US volume).
- `feed="sip"`: the consolidated tape (100% of US volume, all exchanges).
  Real-time on paid plans; the free Basic plan can still query SIP bars, only
  not the most recent 15 minutes, and answers such a request with HTTP 403. The
  provider therefore ends a SIP stock request `sip_delay_minutes` (default 15)
  before now, so `end` may be today, as incremental `update` requests are. Pass
  `sip_delay_minutes=0` on a paid plan.
- `feed="otc"` (over-the-counter) and `feed="boats"` (Blue Ocean overnight) are
  also accepted. Any other value raises `DataValidationError` at construction.
- `adjustment="raw"` (default, Alpaca's own default): stock bars are **not**
  adjusted for splits or dividends. Pass `adjustment="split"`, `"dividend"`, or
  `"all"` for adjusted bars. Crypto has no adjustment concept.

```python
provider = AlpacaDataProvider(feed="sip", adjustment="all")
```

In a `ml4t.data.yaml`, the same settings go under `extra` of a named provider:

```yaml
providers:
  - name: alpaca_sip
    type: alpaca
    extra: {feed: sip, adjustment: all}
```

---

## API Key Setup

```bash
# Environment variables (project convention)
export ALPACA_API_KEY=your_key_here
export ALPACA_API_SECRET=your_secret_here
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

- [Equity](equities.md) and [ETF](etfs.md) source references
- [Alpaca Market Data docs](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Provider README](index.md)
