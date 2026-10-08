# ForecastEx Provider

**Provider**: `ForecastExProvider`
**Website**: [forecastex.com](https://forecastex.com/data)
**API Key**: Not required
**Free Tier**: Free daily files from 2024-08-01

---

## Overview

ForecastEx LLC is a CFTC-registered exchange (DCM) and clearinghouse (DCO) affiliated with
Interactive Brokers. Each Forecast Market lists a YES and a NO contract that settle at $1.00 or
$0.00. ForecastEx publishes daily CSV files in the public S3 bucket `forecastex-public-data`;
the provider reads them directly.

| File | Content | Provider method |
|------|---------|-----------------|
| `prices/daily_prices_YYYYMMDD.csv` | Daily open, high, low, close, settlement price, paired quantity, open interest and VWAP per contract side | `fetch_prices()` |
| `pairs/pairs_YYYYMMDD.csv` | Every matched YES/NO pair (trade); the current day's file is refreshed every 10 minutes | `fetch_trades()` |
| `daily_summary/summary_YYYYMMDD.csv` | Product names and categories (from 2025-01-01) | `fetch_products()` |

All methods take an inclusive range of trading dates (Central Time). Days without a published
file are skipped, and a range with no files returns an empty, typed frame.

---

## Quick Start

```python
from ml4t.data.providers import ForecastExProvider

provider = ForecastExProvider()

markets = provider.fetch_markets("2026-09-15", "2026-09-16", product="FFDEC")
trades = provider.fetch_trades("2026-09-16", "2026-09-16", product="FFDEC")
prices = provider.fetch_prices("2026-09-16", "2026-09-16", product="FFDEC")

provider.close()
```

`event_contract` (the `ticker` column) encodes product, date and threshold, for example
`FFDEC_091626_E25`. The `product` column is the prefix before the first underscore. Suffix
semantics come from each product's terms PDF under `regulatory/` in the same bucket.

---

## Resolution Outcome

The files carry no outcome column. `fetch_markets()` derives it:

1. It also reads the prices file for the day after `end`. A contract absent from the next
   published file has expired.
2. If the YES `settlement_price` on the contract's last row is 1.00 or 0.00, that is the payout:
   `status="resolved"`, `result` is `"yes"` or `"no"`, and `settlement_value` is the payout.
3. Contracts that expire after the daily file is cut, such as hourly temperature products, drop
   out with their last daily mark instead of the payout. They are reported as
   `status="expired"` with a null `result`, because the files do not publish their outcome.
4. Contracts still listed in the next file, or trading on `end` when the next file is not yet
   published, are `status="open"`.

`settlement_ts` is the contract's expiration time; ForecastEx publishes no separate settlement
timestamp. The clearinghouse settles daily at 13:00 CT for contracts resolved in the previous
24 hours.

| Column | Meaning |
|--------|---------|
| `ticker`, `product`, `expiration` | Event contract, product code, expiration (UTC) |
| `status` | `resolved`, `expired` or `open` |
| `first_date`, `last_date` | First and last trading date seen in the range |
| `result`, `settlement_value`, `settlement_ts` | Normalized outcome, YES payout, expiration time |
| `last_close`, `volume` | Last daily close and paired contracts in the range |

---

## Trades

`fetch_trades()` returns one row per matched pair with the shared prediction-market trade columns
`trade_id`, `ticker`, `timestamp` (UTC), `price` (YES price), `count`, `taker_side` and
`is_block_trade`, plus `no_price`, `product` and `expiration`. ForecastEx pairs have no aggressor
side, so `taker_side` is always null, and `is_block_trade` is always false.

---

## Limits

- No order-book history; the live book is available only through an FCM such as IBKR.
- No documented rate limit; the provider defaults to 10 requests per second.
- High and low are 0.00 on days without trades, while open and close carry the daily marks.

---

## See Also

- [Prediction-market data sources](prediction_markets.md)
- [Kalshi Provider](kalshi.md)
- [Polymarket US Provider](polymarket_us.md)
- [ForecastEx data page](https://forecastex.com/data)
