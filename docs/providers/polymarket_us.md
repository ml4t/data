# Polymarket US Provider

**Provider**: `PolymarketUSProvider`
**Website**: [polymarket.us](https://polymarket.us)
**API Key**: Not required
**Free Tier**: Free public gateway

---

## Overview

Polymarket US is the CFTC-regulated exchange (DCM and DCO) operated by QCX LLC d/b/a Polymarket
US. It is a separate venue from the global, crypto-settled Polymarket served by
[`PolymarketProvider`](polymarket.md), with separate markets, order books and identifiers.
Markets are identified by slug.

The provider reads the public gateway `https://gateway.polymarket.us`:

| Endpoint | Content | Provider method |
|----------|---------|-----------------|
| `GET /v1/markets` | Markets with status, sides, outcomes and final prices; offset pagination, at most 500 per page | `iter_markets()`, `fetch_markets()` |
| `GET /v1/markets/{slug}/settlement` | Settlement price of a resolved market | `get_settlement()` |
| `GET /v1/price-history` | Book-derived YES and NO display prices | `fetch_price_history()` |

There is no public trade history: the trade-report endpoints return only the caller's own trades.

---

## Quick Start

```python
from ml4t.data.providers import PolymarketUSProvider

provider = PolymarketUSProvider()

resolved = provider.fetch_markets(closed=True, categories=["macro"])
print(resolved.select("ticker", "result", "settlement_value", "winning_outcome"))

history = provider.fetch_price_history(resolved["ticker"][0])
minutes = provider.fetch_price_history(
    resolved["ticker"][0], start="2026-05-11", end="2026-05-12"
)

provider.close()
```

`fetch_markets()` filters on `closed`, `active`, `categories` (`sports`, `macro`, `politics`,
...), `slugs` and an end-date window. `max_pages` caps the number of pages for sampling.

---

## Resolution Outcome

Each market is one instrument: the long market side is the YES contract and the short side the
NO contract. For a market with status `MARKET_STATUS_RESOLVED`, the final long-side price in
`marketSides` is the payout per YES contract:

| Payout | `result` |
|--------|----------|
| 1 | `yes` |
| 0 | `no` |
| Other, e.g. 0.5 on a cancelled game | `other` |
| Market not resolved | null |

`winning_outcome` names the winning side (`"Yes"`, `"No"`, or a team for sports markets) and
`yes_outcome` names the long side. `settlement_ts` is always null because the gateway publishes
no settlement time. `get_settlement(slug)` cross-checks the payout and returns None when the
gateway answers 404, which it does for unknown and unsettled markets.

!!! warning "Outcome labels are not ordered like the prices"
    `outcomePrices` is long-side first, but the `outcomes` labels are sometimes listed in the
    other order, for example `["No","Yes"]` with `["1","0"]` for a market whose YES side won.
    The provider reads the outcome from the long market side and never zips labels with prices.

---

## Price History

Without `start` and `end`, `fetch_price_history()` requests the full history
(`fixedInterval=INTERVAL_ALL`: three-hourly points for newer markets, daily points for longer
ones). With both, it requests 24-hour windows at one-minute fidelity, the gateway's limit for
custom ranges. Columns: `timestamp` (UTC), `ticker`, `long_price` (YES display price, from the
best ask) and `short_price` (NO display price, one minus the best bid). These are quotes, not
trades, and the two can sum to more than 1. The public limit is 20 requests per second per IP,
which is the provider's default rate limit.

---

## See Also

- [Prediction-market data sources](prediction_markets.md)
- [Polymarket Provider](polymarket.md) (global venue)
- [Kalshi Provider](kalshi.md)
- [ForecastEx Provider](forecastex.md)
- [Polymarket US API docs](https://docs.polymarket.us/api-reference/introduction)
