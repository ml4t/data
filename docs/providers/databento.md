# Databento Provider

**Provider**: `DataBentoProvider`
**Website**: [databento.com](https://databento.com)
**API Key**: Required
**Free Tier**: $125 credit

---

## Overview

Databento provides institutional-grade market data across 45+ exchanges with
15+ years of history. `DataBentoProvider` exposes OHLCV-oriented historical
fetches, continuous futures helpers, OPRA option helpers, schema discovery, and
direct access to the native `databento.Historical` client for advanced workflows.

**Best For**: Professional futures research, institutional-quality data

**Pricing**:
| Tier | Price | Features |
|------|-------|----------|
| Free Trial | $125 credit | Historical data only |
| Usage-based | Pay as you go | Historical data, $/GB |
| Standard | $179/mo | Live data, 15+ years core |
| Plus | $1,500/mo + fees | External distribution |
| Unlimited | $4,000/mo + fees | All schemas |

---

## Quick Start

```python
import os
os.environ["DATABENTO_API_KEY"] = "your_key_here"

from ml4t.data.providers import DataBentoProvider

provider = DataBentoProvider()

# Futures (CME)
df = provider.fetch_ohlcv("ES", "2024-01-01", "2024-06-01", frequency="daily")

# Multiple schemas for one symbol
df = provider.fetch_multiple_schemas(
    symbol="ES",
    start="2024-01-01",
    end="2024-06-01",
    schemas=["ohlcv-1d", "ohlcv-1h"],
)
```

---

## OPRA Options

The wrapper provides a small OPRA workflow for listed-options research:

1. Estimate request cost before downloading broad quote or bar data.
2. Discover option contracts from Databento definitions.
3. Fetch OHLCV bars or consolidated OPRA quotes for selected contracts.

```python
from ml4t.data.providers import DataBentoProvider

provider = DataBentoProvider()

estimate = provider.estimate_opra_cost(
    symbols=["SPY   240119C00480000"],
    start="2024-01-02",
    end="2024-01-02",
    schema="cbbo-1m",
)

chain = provider.fetch_option_chain(
    underlying="SPY",
    session_date="2024-01-02",
    expiry="2024-01-19",
    right="call",
    min_strike=440,
    max_strike=500,
)

bars = provider.fetch_option_ohlcv(
    contract="SPY   240119C00480000",
    start="2024-01-02",
    end="2024-01-02",
    frequency="daily",
)

quotes = provider.fetch_option_quotes(
    contract="SPY   240119C00480000",
    start="2024-01-02",
    end="2024-01-02",
    schema="cbbo-1m",
)
```

`fetch_option_quotes()` keeps Databento's consolidated OPRA publisher when
`publisher_id` is present. Set `consolidated_only=False` to retain
publisher-level quote rows.

---

## Supported Schemas

| Schema | Description | Use Case |
|--------|-------------|----------|
| `ohlcv-1d` | Daily OHLCV | End-of-day analysis |
| `ohlcv-1h` | Hourly OHLCV | Intraday patterns |
| `ohlcv-1m` | Minute OHLCV | Short-term strategies |
| `trades` | Tick trades | Microstructure |
| `cbbo-1m` | OPRA consolidated quotes | Listed-options quotes |
| `mbp-10` | 10-level depth | Order book analysis via native SDK |

---

## Exchange Coverage

| Category | Exchanges |
|----------|-----------|
| Equity Index | CME (ES, NQ, YM, RTY) |
| Energy | NYMEX (CL, NG, HO, RB) |
| Metals | COMEX (GC, SI, HG) |
| Rates | CBOT (ZN, ZB, ZF, ZT) |
| FX | CME (6E, 6J, 6B, 6A) |
| Agriculture | CBOT (ZC, ZW, ZS) |

45+ exchanges, 650,000+ symbols, 15+ years history.

---

## Continuous Futures

```python
# Fetch continuous front-month contract
df = provider.fetch_continuous_futures(
    root_symbol="ES",
    start="2020-01-01",
    end="2024-12-01",
    frequency="daily",
    version=0,
)
```

---

## API Key Setup

```bash
# .env file
DATABENTO_API_KEY=your_api_key_here
```

Get your API key at [databento.com](https://databento.com).

---

## Cost Estimation

| Data Type | Approximate Cost |
|-----------|------------------|
| Daily OHLCV | $0.01-0.05 per symbol-month |
| Minute OHLCV | $0.10-0.50 per symbol-month |
| Trades | $1-5 per symbol-month |
| L2 Depth | $5-20 per symbol-month |

Use the $125 free credit to explore before committing.

---

## Advanced Workflows

| Feature | Priority | Notes |
|---------|----------|-------|
| MBO (Market by Order) | LOW | Full order book |
| WebSocket streaming | NOT PLANNED | Use native SDK |
| Symbology API | LOW | Symbol resolution |

For advanced Databento schemas, batch jobs, symbology resolution, and live
streaming, use `provider.client` directly.

---

## See Also

- [Databento Pricing](https://databento.com/pricing)
- [Databento Reference](databento_reference.md) - Detailed schema guide
- [Provider README](README.md)
