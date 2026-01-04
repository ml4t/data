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

## Regulatory Status

Kalshi is regulated by the CFTC (Commodity Futures Trading Commission) as a Designated Contract Market (DCM).

---

## See Also

- [Kalshi Developer Docs](https://kalshi.com/developer)
- [Polymarket Provider](polymarket.md)
- [Provider README](README.md)
