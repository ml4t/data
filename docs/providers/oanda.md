# Oanda Provider

**Provider**: `OandaProvider`
**Website**: [oanda.com](https://www.oanda.com)
**API Key**: Required
**Free Tier**: Practice account available

---

## Overview

Oanda provides institutional-grade forex data with comprehensive currency pair coverage.

**Best For**: Forex trading, institutional FX data

---

## Quick Start

```python
import os
os.environ["OANDA_API_KEY"] = "your_key_here"
os.environ["OANDA_ACCOUNT_ID"] = "your_account_id"

from ml4t.data.providers import OandaProvider

provider = OandaProvider()
df = provider.fetch_ohlcv("EUR_USD", "2024-01-01", "2024-12-01", frequency="daily")
provider.close()
```

---

## Symbol Format

Use Oanda's instrument format with underscore:
- `EUR_USD`, `GBP_USD`, `USD_JPY`, etc.

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | ✅ |
| `5m` | ✅ |
| `1h` | ✅ |
| `daily` | ✅ |
| `weekly` | ✅ |

---

## API Key Setup

```bash
# .env file
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id
```

Get credentials by opening a practice account at [oanda.com](https://www.oanda.com).

---

## Coverage

- Major pairs: EUR/USD, GBP/USD, USD/JPY, etc.
- Minor pairs: EUR/GBP, GBP/JPY, etc.
- Exotic pairs: USD/ZAR, USD/MXN, etc.
- Metals: XAU/USD, XAG/USD

---

## See Also

- [Oanda API](https://developer.oanda.com)
- [Provider README](README.md)
