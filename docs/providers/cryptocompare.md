# CryptoCompare Provider

**Provider**: `CryptoCompareProvider`
**Website**: [cryptocompare.com](https://www.cryptocompare.com)
**API Key**: Required
**0.1.0 status**: Included for evaluation; not release-qualified

---

## Overview

CryptoCompare account registration was unavailable during the 0.1.0 release review. The adapter is
included for evaluation, but it has no successful live contract evidence for this release. Do not
treat it as a release-qualified provider until a later release records a successful contract run.

**Best For**: Crypto historical data, alternative to Binance

---

## Quick Start

```python
from ml4t.data.providers import CryptoCompareProvider

# Reads CRYPTOCOMPARE_API_KEY from the environment
provider = CryptoCompareProvider()
df = provider.fetch_ohlcv("BTC", "2024-01-01", "2024-12-01", frequency="daily")
provider.close()
```

---

## Symbol Format

Use base currency symbols:
- `BTC`, `ETH`, `SOL`, `ADA`, etc.

Quote currency defaults to USD.

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | ✅ |
| `1h` | ✅ |
| `daily` | ✅ |

---

## API Key Setup

```bash
# Environment variable
export CRYPTOCOMPARE_API_KEY=your_api_key_here
```

Get your API key at [cryptocompare.com/cryptopian/api-keys](https://www.cryptocompare.com/cryptopian/api-keys).

---

## Rate Limits

Consult CryptoCompare's current terms before use. Access and limits were not verified for 0.1.0.

---

## See Also

- [CryptoCompare Pricing](https://min-api.cryptocompare.com/pricing)
- [Provider reference](index.md)
