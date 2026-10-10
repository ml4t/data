# CryptoCompare Provider

**Provider**: `CryptoCompareProvider`
**Website**: [cryptocompare.com](https://www.cryptocompare.com)
**API Key**: Required
**0.2.0 status**: Adapter included; live contract unverified

---

## Overview

The 0.2.0 package includes the adapter and its offline contract tests. Release validation did not
have a configured `CRYPTOCOMPARE_API_KEY`, so no successful live provider contract was recorded.
Verify access with your own account before relying on the adapter for a production dataset.

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

Consult CryptoCompare's current terms before use. Account access and limits were not verified for
the 0.2.0 release.

---

## See Also

- [Cryptocurrency data sources](crypto.md)
- [CryptoCompare Pricing](https://min-api.cryptocompare.com/pricing)
- [Provider reference](index.md)
