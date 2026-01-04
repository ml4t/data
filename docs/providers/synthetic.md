# Synthetic Provider

**Provider**: `SyntheticProvider`
**API Key**: Not required
**Free Tier**: N/A (generates data)

---

## Overview

Generates synthetic OHLCV data for testing, demos, and development without requiring network access or API keys.

**Best For**: Testing, demos, development

---

## Quick Start

```python
from ml4t.data.providers import SyntheticProvider

provider = SyntheticProvider()

# Generate synthetic data
df = provider.fetch_ohlcv("DEMO", "2024-01-01", "2024-12-01", frequency="daily")

print(df.head())
# Synthetic OHLCV data with realistic patterns

provider.close()
```

---

## Configuration

```python
provider = SyntheticProvider(
    base_price=100.0,      # Starting price
    volatility=0.02,       # Daily volatility (2%)
    trend=0.0001,          # Daily drift
    seed=42                # Reproducible results
)
```

---

## Use Cases

1. **Unit Tests**: Test data pipelines without API calls
2. **Demos**: Show functionality without credentials
3. **Development**: Fast iteration without rate limits
4. **Documentation**: Reproducible examples

---

## Generated Data

- Realistic OHLCV patterns (geometric Brownian motion)
- Proper OHLC relationships (High >= Open, Close, Low)
- Volume follows log-normal distribution
- Weekdays only (no weekends)

---

## See Also

- [Mock Provider](mock.md)
- [Provider README](README.md)
