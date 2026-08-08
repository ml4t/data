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
    base_price=100.0,
    annual_return=0.08,
    annual_volatility=0.20,
    calendar_mode="equity",
    seed=42,
)
```

`calendar_mode="equity"` emits a simplified weekday 09:30-16:00 UTC session. It does
not model exchange holidays, early closes, or daylight-saving changes. Intraday
timestamps are bar starts and exclude 16:00. Daily bars are labelled at 16:00 UTC.
Use `calendar_mode="continuous"` for 24-hour UTC sessions that include weekends;
continuous daily bars are labelled at 00:00 UTC. Weekly and monthly bars use period-end
labels. Annual return and volatility use 261 weekdays for equity mode and 365 days for
continuous mode, with 52 weekly or 12 monthly periods in either mode.

With a seed, the provider derives a symbol-specific stream using BLAKE2b and NumPy's
PCG64 generator. Identical seed, symbol, date, frequency, model, and calendar inputs
produce identical results across interpreter processes and supported platforms.

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
- Equity-session or continuous UTC calendars

## Learned Samples

`LearnedSyntheticProvider` converts pre-generated model samples into the same OHLCV contract. It
accepts a non-pickle NumPy array with shape `(n_samples, sequence_length, n_features)` or an artifact
directory containing `samples.npy` and `metadata.json`.

```python
from ml4t.data.providers import LearnedSyntheticProvider

provider = LearnedSyntheticProvider.from_samples("timegan_sequences.npy", seed=42)
df = provider.fetch_ohlcv("SYNTH_TIMEGAN", "2024-01-01", "2024-12-31", "daily")
```

Executable model checkpoints are not loaded. Generate `samples.npy` during model training before
constructing the provider.

---

## See Also

- [Mock Provider](mock.md)
- [Provider reference](index.md)
