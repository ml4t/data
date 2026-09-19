# Generate OHLCV data without an external service

This tutorial verifies the installation and introduces the provider interface with deterministic,
locally generated data.

## Install the package

Use CPython 3.12, 3.13, or 3.14.

```bash
pip install ml4t-data
```

## Generate a daily series

```python
from ml4t.data.providers import SyntheticProvider

provider = SyntheticProvider(seed=42)
data = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")

assert not data.is_empty()
assert {"timestamp", "symbol", "open", "high", "low", "close", "volume"} <= set(data.columns)
print(data)
```

`SyntheticProvider` follows the same OHLCV interface as network-backed providers. The fixed seed
makes the generated series repeatable, so this example works in a clean environment without
credentials.

## Fetch external data

Choose a provider only after checking its dependency and service requirements. For example, Yahoo
Finance support requires the `yahoo` extra:

```bash
pip install "ml4t-data[yahoo]"
```

```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31", "daily")
```

This second example requires network access and is subject to the provider's availability and usage
terms. See [provider selection](provider-selection.md) before using an adapter in a recurring
workflow.

## Next steps

- [Configure local storage](../user-guide/configuration.md)
- [Update an existing dataset](../user-guide/incremental-updates.md)
- [Validate market data](../user-guide/data-quality.md)
- [Inspect the provider API](../api/index.md)
