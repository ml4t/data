# Installation

## Quick Install

=== "pip"

    ```bash
    pip install ml4t-data
    ```

=== "uv (recommended)"

    ```bash
    uv add ml4t-data
    ```

=== "From source"

    ```bash
    git clone https://github.com/stefan-jansen/ml4t-data.git
    cd ml4t-data
    uv sync
    ```

## Provider Dependencies

Some providers require additional dependencies:

```bash
# Yahoo Finance (included by default)
pip install "ml4t-data[yahoo]"

# DataBento (institutional futures/options)
pip install "ml4t-data[databento]"

# OANDA (forex)
pip install "ml4t-data[oanda]"

# All providers
pip install "ml4t-data[all-providers]"
```

## Development Installation

For contributing or running tests:

```bash
git clone https://github.com/stefan-jansen/ml4t-data.git
cd ml4t-data
uv sync --all-extras
pre-commit install
```

## Verify Installation

```python
from ml4t.data import __version__
print(f"ml4t-data version: {__version__}")

# Test a provider
from ml4t.data.providers import YahooFinanceProvider
provider = YahooFinanceProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
print(f"Fetched {len(df)} rows")
```

## Requirements

- Python 3.9+
- Polars (automatically installed)
- Provider-specific SDKs (optional)
