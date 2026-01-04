# Mock Provider

**Provider**: `MockProvider`
**API Key**: Not required
**Free Tier**: N/A (for testing)

---

## Overview

Mock provider for unit testing. Returns predefined data or raises configured errors.

**Best For**: Unit tests, integration tests

---

## Quick Start

```python
from ml4t.data.providers import MockProvider
import polars as pl

# Create with predefined data
test_data = pl.DataFrame({
    "timestamp": ["2024-01-01", "2024-01-02"],
    "symbol": ["TEST", "TEST"],
    "open": [100.0, 101.0],
    "high": [102.0, 103.0],
    "low": [99.0, 100.0],
    "close": [101.0, 102.0],
    "volume": [1000000.0, 1100000.0],
})

provider = MockProvider(data=test_data)
df = provider.fetch_ohlcv("TEST", "2024-01-01", "2024-01-02")
```

---

## Testing Error Handling

```python
from ml4t.data.core.exceptions import SymbolNotFoundError

# Configure to raise errors
provider = MockProvider(error=SymbolNotFoundError("TEST"))

try:
    df = provider.fetch_ohlcv("TEST", "2024-01-01", "2024-01-02")
except SymbolNotFoundError:
    print("Correctly caught error")
```

---

## Use Cases

1. **Unit Tests**: Test without network
2. **Error Handling Tests**: Verify exception handling
3. **Performance Tests**: Known data size
4. **CI/CD Pipelines**: No external dependencies

---

## See Also

- [Synthetic Provider](synthetic.md)
- [Provider README](README.md)
