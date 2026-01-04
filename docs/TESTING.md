# Testing Guide

## Quick Start

### Run Tests (Parallel by Default)
```bash
# Default: runs tests in parallel using all CPU cores (FAST!)
pytest

# Verbose output
pytest -v

# Disable parallel execution (slower, but better for debugging)
pytest -n 0

# Use specific number of workers
pytest -n 4
```

**Performance**: Tests run in parallel by default using `pytest-xdist`:
- **24 cores**: ~7 seconds for 345 tests (130x speedup vs sequential)
- **Sequential** (`-n 0`): ~15 minutes for full suite

### Run Integration Tests
```bash
# Run ALL tests including integration (if you have API keys)
pytest -m ""

# Run only integration tests
pytest -m integration

# Run specific provider integration tests
pytest tests/integration/test_coingecko.py -m ""
```

### Run API Tests
```bash
# Install API dependencies first
pip install -e ".[api]"

# Run API tests
pytest tests/test_api/ -m ""
```

## Test Categories

### Unit Tests (Fast)
- **Location**: Mostly in `tests/` root, some in `tests/unit/`
- **Duration**: <1 minute for all unit tests
- **Dependencies**: Core dependencies only
- **Markers**: None (run by default)

### Integration Tests (Slow)
- **Location**: `tests/integration/`
- **Duration**: ~15 minutes
- **Dependencies**: Real APIs, may require API keys
- **Markers**: `@pytest.mark.integration`

### API Tests (Optional)
- **Location**: `tests/test_api/`, `tests/test_websocket*.py`
- **Duration**: <1 minute
- **Dependencies**: Requires `slowapi` (`pip install -e ".[api]"`)
- **Markers**: Skipped if slowapi not installed

## API Keys for Integration Tests

Integration tests for provider APIs require API keys:

```bash
# Free tier (no key required)
export COINGECKO_API_KEY=""  # Optional

# Requires API keys
export POLYGON_API_KEY="your_key_here"
export TWELVE_DATA_API_KEY="your_key_here"
export CRYPTOCOMPARE_API_KEY="your_key_here"
export FINNHUB_API_KEY="your_key_here"
export TIINGO_API_KEY="your_key_here"
export EODHD_API_KEY="your_key_here"
export ALPHA_VANTAGE_API_KEY="your_key_here"
export OANDA_API_KEY="your_key_here"
export DATABENTO_API_KEY="your_key_here"
```

## Test Markers

| Marker | Description | Skip by Default? |
|--------|-------------|------------------|
| `integration` | Real API calls, slow tests | ✅ Yes |
| `slow` | Tests taking >10 seconds | ✅ Yes |
| `requires_api_key` | Needs specific API key | ⚠️ If key missing |
| `expensive` | High API costs | ⚠️ In CI only |

## Common Test Commands

```bash
# Run specific test file
pytest tests/test_core_models.py

# Run specific test
pytest tests/test_core_models.py::test_function_name

# Run tests matching pattern
pytest -k "yahoo"

# Stop after first failure
pytest -x

# Show print statements
pytest -s

# Generate coverage report
pytest --cov=ml4t-data --cov-report=html
open htmlcov/index.html
```

## CI/CD

GitHub Actions runs:
- **PR checks**: Unit tests only (fast)
- **Main branch**: Unit + integration tests (with API keys)

## Troubleshooting

### "ModuleNotFoundError: No module named 'slowapi'"
```bash
pip install -e ".[api]"
```

### "ModuleNotFoundError: No module named 'databento'"
```bash
pip install -e ".[databento]"
```

### Tests taking too long
```bash
# Make sure you're running unit tests only
pytest -m "not integration and not slow"
```

### Integration tests skipped
```bash
# Check if API keys are set
env | grep API_KEY

# Run with specific marker
pytest -m integration
```

### Parallel execution issues (debugging)
```bash
# Disable parallel execution for clearer error messages
pytest -n 0

# Use less workers to reduce resource contention
pytest -n 4

# Run single test file
pytest tests/test_specific_file.py -n 0
```

### Test output not showing (parallel mode)
```bash
# Use -s with sequential execution to see print statements
pytest -n 0 -s
```
