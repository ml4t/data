# ML4T Data Installation Guide

## Quick Start

### Standard Installation

```bash
# Clone or navigate to ml4t-data
cd /path/to/ml4t-data

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install ml4t-data in editable mode with all dependencies
pip install -e .
```

### Verify Installation

```bash
# Check ml4t-data CLI is available
ml4t-data --version

# Verify session management dependencies
python -c "import pandas_market_calendars; print(f'✓ pandas-market-calendars {pandas_market_calendars.__version__}')"

# Check providers
ml4t-data providers
```

## Dependencies

ML4T Data has several categories of dependencies:

### Core Dependencies (Always Installed)

These are installed automatically with `pip install -e .`:

- **Data Processing**: polars, pandas, numpy, pyarrow
- **HTTP/Networking**: httpx, tenacity, pybreaker
- **Configuration**: pyyaml, click, python-dotenv, pydantic-settings
- **Utilities**: structlog, platformdirs, filelock, rich
- **Session Management**: pandas-market-calendars (≥4.3.0)
- **REST API**: fastapi, uvicorn, websockets

### Provider-Specific Dependencies (Optional)

Install only the providers you need:

```bash
# Yahoo Finance
pip install -e ".[yahoo]"

# DataBento
pip install -e ".[databento]"

# OANDA
pip install -e ".[oanda]"

# CryptoCompare
pip install -e ".[cryptocompare]"

# EODHD
pip install -e ".[eodhd]"

# Binance
pip install -e ".[binance]"

# Install multiple providers
pip install -e ".[yahoo,databento,cryptocompare]"

# Install ALL providers
pip install -e ".[all]"
```

### Development Dependencies (Optional)

For contributing to ml4t-data:

```bash
pip install -e ".[dev]"
```

Includes: pytest, pytest-cov, ruff, mypy, black, pre-commit

## Updating Installation

If you've already installed ml4t-data but the venv is missing new dependencies (like `pandas-market-calendars`), you need to reinstall:

```bash
# Option 1: Reinstall in editable mode (preserves existing packages)
pip install -e . --force-reinstall --no-deps
pip install -e .

# Option 2: Recreate venv from scratch (clean slate)
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Session Management Requirement

**Important**: Session management features (session date assignment, session completion) require `pandas-market-calendars`.

This is a **core dependency** and should be installed automatically. If you see:

```
ModuleNotFoundError: No module named 'pandas_market_calendars'
```

Then your installation is incomplete. Reinstall ml4t-data:

```bash
pip install -e .
```

## Platform-Specific Notes

### Linux/macOS

Standard installation should work out of the box:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows

Use PowerShell or Command Prompt:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### Docker

If running in Docker container:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

CMD ["ml4t-data", "--help"]
```

## Troubleshooting

### "No module named 'pip'"

Your venv is broken. Recreate it:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### "ml4t-data: command not found"

The ml4t-data CLI is not in PATH. Either:

1. Activate the virtual environment: `source .venv/bin/activate`
2. Use the full path: `.venv/bin/ml4t-data`
3. Reinstall: `pip install -e .`

### "pandas-market-calendars not found"

This is a core dependency. Reinstall:

```bash
pip install -e . --force-reinstall
```

### Import Errors for Optional Dependencies

If you see errors like:

```
ImportError: yfinance is required for YahooProvider
```

Install the provider-specific dependencies:

```bash
pip install -e ".[yahoo]"
```

### Slow Installation

If `pip install` is slow, try using a faster resolver:

```bash
pip install -e . --use-feature=fast-deps
```

Or use a local mirror/cache:

```bash
pip install -e . --no-index --find-links=/path/to/packages
```

## Verifying Your Installation

Run this verification script:

```bash
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import polars
    print(f'✓ polars {polars.__version__}')
except ImportError as e:
    print(f'✗ polars: {e}')

try:
    import pandas_market_calendars as mcal
    print(f'✓ pandas-market-calendars {mcal.__version__}')
except ImportError as e:
    print(f'✗ pandas-market-calendars: {e}')

try:
    from ml4t-data import DataManager
    print('✓ ml4t-data.DataManager')
except ImportError as e:
    print(f'✗ ml4t-data.DataManager: {e}')

try:
    from ml4t.data.sessions import SessionAssigner
    print('✓ ml4t-data.sessions.SessionAssigner')
except ImportError as e:
    print(f'✗ ml4t-data.sessions.SessionAssigner: {e}')

print('\n✓ All core dependencies installed correctly')
"
```

Expected output:

```
Python: 3.11.x
✓ polars 0.20.x
✓ pandas-market-calendars 4.3.x
✓ ml4t-data.DataManager
✓ ml4t-data.sessions.SessionAssigner

✓ All core dependencies installed correctly
```

## Next Steps

After installation:

1. **Configure API keys** (for providers that need them):
   ```bash
   export DATABENTO_API_KEY="your_key"
   export CRYPTOCOMPARE_API_KEY="your_key"
   ```

2. **Test the CLI**:
   ```bash
   ml4t-data --help
   ml4t-data providers
   ml4t-data fetch --symbol BTC --start 2024-01-01 --end 2024-12-31
   ```

3. **Run examples**:
   ```bash
   python examples/wyden_cme_sessions_complete_workflow.py
   python examples/nasdaq_bars_sessions.py
   ```

4. **Read documentation**:
   - `README.md` - Overview and quick start
   - `docs/SESSION_MANAGEMENT.md` - Session date assignment
   - `docs/PROVIDERS.md` - Provider-specific guides

## Support

If installation issues persist:

1. Check Python version: `python --version` (requires ≥3.9)
2. Check pip version: `pip --version` (update with `pip install --upgrade pip`)
3. Try creating a fresh venv in a different location
4. Check for conflicting system packages
5. Review error messages for specific missing dependencies

For provider-specific issues, see `docs/PROVIDERS.md`.
