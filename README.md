# ml4t-data

[![PyPI](https://img.shields.io/pypi/v/ml4t-data)](https://pypi.org/project/ml4t-data/)
[![Python 3.12-3.14](https://img.shields.io/badge/python-3.12--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Market data acquisition, storage, and update workflows for machine learning for trading.

Use `ml4t-data` to fetch and validate market data, keep local datasets current, and provide
repeatable inputs to research and trading workflows. It includes adapters for equities, futures,
foreign exchange, crypto assets, macroeconomic series, factors, and prediction markets.

## Installation

`ml4t-data` supports CPython 3.12, 3.13, and 3.14 on Linux, macOS, and Windows. Python 3.15 is
temporarily excluded while upstream dependencies complete their compatibility work.

```bash
pip install ml4t-data
```

Provider-specific integrations are optional. Install `ml4t-data[yahoo]`,
`ml4t-data[databento]`, `ml4t-data[oanda]`, or `ml4t-data[cot]` when those adapters are
needed. Some providers require network access, an account, credentials, or metered data. The
[provider guide](https://www.ml4trading.io/docs/data/providers/) records each boundary.

## Quick start

The synthetic provider exercises the public OHLCV interface without network access or credentials.

```python
from ml4t.data.providers import SyntheticProvider

provider = SyntheticProvider(seed=42)
data = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")

assert not data.is_empty()
assert {"timestamp", "symbol", "open", "high", "low", "close", "volume"} <= set(data.columns)
```

Set `ML4T_DATA_PATH` to choose the root directory for local datasets. Without an explicit path,
the library uses `./data` in the current working directory.

## Migration from QLDM names

Existing deployments may continue to use `QLDM_DATA_ROOT` and import `QldmError`. Both names
emit `DeprecationWarning` and will not be removed before version 1.0.

- Replace `QLDM_DATA_ROOT` with `ML4T_DATA_PATH`.
- Replace `QldmError` with `ML4TDataError` from `ml4t.data.core.exceptions`.

## Documentation and support

- [Documentation](https://www.ml4trading.io/docs/data/)
- [Issue tracker](https://github.com/ml4t/data/issues)
- [Release notes](https://github.com/ml4t/data/releases)
- [License](LICENSE)

The library supplies data to [ml4t-engineer](https://github.com/ml4t/engineer) feature workflows and
[ml4t-backtest](https://github.com/ml4t/backtest) simulations. Neither package is required to use
`ml4t-data`.

## Development

```bash
git clone https://github.com/ml4t/data.git
cd data
uv sync --all-extras --all-groups
uv run pre-commit install
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check
uv run pytest tests/ -q
uv run mkdocs build --strict
uv build
```

Pull requests must preserve offline deterministic tests. Provider contract tests that require
credentials run in explicit integration lanes.
