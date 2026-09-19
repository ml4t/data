# Installation

## Requirements

`ml4t-data` supports CPython 3.12, 3.13, and 3.14 on Linux, macOS, and Windows.
Python 3.15 is temporarily excluded while upstream dependencies complete compatibility work.

## Install from PyPI

```bash
pip install ml4t-data
```

With uv:

```bash
uv add ml4t-data
```

## Install provider extras

The core package includes the synthetic provider and adapters that do not require separate Python
packages. Install only the extras needed by your data sources:

```bash
pip install "ml4t-data[yahoo]"
pip install "ml4t-data[databento]"
pip install "ml4t-data[oanda]"
pip install "ml4t-data[cot]"
```

`ml4t-data[all-providers]` installs all published provider extras. Python packages do not supply
vendor accounts, credentials, data licenses, or network access.

## Verify the installation

```python
import ml4t.data
from ml4t.data.providers import SyntheticProvider

data = SyntheticProvider(seed=42).fetch_ohlcv(
    "SYNTH", "2024-01-01", "2024-01-10", "daily"
)
assert ml4t.data.__version__
assert not data.is_empty()
```

## Install for development

```bash
git clone https://github.com/ml4t/data.git
cd data
uv sync --all-extras --all-groups
uv run pre-commit install
```

Run the repository gates before submitting a change:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check
uv run pytest tests/ -q
uv run mkdocs build --strict
uv build
```
