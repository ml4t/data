---
hide:
  - navigation
---

# ML4T Data

Market data acquisition, storage, and update workflows for machine learning for trading.

`ml4t-data` provides one interface for acquiring provider data, validating observations, writing
local datasets, and updating those datasets incrementally. Provider adapters cover exchange,
vendor, public, and synthetic data sources. Each provider documents its own credentials, rate
limits, licensing, and optional dependencies.

<div class="grid cards" markdown>

-   :material-play-circle:{ .lg .middle } __First successful use__

    ---

    Install the wheel and generate deterministic OHLCV data without credentials or network access.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-database:{ .lg .middle } __Choose a provider__

    ---

    Compare supported asset classes, authentication requirements, and provider capabilities.

    [:octicons-arrow-right-24: Provider selection](getting-started/provider-selection.md)

-   :material-folder-sync:{ .lg .middle } __Maintain datasets__

    ---

    Configure storage, detect gaps, and update existing data without replacing valid history.

    [:octicons-arrow-right-24: Incremental updates](user-guide/incremental-updates.md)

-   :material-code-braces:{ .lg .middle } __Use the public interface__

    ---

    Inspect provider protocols, configuration models, storage interfaces, and exceptions.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## Quick example

```python
from ml4t.data.providers import SyntheticProvider

provider = SyntheticProvider(seed=42)
data = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")

assert not data.is_empty()
```

The synthetic provider is local and deterministic. Live providers require network access and may
require an account, credentials, paid data, or an optional package.

## Start here

- [Installation](getting-started/installation.md) describes supported Python versions and extras.
- [Quickstart](getting-started/quickstart.md) teaches the first offline workflow.
- [User guide](user-guide/index.md) contains task-oriented storage, update, and validation guides.
- [Provider guide](providers/index.md) records provider capabilities and external-service boundaries.
- [Migration guide](getting-started/migration.md) maps former QLDM names to current interfaces.
- [Contributing](contributing/index.md) explains development setup and quality checks.
