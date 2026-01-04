# User Guide

Complete documentation for ML4T Data features.

## Core Features

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } __CLI Reference__

    ---

    Command-line interface for fetching and managing data.

    [:octicons-arrow-right-24: CLI Reference](cli-reference.md)

-   :material-cog:{ .lg .middle } __Configuration__

    ---

    YAML configuration for automated data pipelines.

    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-sync:{ .lg .middle } __Incremental Updates__

    ---

    Efficient daily updates with gap detection.

    [:octicons-arrow-right-24: Updates](incremental-updates.md)

-   :material-shield-check:{ .lg .middle } __Data Quality__

    ---

    OHLC validation and anomaly detection.

    [:octicons-arrow-right-24: Quality](data-quality.md)

-   :material-database:{ .lg .middle } __Storage__

    ---

    Hive-partitioned Parquet storage system.

    [:octicons-arrow-right-24: Storage](storage.md)

</div>

## Performance

ML4T Data is optimized for speed:

| Operation | Performance |
|-----------|-------------|
| Sequential fetch (100 symbols) | 50-100s |
| Async batch fetch (100 symbols) | 5-15s |
| Load from storage (100 symbols) | 0.165s |

## Next Steps

- [API Reference](../api/index.md) - Auto-generated from source
- [Tutorials](../tutorials/index.md) - Step-by-step guides
- [Contributing](../contributing/index.md) - Add your own provider
