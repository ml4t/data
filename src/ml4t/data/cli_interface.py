"""Command-line interface for ML4T Data unified data management.

This module provides backwards compatibility by re-exporting from the new
modular CLI package structure.

The CLI has been refactored into focused modules:
    - cli/utils.py: Shared utilities (console, validation, file saving)
    - cli/core.py: Core data commands (fetch, update, validate, status, export, info, list)
    - cli/batch.py: Batch operations (update-all)
    - cli/futures.py: Futures commands (download-futures, update-futures)
    - cli/cot.py: COT data commands (download-cot)
    - cli/config.py: Configuration commands (version, providers, config)
"""

from __future__ import annotations

# Re-export the CLI and main entry point for backwards compatibility
from ml4t.data.cli import cli, main

# Re-export commonly used utilities for any code that imported from here
from ml4t.data.cli.utils import (
    console,
    create_progress_bar,
    load_symbols_from_file,
    print_error,
    print_success,
    print_warning,
    save_batch_results,
    save_dataframe,
    validate_date,
)

# Re-export core classes for backward compatibility with tests that patch these
from ml4t.data.data_manager import DataManager
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.metadata_tracker import MetadataTracker
from ml4t.data.update_manager import IncrementalUpdater

__all__ = [
    "cli",
    "main",
    "console",
    "validate_date",
    "save_dataframe",
    "save_batch_results",
    "load_symbols_from_file",
    "create_progress_bar",
    "print_error",
    "print_success",
    "print_warning",
    # Core classes for backward compatibility
    "DataManager",
    "HiveStorage",
    "MetadataTracker",
    "IncrementalUpdater",
]

if __name__ == "__main__":
    cli()
