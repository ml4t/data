"""Batch operations CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml

from ml4t.data.data_manager import DataManager
from ml4t.data.storage import create_storage

from .utils import console, load_symbols_from_file


def _as_date_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_config_path(path: str | Path, config_path: Path) -> Path:
    """Resolve a path from the YAML file."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = config_path.parent / resolved
    return resolved.resolve()


def _build_storage_from_config(config_data: dict[str, Any], config_path: Path):
    """Create storage using the YAML storage section."""
    storage_config = dict(config_data.get("storage", {}))
    storage_path_value = storage_config.pop("path", None)
    if storage_path_value is None:
        storage_path_value = storage_config.pop("base_path", "./data")
    else:
        storage_config.pop("base_path", None)
    storage_path = _resolve_config_path(storage_path_value, config_path)
    strategy = storage_config.pop("strategy", "hive")

    allowed_options = {
        "compression",
        "partition_granularity",
        "partition_cols",
        "atomic_writes",
        "enable_locking",
        "metadata_tracking",
        "generate_profile",
    }
    storage_options = {key: storage_config[key] for key in allowed_options if key in storage_config}

    if str(storage_options.get("compression", "")).lower() in {"none", "null"}:
        storage_options["compression"] = None

    return create_storage(storage_path, strategy=strategy, **storage_options), storage_path


@click.command("update-all")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file (YAML)",
)
@click.option("--dataset", "-d", help="Update specific dataset (e.g., 'futures', 'spot')")
@click.option("--dry-run", is_flag=True, help="Show what would be updated without updating")
@click.pass_context
def update_all(ctx, config, dataset, dry_run):
    """Update all datasets from configuration file.

    Examples:

        # Update everything from config
        ml4t-data update-all -c ml4t-data.yaml

        # Update only futures
        ml4t-data update-all -c ml4t-data.yaml --dataset futures

        # Dry run to see what would be updated
        ml4t-data update-all -c ml4t-data.yaml --dry-run

    Dataset configuration supports two formats for symbols:

        # Inline list (good for small datasets)
        datasets:
          demo:
            provider: yahoo
            symbols: [AAPL, MSFT, GOOGL]

        # File reference (good for large datasets like S&P 500)
        datasets:
          sp500:
            provider: yahoo
            symbols_file: sp500.txt  # Relative to config file
    """
    verbose = ctx.obj.get("verbose", False)
    config_path = Path(config)

    try:
        # Load config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        storage, storage_path = _build_storage_from_config(cfg, config_path)
        console.print(f"[cyan]Storage:[/cyan] {storage_path}")

        manager = DataManager(storage=storage)

        # Get datasets to update
        datasets = cfg.get("datasets", {})
        if dataset:
            if dataset not in datasets:
                console.print(f"[red]Dataset '{dataset}' not found in config[/red]")
                raise click.Abort()
            datasets = {dataset: datasets[dataset]}

        console.print(f"\n[bold]Updating {len(datasets)} dataset(s)[/bold]\n")

        # Update each dataset
        for ds_name, ds_config in datasets.items():
            console.print(f"[bold cyan]=== {ds_name.upper()} ===[/bold cyan]")

            provider = ds_config["provider"]

            # Load symbols from inline list or file
            if "symbols" in ds_config:
                symbols = ds_config["symbols"]
            elif "symbols_file" in ds_config:
                symbols_file = ds_config["symbols_file"]
                console.print(f"Loading symbols from: {symbols_file}")
                try:
                    symbols = load_symbols_from_file(symbols_file, config_path.parent)
                    console.print(f"Loaded {len(symbols)} symbols")
                except FileNotFoundError as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue
            else:
                console.print(
                    f"[red]Dataset '{ds_name}' must have 'symbols' or 'symbols_file'[/red]"
                )
                continue

            console.print(f"Provider: {provider}")
            if len(symbols) <= 10:
                console.print(f"Symbols: {', '.join(symbols)}")
            else:
                console.print(
                    f"Symbols: {len(symbols)} symbols ({symbols[0]}, {symbols[1]}, ..., {symbols[-1]})"
                )

            if dry_run:
                console.print("[yellow]  (dry run - no updates performed)[/yellow]\n")
                continue

            # Extract additional config options
            frequency = ds_config.get("frequency", "daily")
            asset_class = ds_config.get("asset_class", "equities")
            lookback_days = ds_config.get("lookback_days", 7)
            fill_gaps = ds_config.get("fill_gaps", True)
            initial_start = _as_date_string(ds_config.get("start") or ds_config.get("start_date"))
            initial_end = _as_date_string(ds_config.get("end") or ds_config.get("end_date"))
            initial_load_days = ds_config.get("initial_load_days", 365)
            if initial_load_days is None:
                initial_load_days = 365

            # Update each symbol
            for symbol in symbols:
                console.print(f"\n  [cyan]>[/cyan] {symbol}...", end=" ")

                try:
                    key = manager.update(
                        symbol,
                        frequency=frequency,
                        asset_class=asset_class,
                        provider=provider,
                        lookback_days=lookback_days,
                        fill_gaps=fill_gaps,
                        initial_start=initial_start,
                        initial_end=initial_end,
                        initial_load_days=initial_load_days,
                    )
                    console.print(f"[green]OK[/green] {key}")

                except Exception as e:
                    console.print(f"[red]FAIL {e}[/red]")
                    if verbose:
                        console.print(f"[dim]{e}[/dim]")

            console.print()

        console.print("[bold green]OK Update complete![/bold green]")

    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config}[/red]")
        raise click.Abort()
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML config: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise click.Abort()
