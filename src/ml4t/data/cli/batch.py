"""Batch operations CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from ml4t.data.config import DataConfig, load_config
from ml4t.data.config.models import StorageConfig
from ml4t.data.data_manager import DataManager
from ml4t.data.storage import create_storage

from .utils import console, load_symbols_from_file


def _as_date_string(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _resolve_config_path(path: str | Path, config_path: Path) -> Path:
    """Resolve a path from the YAML file."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = config_path.parent / resolved
    return resolved.resolve()


def _build_storage_from_config(config_data: DataConfig, config_path: Path):
    """Create storage using the YAML storage section."""
    if not isinstance(config_data.storage, StorageConfig):
        raise ValueError("Unsupported storage configuration")
    configured = config_data.storage
    storage_path = _resolve_config_path(configured.base_path, config_path)
    strategy = configured.strategy.value
    storage_options: dict[str, Any] = {}
    for field in (
        "compression",
        "partition_granularity",
        "partition_cols",
        "atomic_writes",
        "metadata_tracking",
    ):
        if field in configured.model_fields_set:
            value = getattr(configured, field)
            storage_options[field] = value.value if hasattr(value, "value") else value

    if storage_options.get("compression") == "none":
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
          - name: demo
            provider: yahoo
            symbols: [AAPL, MSFT, GOOGL]

        # File reference (good for large datasets like S&P 500)
        datasets:
          - name: sp500
            provider: yahoo
            symbols_file: sp500.txt  # Relative to config file
    """
    verbose = ctx.obj.get("verbose", False)
    config_path = Path(config)

    try:
        cfg = load_config(config_path)

        storage, storage_path = _build_storage_from_config(cfg, config_path)
        console.print(f"[cyan]Storage:[/cyan] {storage_path}")

        manager = DataManager(config_path=str(config_path), storage=storage)

        # Get datasets to update
        datasets = {configured.name: configured for configured in cfg.datasets}
        if dataset:
            if dataset not in datasets:
                console.print(f"[red]Dataset '{dataset}' not found in config[/red]")
                raise click.Abort()
            datasets = {dataset: datasets[dataset]}

        console.print(f"\n[bold]Updating {len(datasets)} dataset(s)[/bold]\n")

        # Update each dataset
        for ds_name, ds_config in datasets.items():
            console.print(f"[bold cyan]=== {ds_name.upper()} ===[/bold cyan]")

            provider = ds_config.provider

            # Load symbols from inline list or file
            if ds_config.symbols:
                symbols = ds_config.symbols
            elif ds_config.symbols_file:
                symbols_file = ds_config.symbols_file
                console.print(f"Loading symbols from: {symbols_file}")
                try:
                    symbols = load_symbols_from_file(symbols_file, config_path.parent)
                    console.print(f"Loaded {len(symbols)} symbols")
                except FileNotFoundError as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue
            elif ds_config.universe:
                universe = cfg.get_universe(ds_config.universe)
                if universe is None:
                    raise ValueError(
                        f"Dataset '{ds_name}' references unknown universe '{ds_config.universe}'"
                    )
                symbols = universe.symbols
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
            frequency = ds_config.frequency.value
            asset_class = ds_config.asset_class.value
            lookback_days = ds_config.lookback_days
            fill_gaps = ds_config.fill_gaps
            initial_start = _as_date_string(ds_config.start_date)
            initial_end = _as_date_string(ds_config.end_date)
            initial_load_days = ds_config.initial_load_days

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
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise click.Abort()
