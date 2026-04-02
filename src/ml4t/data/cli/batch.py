"""Batch operations CLI commands."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from ml4t.data.data_manager import DataManager
from ml4t.data.paths import default_ml4t_data_path, resolve_ml4t_data_path
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage

from .utils import console, load_symbols_from_file


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

        # Get storage path
        storage_path = resolve_ml4t_data_path(
            ".",
            default_path=default_ml4t_data_path(),
            configured_path=cfg.get("storage", {}).get("path"),
            config=cfg,
            config_dir=config_path.parent,
        )
        console.print(f"[cyan]Storage:[/cyan] {storage_path}")

        # Initialize storage and manager
        storage = HiveStorage(config=StorageConfig(base_path=str(storage_path)))
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

            # Update each symbol
            for symbol in symbols:
                console.print(f"\n  [cyan]>[/cyan] {symbol}...", end=" ")

                try:
                    key = manager.update(
                        symbol,
                        frequency=frequency,
                        asset_class=provider,
                        provider=provider,
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
