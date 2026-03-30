"""Configuration and system CLI commands."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich import box
from rich.table import Table

from ml4t.data import __version__
from ml4t.data.storage.metadata_tracker import MetadataTracker

from .utils import console


@click.command()
def version():
    """Show version information."""
    console.print(f"[bold]ML4T Data version:[/bold] {__version__}")
    console.print(f"[bold]Python:[/bold] {sys.version.split()[0]}")


@click.command()
def providers():
    """List available data providers."""
    table = Table(title="Available Data Providers", box=box.ROUNDED)

    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("API Key", style="yellow")

    # List all available providers (matches DataManager's provider registry)
    providers_list = [
        ("yahoo", "Yahoo Finance - Free US/Global equities", "No"),
        ("coingecko", "CoinGecko - Free crypto historical", "No"),
        ("binance_api", "Binance REST API - crypto spot/futures", "No"),
        ("binance_bulk", "Binance bulk archives - historical spot/futures", "No"),
        ("cryptocompare", "CryptoCompare - Crypto + paid features", "Optional"),
        ("databento", "DataBento - Professional derivatives", "Yes"),
        ("oanda", "OANDA - Professional forex", "Yes"),
        ("mock", "Mock Provider - Testing only", "No"),
    ]

    for provider, description, api_key in providers_list:
        table.add_row(provider, description, api_key)

    console.print(table)
    console.print("\n[bold]Usage:[/bold] ml4t-data fetch --provider <name> --symbol <symbol> ...")


@click.command("config")
@click.pass_context
def show_config(ctx):
    """Show current configuration."""
    storage_path = ctx.params.get("storage_path") or Path.cwd() / "data"

    table = Table(title="ML4T Data Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Storage Path", str(storage_path))
    table.add_row("Storage Strategy", "HiveStorage (partitioned Parquet)")
    table.add_row("Version", __version__)

    console.print(table)


@click.command()
@click.option("--storage-path", default=None, help="Storage directory")
@click.option("--stale-days", "-d", default=7, type=int, help="Days before data considered stale")
@click.option("--detailed", is_flag=True, help="Show detailed information")
def health(storage_path, stale_days, detailed):
    """Check health status of all datasets."""
    try:
        storage_path = Path(storage_path) if storage_path else Path.cwd() / "data"
        tracker = MetadataTracker(base_path=storage_path)

        summary = tracker.get_summary()

        if summary["total_datasets"] == 0:
            console.print("[yellow]No datasets found[/yellow]")
            return

        # Display summary
        table = Table(title="Dataset Health Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Datasets", str(summary["total_datasets"]))
        table.add_row("Total Updates", str(summary["total_updates"]))
        table.add_row("Unique Providers", str(summary["unique_providers"]))
        table.add_row("Unique Symbols", str(summary["unique_symbols"]))

        console.print(table)

        if detailed:
            # Show per-symbol status
            updates = tracker.list_updates()
            symbol_table = Table(title="Per-Symbol Status", box=box.SIMPLE)
            symbol_table.add_column("Symbol", style="cyan")
            symbol_table.add_column("Provider", style="white")
            symbol_table.add_column("Last Updated", style="yellow")
            symbol_table.add_column("Status", style="white")

            now = datetime.now()
            stale_threshold = now - timedelta(days=stale_days)

            for update in sorted(updates, key=lambda x: x.symbol):
                status = "OK Fresh" if update.timestamp > stale_threshold else "! Stale"
                symbol_table.add_row(
                    update.symbol, update.provider, update.timestamp.strftime("%Y-%m-%d"), status
                )

            console.print("\n")
            console.print(symbol_table)

    except Exception as e:
        console.print(f"[red]Error checking health: {e}[/red]")
        raise click.Abort()


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development)")
def server(host, port, reload):
    """Start the ML4T Data REST API server."""
    console.print(f"[bold]Starting ML4T Data API server on {host}:{port}[/bold]")

    if reload:
        console.print("[yellow]Auto-reload enabled (development mode)[/yellow]")

    try:
        from ml4t.data.api.main import run_server  # type: ignore[import-not-found]

        run_server(host=host, port=port, reload=reload)

    except ImportError:
        console.print(
            "[red]API dependencies not installed. Install with: pip install 'ml4t-data[api]'[/red]"
        )
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise click.Abort()


@click.command("show-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def show_completion(shell):
    """Show shell completion script.

    To enable completion:

    Bash:
        eval "$(_ML4T_DATA_COMPLETE=bash_source ml4t-data)"

    Zsh:
        eval "$(_ML4T_DATA_COMPLETE=zsh_source ml4t-data)"

    Fish:
        eval (env _ML4T_DATA_COMPLETE=fish_source ml4t-data)
    """
    import os
    import subprocess

    env = os.environ.copy()
    env["_QDATA_COMPLETE"] = f"{shell}_source"

    result = subprocess.run(
        [sys.executable, "-m", "mlquant.data.cli_interface"],
        env=env,
        capture_output=True,
        text=True,
    )

    console.print(result.stdout)
