"""Configuration and system CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich import box
from rich.table import Table

from ml4t.data import __version__
from ml4t.data.providers.registry import advertised_provider_specs

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

    for spec in advertised_provider_specs():
        table.add_row(spec.name, spec.description, spec.access_label)

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
    from click.shell_completion import get_completion_class

    completion_class = get_completion_class(shell)
    if completion_class is None:
        raise click.ClickException(f"Shell completion is not available for {shell}")

    from ml4t.data.cli import cli

    completion = completion_class(cli, {}, "ml4t-data", "_ML4T_DATA_COMPLETE")
    click.echo(completion.source())
