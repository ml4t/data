"""CLI package for ML4T Data.

This package provides the command-line interface for ml4t-data, organized into
focused modules by functionality.

Modules:
    utils: Shared utilities (console, validation, file saving)
    core: Core data commands (fetch, update, validate, status, export, info, list)
    batch: Batch operations (update-all)
    futures: Futures commands (download-futures, update-futures)
    cot: COT data commands (download-cot)
    config: Configuration commands (version, providers, config, health)
"""

import click

from ml4t.data import __version__

from . import batch, config, core, cot, futures
from .utils import console


@click.group()
@click.version_option(version=__version__, prog_name="ml4t-data")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """ML4T Data - Unified Financial Data Management.

    A comprehensive tool for fetching, updating, and managing financial data
    from multiple providers with intelligent caching and incremental updates.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    if not quiet:
        console.print(f"[bold cyan]ML4T Data v{__version__}[/bold cyan]")


# Register commands from submodules
cli.add_command(core.fetch)
cli.add_command(core.update)
cli.add_command(core.validate)
cli.add_command(core.status)
cli.add_command(core.export)
cli.add_command(core.info)
cli.add_command(core.list_data, name="list")

cli.add_command(batch.update_all, name="update-all")

cli.add_command(futures.download_futures, name="download-futures")
cli.add_command(futures.update_futures, name="update-futures")

cli.add_command(cot.download_cot, name="download-cot")

cli.add_command(config.version)
cli.add_command(config.providers)
cli.add_command(config.show_config, name="config")
cli.add_command(config.health)
cli.add_command(config.show_completion, name="show-completion")


def main():
    """Entry point for the CLI."""
    cli()


__all__ = ["cli", "main"]
