"""Futures data CLI commands."""

from __future__ import annotations

import click
from rich import box
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from .utils import console, validate_date


@click.command("download-futures")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="YAML configuration file for futures download",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show cost estimate without downloading",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-download existing data",
)
@click.option(
    "--product",
    "-p",
    multiple=True,
    help="Download specific product(s) only",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel downloads (default: 1, recommended: 4)",
)
@click.pass_context
def download_futures(ctx, config, dry_run, force, product, parallel):
    """Download futures data from Databento.

    Downloads historical futures data including OHLCV bars, instrument
    definitions, and statistics (settlement prices, open interest).

    Examples:

        # Preview download cost
        ml4t-data download-futures -c configs/futures_download.yaml --dry-run

        # Full download
        ml4t-data download-futures -c configs/futures_download.yaml

        # Download specific products only
        ml4t-data download-futures -c configs/futures_download.yaml -p ES -p CL

        # Re-download existing data
        ml4t-data download-futures -c configs/futures_download.yaml --force
    """
    from ml4t.data.futures import load_yaml_config

    quiet = ctx.obj.get("quiet", False)

    try:
        # Load configuration
        download_config = load_yaml_config(config)

        import ml4t.data.futures as futures

        futures.require_databento("FuturesDownloader")
        downloader_class = vars(futures)["FuturesDownloader"]

        # Override products if specific ones requested
        if product:
            download_config.products = list(product)

        # Initialize downloader
        downloader = downloader_class(download_config)

        # Show cost estimate
        cost = downloader.estimate_cost()

        if not quiet:
            console.print("\n[bold cyan]Futures Download Configuration[/bold cyan]")
            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Products", str(cost["products"]))
            table.add_row("Schemas", str(cost["schemas"]))
            table.add_row("Years", str(cost["years"]))
            table.add_row("Date Range", f"{download_config.start} -> {download_config.end}")
            table.add_row("Storage Path", str(download_config.storage_path))
            table.add_row("Estimated Cost", f"${cost['estimated_total_usd']:.2f}")
            console.print(table)

        if dry_run:
            console.print("\n[yellow]Dry run - no data will be downloaded[/yellow]")

            # Show existing data
            existing = downloader.list_downloaded()
            if any(existing.values()):
                console.print("\n[bold]Existing Data:[/bold]")
                for schema, products_list in existing.items():
                    if products_list:
                        console.print(f"  {schema}: {len(products_list)} product(s)")
            return

        # Confirm download
        if not quiet:
            console.print()
            if not click.confirm("Proceed with download?", default=True):
                console.print("[yellow]Download cancelled[/yellow]")
                return

        # Run download
        console.print("\n[bold]Starting download...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading", total=len(download_config.get_product_list()))

            # Custom progress callback
            def progress_callback(_product_name: str, _bytes_downloaded: int = 0) -> None:
                progress.advance(task)

            downloader.progress.on_complete = progress_callback

            # Run download (parallel or sequential)
            if parallel > 1:
                result = downloader.download_all_parallel(
                    max_workers=parallel,
                    skip_existing=not force,
                )
            else:
                result = downloader.download_all(
                    skip_existing=not force,
                    continue_on_error=True,
                )

        # Show summary
        console.print("\n[bold cyan]Download Summary[/bold cyan]")
        summary_table = Table(show_header=True, box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Completed", str(len(result.completed_products)))
        summary_table.add_row("Failed", str(len(result.failed_products)))
        console.print(summary_table)

        if result.failed_products:
            console.print("\n[red]Failed Products:[/red]")
            for prod, error in result.failed_products.items():
                console.print(f"  {prod}: {error}")

    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise click.Abort()


@click.command("update-futures")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="YAML configuration file for futures",
)
@click.option(
    "--end-date",
    callback=validate_date,
    default=None,
    help="End date for update (default: today)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be updated without downloading",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.pass_context
def update_futures(ctx, config, end_date, dry_run, yes):
    """Update existing futures data with latest available data.

    This command incrementally updates futures data by:
    1. Finding the latest date in existing data
    2. Downloading only new data since that date
    3. Merging new data with existing data

    This is much cheaper than re-downloading the full dataset!

    Examples:

        # Update to today
        ml4t-data update-futures -c configs/ml4t_futures.yaml

        # Update to specific date
        ml4t-data update-futures -c configs/ml4t_futures.yaml --end-date 2025-12-31

        # Preview update without downloading
        ml4t-data update-futures -c configs/ml4t_futures.yaml --dry-run

    Book readers in 2027+ can use this to get the latest data:

        ml4t-data update-futures -c configs/ml4t_futures.yaml

    This will automatically fetch all data since the last download.
    """
    from datetime import datetime, timedelta

    from ml4t.data.futures import load_yaml_config

    quiet = ctx.obj.get("quiet", False)
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load configuration
        download_config = load_yaml_config(config)

        import ml4t.data.futures as futures

        futures.require_databento("FuturesDownloader")
        downloader_class = vars(futures)["FuturesDownloader"]

        # Initialize downloader
        downloader = downloader_class(download_config)

        # Find latest existing date
        latest_date = downloader.get_latest_date()

        if not quiet:
            console.print("\n[bold cyan]Futures Update Status[/bold cyan]")
            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Storage Path", str(download_config.storage_path))
            table.add_row("Products", str(len(download_config.get_product_list())))

            if latest_date:
                table.add_row("Latest Data", latest_date.strftime("%Y-%m-%d"))
                table.add_row("Update To", end_date or datetime.now().strftime("%Y-%m-%d"))
            else:
                table.add_row("Status", "[yellow]No existing data - will do full download[/yellow]")

            console.print(table)

        if dry_run:
            console.print("\n[yellow]Dry run - no data will be downloaded[/yellow]")

            # Show existing data summary
            existing = downloader.list_downloaded()
            if any(existing.values()):
                console.print("\n[bold]Existing Data:[/bold]")
                for schema, products_list in existing.items():
                    if products_list:
                        console.print(f"  {schema}: {len(products_list)} product(s)")
            return

        if latest_date is None:
            if not quiet:
                console.print("\n[yellow]No existing data found. Running full download...[/yellow]")
                console.print("[dim]For initial download, use: ml4t-data download-futures[/dim]")
            return

        # Check if already up to date
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        next_day = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")

        if next_day >= end:
            console.print(
                f"\n[green]OK Data already up to date (latest: {latest_date.strftime('%Y-%m-%d')})[/green]"
            )
            return

        # Confirm update
        if not quiet and not yes:
            console.print()
            if not click.confirm(f"Update from {next_day} to {end}?", default=True):
                console.print("[yellow]Update cancelled[/yellow]")
                return

        # Run update
        console.print("\n[bold]Updating futures data...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Updating", total=len(download_config.get_product_list()))

            # Custom progress callback
            def progress_callback(_product_name: str, _bytes_downloaded: int = 0) -> None:
                progress.advance(task)

            downloader.progress.on_complete = progress_callback

            result = downloader.update(end_date=end)

        # Show summary
        console.print("\n[bold cyan]Update Summary[/bold cyan]")
        summary_table = Table(show_header=True, box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Updated", str(len(result.completed_products)))
        summary_table.add_row("Failed", str(len(result.failed_products)))

        # Get new latest date
        new_latest = downloader.get_latest_date()
        if new_latest:
            summary_table.add_row("New Latest Date", new_latest.strftime("%Y-%m-%d"))

        console.print(summary_table)

        if result.failed_products:
            console.print("\n[red]Failed Products:[/red]")
            for prod, error in result.failed_products.items():
                console.print(f"  {prod}: {error}")

    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise click.Abort()
