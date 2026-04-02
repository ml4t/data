"""COT (Commitment of Traders) data CLI commands."""

from __future__ import annotations

from pathlib import Path

import click
import polars as pl
from rich import box
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from ml4t.data.paths import default_ml4t_data_path, resolve_ml4t_data_path

from .utils import console


@click.command("download-cot")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="YAML configuration file for COT download",
)
@click.option(
    "--products",
    "-p",
    multiple=True,
    help="Product code(s) to download (e.g., ES, CL, GC)",
)
@click.option(
    "--start-year",
    type=int,
    default=2020,
    help="Start year (default: 2020)",
)
@click.option(
    "--end-year",
    type=int,
    default=None,
    help="End year (default: current year)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./data/cot)",
)
@click.option(
    "--list-products",
    is_flag=True,
    help="List all available product codes",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be downloaded without downloading",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-download existing data",
)
@click.pass_context
def download_cot(
    ctx, config, products, start_year, end_year, output, list_products, dry_run, force
):
    """Download CFTC Commitment of Traders (COT) data.

    COT reports provide weekly positioning data broken down by trader type
    (hedge funds, commercials, etc.). This data is FREE from the CFTC.

    Examples:

        # List available products
        ml4t-data download-cot --list-products

        # Download specific products
        ml4t-data download-cot -p ES -p CL -p GC

        # Download from 2020 to current year
        ml4t-data download-cot -p ES --start-year 2020

        # Download from config file
        ml4t-data download-cot -c configs/cot_download.yaml

        # Preview what would be downloaded
        ml4t-data download-cot -p ES -p CL --dry-run
    """
    from ml4t.data.cot import PRODUCT_MAPPINGS, COTConfig, COTFetcher

    quiet = ctx.obj.get("quiet", False)
    verbose = ctx.obj.get("verbose", False)

    # List products if requested
    if list_products:
        console.print("\n[bold cyan]Available COT Product Codes[/bold cyan]\n")

        # Group by report type
        financial = []
        commodity = []

        for code, mapping in PRODUCT_MAPPINGS.items():
            info = (code, mapping.description, mapping.cot_name[:50])
            if "financial_futures" in mapping.report_type:
                financial.append(info)
            else:
                commodity.append(info)

        # Financial futures table
        console.print("[bold]Financial Futures (TFF Report)[/bold]")
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Code", style="cyan", width=6)
        table.add_column("Description", style="white", width=20)
        table.add_column("CFTC Market Name", style="dim", width=50)

        for code, desc, cot_name in sorted(financial):
            table.add_row(code, desc, cot_name)
        console.print(table)

        # Commodity futures table
        console.print("\n[bold]Commodity Futures (Disaggregated Report)[/bold]")
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Code", style="cyan", width=6)
        table.add_column("Description", style="white", width=20)
        table.add_column("CFTC Market Name", style="dim", width=50)

        for code, desc, cot_name in sorted(commodity):
            table.add_row(code, desc, cot_name)
        console.print(table)

        console.print(f"\n[dim]Total: {len(PRODUCT_MAPPINGS)} products available[/dim]")
        return

    try:
        # Build config from CLI options or file
        if config:
            from ml4t.data.cot import load_cot_config

            cot_config = load_cot_config(config)
            # Override with CLI options if provided
            if products:
                cot_config.products = list(products)
            if end_year:
                cot_config.end_year = end_year
            if output:
                cot_config.storage_path = Path(output).expanduser()
        else:
            # Build config from CLI options
            if not products:
                console.print("[red]Error: Specify products with -p or use --config[/red]")
                console.print("[dim]Use --list-products to see available codes[/dim]")
                raise click.Abort()

            from datetime import datetime

            cot_config = COTConfig(
                products=list(products),
                start_year=start_year,
                end_year=end_year or datetime.now().year,
                storage_path=(
                    Path(output).expanduser()
                    if output is not None
                    else resolve_ml4t_data_path(
                        "cot",
                        default_path=default_ml4t_data_path("cot"),
                    )
                ),
            )

        # Validate products
        invalid_products = [p for p in cot_config.products if p not in PRODUCT_MAPPINGS]
        if invalid_products:
            console.print(f"[red]Unknown product codes: {invalid_products}[/red]")
            console.print("[dim]Use --list-products to see available codes[/dim]")
            raise click.Abort()

        # Show configuration
        if not quiet:
            console.print("\n[bold cyan]COT Download Configuration[/bold cyan]")
            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Products", ", ".join(cot_config.products))
            table.add_row("Years", f"{cot_config.start_year} - {cot_config.end_year}")
            table.add_row("Storage Path", str(cot_config.storage_path))
            table.add_row("Cost", "[bold green]FREE[/bold green] (CFTC public data)")
            console.print(table)

        if dry_run:
            console.print("\n[yellow]Dry run - no data will be downloaded[/yellow]")

            # Show what would be downloaded
            console.print("\n[bold]Products to download:[/bold]")
            for product in cot_config.products:
                mapping = PRODUCT_MAPPINGS[product]
                console.print(f"  {product}: {mapping.description} ({mapping.report_type})")

            # Check existing data
            existing = []
            for product in cot_config.products:
                path = cot_config.storage_path / f"product={product}" / "data.parquet"
                if path.exists():
                    existing.append(product)

            if existing:
                console.print(f"\n[dim]Already downloaded: {', '.join(existing)}[/dim]")
            return

        # Initialize fetcher and download
        fetcher = COTFetcher(cot_config)

        console.print("\n[bold]Downloading COT data...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading", total=len(cot_config.products))

            results = {}
            errors = {}

            for product in cot_config.products:
                progress.update(task, description=f"Fetching {product}...")

                output_path = cot_config.storage_path / f"product={product}" / "data.parquet"

                if not force and output_path.exists():
                    progress.update(task, advance=1)
                    results[product] = output_path
                    continue

                try:
                    df = fetcher.fetch_product(product)
                    if not df.is_empty():
                        path = fetcher.save_to_hive(df, product)
                        results[product] = path
                    else:
                        errors[product] = "No data found"
                except Exception as e:
                    errors[product] = str(e)
                    if verbose:
                        import traceback

                        console.print(f"\n[red]{traceback.format_exc()}[/red]")

                progress.advance(task)

        # Show summary
        console.print("\n[bold cyan]Download Summary[/bold cyan]")
        summary_table = Table(show_header=True, box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Completed", str(len(results)))
        summary_table.add_row("Failed", str(len(errors)))
        summary_table.add_row("Storage Path", str(cot_config.storage_path))
        console.print(summary_table)

        if errors:
            console.print("\n[red]Failed Products:[/red]")
            for product, error in errors.items():
                console.print(f"  {product}: {error}")

        if results and not quiet:
            console.print("\n[bold]Downloaded Data:[/bold]")
            for product, path in results.items():
                # Get row count
                df = pl.read_parquet(path)
                console.print(f"  {product}: {len(df)} rows ({path})")

    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise click.Abort()
