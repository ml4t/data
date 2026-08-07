"""Core CLI commands for data operations.

Commands: fetch, update, validate, status, export, info, list
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click
import pandas as pd
import polars as pl
from rich import box
from rich.table import Table

from ml4t.data.config import load_config
from ml4t.data.data_manager import DataManager
from ml4t.data.managers.metadata_manager import MetadataManager
from ml4t.data.storage.backend import StorageBackend, StorageConfig
from ml4t.data.storage.hive import HiveStorage

from .batch import _as_date_string, _build_storage_from_config
from .utils import (
    console,
    create_progress_bar,
    load_symbols_from_file,
    print_error,
    print_success,
    save_batch_results,
    save_dataframe,
    validate_date,
)


def _resolve_storage(
    config_path: str | None,
    storage_path: str | None,
) -> tuple[StorageBackend, Path]:
    """Build configured storage for a CLI command."""
    if config_path and storage_path:
        raise click.UsageError("Use either --config or --storage-path, not both.")
    if config_path:
        resolved_config = Path(config_path).resolve()
        cfg = load_config(resolved_config)
        return _build_storage_from_config(cfg, resolved_config)
    resolved_storage_path = Path(storage_path or "./data").expanduser()
    return HiveStorage(StorageConfig(base_path=resolved_storage_path)), resolved_storage_path


def _collect_dataframe(data: Any) -> pl.DataFrame:
    """Materialize a DataManager result for CLI rendering and file output."""
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    raise TypeError(f"Expected a Polars DataFrame, received {type(data).__name__}")


def _parse_row_count(value: object) -> int | None:
    """Return a non-negative integral storage row count when one is encoded."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not parsed.is_integer() or parsed < 0:
        return None
    return int(parsed)


def _format_row_count(value: object) -> str:
    """Format a validated storage row count."""
    parsed = _parse_row_count(value)
    if parsed is None:
        return "-"
    return f"{parsed:,}"


def _storage_row_count(storage: StorageBackend, key: str) -> int:
    """Count stored rows without materializing the full dataset."""
    return int(storage.read(key).select(pl.len()).collect().item())


@click.command()
@click.option("--symbol", "-s", multiple=True, help="Symbol(s) to fetch")
@click.option(
    "--symbols-file",
    "-f",
    type=click.Path(exists=True),
    help="File containing symbols (one per line)",
)
@click.option("--start", callback=validate_date, help="Start date (YYYY-MM-DD)")
@click.option("--end", callback=validate_date, help="End date (YYYY-MM-DD)")
@click.option(
    "--frequency",
    default=None,
    type=click.Choice(["daily", "hourly", "weekly"]),
    help="Data frequency (default: configured value or daily)",
)
@click.option("--provider", "-p", help="Specific provider to use")
@click.option("--output", "-o", type=click.Path(), help="Output file path (.parquet or .csv)")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--dataset", help="Dataset from the configuration file")
@click.option("--progress", is_flag=True, help="Show progress bar")
@click.pass_context
def fetch(
    ctx,
    symbol,
    symbols_file,
    start,
    end,
    frequency,
    provider,
    output,
    config,
    dataset,
    progress,
):
    """Fetch financial data from providers.

    Examples:
        ml4t-data fetch -s BTC --provider cryptocompare --start 2024-01-01 --end 2024-01-31
        ml4t-data fetch -s BTC -s ETH -p cryptocompare --start 2024-01-01 --end 2024-01-31
        ml4t-data fetch -f symbols.txt --provider yahoo --start 2024-01-01 --end 2024-01-31
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    # Load configuration if provided
    if config:
        if not quiet:
            console.print(f"Loading configuration from {config}")
        configured = load_config(Path(config))
        if dataset:
            configured_dataset = configured.get_dataset(dataset)
            if configured_dataset is None:
                raise click.UsageError(f"Dataset '{dataset}' not found in {config}.")
        elif len(configured.datasets) == 1:
            configured_dataset = configured.datasets[0]
        else:
            raise click.UsageError("Use --dataset when the configuration has multiple datasets.")

        if not symbol and not symbols_file:
            symbol = list(configured_dataset.symbols)
            if configured_dataset.symbols_file:
                symbol.extend(load_symbols_from_file(configured_dataset.symbols_file))
            if configured_dataset.universe:
                universe = configured.get_universe(configured_dataset.universe)
                if universe is None:
                    raise click.UsageError(
                        f"Dataset '{configured_dataset.name}' references unknown universe "
                        f"'{configured_dataset.universe}'."
                    )
                symbol.extend(universe.symbols)
        start = start or _as_date_string(configured_dataset.start_date)
        end = end or _as_date_string(configured_dataset.end_date)
        frequency = frequency or configured_dataset.frequency.value
        provider = provider or configured_dataset.provider

    frequency = frequency or "daily"

    missing_options = [name for name, value in (("--start", start), ("--end", end)) if not value]
    if missing_options:
        options = " and ".join(f"'{name}'" for name in missing_options)
        raise click.UsageError(f"Missing option {options} unless supplied by --config.")
    assert isinstance(start, str)
    assert isinstance(end, str)

    # Collect symbols
    symbols = list(symbol)
    if symbols_file:
        with open(symbols_file) as f:
            file_symbols = [line.strip() for line in f if line.strip()]
            symbols.extend(file_symbols)
        if not quiet:
            console.print(f"Fetching {len(file_symbols)} symbols from file")

    if not symbols:
        console.print("[red]Error: No symbols specified[/red]")
        ctx.exit(1)

    try:
        dm = DataManager(config_path=config)

        if len(symbols) == 1:
            sym = symbols[0]
            if not quiet:
                console.print(f"Fetching {sym} from {start} to {end}")

            df = _collect_dataframe(
                dm.fetch(sym, start, end, frequency=frequency, provider=provider)
            )

            if not quiet:
                print_success(f"Fetched {len(df)} rows")

            if output:
                save_dataframe(df, output)
                if not quiet:
                    console.print(f"[green]Saved to {output}[/green]")
        else:
            if not quiet:
                console.print(f"Fetching {len(symbols)} symbols")

            failures: dict[str, str] = {}
            dm.validate_routes(list(symbols), provider)

            if progress and not quiet:
                with create_progress_bar() as progress_bar:
                    task = progress_bar.add_task("Fetching...", total=len(symbols))
                    results: dict[str, pl.DataFrame | None] = {}
                    for sym in symbols:
                        try:
                            results[sym] = _collect_dataframe(
                                dm.fetch(
                                    sym,
                                    start,
                                    end,
                                    frequency=frequency,
                                    provider=provider,
                                )
                            )
                            progress_bar.update(task, advance=1, description=f"Fetched {sym}")
                        except Exception as e:
                            failures[sym] = str(e)
                            if verbose:
                                console.print(
                                    f"[yellow]Warning: Failed to fetch {sym}: {e}[/yellow]"
                                )
                            results[sym] = None
                            progress_bar.update(task, advance=1)
            else:
                results = {}
                for sym in symbols:
                    try:
                        results[sym] = _collect_dataframe(
                            dm.fetch(
                                sym,
                                start,
                                end,
                                frequency=frequency,
                                provider=provider,
                            )
                        )
                    except Exception as e:
                        failures[sym] = str(e)
                        results[sym] = None
                        if verbose:
                            console.print(f"[yellow]Warning: Failed to fetch {sym}: {e}[/yellow]")

            successful_results = {
                symbol: data for symbol, data in results.items() if data is not None
            }
            if not successful_results:
                detail = next(iter(failures.values()), "all provider requests failed")
                raise ValueError(f"No symbols were fetched: {detail}")
            if not quiet:
                print_success(f"Successfully fetched {len(successful_results)} symbols")

            if output:
                save_batch_results(successful_results, output)
                if not quiet:
                    console.print(f"[green]Saved to {output}[/green]")

    except Exception as e:
        print_error(str(e), verbose, e)
        ctx.exit(1)


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to update")
@click.option(
    "--frequency",
    type=click.Choice(["daily", "hourly", "weekly"]),
    default="daily",
    show_default=True,
)
@click.option(
    "--asset-class",
    type=click.Choice(["equities", "crypto", "forex", "futures"]),
    default="equities",
    show_default=True,
)
@click.option("--lookback-days", type=click.IntRange(min=0), default=7, show_default=True)
@click.option("--fill-gaps/--no-fill-gaps", default=True, show_default=True)
@click.option("--provider", "-p", help="Provider to use for fetching")
@click.option(
    "--initial-start",
    "--start",
    "initial_start",
    callback=validate_date,
    help="Start date used only when the dataset does not exist",
)
@click.option(
    "--initial-end",
    "--end",
    "initial_end",
    callback=validate_date,
    help="End date used only when the dataset does not exist",
)
@click.option("--initial-load-days", type=click.IntRange(min=1), default=365, show_default=True)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Configuration file")
@click.option("--storage-path", type=click.Path(), help="Hive storage directory (default: ./data)")
@click.pass_context
def update(
    ctx,
    symbol,
    frequency,
    asset_class,
    lookback_days,
    fill_gaps,
    provider,
    initial_start,
    initial_end,
    initial_load_days,
    config_path,
    storage_path,
):
    """Perform incremental data updates."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        storage, _ = _resolve_storage(config_path, storage_path)
        key = f"{asset_class}/{frequency}/{symbol}"
        key_exists = storage.exists(key)
        if key_exists and (initial_start is not None or initial_end is not None):
            raise click.UsageError(
                "--initial-start and --initial-end apply only when the dataset does not exist."
            )
        rows_before = _storage_row_count(storage, key) if key_exists and not quiet else 0
        manager = DataManager(config_path=config_path, storage=storage)
        updated_key = manager.update(
            symbol,
            frequency=frequency,
            asset_class=asset_class,
            lookback_days=lookback_days,
            fill_gaps=fill_gaps,
            provider=provider,
            initial_start=initial_start,
            initial_end=initial_end,
            initial_load_days=initial_load_days,
        )
        rows_after = _storage_row_count(storage, updated_key) if not quiet else 0
        if not quiet:
            print_success(f"Updated {updated_key}")
            console.print(f"   Rows: {rows_before:,} -> {rows_after:,}")

    except click.ClickException:
        raise
    except Exception as e:
        print_error(str(e), verbose, e)
        ctx.exit(1)


@click.command()
@click.option("--symbol", "-s", help="Symbol to validate")
@click.option("--all", "validate_all", is_flag=True, help="Validate all symbols")
@click.option("--anomalies", is_flag=True, help="Run anomaly detection")
@click.option("--save-report", is_flag=True, help="Save anomaly report to disk")
@click.option(
    "--severity",
    default="warning",
    type=click.Choice(["info", "warning", "error", "critical"]),
    help="Minimum severity to display",
)
@click.option(
    "--frequency",
    type=click.Choice(["daily", "hourly", "weekly"]),
    default="daily",
    show_default=True,
    help="Stored data frequency",
)
@click.option(
    "--asset-class",
    type=click.Choice(["equities", "crypto", "forex", "futures"]),
    default="equities",
    show_default=True,
    help="Stored asset class",
)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Configuration file")
@click.option("--storage-path", type=click.Path(), help="Hive storage directory (default: ./data)")
@click.pass_context
def validate(
    ctx,
    symbol,
    validate_all,
    anomalies,
    save_report,
    severity,
    frequency,
    asset_class,
    config_path,
    storage_path,
):
    """Validate data quality and integrity."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        storage, _ = _resolve_storage(config_path, storage_path)

        keys: list[str] = []
        if validate_all:
            keys = storage.list_keys()
        elif symbol:
            keys = [f"{asset_class}/{frequency}/{symbol}"]
        else:
            console.print("[red]Error: Specify --symbol or --all[/red]")
            ctx.exit(1)

        total_issues = 0
        metadata_manager = MetadataManager(storage)

        for key in keys:
            metadata = metadata_manager.get_metadata_for_key(key) or {}
            sym = str(metadata.get("symbol") or key.rsplit("/", 1)[-1])
            if not quiet:
                console.print(f"Validating {sym}...")

            if not storage.exists(key):
                console.print(f"[yellow]  Symbol {sym} not found in storage[/yellow]")
                continue

            df = storage.read(key).collect()
            issues = []

            # Schema check
            required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
            elif not quiet:
                console.print("[green]  ✅ Schema validation passed[/green]")

            # OHLC check
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                invalid_high_low = df.filter(pl.col("high") < pl.col("low"))
                if len(invalid_high_low) > 0:
                    issues.append(f"High < Low in {len(invalid_high_low)} rows")

                if not issues and not quiet:
                    console.print("[green]  ✅ OHLC relationships valid[/green]")

            # Duplicates check
            if "timestamp" in df.columns:
                duplicate_count = len(df) - df["timestamp"].n_unique()
                if duplicate_count > 0:
                    issues.append(f"{duplicate_count} duplicate timestamps")

            if not quiet:
                console.print(f"  Total rows: {len(df):,}")

            if issues:
                console.print("[red]  ❌ Validation issues found:[/red]")
                for issue in issues:
                    console.print(f"    - {issue}")
                total_issues += len(issues)
            elif not quiet:
                console.print("[green]  ✅ All validations passed[/green]")

            # Anomaly detection
            if anomalies:
                if not quiet:
                    console.print("\n  🔍 Running anomaly detection...")

                try:
                    from ml4t.data.anomaly import (
                        AnomalyManager,
                        AnomalySeverity,
                    )

                    manager = AnomalyManager()
                    report = manager.analyze(
                        df,
                        symbol=sym,
                        asset_class=str(metadata.get("asset_class") or asset_class),
                    )

                    if severity != "info":
                        report = manager.filter_by_severity(report, severity)

                    if report.anomalies:
                        console.print(
                            f"  [yellow]⚠️  Found {len(report.anomalies)} anomalies[/yellow]"
                        )

                        severity_emoji = {
                            AnomalySeverity.INFO: "ℹ️",
                            AnomalySeverity.WARNING: "⚠️",
                            AnomalySeverity.ERROR: "❌",
                            AnomalySeverity.CRITICAL: "🚨",
                        }

                        for anom in report.anomalies[:5]:
                            emoji = severity_emoji.get(anom.severity, "❓")
                            console.print(
                                f"    {emoji} [{anom.severity.value.upper()}] {anom.type.value}"
                            )
                            console.print(f"       Date: {anom.timestamp}")
                            console.print(f"       {anom.message}")

                        if len(report.anomalies) > 5:
                            console.print(f"    ... and {len(report.anomalies) - 5} more anomalies")

                        if save_report:
                            report_path = manager.save_report(report, Path("./anomaly_reports"))
                            print_success(f"Report saved to: {report_path}")

                        total_issues += len(report.anomalies)
                    else:
                        console.print("  [green]✅ No anomalies detected[/green]")

                except ImportError:
                    console.print("  [yellow]⚠️  Anomaly detection module not available[/yellow]")

        if total_issues > 0:
            ctx.exit(1)

    except click.ClickException:
        raise
    except Exception as e:
        print_error(str(e), verbose, e)
        ctx.exit(1)


def _metadata_datetime(value: object) -> datetime | None:
    """Parse one storage metadata timestamp as an aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _metadata_health(metadata: dict[str, object], stale_days: int) -> str:
    """Classify a dataset from its canonical observation or update timestamp."""
    observed_through = _metadata_datetime(metadata.get("end_date") or metadata.get("last_updated"))
    if observed_through is None:
        return "error"
    return (
        "stale" if observed_through < datetime.now(UTC) - timedelta(days=stale_days) else "healthy"
    )


@click.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed status")
@click.option("--stale-days", default=7, type=click.IntRange(min=0), show_default=True)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Configuration file")
@click.option("--storage-path", type=click.Path(), help="Hive storage directory (default: ./data)")
@click.pass_context
def status(ctx, detailed, stale_days, config_path, storage_path):
    """Show system overview and health status."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        storage, resolved_storage_path = _resolve_storage(config_path, storage_path)

        metadata_manager = MetadataManager(storage)
        datasets: list[tuple[str, dict[str, object], str]] = []
        total_rows = 0
        for key in storage.list_keys():
            metadata = metadata_manager.get_metadata_for_key(key) or {}
            health = _metadata_health(metadata, stale_days)
            row_count = _parse_row_count(metadata.get("row_count"))
            if row_count is None:
                health = "error"
            else:
                total_rows += row_count
            datasets.append((key, metadata, health))

        status_counts = {
            name: sum(health == name for _, _, health in datasets)
            for name in ("healthy", "stale", "error")
        }

        if not quiet:
            table = Table(title="System Status", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Total Datasets", str(len(datasets)))
            table.add_row("Healthy", f"[green]{status_counts['healthy']}[/green]")
            table.add_row("Stale", f"[yellow]{status_counts['stale']}[/yellow]")
            table.add_row("Error", f"[red]{status_counts['error']}[/red]")
            table.add_row("Total Rows", f"{total_rows:,}")

            console.print(table)

        if detailed:
            detail_table = Table(title="Dataset Status", box=box.ROUNDED)
            detail_table.add_column("Status")
            detail_table.add_column("Key", style="dim")
            detail_table.add_column("Symbol", style="cyan")
            detail_table.add_column("Provider")
            detail_table.add_column("Rows", justify="right")
            detail_table.add_column("Data Through")
            detail_table.add_column("Last Updated")
            colors = {"healthy": "green", "stale": "yellow", "error": "red"}
            for key, metadata, health in datasets:
                row_count = metadata.get("row_count")
                rows = _format_row_count(row_count)
                detail_table.add_row(
                    f"[{colors[health]}]{health.title()}[/{colors[health]}]",
                    key,
                    str(metadata.get("symbol") or key.rsplit("/", 1)[-1]),
                    str(metadata.get("provider") or ""),
                    rows,
                    str(metadata.get("end_date") or "")[:10],
                    str(metadata.get("last_updated") or "")[:19],
                )
            console.print(detail_table)

        if verbose:
            console.print(f"\n[dim]Storage path: {resolved_storage_path}[/dim]")

    except click.ClickException:
        raise
    except Exception as e:
        print_error(str(e), verbose, e)
        ctx.exit(1)


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to export")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option(
    "--format",
    "-f",
    "format_type",
    default="csv",
    type=click.Choice(["csv", "json", "parquet"]),
    help="Export format",
)
@click.option(
    "--frequency",
    type=click.Choice(["daily", "hourly", "weekly"]),
    default="daily",
    show_default=True,
    help="Stored data frequency",
)
@click.option(
    "--asset-class",
    type=click.Choice(["equities", "crypto", "forex", "futures"]),
    default="equities",
    show_default=True,
    help="Stored asset class",
)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Configuration file")
@click.option("--storage-path", type=click.Path(), help="Hive storage directory (default: ./data)")
def export(symbol, output, format_type, frequency, asset_class, config_path, storage_path):
    """Export data to various formats (CSV, JSON, Parquet)."""
    try:
        storage, _ = _resolve_storage(config_path, storage_path)
        key = f"{asset_class}/{frequency}/{symbol}"

        console.print(f"[bold]Reading data for {symbol}...[/bold]")
        if not storage.exists(key):
            raise click.ClickException(f"No data found for {symbol}")
        df = storage.read(key).collect()

        if df.is_empty():
            raise click.ClickException(f"No data found for {symbol}")

        output_path = Path(output)
        console.print(f"[bold]Exporting to {output_path}...[/bold]")

        if format_type == "csv":
            df.write_csv(output_path)
        elif format_type == "json":
            df.write_json(output_path)
        elif format_type == "parquet":
            df.write_parquet(output_path)

        print_success(f"Exported {len(df)} rows to {output_path}")

    except click.ClickException:
        raise
    except Exception as e:
        print_error(str(e))
        raise click.Abort()


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to show info for")
@click.option(
    "--frequency",
    type=click.Choice(["daily", "hourly", "weekly"]),
    default="daily",
    show_default=True,
    help="Stored data frequency",
)
@click.option(
    "--asset-class",
    type=click.Choice(["equities", "crypto", "forex", "futures"]),
    default="equities",
    show_default=True,
    help="Stored asset class",
)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Configuration file")
@click.option("--storage-path", type=click.Path(), help="Hive storage directory (default: ./data)")
def info(symbol, frequency, asset_class, config_path, storage_path):
    """Show information about stored data."""
    try:
        storage, _ = _resolve_storage(config_path, storage_path)
        key = f"{asset_class}/{frequency}/{symbol}"
        if not storage.exists(key):
            raise click.ClickException(f"No data found for {symbol}")

        df = storage.read(key).collect()
        metadata = MetadataManager(storage).get_metadata_for_key(key) or {}

        table = Table(title=f"Data Info: {symbol}", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Symbol", symbol)
        table.add_row("Rows", str(len(df)))
        table.add_row("Date Range", f"{df['timestamp'].min()} to {df['timestamp'].max()}")
        table.add_row("Columns", ", ".join(df.columns))
        table.add_row("Provider", str(metadata.get("provider") or ""))
        table.add_row("Last Updated", str(metadata.get("last_updated") or "")[:19])
        table.add_row("Frequency", str(metadata.get("frequency") or frequency))

        console.print(table)
        console.print("\n[bold]Data Preview:[/bold]")
        console.print(df.head(5))

    except click.ClickException:
        raise
    except Exception as e:
        print_error(str(e))
        raise click.Abort()


@click.command("list")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file (YAML)")
@click.option("--storage-path", type=click.Path(exists=True), help="Storage directory")
@click.pass_context
def list_data(_ctx, config, storage_path):
    """List all stored datasets."""
    if config and storage_path:
        raise click.UsageError("Use either --config or --storage-path, not both.")
    try:
        if config:
            from ml4t.data.config import load_config

            config_path = Path(config).resolve()
            cfg = load_config(config_path)
            storage, storage_path = _build_storage_from_config(cfg, config_path)
        elif storage_path:
            storage_path = Path(storage_path).expanduser()
            storage = HiveStorage(StorageConfig(base_path=storage_path))
        else:
            console.print("[red]Either --config or --storage-path required[/red]")
            raise click.Abort()

        console.print(f"[cyan]Storage:[/cyan] {storage_path}\n")

        keys = storage.list_keys()
        if not keys:
            console.print("[yellow]No data found[/yellow]")
            return

        metadata_manager = MetadataManager(storage)
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Key", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Provider")
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Date Range", style="dim")
        table.add_column("Last Updated", style="dim")

        for key in keys:
            metadata = metadata_manager.get_metadata_for_key(key) or {}
            symbol = str(metadata.get("symbol") or key.rsplit("/", 1)[-1])
            provider = str(metadata.get("provider") or "")
            row_count = metadata.get("row_count")
            rows = _format_row_count(row_count)
            start = str(metadata.get("start_date") or "")[:10]
            end = str(metadata.get("end_date") or "")[:10]
            updated = str(metadata.get("last_updated") or "")[:19]
            table.add_row(key, symbol, provider, rows, f"{start} to {end}", updated)

        console.print(table)
        console.print(f"[bold]Total:[/bold] {len(keys)} dataset(s)")

    except click.ClickException:
        raise
    except Exception as e:
        print_error(str(e))
        raise click.Abort()
