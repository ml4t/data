"""Core CLI commands for data operations.

Commands: fetch, update, validate, status, export, info, list
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import click
import polars as pl
from rich import box
from rich.panel import Panel
from rich.table import Table

from ml4t.data.core.keys import is_storage_key, normalize_storage_key, parse_storage_key
from ml4t.data.data_manager import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.key_codec import decode_storage_key
from ml4t.data.storage.manifest import normalize_manifest_payload
from ml4t.data.storage.metadata_tracker import MetadataTracker
from ml4t.data.update_manager import IncrementalUpdater, UpdateStrategy

from .utils import (
    console,
    create_progress_bar,
    print_error,
    print_success,
    save_batch_results,
    save_dataframe,
    validate_date,
)


@click.command()
@click.option("--symbol", "-s", multiple=True, help="Symbol(s) to fetch")
@click.option(
    "--symbols-file",
    "-f",
    type=click.Path(exists=True),
    help="File containing symbols (one per line)",
)
@click.option("--start", callback=validate_date, required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", callback=validate_date, required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--frequency",
    default="daily",
    type=click.Choice(["daily", "hourly", "weekly"]),
    help="Data frequency",
)
@click.option("--provider", "-p", help="Specific provider to use")
@click.option("--output", "-o", type=click.Path(), help="Output file path (.parquet or .csv)")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file (JSON)")
@click.option("--progress", is_flag=True, help="Show progress bar")
@click.pass_context
def fetch(ctx, symbol, symbols_file, start, end, frequency, provider, output, config, progress):
    """Fetch financial data from providers.

    Examples:
        ml4t-data fetch -s BTC --start 2024-01-01 --end 2024-01-31
        ml4t-data fetch -s BTC -s ETH --start 2024-01-01 --end 2024-01-31
        ml4t-data fetch -f symbols.txt --start 2024-01-01 --end 2024-01-31
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    # Load configuration if provided
    if config:
        if not quiet:
            console.print(f"Loading configuration from {config}")
        with open(config) as f:
            config_data = json.load(f)
            symbol = config_data.get("symbols", list(symbol))
            start = config_data.get("start", start)
            end = config_data.get("end", end)
            frequency = config_data.get("frequency", frequency)
            provider = config_data.get("provider", provider)

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
        dm = DataManager()

        if len(symbols) == 1:
            sym = symbols[0]
            if not quiet:
                console.print(f"Fetching {sym} from {start} to {end}")

            df = dm.fetch(sym, start, end, frequency=frequency, provider=provider)

            if not quiet:
                print_success(f"Fetched {len(df)} rows")

            if output:
                save_dataframe(df, output)
                if not quiet:
                    console.print(f"[green]Saved to {output}[/green]")
        else:
            if not quiet:
                console.print(f"Fetching {len(symbols)} symbols")

            if progress and not quiet:
                with create_progress_bar() as progress_bar:
                    task = progress_bar.add_task("Fetching...", total=len(symbols))
                    results = {}
                    for sym in symbols:
                        try:
                            results[sym] = dm.fetch(
                                sym, start, end, frequency=frequency, provider=provider
                            )
                            progress_bar.update(task, advance=1, description=f"Fetched {sym}")
                        except Exception as e:
                            if verbose:
                                console.print(
                                    f"[yellow]Warning: Failed to fetch {sym}: {e}[/yellow]"
                                )
                            results[sym] = None
                            progress_bar.update(task, advance=1)
            else:
                results = dm.fetch_batch(symbols, start, end, frequency=frequency)

            successful = sum(1 for v in results.values() if v is not None)
            if not quiet:
                print_success(f"Successfully fetched {successful} symbols")

            if output:
                save_batch_results(results, output)
                if not quiet:
                    console.print(f"[green]Saved to {output}[/green]")

    except Exception as e:
        print_error(str(e), verbose, e)
        ctx.exit(1)


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to update")
@click.option("--start", callback=validate_date, help="Start date (YYYY-MM-DD)")
@click.option("--end", callback=validate_date, help="End date (YYYY-MM-DD)")
@click.option(
    "--strategy",
    type=click.Choice(["incremental", "append_only", "full_refresh", "backfill"]),
    default="incremental",
    help="Update strategy",
)
@click.option("--provider", "-p", help="Provider to use for fetching")
@click.option("--storage-path", default="./data", help="Storage directory path")
@click.option("--asset-class", default="equities", help="Asset class for storage key")
@click.option(
    "--frequency",
    default="daily",
    type=click.Choice(["daily", "hourly", "weekly"]),
    help="Data frequency for storage key and fetching",
)
@click.pass_context
def update(ctx, symbol, start, end, strategy, provider, storage_path, asset_class, frequency):
    """Perform incremental data updates."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        storage_config = StorageConfig(base_path=Path(storage_path))
        storage = HiveStorage(storage_config)
        tracker = MetadataTracker(Path(storage_path))

        update_strategy = UpdateStrategy[strategy.upper()]
        updater = IncrementalUpdater(strategy=update_strategy)
        storage_key = normalize_storage_key(
            symbol_or_key=symbol,
            asset_class=asset_class,
            frequency=frequency,
        )
        fetch_symbol = parse_storage_key(storage_key)[2] if is_storage_key(symbol) else symbol

        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        actual_start, actual_end, update_type = updater.determine_update_range(
            storage,
            storage_key,
            datetime.strptime(start, "%Y-%m-%d"),
            datetime.strptime(end, "%Y-%m-%d"),
        )

        if update_type == "none":
            if not quiet:
                print_success(f"Data already up to date for {fetch_symbol}")
            return

        if not quiet:
            console.print(
                f"{'Incremental' if update_type == 'incremental' else 'Full'} update "
                f"from {actual_start.date()} to {actual_end.date()}"
            )

        dm = DataManager()
        new_data = dm.fetch(
            fetch_symbol,
            actual_start.strftime("%Y-%m-%d"),
            actual_end.strftime("%Y-%m-%d"),
            frequency=frequency,
            provider=provider,
        )

        if new_data.is_empty():
            if not quiet:
                console.print("[yellow]No new data available[/yellow]")
            return

        result = updater.update_incremental(
            storage,
            tracker,
            storage_key,
            new_data,
            provider=provider or "auto",
            strategy=update_strategy,
        )

        if result.success:
            if not quiet:
                print_success("Update successful")
                console.print(
                    f"   Added {result.rows_added} rows, updated {result.rows_updated} rows"
                )
                if result.gaps_filled > 0:
                    console.print(f"   Filled {result.gaps_filled} gaps")
        else:
            console.print("[red]❌ Update failed[/red]")
            if result.errors:
                for error in result.errors:
                    console.print(f"   {error}")
            ctx.exit(1)

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
@click.option("--storage-path", default="./data", help="Storage directory path")
@click.pass_context
def validate(ctx, symbol, validate_all, anomalies, save_report, severity, storage_path):
    """Validate data quality and integrity."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        storage_config = StorageConfig(base_path=Path(storage_path))
        storage = HiveStorage(storage_config)

        symbols = []
        if validate_all:
            symbols = storage.list_keys()
        elif symbol:
            symbols = [symbol]
        else:
            console.print("[red]Error: Specify --symbol or --all[/red]")
            ctx.exit(1)

        total_issues = 0

        for sym in symbols:
            if not quiet:
                console.print(f"Validating {sym}...")

            if not storage.exists(sym):
                console.print(f"[yellow]  Symbol {sym} not found in storage[/yellow]")
                continue

            df = storage.read(sym).collect()
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
                        PriceStalenessDetector,
                        ReturnOutlierDetector,
                        VolumeSpikeDetector,
                    )
                    from ml4t.data.anomaly.config import (
                        PriceStalenessConfig,
                        ReturnOutlierConfig,
                        VolumeSpikeConfig,
                    )

                    manager = AnomalyManager()
                    manager.detectors.append(
                        PriceStalenessDetector(PriceStalenessConfig(max_unchanged_days=3))
                    )
                    manager.detectors.append(
                        ReturnOutlierDetector(ReturnOutlierConfig(threshold=5.0))
                    )
                    manager.detectors.append(VolumeSpikeDetector(VolumeSpikeConfig(threshold=10.0)))

                    report = manager.analyze(df, symbol=sym, asset_class="unknown")

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

    except Exception as e:
        print_error(str(e), verbose, e)
        ctx.exit(1)


@click.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed status")
@click.option("--storage-path", default="./data", help="Storage directory path")
@click.pass_context
def status(ctx, detailed, storage_path):
    """Show system overview and health status."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        storage_config = StorageConfig(base_path=Path(storage_path))
        storage = HiveStorage(storage_config)
        tracker = MetadataTracker(Path(storage_path))

        summary = tracker.get_summary()

        if not quiet:
            table = Table(title="System Status", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Total Datasets", str(summary.get("total_datasets", 0)))
            table.add_row("Healthy", f"[green]{summary.get('healthy', 0)}[/green]")
            table.add_row("Stale", f"[yellow]{summary.get('stale', 0)}[/yellow]")
            table.add_row("Error", f"[red]{summary.get('error', 0)}[/red]")
            table.add_row("Total Rows", f"{summary.get('total_rows', 0):,}")
            table.add_row("Total Updates", str(summary.get("total_updates", 0)))

            console.print(table)

            if summary.get("by_asset_class"):
                asset_table = Table(title="By Asset Class", box=box.SIMPLE)
                asset_table.add_column("Asset Class", style="cyan")
                asset_table.add_column("Count", style="white")

                for asset_class, count in summary["by_asset_class"].items():
                    asset_table.add_row(asset_class or "unknown", str(count))

                console.print(asset_table)

        if detailed:
            console.print("\n[bold]Detailed Dataset Information:[/bold]")

            for sym in storage.list_keys():
                metadata = tracker.get_metadata(sym)
                if metadata:
                    status_color = {"healthy": "green", "stale": "yellow", "error": "red"}.get(
                        metadata.health_status, "white"
                    )

                    panel_content = f"""Provider: {metadata.provider}
Frequency: {metadata.frequency}
Rows: {metadata.total_rows:,}
Date Range: {metadata.date_range_start.date()} to {metadata.date_range_end.date()}
Last Update: {metadata.last_update}
Updates: {metadata.update_count}
Status: [{status_color}]{metadata.health_status}[/{status_color}]""".strip()

                    panel = Panel(panel_content, title=f"Dataset: {sym}", border_style="cyan")
                    console.print(panel)

        if verbose:
            console.print(f"\n[dim]Storage path: {storage_path}[/dim]")

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
@click.option("--storage-path", default=None, help="Storage directory")
def export(symbol, output, format_type, storage_path):
    """Export data to various formats (CSV, JSON, Parquet)."""
    try:
        storage_path = Path(storage_path) if storage_path else Path.cwd() / "data"
        config = StorageConfig(base_path=storage_path)
        storage = HiveStorage(config)

        console.print(f"[bold]Reading data for {symbol}...[/bold]")
        df = storage.read(symbol).collect()

        if df.is_empty():
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            return

        output_path = Path(output)
        console.print(f"[bold]Exporting to {output_path}...[/bold]")

        if format_type == "csv":
            df.write_csv(output_path)
        elif format_type == "json":
            df.write_json(output_path)
        elif format_type == "parquet":
            df.write_parquet(output_path)

        print_success(f"Exported {len(df)} rows to {output_path}")

    except Exception as e:
        print_error(str(e))
        raise click.Abort()


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to show info for")
@click.option("--storage-path", default=None, help="Storage directory")
def info(symbol, storage_path):
    """Show information about stored data."""
    try:
        storage_path = Path(storage_path) if storage_path else Path.cwd() / "data"
        config = StorageConfig(base_path=storage_path)
        storage = HiveStorage(config)
        tracker = MetadataTracker(base_path=storage_path)

        if not any(record.symbol == symbol for record in tracker.list_updates()):
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            return

        df = storage.read(symbol).collect()
        updates = [r for r in tracker.list_updates() if r.symbol == symbol]
        latest_update = max(updates, key=lambda x: x.timestamp) if updates else None

        table = Table(title=f"Data Info: {symbol}", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Symbol", symbol)
        table.add_row("Rows", str(len(df)))
        table.add_row("Date Range", f"{df['timestamp'].min()} to {df['timestamp'].max()}")
        table.add_row("Columns", ", ".join(df.columns))

        if latest_update:
            table.add_row("Provider", latest_update.provider)
            table.add_row("Last Updated", latest_update.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Frequency", latest_update.frequency)

        console.print(table)
        console.print("\n[bold]Data Preview:[/bold]")
        console.print(df.head(5))

    except Exception as e:
        print_error(str(e))
        raise click.Abort()


@click.command("list")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file (YAML)")
@click.option("--storage-path", type=click.Path(exists=True), help="Storage directory")
@click.pass_context
def list_data(_ctx, config, storage_path):
    """List all stored datasets."""
    import json as json_module

    import yaml

    try:
        if config:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            storage_path = Path(cfg["storage"]["path"]).expanduser()
        elif storage_path:
            storage_path = Path(storage_path).expanduser()
        else:
            console.print("[red]Either --config or --storage-path required[/red]")
            raise click.Abort()

        console.print(f"[cyan]Storage:[/cyan] {storage_path}\n")

        metadata_dir = storage_path / ".metadata"
        if not metadata_dir.exists():
            console.print("[yellow]No data found[/yellow]")
            return

        futures_data = {}
        spot_data = {}

        for meta_file in metadata_dir.glob("*.json"):
            with open(meta_file) as f:
                meta = json_module.load(f)

            decoded_key = decode_storage_key(meta_file.stem)
            normalized = normalize_manifest_payload(decoded_key, meta)
            provider = normalized.get("provider")
            symbol = normalized.get("symbol")

            if not symbol or not provider:
                continue

            data_range = normalized.get("data_range")
            if not isinstance(data_range, dict):
                data_range = {}

            data_info = {
                "rows": normalized.get("row_count", normalized.get("total_rows", 0)),
                "start": data_range.get("start", ""),
                "end": data_range.get("end", ""),
                "updated": normalized.get("last_updated", ""),
            }

            if provider == "databento":
                futures_data[symbol] = data_info
            elif provider == "cryptocompare":
                spot_data[symbol] = data_info

        if futures_data:
            console.print("[bold]Futures (DataBento)[/bold]")
            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("Symbol", style="cyan")
            table.add_column("Rows", justify="right", style="green")
            table.add_column("Date Range", style="dim")
            table.add_column("Last Updated", style="dim")

            for sym in sorted(futures_data.keys()):
                info_d = futures_data[sym]
                rows = f"{info_d['rows']:,}"
                date_range = f"{info_d['start'][:10]} → {info_d['end'][:10]}"
                updated = info_d["updated"][:19] if info_d["updated"] else ""
                table.add_row(sym, rows, date_range, updated)

            console.print(table)
            console.print()

        if spot_data:
            console.print("[bold]Spot (CryptoCompare)[/bold]")
            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("Symbol", style="cyan")
            table.add_column("Rows", justify="right", style="green")
            table.add_column("Date Range", style="dim")
            table.add_column("Last Updated", style="dim")

            for sym in sorted(spot_data.keys()):
                info_d = spot_data[sym]
                rows = f"{info_d['rows']:,}"
                date_range = f"{info_d['start'][:10]} → {info_d['end'][:10]}"
                updated = info_d["updated"][:19] if info_d["updated"] else ""
                table.add_row(sym, rows, date_range, updated)

            console.print(table)
            console.print()

        total = len(futures_data) + len(spot_data)
        console.print(f"[bold]Total:[/bold] {total} dataset(s)")

    except Exception as e:
        print_error(str(e))
        raise click.Abort()
