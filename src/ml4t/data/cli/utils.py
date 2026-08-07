"""Shared utilities for CLI commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import polars as pl
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

# Initialize Rich console for colored output
console = Console()


def validate_date(_ctx, _param, value):
    """Validate date format callback for Click options.

    Args:
        _ctx: Click context (unused, required by Click)
        _param: Click parameter (unused, required by Click)
        value: Date string to validate

    Returns:
        The validated date string

    Raises:
        click.BadParameter: If date format is invalid
    """
    if value is None:
        return value
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        raise click.BadParameter(f"Invalid date format: {value}. Use YYYY-MM-DD format.")


def save_dataframe(df: pl.DataFrame, path: str) -> None:
    """Save DataFrame to file.

    Args:
        df: Polars DataFrame to save
        path: Output file path (.csv, .parquet, .pq)
    """
    output_path = Path(path)

    if output_path.suffix == ".csv":
        df.write_csv(output_path)
    elif output_path.suffix in [".parquet", ".pq"]:
        df.write_parquet(output_path)
    else:
        # Default to parquet
        df.write_parquet(output_path.with_suffix(".parquet"))


def save_batch_results(results: dict[str, pl.DataFrame], path: str) -> None:
    """Save batch results to file(s).

    Args:
        results: Dictionary mapping symbols to DataFrames
        path: Output file path
    """
    output_path = Path(path)

    if output_path.suffix == ".parquet":
        # Save as single parquet with symbol column
        dfs = []
        for symbol, df in results.items():
            if df is not None:
                df_with_symbol = df.with_columns(pl.lit(symbol).alias("symbol"))
                dfs.append(df_with_symbol)

        if dfs:
            combined_df = pl.concat(dfs)
            combined_df.write_parquet(output_path)
    else:
        # Save each symbol to separate file
        for symbol, df in results.items():
            if df is not None:
                symbol_path = (
                    output_path.parent
                    / f"{output_path.stem}_{symbol}{output_path.suffix or '.parquet'}"
                )
                save_dataframe(df, str(symbol_path))


def load_symbols_from_file(file_path: str | Path, config_dir: Path | None = None) -> list[str]:
    """Load symbols from a file.

    Supports:
    - One symbol per line
    - Lines starting with # are comments
    - Relative paths resolved against config_dir

    Args:
        file_path: Path to symbols file
        config_dir: Base directory for relative paths

    Returns:
        List of symbol strings
    """
    path = Path(file_path)

    # Resolve relative paths
    if not path.is_absolute() and config_dir:
        path = config_dir / path

    if not path.exists():
        raise click.BadParameter(f"Symbols file not found: {path}")

    symbols = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                symbols.append(line)

    return symbols


def create_progress_bar() -> Progress:
    """Create a standard progress bar for CLI operations.

    Returns:
        Rich Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def print_error(message: str, verbose: bool = False, exception: Exception | None = None) -> None:
    """Print an error message with optional traceback.

    Args:
        message: Error message to display
        verbose: If True, print full traceback
        exception: Optional exception for traceback
    """
    console.print(f"[red]Error: {message}[/red]")
    if verbose and exception:
        import traceback

        traceback.print_exc()


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to display
    """
    console.print(f"[green]✅ {message}[/green]")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to display
    """
    console.print(f"[yellow]⚠️ {message}[/yellow]")
