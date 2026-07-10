"""Shared helpers for provider-specific fundamentals endpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any, Literal

import pandas as pd
import polars as pl

StatementType = Literal["income", "balance", "cashflow"]
PeriodType = Literal["annual", "quarterly", "ttm"]

FINANCIAL_STATEMENT_COLUMNS = [
    "symbol",
    "provider",
    "statement_type",
    "period_type",
    "period_end",
    "line_item",
    "value",
    "currency",
    "fiscal_year",
    "fiscal_period",
    "filed_at",
    "source",
]

COMPANY_METRIC_COLUMNS = [
    "symbol",
    "provider",
    "metric",
    "value",
    "period",
    "as_of",
    "currency",
    "source",
]

_FINANCIAL_STATEMENT_SCHEMA = {
    "symbol": pl.String,
    "provider": pl.String,
    "statement_type": pl.String,
    "period_type": pl.String,
    "period_end": pl.String,
    "line_item": pl.String,
    "value": pl.Float64,
    "currency": pl.String,
    "fiscal_year": pl.Int64,
    "fiscal_period": pl.String,
    "filed_at": pl.String,
    "source": pl.String,
}

_COMPANY_METRIC_SCHEMA = {
    "symbol": pl.String,
    "provider": pl.String,
    "metric": pl.String,
    "value": pl.Float64,
    "period": pl.String,
    "as_of": pl.String,
    "currency": pl.String,
    "source": pl.String,
}

_STATEMENT_ALIASES: dict[str, StatementType] = {
    "income": "income",
    "income_statement": "income",
    "ic": "income",
    "balance": "balance",
    "balance_sheet": "balance",
    "bs": "balance",
    "cashflow": "cashflow",
    "cash_flow": "cashflow",
    "cash": "cashflow",
    "cf": "cashflow",
}

_PERIOD_ALIASES: dict[str, PeriodType] = {
    "annual": "annual",
    "yearly": "annual",
    "year": "annual",
    "quarterly": "quarterly",
    "quarter": "quarterly",
    "ttm": "ttm",
    "trailing": "ttm",
}

_METADATA_KEYS = {
    "accessNumber",
    "acceptedDate",
    "calendarYear",
    "cik",
    "currency",
    "currency_symbol",
    "date",
    "endDate",
    "end_date",
    "filedDate",
    "filing_date",
    "filingDate",
    "finalLink",
    "fiscalDateEnding",
    "fiscalPeriod",
    "fiscal_period",
    "fiscalYear",
    "fiscal_year",
    "form",
    "link",
    "period",
    "quarter",
    "reportedCurrency",
    "symbol",
    "year",
}


def normalize_statement_type(statement: str) -> StatementType:
    """Normalize common statement aliases to the shared statement names."""
    key = statement.lower().replace("-", "_").replace(" ", "_")
    try:
        return _STATEMENT_ALIASES[key]
    except KeyError as err:
        supported = ", ".join(sorted(_STATEMENT_ALIASES))
        raise ValueError(f"Unsupported statement '{statement}'. Supported: {supported}") from err


def normalize_period_type(period: str) -> PeriodType:
    """Normalize common period aliases to the shared period names."""
    key = period.lower().replace("-", "_").replace(" ", "_")
    try:
        return _PERIOD_ALIASES[key]
    except KeyError as err:
        supported = ", ".join(sorted(_PERIOD_ALIASES))
        raise ValueError(f"Unsupported period '{period}'. Supported: {supported}") from err


def empty_financials_frame() -> pl.DataFrame:
    """Create an empty financial statements frame with the canonical schema."""
    return pl.DataFrame(schema=_FINANCIAL_STATEMENT_SCHEMA)


def empty_company_metrics_frame() -> pl.DataFrame:
    """Create an empty company metrics frame with the canonical schema."""
    return pl.DataFrame(schema=_COMPANY_METRIC_SCHEMA)


def rows_to_financials_frame(rows: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Convert statement rows to a Polars DataFrame with canonical column order."""
    return _rows_to_frame(rows, _FINANCIAL_STATEMENT_SCHEMA, FINANCIAL_STATEMENT_COLUMNS)


def rows_to_company_metrics_frame(rows: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Convert metric rows to a Polars DataFrame with canonical column order."""
    return _rows_to_frame(rows, _COMPANY_METRIC_SCHEMA, COMPANY_METRIC_COLUMNS)


def wide_pandas_statement_to_financials(
    frame: pd.DataFrame,
    *,
    symbol: str,
    provider: str,
    statement_type: StatementType,
    period_type: PeriodType,
    currency: str | None = None,
    source: str | None = None,
) -> pl.DataFrame:
    """Normalize yfinance-style wide statement data to canonical long form."""
    if frame.empty:
        return empty_financials_frame()

    rows: list[dict[str, Any]] = []
    for line_item in frame.index:
        for period_end, value in frame.loc[line_item].items():
            if not _is_number(value):
                continue
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "provider": provider,
                    "statement_type": statement_type,
                    "period_type": period_type,
                    "period_end": _date_string(period_end),
                    "line_item": str(line_item),
                    "value": float(value),
                    "currency": currency,
                    "source": source,
                }
            )
    return rows_to_financials_frame(rows)


def records_to_financials_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    symbol: str,
    provider: str,
    statement_type: StatementType,
    period_type: PeriodType,
    currency: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize record-oriented statements to canonical row dictionaries."""
    rows: list[dict[str, Any]] = []
    for record in records:
        period_end = _first_present(
            record, "period", "date", "endDate", "end_date", "fiscalDateEnding"
        )
        filed_at = _first_present(record, "filing_date", "filingDate", "filedDate", "acceptedDate")
        fiscal_year = _coerce_int(_first_present(record, "year", "fiscal_year", "fiscalYear"))
        fiscal_period = _first_present(record, "quarter", "fiscal_period", "fiscalPeriod")
        record_currency = _first_present(record, "currency", "currency_symbol", "reportedCurrency")

        for line_item, value in _iter_numeric_leaves(record):
            if line_item in _METADATA_KEYS:
                continue
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "provider": provider,
                    "statement_type": statement_type,
                    "period_type": period_type,
                    "period_end": _date_string(period_end),
                    "line_item": line_item,
                    "value": float(value),
                    "currency": str(record_currency or currency)
                    if record_currency or currency
                    else None,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": str(fiscal_period) if fiscal_period is not None else None,
                    "filed_at": _date_string(filed_at),
                    "source": source,
                }
            )
    return rows


def numeric_mapping_to_metric_rows(
    values: Mapping[str, Any],
    *,
    symbol: str,
    provider: str,
    period: str | None = None,
    as_of: str | None = None,
    currency: str | None = None,
    source: str | None = None,
    metrics: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a flat mapping of numeric values to canonical metric rows."""
    requested = set(metrics) if metrics else None
    rows: list[dict[str, Any]] = []
    for metric, value in values.items():
        if requested is not None and metric not in requested:
            continue
        if not _is_number(value):
            continue
        rows.append(
            {
                "symbol": symbol.upper(),
                "provider": provider,
                "metric": metric,
                "value": float(value),
                "period": period,
                "as_of": as_of,
                "currency": currency,
                "source": source,
            }
        )
    return rows


def nested_mapping_to_metric_rows(
    values: Mapping[str, Any],
    *,
    symbol: str,
    provider: str,
    period: str | None = None,
    as_of: str | None = None,
    currency: str | None = None,
    source: str | None = None,
    metrics: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert nested numeric mappings to canonical metric rows."""
    requested = set(metrics) if metrics else None
    rows: list[dict[str, Any]] = []
    for metric, value in _iter_numeric_leaves(values):
        if metric in _METADATA_KEYS:
            continue
        if requested is not None and metric not in requested:
            continue
        rows.append(
            {
                "symbol": symbol.upper(),
                "provider": provider,
                "metric": metric,
                "value": float(value),
                "period": period,
                "as_of": as_of,
                "currency": currency,
                "source": source,
            }
        )
    return rows


def sequence_or_mapping_values(data: Any) -> list[Mapping[str, Any]]:
    """Return statement records from common list or date-keyed mapping shapes."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping):
        return [item for item in data.values() if isinstance(item, Mapping)]
    return []


def _rows_to_frame(
    rows: Iterable[Mapping[str, Any]],
    schema: Mapping[str, pl.DataType],
    columns: list[str],
) -> pl.DataFrame:
    row_list = [dict(row) for row in rows]
    if not row_list:
        return pl.DataFrame(schema=schema)

    frame = pl.DataFrame(row_list)
    for column, dtype in schema.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(column))
        else:
            frame = frame.with_columns(pl.col(column).cast(dtype, strict=False))
    return frame.select(columns)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if value is None:
        return False
    if not isinstance(value, Real):
        return False
    return not pd.isna(value)


def _date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value)
    return text if text else None


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_numeric_leaves(record: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Real]]:
    for key, value in record.items():
        line_item = f"{prefix}.{key}" if prefix else str(key)
        if _is_number(value):
            yield line_item, value
            continue
        if isinstance(value, Mapping):
            if "value" in value and _is_number(value["value"]):
                yield line_item, value["value"]
            else:
                yield from _iter_numeric_leaves(value, line_item)
