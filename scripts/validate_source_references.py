"""Validate asset-class source-reference pages against the provider registry."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from ml4t.data.providers.registry import ProviderSpec, advertised_provider_specs

SOURCE_REFERENCE_PAGES = (
    "providers/equities.md",
    "providers/etfs.md",
    "providers/futures.md",
    "providers/options.md",
    "providers/fx.md",
    "providers/crypto.md",
    "providers/fixed_income.md",
    "providers/macro.md",
    "providers/fundamentals.md",
    "providers/alternative_data.md",
    "providers/factors.md",
    "providers/prediction_markets.md",
)

REQUIRED_COLUMNS = (
    "Source and product",
    "Coverage or history",
    "Acquisition",
    "Access",
    "Material research caveat",
    "ml4t-data provider",
)

_PROVIDER_CLASS = re.compile(r"`([A-Za-z][A-Za-z0-9_]*Provider)`")


class SourceReferenceError(ValueError):
    """Raised when a source-reference contract is invalid."""


class _MkDocsLoader(yaml.SafeLoader):
    """Load MkDocs configuration without evaluating environment variables."""


def _ignore_env(loader: _MkDocsLoader, node: yaml.Node) -> Any:
    """Return the declared environment fallback as inert configuration data."""
    if isinstance(node, yaml.SequenceNode):
        values = loader.construct_sequence(node)
        return values[-1] if values else None
    return loader.construct_scalar(node)


def _python_name(_loader: _MkDocsLoader, suffix: str, _node: yaml.Node) -> str:
    """Keep MkDocs extension callables as inert dotted names."""
    return suffix


_MkDocsLoader.add_constructor("!ENV", _ignore_env)
_MkDocsLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _python_name)


def _markdown_cells(line: str) -> list[str]:
    """Split one simple Markdown table row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _nav_paths(value: Any) -> set[str]:
    """Collect Markdown paths from a MkDocs navigation tree."""
    if isinstance(value, str):
        return {value} if value.endswith(".md") else set()
    if isinstance(value, list):
        paths: set[str] = set()
        for item in value:
            paths.update(_nav_paths(item))
        return paths
    if isinstance(value, dict):
        paths = set()
        for item in value.values():
            paths.update(_nav_paths(item))
        return paths
    return set()


def _comparison_rows(page: Path) -> list[tuple[int, list[str]]]:
    """Return rows from the required comparison table in one page."""
    lines = page.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if tuple(_markdown_cells(line)) != REQUIRED_COLUMNS:
            continue
        if index + 1 >= len(lines):
            break
        separator = _markdown_cells(lines[index + 1])
        if len(separator) != len(REQUIRED_COLUMNS) or not all(
            cell and set(cell) <= {"-", ":"} for cell in separator
        ):
            raise SourceReferenceError(f"{page}: comparison table has no valid separator row")

        rows: list[tuple[int, list[str]]] = []
        for row_index in range(index + 2, len(lines)):
            row = lines[row_index]
            if not row.lstrip().startswith("|"):
                break
            rows.append((row_index + 1, _markdown_cells(row)))
        if not rows:
            raise SourceReferenceError(f"{page}: comparison table has no source rows")
        return rows
    raise SourceReferenceError(
        f"{page}: missing comparison table with columns {', '.join(REQUIRED_COLUMNS)}"
    )


def validate_source_references(
    project_root: Path,
    *,
    provider_specs: Iterable[ProviderSpec] | None = None,
    source_pages: Iterable[str] = SOURCE_REFERENCE_PAGES,
) -> None:
    """Validate page inventory, navigation, row fields, and provider mappings."""
    project_root = project_root.resolve()
    docs_root = project_root / "docs"
    config_path = project_root / "mkdocs.yml"
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_MkDocsLoader)
    nav_paths = _nav_paths(config.get("nav", []))

    specs = tuple(provider_specs or advertised_provider_specs())
    specs_by_class: Mapping[str, ProviderSpec] = {spec.class_name: spec for spec in specs}

    errors: list[str] = []
    for relative_page in source_pages:
        page = docs_root / relative_page
        if not page.is_file():
            errors.append(f"missing source-reference page: {relative_page}")
            continue
        if relative_page not in nav_paths:
            errors.append(
                f"source-reference page is missing from MkDocs navigation: {relative_page}"
            )

        try:
            rows = _comparison_rows(page)
        except SourceReferenceError as error:
            errors.append(str(error))
            continue

        for line_number, cells in rows:
            location = f"{page}:{line_number}"
            if len(cells) != len(REQUIRED_COLUMNS):
                errors.append(
                    f"{location}: expected {len(REQUIRED_COLUMNS)} fields, found {len(cells)}"
                )
                continue
            missing = [
                REQUIRED_COLUMNS[index] for index, value in enumerate(cells) if not value.strip()
            ]
            if missing:
                errors.append(f"{location}: empty required fields: {', '.join(missing)}")
                continue

            provider_cell = cells[-1]
            if provider_cell == "No":
                continue
            provider_classes = _PROVIDER_CLASS.findall(provider_cell)
            if len(provider_classes) != 1:
                errors.append(
                    f"{location}: provider field must contain one backticked provider class or No"
                )
                continue
            class_name = provider_classes[0]
            spec = specs_by_class.get(class_name)
            if spec is None:
                errors.append(f"{location}: unknown advertised provider class: {class_name}")
                continue
            provider_page = docs_root / "providers" / f"{spec.name}.md"
            if not provider_page.is_file():
                errors.append(
                    f"{location}: provider class {class_name} has no provider page at "
                    f"providers/{spec.name}.md"
                )

    if errors:
        raise SourceReferenceError("\n".join(errors))


def main() -> int:
    """Validate the checkout containing this script."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        validate_source_references(project_root)
    except SourceReferenceError as error:
        print(error)
        return 1
    print("Source-reference contracts are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
