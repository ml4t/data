"""Tests for asset-class source-reference documentation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml4t.data.providers.registry import ProviderSpec
from scripts.validate_source_references import SourceReferenceError, validate_source_references

SOURCE_PAGES = ("providers/equities.md", "providers/macro.md")
TABLE_HEADER = (
    "| Source and product | Coverage or history | Acquisition | Access | "
    "Material research caveat | ml4t-data provider |"
)
TABLE_SEPARATOR = "|---|---|---|---|---|---|"


def _write_fixture(
    root: Path,
    *,
    equity_provider: str = "No",
    include_macro_in_nav: bool = True,
    equity_row: str | None = None,
) -> ProviderSpec:
    """Create the smallest complete source-reference fixture."""
    provider_spec = ProviderSpec(
        name="sample",
        module="sample.module",
        class_name="SampleProvider",
        description="Sample",
        capabilities=frozenset({"ohlcv"}),
    )
    providers = root / "docs" / "providers"
    providers.mkdir(parents=True)
    (providers / "sample.md").write_text("# Sample\n", encoding="utf-8")

    valid_row = "| Source | Coverage | API | Free | Caveat | No |"
    selected_row = equity_row or (
        f"| Source | Coverage | API | Free | Caveat | {equity_provider} |"
    )
    (providers / "equities.md").write_text(
        f"# Equities\n\n{TABLE_HEADER}\n{TABLE_SEPARATOR}\n{selected_row}\n",
        encoding="utf-8",
    )
    (providers / "macro.md").write_text(
        f"# Macro\n\n{TABLE_HEADER}\n{TABLE_SEPARATOR}\n{valid_row}\n",
        encoding="utf-8",
    )

    nav = ["    - Equity: providers/equities.md"]
    if include_macro_in_nav:
        nav.append("    - Macro: providers/macro.md")
    (root / "mkdocs.yml").write_text(
        "nav:\n  - Providers:\n" + "\n".join(nav) + "\n",
        encoding="utf-8",
    )
    return provider_spec


def test_repository_source_references_satisfy_contract():
    validate_source_references(Path(__file__).resolve().parents[1])


def test_unknown_provider_class_is_rejected(tmp_path: Path):
    spec = _write_fixture(tmp_path, equity_provider="`UnknownProvider`")

    with pytest.raises(SourceReferenceError, match="unknown advertised provider class"):
        validate_source_references(
            tmp_path,
            provider_specs=(spec,),
            source_pages=SOURCE_PAGES,
        )


def test_source_page_missing_from_navigation_is_rejected(tmp_path: Path):
    spec = _write_fixture(tmp_path, include_macro_in_nav=False)

    with pytest.raises(SourceReferenceError, match="missing from MkDocs navigation"):
        validate_source_references(
            tmp_path,
            provider_specs=(spec,),
            source_pages=SOURCE_PAGES,
        )


def test_incomplete_source_row_is_rejected(tmp_path: Path):
    spec = _write_fixture(
        tmp_path,
        equity_row="| Source | Coverage | API | | Caveat | No |",
    )

    with pytest.raises(SourceReferenceError, match="empty required fields: Access"):
        validate_source_references(
            tmp_path,
            provider_specs=(spec,),
            source_pages=SOURCE_PAGES,
        )
