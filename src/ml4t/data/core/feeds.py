"""Strict receiving-boundary checks over the shared feed declaration."""

from collections.abc import Mapping
from typing import Any

from ml4t.specs import FeedSpec


def resolve_feed_spec(
    value: Any | None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
) -> FeedSpec:
    """Reject contradictory mappings before delegating declaration semantics to Specs."""
    if isinstance(value, Mapping):
        for aliases in FeedSpec._FIELD_ALIASES.values():
            supplied = [value[name] for name in aliases if name in value]
            if supplied and any(item != supplied[0] for item in supplied[1:]):
                raise ValueError(f"Conflicting feed mappings: {aliases}")
    feed = FeedSpec.from_any(value)
    overrides = {"timestamp_col": timestamp_col, "entity_col": entity_col, "price_col": price_col}
    for name, column in overrides.items():
        if column is not None and value is not None and getattr(feed, name) not in (None, column):
            raise ValueError(f"{name} conflicts with feed_spec")
    return feed.with_overrides(**overrides)
