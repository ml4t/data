"""Shared market-data artifact contracts for ML4T libraries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar

from .base import ArtifactKind, ArtifactProvenance, ArtifactSpec, ArtifactStorage, optional_str


class TimestampSemantics(str, Enum):
    """How timestamps should be interpreted downstream."""

    EVENT_TIME = "event_time"
    BAR_CLOSE = "bar_close"
    SESSION_LABEL = "session_label"


_MISSING = object()


@dataclass(frozen=True, slots=True)
class FeedSpec:
    """Dataset contract shared across ML4T libraries."""

    _FIELD_ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "timestamp_col": ("timestamp_col", "time_col", "datetime_col"),
        "entity_col": ("entity_col", "symbol_col", "group_col", "ticker_col", "asset_col"),
        "open_col": ("open_col",),
        "high_col": ("high_col",),
        "low_col": ("low_col",),
        "volume_col": ("volume_col",),
        "bid_col": ("bid_col",),
        "ask_col": ("ask_col",),
        "mid_col": ("mid_col",),
        "bid_size_col": ("bid_size_col",),
        "ask_size_col": ("ask_size_col",),
        "calendar": ("calendar",),
        "timezone": ("timezone",),
        "data_frequency": ("data_frequency", "frequency"),
        "bar_type": ("bar_type",),
        "timestamp_semantics": ("timestamp_semantics",),
        "session_start_time": ("session_start_time",),
    }

    timestamp_col: str = "timestamp"
    entity_col: str | Sequence[str] | None = None
    price_col: str = "close"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    bid_col: str | None = None
    ask_col: str | None = None
    mid_col: str | None = None
    bid_size_col: str | None = None
    ask_size_col: str | None = None
    calendar: str | None = None
    timezone: str | None = None
    data_frequency: Any | None = None
    bar_type: str | None = None
    timestamp_semantics: TimestampSemantics | str | None = None
    session_start_time: str | None = None

    def __post_init__(self) -> None:
        semantics = self.timestamp_semantics
        if semantics is None:
            return
        if not isinstance(semantics, TimestampSemantics):
            object.__setattr__(self, "timestamp_semantics", TimestampSemantics(str(semantics)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> FeedSpec:
        """Create a feed contract from a generic mapping."""

        def pick(*names: str) -> Any:
            for name in names:
                if name in mapping:
                    return mapping[name]
            return _MISSING

        return cls(**cls._extract_alias_data(pick))

    @classmethod
    def from_object(cls, value: Any) -> FeedSpec:
        """Create a feed contract from another ML4T config or metadata object."""
        if isinstance(value, FeedSpec):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)

        metadata = getattr(value, "metadata", None)
        if metadata is not None:
            return cls.from_object(metadata)

        def pick(*names: str) -> Any:
            for name in names:
                if hasattr(value, name):
                    return getattr(value, name)
            return _MISSING

        return cls(**cls._extract_alias_data(pick))

    @classmethod
    def from_any(cls, value: Any | None) -> FeedSpec:
        """Create a feed contract from a mapping, object, or existing spec."""
        if value is None:
            return cls()
        return cls.from_object(value)

    def with_overrides(self, **overrides: Any) -> FeedSpec:
        """Return a new spec with explicit non-null overrides applied."""
        updates = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **updates) if updates else self

    def resolve(self, columns: Sequence[str], entity_candidates: Sequence[str]) -> FeedSpec:
        """Resolve the entity column against an observed DataFrame schema."""
        if self.timestamp_col not in columns:
            raise ValueError(
                f"timestamp_col={self.timestamp_col!r} not found in columns {list(columns)}"
            )

        entity_col = self._coerce_entity_col(self.entity_col)
        if entity_col is not None:
            if entity_col not in columns:
                raise ValueError(f"entity_col={entity_col!r} not found in columns {list(columns)}")
            return replace(self, entity_col=entity_col)

        for candidate in entity_candidates:
            if candidate in columns:
                return replace(self, entity_col=candidate)

        raise ValueError(
            f"Cannot detect entity column. Expected one of {tuple(entity_candidates)}, "
            f"got columns {list(columns)}"
        )

    @staticmethod
    def _coerce_entity_col(value: str | Sequence[str] | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        values = [str(item) for item in value]
        if not values:
            return None
        if len(values) != 1:
            raise ValueError("ML4T FeedSpec currently supports a single entity column")
        return values[0]

    @classmethod
    def _extract_alias_data(cls, pick: Callable[..., Any]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        cls._apply_price_aliases(data, pick)
        for field_name, aliases in cls._FIELD_ALIASES.items():
            value = pick(*aliases)
            if value is not _MISSING:
                data[field_name] = value
        return data

    @staticmethod
    def _apply_price_aliases(data: dict[str, Any], pick: Callable[..., Any]) -> None:
        price_col = pick("price_col")
        close_col = pick("close_col")
        if price_col is not _MISSING:
            data["price_col"] = price_col
        elif close_col is not _MISSING:
            data["price_col"] = close_col

        if close_col is not _MISSING:
            data["close_col"] = close_col
        elif price_col is not _MISSING:
            data["close_col"] = price_col


@dataclass(frozen=True, slots=True)
class MarketDataSchema:
    """Column mapping for tradable market data."""

    timestamp_col: str = "timestamp"
    entity_col: str = "symbol"
    price_col: str = "close"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    bid_col: str | None = None
    ask_col: str | None = None
    mid_col: str | None = None
    bid_size_col: str | None = None
    ask_size_col: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> MarketDataSchema:
        if mapping is None:
            return cls()
        return cls(
            timestamp_col=str(mapping.get("timestamp_col", "timestamp")),
            entity_col=str(mapping.get("entity_col", "symbol")),
            price_col=str(mapping.get("price_col", "close")),
            open_col=str(mapping.get("open_col", "open")),
            high_col=str(mapping.get("high_col", "high")),
            low_col=str(mapping.get("low_col", "low")),
            close_col=str(mapping.get("close_col", "close")),
            volume_col=str(mapping.get("volume_col", "volume")),
            bid_col=optional_str(mapping.get("bid_col")),
            ask_col=optional_str(mapping.get("ask_col")),
            mid_col=optional_str(mapping.get("mid_col")),
            bid_size_col=optional_str(mapping.get("bid_size_col")),
            ask_size_col=optional_str(mapping.get("ask_size_col")),
        )


@dataclass(frozen=True, slots=True)
class MarketDataSemantics:
    """Temporal and execution semantics for tradable market data."""

    data_frequency: str | None = None
    calendar: str | None = None
    timezone: str | None = None
    timestamp_semantics: TimestampSemantics | str | None = None
    session_start_time: str | None = None
    bar_type: str | None = None

    def __post_init__(self) -> None:
        semantics = self.timestamp_semantics
        if semantics is None or isinstance(semantics, TimestampSemantics):
            return
        object.__setattr__(self, "timestamp_semantics", TimestampSemantics(str(semantics)))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> MarketDataSemantics:
        if mapping is None:
            return cls()
        return cls(
            data_frequency=optional_str(mapping.get("data_frequency")),
            calendar=optional_str(mapping.get("calendar")),
            timezone=optional_str(mapping.get("timezone")),
            timestamp_semantics=mapping.get("timestamp_semantics"),
            session_start_time=optional_str(mapping.get("session_start_time")),
            bar_type=optional_str(mapping.get("bar_type")),
        )


@dataclass(frozen=True, slots=True)
class MarketDataSpec(ArtifactSpec):
    """Shared specification for tradable market data artifacts."""

    kind: ArtifactKind = field(default=ArtifactKind.MARKET_DATA, init=False)
    schema: MarketDataSchema = field(default_factory=MarketDataSchema)
    semantics: MarketDataSemantics = field(default_factory=MarketDataSemantics)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> MarketDataSpec:
        return cls(
            artifact_id=str(mapping["artifact_id"]),
            version=int(mapping.get("version", 1)),
            storage=ArtifactStorage.from_mapping(mapping.get("storage")),
            provenance=ArtifactProvenance.from_mapping(mapping.get("provenance")),
            schema=MarketDataSchema.from_mapping(mapping.get("schema")),
            semantics=MarketDataSemantics.from_mapping(mapping.get("semantics")),
        )


__all__ = [
    "FeedSpec",
    "MarketDataSchema",
    "MarketDataSemantics",
    "MarketDataSpec",
    "TimestampSemantics",
]
