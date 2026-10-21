"""Dated instrument identity and membership over captured native tables."""

from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
import polars as pl

from ml4t.data.assets.asset_class import AssetClass
from ml4t.data.core.feeds import resolve_feed_spec
from ml4t.data.core.revisions import (
    native_frame,
    select_revisions,
    session_label_at,
    session_labels_at,
    utc_column,
    utc_stamp,
)
from ml4t.specs import FeedSpec


def _intervals(frame: pl.DataFrame, groups: list[str]) -> None:
    for group in frame.partition_by(groups, maintain_order=True):
        ordered = group.sort("valid_from").select("valid_from", "valid_to").with_row_index()
        if ordered.filter(
            (pl.col("index") > 0)
            & (
                pl.col("valid_to").shift(1).is_null()
                | (pl.col("valid_from") < pl.col("valid_to").shift(1))
            )
        ).height:
            raise ValueError("Overlapping identity or membership intervals")


def _bounds(frame: pl.DataFrame, feed: FeedSpec) -> pl.DataFrame:
    if "valid_from" not in frame.columns or frame["valid_from"].null_count():
        raise ValueError("valid_from is required for each identity interval")
    dtype = frame.schema["valid_from"]
    if dtype != pl.Date and not isinstance(dtype, pl.Datetime):
        raise ValueError("Identity intervals require Date labels or Datetime instants")
    if "valid_to" not in frame.columns or frame.schema["valid_to"] == pl.Null:
        frame = frame.with_columns(pl.lit(None, dtype=dtype).alias("valid_to"))
    if dtype == pl.Date:
        if frame.schema["valid_to"] != pl.Date:
            raise ValueError("Identity interval endpoints must share Date semantics")
    else:
        frame = utc_column(
            utc_column(frame, "valid_from", feed.timezone), "valid_to", feed.timezone
        )
    if frame.filter(
        pl.col("valid_to").is_not_null() & (pl.col("valid_from") >= pl.col("valid_to"))
    ).height:
        raise ValueError("Identity intervals must be nonempty [valid_from,valid_to)")
    return utc_column(frame, "available_at", feed.timezone)


def _on(
    frame: pl.DataFrame,
    value: date | datetime | str,
    *,
    feed: FeedSpec,
    start_only: bool = False,
) -> pl.DataFrame:
    dtype = frame.schema["valid_from"]
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("on must identify a known date or instant")
    timezone = feed.timezone
    if dtype == pl.Date:
        instant = isinstance(value, datetime) or isinstance(value, str) and len(value) > 10
        if instant and feed.timestamp_semantics == "session_label":
            point = pl.lit(session_label_at(value, feed))
        else:
            if stamp.tzinfo is not None and timezone is not None:
                stamp = stamp.tz_convert(timezone)
            point = pl.lit(stamp.date())
    else:
        stamp = (
            stamp.tz_localize(timezone or "UTC")
            if stamp.tzinfo is None
            else stamp.tz_convert("UTC")
        )
        point = pl.lit(stamp.value).cast(pl.Datetime("ns", "UTC"))
    predicate = pl.col("valid_from") <= point
    if not start_only:
        predicate &= pl.col("valid_to").is_null() | (point < pl.col("valid_to"))
    return frame.filter(predicate)


class InstrumentRegistry:
    """Resolve dated source symbols to stable IDs without building an asset database.

    Identity rows require asset,symbol,exchange,instrument_type,source,revision_id,valid_from and
    optional valid_to/available_at. ``valid_to`` is exclusive. Listing/delisting fields remain
    source facts; interval endpoints,not missing prices,declare identity coverage. Optional
    memberships have asset,universe,source,revision_id and the same interval/availability fields.
    ``as_of=None`` is explicitly retrospective. A causal cutoff requires actual availability.
    """

    def __init__(self, identities: pl.DataFrame, memberships: pl.DataFrame | None, feed: FeedSpec):
        self._identities, self._memberships, self._feed = identities, memberships, feed

    @classmethod
    def from_frame(
        cls,
        identities: pl.DataFrame | pd.DataFrame,
        memberships: pl.DataFrame | pd.DataFrame | None = None,
        *,
        feed_spec: Any | None = None,
    ) -> "InstrumentRegistry":
        """Validate captured identity/membership intervals without changing caller tables."""
        feed = resolve_feed_spec(
            FeedSpec(entity_col="asset", time_contract_version=2)
            if feed_spec is None
            else feed_spec
        )
        asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
        frame = _bounds(native_frame(identities), feed)
        required = {asset, "symbol", "exchange", "instrument_type", "source", "revision_id"}
        if missing := required.difference(frame.columns):
            raise ValueError(f"Missing identity fields: {sorted(missing)}")
        for name in required:
            if frame[name].null_count() or (
                frame.schema[name] == pl.String and frame[name].str.strip_chars().eq("").any()
            ):
                raise ValueError(f"Identity field {name} must be known and nonempty")
        if set(frame["instrument_type"].unique().to_list()).difference(
            item.value for item in AssetClass
        ):
            raise ValueError(
                "Declare a supported instrument_type instead of inferring execution metadata"
            )
        latest = select_revisions(frame, [asset, "source", "valid_from"], None)
        _intervals(latest, [asset])
        _intervals(latest, ["symbol", "exchange"])
        members = None
        if memberships is not None:
            members = _bounds(native_frame(memberships), feed)
            if missing := {asset, "universe", "source", "revision_id"}.difference(members.columns):
                raise ValueError(f"Missing membership fields: {sorted(missing)}")
            if not members[asset].is_in(frame[asset].unique().implode()).all():
                raise ValueError("Membership contains an undeclared stable asset ID")
            _intervals(
                select_revisions(members, [asset, "universe", "source", "valid_from"], None),
                [asset, "universe"],
            )
        return cls(frame, members, feed)

    @property
    def identities(self) -> pl.DataFrame:
        """Return a detached captured identity table,including source revisions."""
        return self._identities.clone()

    @property
    def memberships(self) -> pl.DataFrame | None:
        """Return captured membership provenance,or None when no coverage was supplied."""
        return None if self._memberships is None else self._memberships.clone()

    def _normalize(
        self,
        frame: pl.DataFrame,
        feed: FeedSpec,
        as_of: datetime | str | None,
        mode: Literal["stable_id", "symbol"],
        include_fields: tuple[str, ...] = (),
    ) -> tuple[pl.DataFrame, pl.DataFrame, str]:
        asset = FeedSpec._coerce_entity_col(self._feed.entity_col) or "asset"
        entity = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
        prefix = "__ml4t_identity_"
        if any(name.startswith(prefix) for name in frame.columns):
            raise ValueError("Source columns collide with internal identity keys")
        if mode not in {"stable_id", "symbol"}:
            raise ValueError("Declare stable_id or symbol identity_mode")
        if asset in frame.columns and (asset != entity or mode == "symbol"):
            raise ValueError("Stable identity output would collide with a captured source column")
        if entity not in frame.columns:
            raise ValueError("History requires the declared source entity column")
        lookup = "symbol" if mode == "symbol" else asset
        ids = self._identities.filter(pl.col(lookup).is_in(frame[entity].unique().implode()))[
            asset
        ].unique()
        candidates = self._identities.filter(pl.col(asset).is_in(ids.implode()))
        snapshot_feed = feed.with_overrides(entity_col=asset)
        session_intervals = self._feed.timestamp_semantics == "session_label"
        if session_intervals:
            snapshot_feed = snapshot_feed.with_overrides(timestamp_semantics="session_label")
        identities = InstrumentRegistry(candidates, None, snapshot_feed).lifetimes(as_of=as_of)
        timestamp = feed.timestamp_col
        dtype = frame.schema[timestamp]
        if identities.schema["valid_from"] == pl.Date:
            point = pl.col(timestamp)
            if isinstance(dtype, pl.Datetime):
                if session_intervals and feed.timestamp_semantics != "session_label":
                    point = pl.lit(session_labels_at(frame[timestamp], feed))
                else:
                    if feed.timezone is not None:
                        point = point.dt.convert_time_zone(feed.timezone)
                    point = point.dt.date()
        else:
            if dtype == pl.Date:
                raise ValueError("Date labels cannot establish an instant identity interval")
            point = pl.col(timestamp).dt.convert_time_zone("UTC")
        left = frame.with_row_index(prefix + "row").with_columns(point.alias(prefix + "point"))
        right = identities.rename({name: prefix + name for name in identities.columns})
        left_keys, right_keys = [entity], [prefix + lookup]
        if "exchange" in frame.columns:
            left_keys.append("exchange")
            right_keys.append(prefix + "exchange")
        matched = (
            left.join(right, left_on=left_keys, right_on=right_keys, how="inner", coalesce=False)
            .filter(
                (pl.col(prefix + "valid_from") <= pl.col(prefix + "point"))
                & (
                    pl.col(prefix + "valid_to").is_null()
                    | (pl.col(prefix + "point") < pl.col(prefix + "valid_to"))
                )
            )
            .sort(prefix + "row")
        )
        if matched.height != frame.height or matched[prefix + "row"].n_unique() != frame.height:
            raise ValueError("Missing or ambiguous dated identity;declare source ID,venue and date")
        used = matched.select(
            pl.col(prefix + name).alias(name) for name in identities.columns
        ).unique(maintain_order=True)
        output = matched.select(frame.columns)
        if asset != entity:
            output = output.with_columns(matched[prefix + asset].alias(asset))
        for name in include_fields:
            target = "instrument_" + name
            if target in output.columns:
                raise ValueError("Instrument metadata would collide with a captured source column")
            output = output.with_columns(matched[prefix + name].alias(target))
        return output, used, asset

    def lifetimes(
        self,
        on: date | datetime | str | None = None,
        *,
        as_of: datetime | str | None = None,
        universe: str | None = None,
    ) -> pl.DataFrame:
        """Return declared identity coverage,optionally known-at membership on a date.

        Absence of a declared row does not establish delisting or completeness of an entire
        universe. Original membership intervals and revision provenance remain in ``memberships``.
        """
        asset = FeedSpec._coerce_entity_col(self._feed.entity_col) or "asset"
        frame = self._identities
        if on is not None:
            frame = _on(frame, on, feed=self._feed, start_only=True)
        elif as_of is not None:
            frame = _on(frame, utc_stamp(as_of), feed=self._feed, start_only=True)
        frame = select_revisions(frame, [asset, "source", "valid_from"], as_of)
        _intervals(frame, [asset])
        _intervals(frame, ["symbol", "exchange"])
        if on is not None:
            frame = _on(frame, on, feed=self._feed)
        if universe is not None:
            if self._memberships is None:
                raise ValueError("A historical universe requires captured membership intervals")
            if universe not in self._memberships["universe"].to_list():
                raise ValueError("No captured source coverage for the requested universe")
            if on is None:
                raise ValueError("Specify on for a historical universe membership query")
            members = _on(
                self._memberships.filter(pl.col("universe") == universe),
                on,
                feed=self._feed,
                start_only=True,
            )
            members = select_revisions(members, [asset, "universe", "source", "valid_from"], as_of)
            _intervals(members, [asset, "universe"])
            frame = frame.join(
                _on(members, on, feed=self._feed).select(asset).unique(),
                on=asset,
                how="semi",
            )
        return frame.clone()

    def resolve(
        self,
        symbol: str,
        on: date | datetime | str,
        exchange: str | None = None,
        *,
        as_of: datetime | str | None = None,
    ) -> Any:
        """Resolve exactly one stable ID,or raise for missing/ambiguous dated symbols."""
        asset = FeedSpec._coerce_entity_col(self._feed.entity_col) or "asset"
        ids = self._identities.filter(pl.col("symbol") == symbol)[asset].unique()
        selected = self._identities.filter(pl.col(asset).is_in(ids.implode()))
        candidates = InstrumentRegistry(selected, None, self._feed).lifetimes(on, as_of=as_of)
        candidates = candidates.filter(pl.col("symbol") == symbol)
        if exchange is not None:
            candidates = candidates.filter(pl.col("exchange") == exchange)
        if candidates.height != 1:
            raise ValueError(
                f"Missing or ambiguous identity for {symbol!r} on {on};declare exchange/date"
            )
        return candidates[asset][0]
