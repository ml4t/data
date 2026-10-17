"""Public provider/validator/storage workflows must use the declared schedule."""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.assets import AssetClass, AssetInfo, AssetValidator
from ml4t.data.calendar.crypto import CryptoCalendar, CryptoSessionValidator
from ml4t.data.providers import SyntheticProvider
from ml4t.data.sessions import SessionAssigner
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.validation.cross_validation import CrossValidator
from ml4t.data.validation.rules import ValidationRuleConfig, ValidationRuleSet
from ml4t.specs import UNSET_CALENDAR, FeedSpec


def minute_observation(day: str, clock: str) -> pl.DataFrame:
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("SPY", day, day, "minute")
    selected = frame.filter(pl.col("timestamp").dt.strftime("%H:%M") == clock)
    assert selected.height == 1
    return selected


def temporal_issues(result):
    return [issue for issue in result.issues if issue.check in {"weekend_trading", "market_hours"}]


def test_daily_provider_labels_are_not_after_hours_instants():
    with SyntheticProvider() as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-11", "daily")
    before = frame.clone()
    result = AssetValidator(AssetInfo(symbol="SPY", asset_class=AssetClass.EQUITY)).validate(
        frame, "daily"
    )
    assert not any("market hours" in issue for issue in result.get("equity_specific", []))
    assert_frame_equal(frame, before)


def test_summer_nyse_open_is_not_rejected_by_winter_utc_clock():
    frame = minute_observation("2025-07-07", "13:30")
    assert temporal_issues(CrossValidator().validate(frame, frequency="minute")) == []


@pytest.mark.parametrize("day,clock", [("2025-07-03", "17:01"), ("2025-07-04", "15:00")])
def test_nyse_early_close_and_holiday_are_not_regular_market_hours(day, clock):
    frame = minute_observation(day, clock)
    result = CrossValidator().validate(frame, frequency="minute")
    assert any(issue.check == "market_hours" for issue in temporal_issues(result))


def test_disabled_cross_validator_does_not_reinstate_weekend_rules():
    frame = minute_observation("2025-07-12", "18:00")
    before = frame.clone()
    result = CrossValidator().validate(frame, frequency="minute", calendar=None)
    assert temporal_issues(result) == []
    assert_frame_equal(frame, before)


def test_disabled_crypto_validator_does_not_reinstate_gap_calendar():
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("BTCUSD", "2025-07-07", "2025-07-09", "daily")
    frame = frame.filter(pl.col("timestamp").dt.day() != 8)
    assert CryptoSessionValidator(calendar=None).validate_continuity(frame, "daily") == []


@pytest.mark.parametrize("validator", ["cross", "asset"])
def test_explicit_override_metadata_and_disabled_metadata_have_same_membership(validator):
    frame = minute_observation("2025-07-12", "18:00")
    feed = FeedSpec(calendar="24/7", timestamp_semantics="event_time", time_contract_version=2)
    if validator == "cross":
        instance = CrossValidator()

        def has_temporal(**kw):
            return bool(temporal_issues(instance.validate(frame, **kw)))
    else:
        instance = AssetValidator(AssetInfo(symbol="SPY", asset_class=AssetClass.EQUITY))

        def has_temporal(**kw):
            return bool(instance.validate(frame, **kw).get("equity_specific"))

    assert not has_temporal(frequency="minute", feed_spec=feed)
    assert has_temporal(frequency="minute", calendar="NYSE", feed_spec=feed)
    assert not has_temporal(frequency="minute", calendar=None, feed_spec=feed)
    assert not has_temporal(
        frequency="minute", feed_spec=FeedSpec(calendar=None, time_contract_version=2)
    )
    assert has_temporal(frequency="minute", feed_spec=FeedSpec())


def test_disabled_temporal_validation_keeps_price_and_volume_checks():
    frame = minute_observation("2025-07-12", "18:00").with_columns(
        pl.lit(-1.0).alias("volume"), pl.lit(-1.0).alias("close")
    )
    result = AssetValidator(AssetInfo(symbol="SPY", asset_class=AssetClass.EQUITY)).validate(
        frame, "minute", calendar=None
    )
    assert any("non-positive" in item for item in result.get("equity_specific", []))
    assert result.get("data_quality")
    zero_volume = frame.with_columns(pl.lit(0.0).alias("volume"))
    assert CryptoSessionValidator(None).validate_volume_profile(zero_volume) == [
        "Found 1 periods with zero volume"
    ]


def test_declared_daily_event_instant_overrides_frequency_and_date_label_heuristics():
    frame = minute_observation("2025-07-06", "23:30")
    feed = FeedSpec(calendar="CME Globex Crypto", timestamp_semantics="event_time")
    assert temporal_issues(CrossValidator(feed_spec=feed).validate(frame, frequency="daily")) == []


def test_closed_instant_weekend_reporting_uses_selected_local_day_not_utc_date():
    frame = minute_observation("2025-07-12", "01:00")
    result = CrossValidator().validate(frame, frequency="minute")
    assert [issue.check for issue in temporal_issues(result)] == ["market_hours"]


def test_cross_validator_daily_feed_timezone_preserves_the_session_label():
    with SyntheticProvider() as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-07", "daily")
    frame = frame.with_columns(pl.col("timestamp").dt.replace(hour=0))
    local = frame.with_columns(pl.col("timestamp").dt.replace_time_zone("Asia/Tokyo"))
    utc = local.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    assert utc["timestamp"].dt.date().to_list() == [date(2025, 7, 6)]
    feed = FeedSpec(
        calendar="NYSE",
        timezone="Asia/Tokyo",
        data_frequency="daily",
        timestamp_semantics="session_label",
    )
    assert temporal_issues(CrossValidator(feed_spec=feed).validate(utc)) == []


def test_explicit_legacy_crypto_object_retains_its_continuous_gap_contract():
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("BTCUSD", "2025-07-11", "2025-07-14", "daily")
    observed = frame.filter(pl.col("timestamp").dt.weekday() <= 5)
    assert len(CryptoSessionValidator(CryptoCalendar()).validate_continuity(observed)) == 1
    assert CryptoSessionValidator().validate_continuity(observed) == []
    assert len(CryptoSessionValidator("24/7").validate_continuity(observed)) == 1


def test_legacy_crypto_hint_cannot_override_explicit_or_declared_calendar():
    frame = minute_observation("2025-07-12", "18:00")
    instance = CrossValidator(is_crypto=True)
    assert temporal_issues(instance.validate(frame, frequency="minute")) == []
    assert temporal_issues(instance.validate(frame, frequency="minute", calendar="NYSE"))
    assert temporal_issues(
        instance.validate(frame, frequency="minute", feed_spec=FeedSpec(calendar="NYSE"))
    )


@pytest.mark.parametrize("storage_class", [FlatStorage, HiveStorage])
@pytest.mark.parametrize("operation", ["load", "import"])
@pytest.mark.parametrize("calendar", ["NYSE", "24/7", None])
def test_real_storage_validation_receives_frequency_and_effective_feed(
    tmp_path, monkeypatch, storage_class, operation, calendar
):
    reports = []
    validate = CrossValidator.validate

    def record(self, frame, *args, **kwargs):
        report = validate(self, frame, *args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(CrossValidator, "validate", record)
    storage = storage_class(StorageConfig(base_path=tmp_path))
    providers = {"synthetic": {"extra": {"calendar_mode": "continuous", "seed": 42}}}
    feed = FeedSpec(calendar="24/7", timestamp_semantics="event_time", time_contract_version=2)
    with DataManager(providers=providers, storage=storage) as manager:
        if operation == "load":
            manager.load(
                "SPY",
                "2025-07-12",
                "2025-07-12",
                provider="synthetic",
                frequency="minute",
                asset_class="crypto",
                calendar=calendar,
                feed_spec=feed,
            )
        else:
            frame = minute_observation("2025-07-12", "18:00")
            manager.import_data(
                frame,
                "SPY",
                provider="synthetic",
                frequency="minute",
                asset_class="crypto",
                calendar=calendar,
                feed_spec=feed,
            )
    assert len(reports) == 1
    assert bool(temporal_issues(reports[0])) == (calendar == "NYSE")


@pytest.mark.parametrize("calendar", [UNSET_CALENDAR, None, "NYSE", "24/7"])
def test_validation_rule_json_yaml_copy_preserves_omitted_disabled_and_named_calendars(
    tmp_path, calendar
):
    frame = minute_observation("2025-07-12", "18:00")
    config = ValidationRuleConfig(calendar=calendar)
    wire = config.model_dump(mode="json")
    assert ("calendar" in wire) == (calendar is not UNSET_CALENDAR)
    rules = ValidationRuleSet(name="calendar_rules", default_rule=config)
    path = tmp_path / "rules.yaml"
    rules.save(path)
    restored = ValidationRuleSet.load(path).get_rules("SPY")
    copies = [
        config.model_copy(),
        ValidationRuleConfig.model_validate_json(config.model_dump_json()),
        restored,
    ]
    for copy in copies:
        assert copy.calendar is calendar or copy.calendar == calendar
        issues = temporal_issues(CrossValidator(feed_spec=copy).validate(frame, frequency="minute"))
        assert bool(issues) == (calendar is UNSET_CALENDAR or calendar == "NYSE")
        assert not temporal_issues(
            CrossValidator(feed_spec=copy).validate(frame, frequency="minute", calendar=None)
        )


def test_legacy_rule_null_calendar_still_inherits_default_instead_of_disabling():
    config = ValidationRuleConfig.model_validate({"calendar": None, "time_contract_version": 1})
    assert FeedSpec.from_any(config).resolve_calendar() == "NYSE"
    frame = minute_observation("2025-07-12", "18:00")
    assert temporal_issues(CrossValidator(feed_spec=config).validate(frame, frequency="minute"))


@pytest.mark.parametrize("frequency", ["daily", "1d", "1day"])
def test_crypto_continuity_preserves_supported_daily_frequency_aliases(frequency):
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("BTCUSD", "2025-07-11", "2025-07-14", "daily")
    observed = frame.filter(pl.col("timestamp").dt.weekday() <= 5)
    assert len(CryptoSessionValidator("24/7").validate_continuity(observed, frequency)) == 1


@pytest.mark.parametrize("calendar,last_minute", [("NYSE", "19:59"), ("HKEX", "03:59")])
@pytest.mark.parametrize("offset_ns", [-1, 0, 1])
def test_nanosecond_close_labels_respect_exact_session_and_break_boundaries(
    calendar, last_minute, offset_ns
):
    frame = minute_observation("2025-07-07", last_minute).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))
        + pl.duration(minutes=1, nanoseconds=offset_ns)
    )
    before = frame.clone()
    feed = FeedSpec(calendar=calendar, timestamp_semantics="bar_close")
    assigner = SessionAssigner(feed_spec=feed)
    assigned = assigner.assign_sessions(frame, bar_frequency="intraday")
    assert assigned["session_date"].to_list() == ([None] if offset_ns > 0 else [date(2025, 7, 7)])
    report = CrossValidator(feed_spec=feed).validate(frame, frequency="minute")
    assert bool(temporal_issues(report)) == (offset_ns > 0)
    if offset_ns > 0:
        with pytest.raises(ValueError, match="outside a trading session"):
            assigner.assign_sessions(frame, bar_frequency="intraday", outside_session="raise")
        assert assigner.assign_sessions(
            frame, bar_frequency="intraday", outside_session="drop"
        ).is_empty()
    assert_frame_equal(frame, before)


def test_nullable_nanosecond_instants_keep_original_precision_and_null_policy():
    frame = minute_observation("2025-07-07", "13:30").with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC")) + pl.duration(nanoseconds=1)
    )
    missing = frame.with_columns(pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias("timestamp"))
    observed = pl.concat([frame, missing])
    before = observed.clone()
    assigned = SessionAssigner(
        feed_spec=FeedSpec(timestamp_semantics="event_time")
    ).assign_sessions(observed)
    assert assigned["session_date"].to_list() == [date(2025, 7, 7), None]
    assert_frame_equal(assigned.select(observed.columns), before)
