"""Portable snapshot contracts through real storage generations and files."""

import hashlib
import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from threading import Event
from unittest.mock import patch

import polars as pl
import pytest
from filelock import Timeout
from polars.testing import assert_frame_equal

from ml4t.data.storage import (
    FlatStorage,
    HiveStorage,
    SnapshotTable,
    StorageConfig,
    export_snapshot,
    read_snapshot,
)
from ml4t.specs import ArtifactStorage, FeedSpec, MarketDataSpec


def declaration(**overrides):
    return {
        "source_revision": "captured-v1",
        "identity_namespace": "test-market",
        "units": {"close": "USD/share", "volume": "shares"},
        "as_of": "2025-07-08T16:00:00Z",
        "price_basis": "raw",
        "share_basis": "source",
        "revisions": {"identity": "i1", "actions": "a1", "chain": None},
        **overrides,
    }


def observations():
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [1751554800000000001, 1751554800000000002, 1751900400000000003],
                dtype=pl.Datetime("ns", "UTC"),
            ),
            "asset_id": ["same-id"] * 3,
            "close": [100.0, None, 101.0],
            "volume": [0, None, 200],
        }
    )


def table(frame=None, **overrides):
    return SnapshotTable(
        observations() if frame is None else frame,
        "prices",
        declaration(),
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar="NYSE",
            entity_col="asset_id",
            data_frequency="minute",
            timestamp_semantics="event_time",
        ),
        **overrides,
    )


def export(path, tables=None, **overrides):
    options = {
        "effective_config": {"consumer": "supplied", "costs": {"commission": 0.01}},
        "artifacts": {"consumer-wheel": "1" * 64},
        **overrides,
    }
    return export_snapshot(path, {"prices": table()} if tables is None else tables, **options)


@pytest.mark.parametrize("backend_type", [FlatStorage, HiveStorage])
def test_snapshot_survives_generation_pruning_and_relocation_without_ns_or_null_loss(
    tmp_path, backend_type
):
    backend = backend_type(StorageConfig(base_path=tmp_path / "source"))
    frame = observations()
    metadata = {**declaration(), **table().feed_spec.to_dict()}
    backend.write(frame, "prices/SPY", metadata)
    source = SnapshotTable(backend, "prices", {}, key="prices/SPY")
    captured = export(tmp_path / "capture", {"prices": source})
    original_generation = backend._current_commit("prices/SPY").generation_path
    for index in range(5):
        backend.write(
            frame.with_columns(pl.lit(1000.0 + index).alias("close")), "prices/SPY", metadata
        )
    assert not original_generation.exists()
    relocated = tmp_path / "different-machine-directory"
    shutil.move(tmp_path / "capture", relocated)
    restored = read_snapshot(relocated)
    assert restored.snapshot_id == captured.snapshot_id
    assert_frame_equal(restored.tables["prices"], frame)
    assert restored.tables["prices"]["timestamp"].cast(pl.Int64).to_list() == [
        1751554800000000001,
        1751554800000000002,
        1751900400000000003,
    ]
    clock = restored.manifest["content"]["tables"]["prices"]["clock"]
    assert clock["resolved_calendar"] == "NYSE"
    assert clock["schedule_provider"] == "pandas-market-calendars"
    assert clock["schedule_provider_version"]
    assert str(tmp_path) not in json.dumps(restored.manifest)


def test_snapshot_preserves_local_session_labels_and_calendar_omission_vs_null(tmp_path):
    frame = pl.DataFrame({"day": [date(2025, 7, 3), date(2025, 7, 7)], "close": [50.0, 51.0]})
    source = table(frame)
    omitted = replace(
        source,
        feed_spec=FeedSpec(
            time_contract_version=2, timestamp_col="day", timestamp_semantics="session_label"
        ),
    )
    disabled = replace(omitted, feed_spec=omitted.feed_spec.with_overrides(calendar=None))
    first = export(tmp_path / "omitted", {"prices": omitted})
    second = export(tmp_path / "disabled", {"prices": disabled})
    explicit = export(tmp_path / "explicit", {"prices": disabled}, calendar="24/7")
    assert_frame_equal(first.tables["prices"], frame)
    clocks = [
        item.manifest["content"]["tables"]["prices"]["clock"] for item in (first, second, explicit)
    ]
    assert "calendar" not in clocks[0]["feed_spec"]
    assert clocks[0]["resolved_calendar"] == "NYSE"
    assert clocks[1]["feed_spec"]["calendar"] is None
    assert clocks[1]["resolved_calendar"] is None
    assert clocks[1]["schedule_provider"] is None
    assert clocks[2]["resolved_calendar"] == "24/7"
    assert clocks[2]["schedule_provider_version"]
    assert len({item.snapshot_id for item in (first, second, explicit)}) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("as_of", "2025-07-09T16:00:00Z"),
        ("source_revision", "captured-v2"),
        ("price_basis", "split-adjusted"),
        ("share_basis", "perspective"),
        ("identity_namespace", "another-source"),
        ("revisions", {"identity": "i2", "actions": "a1", "chain": "c1"}),
        ("units", {"close": "cents/share"}),
    ],
)
def test_snapshot_identity_changes_with_financial_input_declarations(tmp_path, field, value):
    before = export(tmp_path / "before")
    changed = replace(table(), metadata=declaration(**{field: value}))
    after = export(tmp_path / "after", {"prices": changed})
    assert after.snapshot_id != before.snapshot_id


def test_snapshot_identity_includes_actual_input_values_and_effective_consumer_policies(tmp_path):
    first = export(tmp_path / "first")
    data_changed = export(
        tmp_path / "data",
        {"prices": table(observations().with_columns(pl.lit(2.0).alias("close")))},
    )
    policy_changed = export(tmp_path / "policy", effective_config={"costs": {"commission": 0.02}})
    assert len({item.snapshot_id for item in (first, data_changed, policy_changed)}) == 3


def test_snapshot_identity_ignores_capture_time_json_key_order_and_storage_generation_ids(tmp_path):
    backend = FlatStorage(StorageConfig(base_path=tmp_path / "source"))
    metadata = {**declaration(), **table().feed_spec.to_dict()}
    backend.write(observations(), "bars", metadata)
    source = SnapshotTable(backend, "prices", {}, key="bars")
    first = export(tmp_path / "first", {"prices": source})
    backend.write(observations(), "bars", dict(reversed(list(metadata.items()))))
    second = export(
        tmp_path / "second",
        {"prices": source},
        effective_config={"costs": {"commission": 0.01}, "consumer": "supplied"},
    )
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest["captured_at"] != second.manifest["captured_at"]
    assert (
        first.manifest["captures"]["prices"]["generation"]
        != second.manifest["captures"]["prices"]["generation"]
    )
    assert export(tmp_path / "first", {"prices": source}).snapshot_id == first.snapshot_id


def test_snapshot_records_a_per_table_revision_tuple_not_an_invented_common_vintage(tmp_path):
    backend = FlatStorage(StorageConfig(base_path=tmp_path / "source"))
    for key, revision, as_of in (("prices", "p1", "2025-07-08T16:00:00Z"), ("actions", "a7", None)):
        backend.write(
            observations(),
            key,
            {**declaration(source_revision=revision, as_of=as_of), **table().feed_spec.to_dict()},
        )
    sources = {key: SnapshotTable(backend, key, {}, key=key) for key in ("prices", "actions")}
    result = export(tmp_path / "snapshot", sources)
    entries = result.manifest["content"]["tables"]
    assert entries["prices"]["metadata"]["source_revision"] == "p1"
    assert entries["actions"]["metadata"]["source_revision"] == "a7"
    assert entries["prices"]["metadata"]["as_of"] == "2025-07-08T16:00:00Z"
    assert entries["actions"]["metadata"]["as_of"] is None
    assert set(result.manifest["captures"]) == {"prices", "actions"}
    assert (
        result.manifest["captures"]["prices"]["commit"]
        != result.manifest["captures"]["actions"]["commit"]
    )


def test_all_snapshot_file_hashes_are_checked_before_any_table_is_parsed(tmp_path):
    export(tmp_path / "snapshot", {"a": table(), "z": table()})
    path = tmp_path / "snapshot/tables/z/data.parquet"
    path.write_bytes(b"interrupted or corrupt file")
    with (
        patch(
            "ml4t.data.storage.snapshot.pl.read_parquet",
            side_effect=AssertionError("parsed before all hashes checked"),
        ),
        pytest.raises(ValueError, match="hash mismatch"),
    ):
        read_snapshot(tmp_path / "snapshot")


def test_snapshot_rejects_manifest_tampering_without_hiding_coverage_or_source_tuple(tmp_path):
    export(tmp_path / "snapshot")
    path = tmp_path / "snapshot/manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["captures"] = {"prices": {"commit": "invented"}}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checksum"):
        read_snapshot(tmp_path / "snapshot")


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.parquet",
        "/outside.parquet",
        "tables/prices/../../outside.parquet",
        "tables/prices/./data.parquet",
        "tables\\prices\\data.parquet",
    ],
)
def test_snapshot_rejects_manifest_path_escapes_even_if_manifest_is_rehashed(tmp_path, relative):
    export(tmp_path / "snapshot")
    path = tmp_path / "snapshot/manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["content"]["tables"]["prices"]["files"][0]["path"] = relative

    def canonical(item):
        return json.dumps(item, sort_keys=True, separators=(",", ":")).encode()

    manifest["snapshot_id"] = hashlib.sha256(canonical(manifest["content"])).hexdigest()
    del manifest["manifest_sha256"]
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_snapshot(tmp_path / "snapshot")


def test_snapshot_rejects_internal_file_symlinks_instead_of_following_them(tmp_path):
    export(tmp_path / "snapshot")
    file = tmp_path / "snapshot/tables/prices/data.parquet"
    other = tmp_path / "same-bytes.parquet"
    file.rename(other)
    file.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        read_snapshot(tmp_path / "snapshot")


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_revision": "incomplete"},
        declaration(source_revision=None),
        declaration(api_key="do-not-serialize"),
        declaration(source="https://source.invalid/download?api_key=do-not-serialize"),
        declaration(local_path="/private/source"),
        declaration(quality=float("nan")),
    ],
)
def test_invalid_snapshot_declarations_leave_a_previous_valid_result_untouched(tmp_path, metadata):
    before = export(tmp_path / "snapshot")
    manifest = (tmp_path / "snapshot/manifest.json").read_bytes()
    with pytest.raises((ValueError, TypeError)):
        export(tmp_path / "snapshot", {"prices": replace(table(), metadata=metadata)})
    assert (tmp_path / "snapshot/manifest.json").read_bytes() == manifest
    assert read_snapshot(tmp_path / "snapshot").snapshot_id == before.snapshot_id
    assert not list(tmp_path.glob(".snapshot.staging-*"))


def test_snapshot_copy_interruption_does_not_replace_previous_valid_output(tmp_path):
    backend = FlatStorage(StorageConfig(base_path=tmp_path / "source"))
    backend.write(observations(), "prices", declaration())
    before = export(tmp_path / "snapshot")
    with (
        patch("ml4t.data.storage.snapshot.shutil.copyfile", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        export(
            tmp_path / "snapshot",
            {"prices": SnapshotTable(backend, "prices", {}, key="prices")},
        )
    assert read_snapshot(tmp_path / "snapshot").snapshot_id == before.snapshot_id
    assert not list(tmp_path.glob(".snapshot.staging-*"))


def test_snapshot_never_overwrites_a_valid_different_immutable_input_set(tmp_path):
    first = export(tmp_path / "snapshot")
    with pytest.raises(FileExistsError, match="immutable"):
        export(tmp_path / "snapshot", effective_config={"commission": 100})
    assert read_snapshot(tmp_path / "snapshot").snapshot_id == first.snapshot_id


def test_snapshot_preserves_shared_artifact_declaration_with_a_portable_storage_location(tmp_path):
    artifact = MarketDataSpec("prices-v1", storage=ArtifactStorage(path="/old-machine/data"))
    result = export(
        tmp_path / "snapshot",
        {"prices": replace(table(), artifact=artifact)},
        unavailable={"merger_terms": "not supplied by source"},
    )
    content = result.manifest["content"]
    stored = content["tables"]["prices"]["artifact"]
    assert stored["artifact_id"] == "prices-v1"
    assert stored["kind"] == "market_data"
    assert stored["storage"]["path"] == "tables/prices"
    assert content["unavailable"] == {"merger_terms": "not supplied by source"}


@pytest.mark.parametrize("stored", [False, True])
def test_csv_snapshot_records_default_and_metadata_only_calendars_without_a_feed_object(
    tmp_path, stored
):
    path = tmp_path / "observations.csv"
    path.write_text("timestamp,symbol,close\n2025-07-04T12:00:00Z,SPY,100\n")
    frame = pl.read_csv(path, try_parse_dates=True)
    cases = [
        ({}, {}, "NYSE"),
        ({"calendar": None}, {}, None),
        ({"calendar": "24/7"}, {}, "24/7"),
        ({"calendar": "24/7"}, {"calendar": None}, None),
        ({"calendar": None}, {"calendar": "24/7"}, "24/7"),
    ]
    identities = []
    for index, (declared, explicit, expected) in enumerate(cases):
        metadata = declaration(**declared)
        if stored:
            backend = FlatStorage(StorageConfig(base_path=tmp_path / f"source-{index}"))
            backend.write(frame, "prices", metadata)
            source = SnapshotTable(backend, "prices", {}, key="prices")
        else:
            source = SnapshotTable(frame, "prices", metadata)
        captured = export(tmp_path / f"snapshot-{index}", {"prices": source}, **explicit)
        clock = captured.manifest["content"]["tables"]["prices"]["clock"]
        assert clock is not None
        assert clock["resolved_calendar"] == expected
        assert (clock["schedule_provider"] is None) == (expected is None)
        assert_frame_equal(captured.tables["prices"], frame)
        identities.append(captured.snapshot_id)
    assert len(set(identities)) == len(cases)


def test_snapshot_copies_a_generation_while_holding_the_actual_source_writer_lock(tmp_path):
    backend = FlatStorage(StorageConfig(base_path=tmp_path / "source"))
    backend.write(observations(), "prices", declaration())
    entered, release = Event(), Event()
    copy = shutil.copyfile

    def pause_during_copy(source, destination):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("source-copy test did not release its bounded barrier")
        return copy(source, destination)

    source = SnapshotTable(backend, "prices", {}, key="prices")
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        patch("ml4t.data.storage.snapshot.shutil.copyfile", side_effect=pause_during_copy),
    ):
        pending = executor.submit(export, tmp_path / "snapshot", {"prices": source})
        try:
            assert entered.wait(10)
            with pytest.raises(Timeout), backend._key_lock("prices").acquire(timeout=0):
                pytest.fail("snapshot copy did not hold the source writer lock")
        finally:
            release.set()
        captured = pending.result(timeout=10)
    backend.write(
        observations().with_columns(pl.lit(10000.0).alias("close")), "prices", declaration()
    )
    assert_frame_equal(captured.tables["prices"], observations())


@pytest.mark.parametrize("field", ["FRED_API_KEY", "access_token", "client_secret"])
def test_snapshot_rejects_provider_credential_fields_before_publishing_a_fresh_archive(
    tmp_path, field
):
    path = tmp_path / "new-snapshot"
    with pytest.raises(ValueError, match="credentials"):
        export(
            path,
            {"prices": replace(table(), metadata=declaration(**{field: "not-a-real-credential"}))},
        )
    assert not path.exists()


def test_csv_produced_snapshot_retains_table_role_schema_units_and_reported_missing_cells(tmp_path):
    source = tmp_path / "observations.csv"
    source.write_text(
        "timestamp,asset_id,close,volume\n2025-01-02,A,100,0\n2025-01-03,A,,\n2025-01-04,A,102,200\n"
    )
    frame = pl.read_csv(source, try_parse_dates=True, schema_overrides={"close": pl.Float64})
    feed = FeedSpec(
        time_contract_version=2,
        calendar=None,
        entity_col="asset_id",
        timestamp_semantics="session_label",
        data_frequency="daily",
    )
    result = export(
        tmp_path / "snapshot",
        {"observations": SnapshotTable(frame, "raw_observations", declaration(), feed_spec=feed)},
    )
    entry = result.manifest["content"]["tables"]["observations"]
    assert entry["role"] == "raw_observations"
    assert entry["schema"] == [
        ["timestamp", "Date"],
        ["asset_id", "String"],
        ["close", "Float64"],
        ["volume", "Int64"],
    ]
    assert entry["row_count"] == 3
    assert entry["metadata"]["units"] == {"close": "USD/share", "volume": "shares"}
    assert result.tables["observations"]["asset_id"].to_list() == ["A", "A", "A"]
    assert result.tables["observations"]["close"].to_list() == [100.0, None, 102.0]
    assert result.tables["observations"]["volume"].to_list() == [0, None, 200]


def test_snapshot_data_flush_failure_does_not_publish_a_manifest_or_replace_a_previous_archive(
    tmp_path,
):
    previous = export(tmp_path / "previous")
    sync = os.fsync

    def fail_data_flush(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            position = os.lseek(descriptor, 0, os.SEEK_CUR)
            os.lseek(descriptor, 0, os.SEEK_SET)
            header = os.read(descriptor, 4)
            os.lseek(descriptor, position, os.SEEK_SET)
            if header == b"PAR1":
                raise OSError("data volume full during flush")
        sync(descriptor)

    with (
        patch("os.fsync", side_effect=fail_data_flush),
        pytest.raises(OSError, match="data volume full"),
    ):
        export(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert read_snapshot(tmp_path / "previous").snapshot_id == previous.snapshot_id
