# Migrate existing data workflows

## Move a long price table to a pandas analytical panel

Keep source observations in Data storage. Use `to_price_panel` when an analytical consumer needs
a pandas DataFrame with timestamps on its index and stable asset IDs on its columns. Use
`from_price_panel` to reconstruct native price observations. Neither function computes returns,
changes adjustment basis, fills prices, infers sessions, or downloads data.

Previously, a caller might reshape prices directly:

```python
panel = prices.pivot(index="timestamp", columns="asset", values="close")
```

Replace the reshape with a declared handoff:

```python
from ml4t.data.utils.conversion import from_price_panel, to_price_panel
from ml4t.specs import FeedSpec

feed = FeedSpec(
    time_contract_version=2,
    calendar="NYSE",
    entity_col="asset",
    timestamp_semantics="event_time",
)
panel = to_price_panel(prices, feed_spec=feed, metadata={"price_basis": "raw"})
native_prices = from_price_panel(panel)
```

Run the offline CSV-to-disk example from a source checkout:

```bash
uv run python examples/price_panel_migration.py
```

The example retains July 2, 3, 7 and 8, including B's missing July 7 quote. A null price remains
null. An absent asset/timestamp pair remains absent on inverse conversion. The panel retains a
presence mask, feed declaration, original native dtypes and caller-supplied metadata in
`panel.attrs['ml4t_price_panel']`. Keep those attrs for a sparse inverse. If a consumer changes
the panel axes without updating the mask, inverse conversion fails rather than guessing.
An ordinary pandas panel without these attrs treats every cell, including nulls, as an observation.
The inverse emits clock-major rows and retains the first-seen clock and asset order.

This API projects one price field. Keep OHLCV, per-observation availability, source revisions and
corporate-action facts in their original tables. A null quote does not establish a holiday,
delisting or zero return. Duplicate keys and conflicting mappings fail without changing the input.
Pass existing physical names through `feed_spec` rather than renaming stored datasets.

Storage reads preserve datetime precision. Undeclared instant timestamps retain the existing UTC
read behavior. Naive instants with a declared source timezone are interpreted in that timezone
before conversion to UTC. Declared session labels with a timezone retain that declaration when
represented in UTC; labels without a declared timezone retain their source timezone. Naive bounds on labels use
the declared label clock; instant bounds use UTC. Read ranges are start-inclusive/end-exclusive.
Calendar omission, explicit `None` and a named `24/7` schedule remain distinct. Panel conversion
does not invoke a calendar resolver or create an expected clock.

These adapters are unreleased source functionality. Isolated wheel and source-distribution checks
qualified Data `0.2.2.dev14+gbe73c5d86`, Specs `0.2.0`, Diagnostic `0.1.9` and Engineer
`0.1.7.dev29+gc69e7bada` from immutable source
[`c69e7bada`](https://github.com/ml4t/engineer/commit/c69e7badaac012ad98b1c4b8aa9a67a952b08f5c).
The checks used pandas `3.0.6` and Polars `2.0.0`. Diagnostic's public
`metrics.basic.compute_forward_returns` consumed the reconstructed native table without dropping
B's missing quote. Engineer's public `labeling.fixed_time_horizon_labels` consumed A's complete
price history. Both produced the independently expected one-observation outcomes.

Engineer `0.1.7` is not yet a published release; published `0.1.6` requires Specs `<0.2` and
cannot be installed alongside current Data. Use the qualified immutable Engineer source for this
integration until a compatible Engineer release is available. This is source integration evidence,
not a claim of released-package migration compatibility. Neither consumer nor any Reloaded package
is required for the Data-only example.

## Names preserved through the 0.x series

The package preserves two former QLDM names during the 0.x release series. Their use emits
`DeprecationWarning`. They will not be removed before version 1.0.

## Data root

Replace:

```bash
export QLDM_DATA_ROOT=/srv/market-data
```

with:

```bash
export ML4T_DATA_PATH=/srv/market-data
```

## Base exception

Replace:

```python
from ml4t.data.core.exceptions import QldmError
```

with:

```python
from ml4t.data.core.exceptions import ML4TDataError
```

The former exception name resolves to the same class during the migration period. Code that treats
deprecation warnings as errors should migrate before upgrading.

## Changes for version 0.2

Version 0.2 removes three compatibility interfaces that emitted `DeprecationWarning` in the 0.1
series. Update callers and saved configuration before upgrading.

### Massive provider name

Replace `PolygonProvider` imports and the `polygon` registry name with `MassiveProvider` and
`massive`:

```python
from ml4t.data.providers import MassiveProvider

provider = MassiveProvider()
```

The service account can still use `POLYGON_API_KEY`; new environments should set
`MASSIVE_API_KEY`.

### Negative-price validation

Replace `check_negative_prices=True` with `negative_price_policy="forbid"`, and replace
`check_negative_prices=False` with `negative_price_policy="allow"`. Apply the same substitution in
saved `ValidationRuleConfig` YAML.

Use `negative_price_policy="warn"` when negative prices should produce validation warnings without
failing validation.

### Synthetic GARCH configuration

Remove the `garch_omega` argument from `SyntheticProvider`. The provider derives the per-period
constant from `annual_volatility`, the requested frequency, `garch_alpha`, and `garch_beta` so the
configured unconditional variance is preserved.
