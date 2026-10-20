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

## Import corporate-action facts without changing their financial definition

Use `prepare_action_inputs` for captured event tables. Each row needs stable `asset`, `source`,
`event_id`, `revision_id` and `action_kind` fields. Dates, source identifiers and raw terms remain
in the result. Map existing `sid` or clock columns through `FeedSpec`; a ticker is not a stable ID.
Pass a captured unique-ID instrument table when its membership must be checked. Dated identity
resolution is a separate operation.

Zipline split ratios multiply historical prices. Declare that convention when importing them:

```python
from ml4t.data.adjustments import prepare_action_inputs

facts = prepare_action_inputs(actions, instruments, split_convention="price_factor")
```

A source ratio of `0.5` becomes native new-shares-per-old-share ratio `2`. The original ratio and
source convention remain in `raw_split_ratio` and `source_split_convention`. A native prepared
table can be validated again without reciprocating twice. Rational numerator/denominator terms
are retained and must agree with a supplied ratio.

Unknown availability, destination IDs, consideration and payment dates stay null and appear in
`missing_terms`. Use `required_terms` to reject unavailable facts needed by a particular consumer.
Requirements apply to the relevant action kinds; a split does not require a cash payment date.
Absence of an event does not establish source coverage or prove that no action occurred.
Cash/stock distributions and mergers remain facts, without account or position effects.

Run the offline CSV, validation and storage recipe:

```bash
uv run python examples/action_input_migration.py
```

The source explicitly marks a no-action observation. The recipe validates all applicable price
adjustment terms before aggregation; it never replaces missing prices. The existing native
`apply_corporate_actions` uses the ex-date close. Raw close `200` followed by `98`, a `2:1` split
and `USD3` cash per post-event share yield factor `98/202`, prior adjusted volume `2000` and gross
shareholder return `1.01`. The event row stays unchanged. These native defaults are preserved.

Use `apply_adjustment_factors(prices, factors, method="vendor_prior_close")` to retain imported
per-observation factors instead. The recipe's supplied factor `0.485` gives prior adjusted close
`97`, rather than the native `97.029703...`. Neither function claims these definitions are equal.
Factor keys must exactly cover the raw observations. Volume is adjusted only when its own factor
is supplied. Raw prices, volume and factor provenance remain available; known conflicting currency
units, reserved output fields and incomplete or duplicate keys fail without input mutation.

These are unreleased source APIs. The Data-only recipe needs no Reloaded, Engineer or Backtest
dependency. Prepared adjusted research prices are not execution marks with an additional copy of
the same dividend cash event. The consuming broker owns entitlement, payment, orders and positions.
Source availability remains separate from effective time; this preparation step makes no causal
as-of claim.

Isolated wheel and source-distribution checks qualified Data `0.2.2.dev23+g0917c8401`,
Specs `0.2.0`, Diagnostic `0.1.9` and the immutable Engineer source candidate identified above.
The installed Data recipe produced two versioned event facts. Both consumers' public return/label
APIs produced `0.01` for native adjusted prices and `1/97` for the supplied-factor prices, while
retaining raw prices `200, 98` and volumes `1000, 7`. These checks qualify preparation and
analytical input consumption, not Backtest's corporate-action account transitions or published
Engineer-package compatibility.

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
