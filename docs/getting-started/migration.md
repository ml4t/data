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

## Prepare dated identities and known-at history

A mutable ticker is not an instrument ID. Replace static symbol dictionaries with captured
identity intervals when normalizing saved data:

```python
from ml4t.data.assets import InstrumentRegistry

registry = InstrumentRegistry.from_frame(identities, memberships, feed_spec=identity_feed)
asset = registry.resolve("OLD", "2025-07-03", exchange="NYSE", as_of="2025-07-03T18:00:00Z")
universe = registry.lifetimes(
    "2025-07-03", as_of="2025-07-03T18:00:00Z", universe="captured_index"
)
```

Identity rows require stable `asset`,source `symbol`, `exchange`, `instrument_type`, `source`,
`revision_id` and `valid_from`. `valid_to` is exclusive. Use `entity_col` for an existing stable-ID
field such as `sid`. Source listing/delisting columns are retained. Memberships are a separate
captured table keyed by asset,universe,source and interval. Neither missing quotes nor absence
from an incomplete table establishes delisting or universe completeness. Multiple venues require
an exchange; overlapping or ambiguous intervals fail before output.

Do not replace a runtime Zipline `data.history(...)` call with a Data history engine. Backtest
owns runtime history requests. Data prepares the source inputs and preserves pre-entry observations
for that consumer. Replace retrospective whole-file adjustment during input preparation with an
explicit cutoff:

```python
from ml4t.data.adjustments import prepare_adjusted_history

prepared = prepare_adjusted_history(
    prices, actions, as_of="2025-07-02T21:00:00Z", feed_spec=price_feed,
    instruments=registry, identity_mode="symbol", action_coverage="complete",
)
native_prices = prepared.prices
consumer_feed = prepared.metadata["feed_spec"]
```

In symbol mode, `price_feed.entity_col` names the captured source symbol column. It is preserved;
the registry's stable-ID column is added. Use a distinct physical symbol name,not `asset` for both
meanings. Stable-ID mode is the default and validates dated identity coverage when a registry is
supplied. Without a registry,caller-supplied stable IDs remain usable. A separate `action_feed_spec`
maps existing action entity/effective-time fields without renaming source columns on disk.

Each causal price/action/identity revision needs actual `available_at`, `source` and `revision_id`.
Observation time,effective time and availability are different. Only revisions available by the
cutoff are eligible. Future symbol changes,price corrections and effective actions cannot change
earlier causal output. Unknown relevant availability fails. `as_of=None` explicitly selects latest
retrospective revisions and may change when later facts arrive; no historical availability is
invented. Timestamps retain nanosecond precision. Date identity intervals and price session labels
use their declared calendar; instant identity boundaries are not inferred from civil dates.

The native default is total-return research history on the latest applicable known share basis.
`basis="split_adjusted"` excludes cash factors. `basis="raw"` retains raw prices and does not require
unused action facts. Adjusted history requires the caller to declare complete source action
coverage for the supplied range. That declaration cannot be deduced from an empty event table.
Native cash factors need an actually published ex-date close,not a provisional quote before the
selected exchange close. Intraday preparation needs a separate captured `ex_date_closes` table
with matching currency,revision and availability fields. Explicit `calendar=None` disables schedule
checks but still requires an explicit available close fact for causal cash adjustment. Named
`24/7` remains a schedule. Explicit argument,declared feed and NYSE retain their precedence.

Imported factors keep their source definition through `factors=...`,an explicit `basis` and
`factor_method`. They must exactly cover requested observations. A causal supplied-factor history
requires actual version/availability and one known `basis_at` instant per asset. A future basis,
a basis preceding applicable known actions,or conflicting source method/currency fails. An
unversioned legacy factor capture remains usable only through the explicitly retrospective route,
with unknown provenance recorded. Supplied merger/stock-distribution research factors do not
invent destination consideration,cash,positions or account transitions. Volume adjusts only when
its factor is supplied.

Raw columns remain alongside `adj_*` columns. `prepared.metadata['feed_spec']` points consumers
to the requested adjusted price/volume fields and stable IDs; source/input feed declarations remain
separate. `prepared.actions` contains selected facts and `prepared.identities` the matched identity
provenance. Coverage distinguishes observed null prices from absent scheduled observations between
the supplied endpoints. It does not fill either kind of gap or claim completeness outside that
range. Intraday coverage requires the complete declared version-2 bar grid.

Run the Data-only CSV-to-disk recipe:

```bash
uv run python examples/dated_history_migration.py
```

The recipe retains July1/2 warmup prices `198,200` at the July2 cutoff. A later correction to `999`
does not change that input. The July3 split/cash facts produce the independent native shareholder
return `1.01`; a July7 symbol rename retains ID17. July4 is an exchange closure,not a missing
observation. The example needs no Reloaded,Engineer,Diagnostic or Backtest runtime dependency.
These are unreleased source APIs. Isolated wheel and source-distribution checks qualified Data
`0.2.2.dev35+g2c4ff2050`, Specs `0.2.0`, Diagnostic `0.1.9` and the immutable Engineer source
candidate identified above, with pandas `3.0.6` and Polars `2.0.0`. The installed Engineer public
`features.trend.sma` consumed Data's two pre-entry observations and produced `null,199`; the
price-sensitive decision `200 > 199` remained true after the future correction. Engineer's public
label API and Diagnostic's forward-return API consumed the later adjusted prices and produced
`1/99,0.01,3/98,null` while retaining the raw prices, stable ID and renamed source symbol. These
checks qualify Data's producer and analytical handoff, not Backtest runtime/account behavior or
released Engineer-package compatibility. Subsequent action-ratio validation rejects nonfinite
derived share terms; it does not change these native adjustment definitions.

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

## Prepare individual futures inputs

Keep the existing `ContinuousContractBuilder` for continuous research prices. To prepare the
individual contracts needed by a simulation, load saved observations and captured dated definitions,
then request the source facts known at each actual decision cutoff:

```python
from ml4t.data.futures import HighestVolumeRoll, prepare_futures_inputs

prepared = prepare_futures_inputs(
    bars, definitions, as_of="2025-07-09T21:30:00Z",
    roll_strategy=HighestVolumeRoll(), feed_spec=feed,
)
```

Run the complete offline file-loader, continuous-builder and prepared-input comparison:

```bash
uv run python examples/futures_input_migration.py
```

The fictional TEST contracts use the CME Equity schedule, not a claim about a real contract's
specification. The earlier cutoff selects ID17. Previous-observation volume later selects ID23;
actual paired marks are4020/4030. Delayed ID23 open interest does not qualify the same decision,
so the OI method still selects17. Source files and caller tables remain unchanged. Neither route
fills missing observations. The legacy continuous builder does not itself certify historical
field-publication times.

Stable IDs identify contracts; they do not replace source names in financial tie ordering.
Prepared selections use each observation's dated definition symbol for ties, with the stable ID
string as a final disambiguator. Native `select_contracts(..., tie_breaker_col=...)` exposes this
mapping explicitly; its default remains `symbol`. Equal-expiry time selections now use a
deterministic symbol tie rather than interpreter-dependent set order. A rename alone is not a
contract switch, and a current alias does not replace an earlier ranking observation's name.

Input bars need declared timestamp/entity mappings, raw reference prices, `source`, `revision_id`,
`available_at`, `currency`, `price_quote_unit` and nonempty `mark_kind`. Ranking metrics need their
units in the definitions. Volume defaults to the bar's publication time unless
`volume_available_at` is supplied. OI needs its own `open_interest_available_at` for causal ranking;
do not copy a price's time into it. Supply `<metric>_revision_id` when metric revisions have their
own identities. Repeated unchanged metrics retain their original capture; indistinguishable
publication ties fail. `metrics` contains eligible field vintages. Raw `bars` deliberately retain
captured OI values even when that field was published later; do not use them as causal OI features.

Definitions extend the dated identity contract with root, venue, future instrument type, currency,
multiplier, tick size, quote unit, settlement type and expiration. First-notice, last-trade and
auto-close facts stay unknown unless supplied. Lifecycle fields retain Date versus Datetime
semantics and precision. Known expiration/last-trade instants end eligibility at that instant;
Date facts do not invent an intraday cutoff. A current selection also requires a currently valid
identity interval. Stale bars cannot backdate a time/notice trigger. The request fails rather than
guessing an unobserved replacement contract or an execution.

Unknown contract terms are checked only for definitions actually used by observations or the
current selected contract. An unrelated or future-effective definition cannot block an earlier
valid request. For instant-labelled bars, both observation and publication must be at or before
the cutoff. Publishing a preliminary end-labelled bar early does not make its future close visible.

Use `definitions_feed_spec` when definitions use different identity/temporal mappings, and
`identity_mode="symbol"` for explicit dated symbol-to-ID resolution. Calendar precedence remains
explicit `calendar`, declared feed, then NYSE. Explicit `None` disables scheduling; it does not
enable a session-count expiry policy without a schedule. Daily period-open/close instants use the
selected authoritative session, including CME overnight trading. This preparation accepts one
declared observation per contract/session; it is not an intraday execution engine.

`definitions` and `chain` retain the facts used for observations and current decisions. Selection
`definition_*` fields identify the current facts separately from each bar's `instrument_*` facts.
Roll rows retain both raw marks, timestamps, publication/source/revision IDs, currencies and
normalized marks/multipliers. Missing paired marks or conflicting mark kinds/currencies fail.
`coverage` counts observed rows, not proof of a complete source chain or a calendar-complete price
history. Use the calendar gap report for missing observations.

Raw prices and volumes stay unchanged. With no conversion requested, normalized prices preserve
their source dtype, including large integers. Explicit `price_quote_unit` conversion creates
floating-point price fields with declared factors and adjusted multipliers/ticks; it performs no
FX conversion. Read `metadata['normalized_feed_spec']` to consume those fields and
`metadata['feed_spec']` for raw fields. A multiplier50 and price change10 imply USD500 per contract.
The same calculation after cents-to-dollars conversion uses price change0.1 and multiplier5000.
Neither notional value nor these marks establish cash settlement, margin or fills.

These are unreleased source APIs. Isolated wheel and sdist checks qualified Data
`0.2.2.dev57+ge83e6b086`, Specs `0.2.0`, pandas `3.0.6`, Polars `2.0.0` and immutable Backtest
[`12a2c483f`](https://github.com/ml4t/backtest/commit/12a2c483f8f1fda54244af187db8bf44e6f85fc9)
(`0.1.13.dev22+g12a2c483f`) on Python `3.12.11`. Actual Backtest `ContractSpec` and `Position`
consumed the produced raw and converted fields, retaining IDs, paired marks and units; both
produced the independently expected USD500 price P&L. These checks do not qualify simulated
roll fills, settlement, margin or a complete released-package migration.
Data does not replace Backtest's runtime history, account, roll execution or auto-close policy.
The recipe needs no credentials, network, Reloaded package or Backtest installation.

This source candidate requires Polars `>=1.36.1` and pandas-market-calendars `>=5.4.0`.
Isolated runs using the former advertised minima failed during ordinary Data import or calendar
construction. The saved-file recipe passes with Polars `1.36.1`, pandas-market-calendars `5.4.0`
and pandas `2.3.3`; those checks do not certify every possible dependency-version combination.
The installed minimum pair passed 415 input/calendar/conversion cases with Arrow and 405 with 10
optional Arrow skips without it. Captured Arrow Parquet timestamps retain adjacent nanoseconds;
Arrow Date32/64 values, including nullable/empty inputs, remain civil dates.

Built-in selectors use the dated source symbol to break ties, then stable ID strings. Identity
does not change when a source symbol changes. Custom strategy subclasses retain the original
`select_contracts(data, contract_spec=None)` call contract and their own tie policy; they are not
required to accept the built-in selectors' optional `tie_breaker_col` keyword.

## Empyrical acquisition utilities

Change acquisition imports without installing a legacy analytics runtime:

```python
# Previously: from empyrical.utils import get_fama_french, default_returns_func
from ml4t.data.compat.empyrical import get_fama_french, default_returns_func

factors = get_fama_french("2024-01-02", "2024-01-05")
benchmark = default_returns_func("SPY", start="2024-01-02", end="2024-01-06")
```

These are data utilities only. Diagnostic owns analytical Empyrical replacements. Supplied-table
analysis does not import Data or fetch data unless the caller explicitly selects acquisition.
The Yahoo path needs the optional `ml4t-data[yahoo]` dependency. Factor downloads need no key;
Treasury acquisition can use public FRED CSV without a key. When `FRED_API_KEY` is configured,
API errors propagate rather than silently falling back to anonymous access.
Yahoo acquisition failures retain Data's provider exception types and emit the legacy
`Yahoo Finance read failed` warning. An unavailable Yahoo extra raises an actionable import error.

| Helper | Retained behavior and material differences |
|---|---|
| `get_fama_french(start="1-1-1970", end=None, datasets=...)` | Default FF3 plus momentum, ordered `Mkt-RF,SMB,HML,RF,Mom`, decimal returns and complete-case UTC date sample. Supported source names map to existing provider datasets; unknown names fail. Monthly bounds select the labeled period, including a month whose label precedes a mid-month start. Native provider filtering is unchanged. |
| `load_portfolio_risk_factors(filepath_prefix=None, start=None, end=None)` | `filepath_prefix` is the actual CSV filename. Slice the requested dates after legacy file-mtime cache selection. |
| `get_treasury_yield(start=None, end=None, period="3MO")` | One-column pandas frame, mapped `DGS1MO/3MO/6MO/1/5/10`, annual percent yield. Legacy forward-fill only affects existing reported rows; initial missing values and unreported dates are not invented. `source_missing_mask` retains the reported missing cells. Native `fetch_series` never fills them. |
| `get_symbol_returns_from_yahoo(symbol, start=None, end=None)` | One-column adjusted-close simple-return frame, UTC dates and upstream exclusive end. No price filling or second dividend adjustment; a missing price also prevents a return through that gap. Infinite percentage returns fail. |
| `default_returns_func(symbol, start=None, end=None)` | Named pandas Series; SPY uses `spy.csv`, other symbols use the same direct return route. Suitable for Pyfolio's existing return-function hook. |
| `get_returns_cached(filepath, update_func, latest_dt, **kwargs)` | DataFrame callback, unchanged keyword forwarding, actual caller CSV path and file-mtime freshness. A fresh file is not proof of observation coverage. Validate keys/values before atomic replacement; a failed refresh preserves the old file. A cache-write failure warns and returns valid fetched data without claiming it was persisted. |
| `cache_dir(environ=...)`, `data_path(name)`, `ensure_directory(path)` | Preserve `EMPYRICAL_CACHE_DIR`, then `XDG_CACHE_HOME/empyrical`, then `~/.cache/empyrical`. Native Data paths are unchanged. A file/directory collision fails explicitly. |

RF factor values are returns for their source frequency: `0.02%` becomes `0.0002`. A Treasury
yield such as `5.46` remains an annual percent yield, not that day's risk-free return. Source
observation dates, vintage requests and download times do not establish historical publication.
Native scalar inputs retain unknown availability, route and unit provenance explicitly.

The default previous-weekday recipe uses pandas `BDay`, not an exchange calendar. For a Monday
following an exchange holiday on Friday, it still selects Friday. Prefer explicit bounds for new
work. Pandas index names and period/SDK clock units follow the original data presentation for the
installed pandas version. CSV parsing retains its legacy UTC normalization; CSV does not preserve
the full native metadata declaration. Use portable snapshots when that declaration must survive.

Direct compatibility factor calls bypass the native Parquet cache so an outer CSV refresh cannot
read stale inner factors. Native callers can choose refresh/offline/age policy separately.
Native factor tables retain source nulls and canonical `MOM`; only this opt-in presentation uses
`Mom` and the legacy complete-case sample. No analytical cleaner, feature engine or broker is added.

For existing Pyfolio workflows, register the same callable through its own public hook:

```python
from pyfolio.utils import register_return_func
from ml4t.data.compat.empyrical import default_returns_func

register_return_func(default_returns_func)
```

The offline [acquisition migration example](https://github.com/ml4t/data/blob/3dd71cb394c390316ff0bb36ad1d7d0f3dd6aefc/examples/acquisition_migration.py)
replays retained public response captures through actual parsers, providers and CSV files. It runs
without credentials or network and does not require any Reloaded package. This is an unreleased
source workflow, not a claim that a released successor stack already replaces every legacy option.

Isolated wheel and source-distribution checks qualified Data `0.2.2.dev27+g5ff44e4ac`,
Specs `0.2.0`, Diagnostic `0.1.9`, Pyfolio Reloaded `0.10.0` at immutable commit
`365e649f591c39cef5d0170958280e49b8eeb953`, and Alphalens Reloaded `0.5.0` at
`1bc181de4aafbe33437ed0e362ff96aa7ce20acd`. The checks exercised actual Diagnostic
`FactorData`, Pyfolio's registered return callback and Alphalens' forward-return calculation from
Data-produced inputs. They used Python `3.12`, pandas `2.3.3`, Polars `2.0.0` and yfinance
`1.7.0`. The selected dependency-floor lane separately passed with Polars `1.36.1`,
pandas-market-calendars `5.4.0` and yfinance `1.5.2`. Older yfinance versions admitted by the
former `>=0.2.0` declaration lack the exception API used by Data's empty-download recovery;
the Yahoo extra now requires the qualified `>=1.5.2` range.

## Reconstruct inputs after source pruning

Retain the input tables and declarations instead of referring to mutable source paths. Run the
offline [portable snapshot example](https://github.com/ml4t/data/blob/a98cede0a36ba6856a010456f302e3e987d255de/examples/portable_snapshot_migration.py)
from a source checkout:

```bash
uv run python examples/portable_snapshot_migration.py
```

The recipe reuses the dated-history and individual-futures examples. It captures six tables,
their separate source cutoffs and revisions, stable identity namespaces, units, raw price/share
basis, feed declarations and effective selection configuration. It then prunes source generations,
changes the original futures file, relocates the archive, and reconstructs the same prepared inputs.
The checks compare equity keys, ticker changes, action facts, adjusted prices/volume, selected
contracts and paired roll marks. Missing quotes remain missing. No Reloaded package, credentials
or network is needed. Read the [snapshot storage contract](../user-guide/storage.md#portable-input-snapshots)
for required metadata and integrity rules.

Effective configuration must be supplied by the caller. A Data snapshot does not establish a
common historical publication time for independently sourced tables, discover omitted execution
policies, or certify an account result. Backtest owns runtime history, execution and account
reconstruction; Engineer owns feature computation and Diagnostic owns analytical fill views.

### Convert an existing Zipline bundle once

In a separate conversion environment with Zipline Reloaded installed, load the original bundle
through its standard API and export selected stable source IDs and inclusive reader bounds:

```python
from zipline.data import bundles
from ml4t.data.compat.zipline import export_bundle

with bundles.load("my-existing-bundle") as bundle:
    snapshot = export_bundle(
        bundle,
        "captured-bundle",
        namespace="original-bundle-permanent-ids",
        source_revision="original-ingestion-id",
        sids=[1, 2],
        start="2024-01-02",
        end="2024-01-04",
        frequencies=("daily",),
    )
```

Keep the namespace stable across captures; exported identities combine it with the original sid.
The bridge reads standard asset, bar and adjustment readers without modifying the source bundle.
Native tables include raw observations, source history-adjustment factors, dated asset definitions,
symbol lookup intervals, and corporate-action facts. A bounded stock-dividend selection also retains
the payment asset's identity metadata, without creating price observations for it.

Daily bounds are civil session labels. Minute bounds are UTC instants; use `frequency_bounds`
when selecting daily and minute data together. Minute timestamps retain the source's right-labeled
bar-close convention. Source reader zeros in volume remain zeros without an inferred no-trade
status. Source schedule/provider declarations remain separate from Data's selected calendar;
`calendar=None` disables the native schedule without changing the captured source grid.

Zipline split price factors are converted to native new-shares-per-old-share terms. Source
history factors remain a declared, supplied adjustment method; they are not recomputed as a
different vendor's total-return definition. Publication times, quote currency, merger consideration
and original listing endpoints that public readers cannot establish remain explicitly unavailable.
Symbol ownership lookup intervals must not be interpreted as original listing intervals.
Capturing unknown publication times does not make a bundle suitable for causal known-at claims.

For staged migration, `make_bundle_ingest("captured-bundle")` returns a standard Zipline bundle
ingestion callback. Register it with the same source calendar and a range covering the captured
bars, then use normal `bundles.ingest` and `bundles.load`. The callback verifies all files and
prepares all tables before invoking writers. Asset metadata, source action-table indexes and raw
daily/minute values round-trip through standard writers. A target minute writer must have enough
price precision for the source; incompatible scales fail before writes instead of rounding prices.
Configure `BcolzMinuteBarWriter`'s per-sid scale when needed. Native snapshots impose no legacy
integer price scale. Reverse ingestion is optional and is not required for native reconstruction.

The actual optional conversion checks use immutable Zipline Reloaded `3.2.0` source
[`cf649ac06`](https://github.com/stefan-jansen/zipline-reloaded/commit/cf649ac06fddce76f6caa2403cd8494ac84f72ad),
Python `3.12`, pandas `2.3.3`, Specs `0.2.0` and Polars `2.0.0`. They construct a real bundle
with renamed equities, two venues, two futures, missing quotes, split/cash/stock actions and daily
and minute observations, then compare native preparation and standard re-ingestion. The native
recipe requires no legacy runtime. This is unreleased source functionality; it does not claim that
published Data `0.2.1` provides these APIs or that Data replaces Zipline's runtime or broker.

Clean wheel and source-distribution installs qualified Data `0.2.2.dev27+g90c06ba9c` for native
reconstruction, the optional conversion and the selected dependency floors. Reconstructed tables
also passed actual Engineer labeling, Diagnostic forward returns and Backtest contract/position
checks using Engineer `0.1.7.dev29+gc69e7bada`, Diagnostic `0.1.9` and Backtest
`0.1.13.dev22+g12a2c483f`. That Backtest check verifies a USD500 marked position result, not
cash settlement or corporate-action account execution. The source recipes and input contracts
remain usable without any of those consumers. CI's separate `Installed bundle migration` job
keeps the optional conversion qualified against the pinned Zipline source without adding it to
Data's production dependencies.
