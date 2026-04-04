# futures/ - Databento Futures Data

CME futures data downloading and continuous contract construction.

## Downloader Classes

| Class                  | Symbology           | Schema     | Use Case                            |
| ---------------------- | ------------------- | ---------- | ----------------------------------- |
| `ContinuousDownloader` | `ES.v.0` (volume)   | `ohlcv-1h` | Pre-rolled continuous, Hive output  |
| `IndividualDownloader` | `ESH24` (raw)       | `ohlcv-1h` | Specific contracts for roll demo    |
| `FuturesDownloader`    | `ES.FUT` (parent)   | `ohlcv-1d` | All contracts, bulk download        |
| `FuturesDataManager`   | `ES.FUT` (parent)   | `ohlcv-1d` | High-level interface + profiling    |

## Key Files

| File                      | Purpose                                      |
| ------------------------- | -------------------------------------------- |
| `continuous_downloader.py` | ContinuousDownloader class                   |
| `individual_downloader.py` | IndividualDownloader class                   |
| `downloader.py`            | FuturesDownloader class                      |
| `book_downloader.py`       | FuturesDataManager for book readers          |
| `databento_parser.py`      | Contract symbol parsing (MONTH_CODES, etc.)  |
| `continuous.py`            | ContinuousContractBuilder                    |
| `roll.py`                  | Roll strategies (VolumeBasedRoll, etc.)      |
| `adjustment.py`            | Price adjustment methods (BackAdjustment)    |
| `config.py`                | Configuration dataclasses                    |
| `schema.py`                | Contract specifications (ContractSpec)       |
| `definitions.py`           | DefinitionsDownloader                        |

## Contract Symbol Formats

| Format       | Example    | Description                        |
| ------------ | ---------- | ---------------------------------- |
| Continuous   | `ES.v.0`   | Volume-rolled front month          |
| Continuous   | `ES.c.1`   | Calendar-rolled second month       |
| Individual   | `ESH24`    | March 2024 contract                |
| Parent       | `ES.FUT`   | All contracts for product          |

## Month Codes

| Code | Month     | Code | Month     |
| ---- | --------- | ---- | --------- |
| F    | January   | N    | July      |
| G    | February  | Q    | August    |
| H    | March     | U    | September |
| J    | April     | V    | October   |
| K    | May       | X    | November  |
| M    | June      | Z    | December  |

## Usage Examples

```python
# Continuous contracts
from ml4t.data.futures import ContinuousDownloader, ContinuousDownloadConfig

config = ContinuousDownloadConfig(
    products=["ES", "CL"],
    start="2020-01-01",
    end="2025-12-31",
    tenors=[0, 1, 2],
    schema="ohlcv-1h",
)
downloader = ContinuousDownloader(config)
downloader.download_all()

# Individual contracts
from ml4t.data.futures import IndividualDownloader, IndividualDownloadConfig

config = IndividualDownloadConfig(
    products={
        "ES": {"months": [3, 6, 9, 12]},   # Quarterly
        "CL": {"months": list(range(1, 13))},  # Monthly
    },
    years=[2024, 2025],
    schema="ohlcv-1h",
)
downloader = IndividualDownloader(config)
downloader.download_all()
```

## Data Profiling (FuturesDataManager)

```python
from ml4t.data.futures import FuturesDataManager

manager = FuturesDataManager.from_config("config.yaml")

# Generate profile for specific product
profile = manager.generate_profile("ES")
print(profile.summary())

# Load existing profile
profile = manager.load_profile("ES")

# Generate profiles for all products
profiles = manager.generate_all_profiles()
```

## Output Structure

```
futures/
├── continuous/              # ContinuousDownloader output
│   └── product={PRODUCT}/
│       └── year={YEAR}/
│           └── data.parquet
├── individual/              # IndividualDownloader output
│   └── {PRODUCT}/
│       └── data.parquet
├── ohlcv_1d/                # FuturesDataManager output
│   └── product={PRODUCT}/
│       ├── year={YEAR}/
│       │   └── data.parquet
│       └── _profile.json    # Per-product profile
└── definitions.parquet      # Contract specifications
```
