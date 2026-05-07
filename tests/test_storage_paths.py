from __future__ import annotations

from pathlib import Path

from ml4t.data.config.models import DataConfig, StorageConfig
from ml4t.data.core.config import Config, resolve_data_root, resolve_storage_path
from ml4t.data.cot.fetcher import COTConfig
from ml4t.data.crypto.downloader import CryptoConfig
from ml4t.data.etfs.downloader import ETFConfig
from ml4t.data.futures.book_downloader import FuturesConfig
from ml4t.data.futures.config import FuturesDownloadConfig
from ml4t.data.futures.continuous_downloader import ContinuousDownloadConfig
from ml4t.data.futures.individual_downloader import IndividualDownloadConfig
from ml4t.data.macro.downloader import MacroConfig
from ml4t.data.providers.aqr import AQRFactorProvider
from ml4t.data.providers.fama_french import FamaFrenchProvider
from ml4t.data.providers.nasdaq_itch import ITCHSampleProvider
from ml4t.data.providers.wiki_prices import WikiPricesProvider


def test_resolve_data_root_prefers_ml4t_data_path(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ml4t-data-root"
    monkeypatch.setenv("ML4T_DATA_PATH", str(root))
    monkeypatch.delenv("ML4T_DATA_DIR", raising=False)
    monkeypatch.delenv("QLDM_DATA_ROOT", raising=False)

    assert resolve_data_root() == root.resolve()


def test_resolve_data_root_falls_back_to_local_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ML4T_DATA_PATH", raising=False)
    monkeypatch.delenv("ML4T_DATA_DIR", raising=False)
    monkeypatch.delenv("QLDM_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert resolve_data_root() == (tmp_path / "data").resolve()


def test_default_storage_paths_follow_ml4t_data_path(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "configured-data"
    monkeypatch.setenv("ML4T_DATA_PATH", str(root))

    assert resolve_storage_path(None, "futures") == root.resolve() / "futures"
    assert ETFConfig().storage_path == root.resolve() / "etfs"
    assert CryptoConfig().storage_path == root.resolve() / "crypto"
    assert MacroConfig().storage_path == root.resolve() / "macro"
    assert FuturesConfig().storage_path == root.resolve() / "futures"
    assert FuturesDownloadConfig().storage_path == root.resolve() / "futures"
    assert ContinuousDownloadConfig().storage_path == root.resolve() / "futures" / "continuous"
    assert IndividualDownloadConfig().storage_path == root.resolve() / "futures" / "individual"
    assert COTConfig(products=["ES"]).storage_path == root.resolve() / "cot"


def test_config_models_follow_ml4t_data_path(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "configured-data"
    monkeypatch.setenv("ML4T_DATA_PATH", str(root))

    assert StorageConfig().base_path == root.resolve()
    assert DataConfig().base_dir == root.resolve()
    assert Config().data_root == root.resolve()


def test_provider_default_paths_follow_ml4t_data_path(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "configured-data"
    monkeypatch.setenv("ML4T_DATA_PATH", str(root))

    assert FamaFrenchProvider.default_cache_path() == root.resolve() / "factors" / "fama-french"
    assert AQRFactorProvider.default_data_path() == root.resolve() / "factors" / "aqr"
    assert WikiPricesProvider.default_download_path() == root.resolve() / "wiki"
    assert WikiPricesProvider.default_paths() == [
        root.resolve() / "wiki" / "wiki_prices.parquet",
        root.resolve() / "equities" / "nasdaq" / "wiki_prices.parquet",
        Path("wiki_prices.parquet").resolve(),
    ]
    assert ITCHSampleProvider.default_download_path() == root.resolve() / "equities" / "nasdaq_itch"
    assert ITCHSampleProvider.default_parsed_path() == (
        root.resolve() / "equities" / "nasdaq_itch" / "messages"
    )
