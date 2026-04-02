"""Tests for shared path resolution helpers."""


from ml4t.data import paths


class TestPathHelpers:
    """Tests for config-aware path resolution."""

    def test_find_config_path_prefers_env_override(self, tmp_path, monkeypatch):
        """ML4T_DATA_CONFIG should override automatic search."""
        config_path = tmp_path / "custom-config.yaml"
        config_path.write_text("version: '1.0'")
        monkeypatch.setenv("ML4T_DATA_CONFIG", str(config_path))

        assert paths.find_ml4t_data_config_path(tmp_path) == config_path

    def test_find_config_path_supports_home_location(self, tmp_path, monkeypatch):
        """Home config remains discoverable."""
        home_config = tmp_path / ".config" / "ml4t-data" / "config.yaml"
        home_config.parent.mkdir(parents=True)
        home_config.write_text("version: '1.0'")
        monkeypatch.setattr(paths, "ML4T_DATA_HOME_CONFIG_PATH", home_config)

        assert paths.find_ml4t_data_config_path(tmp_path) == home_config

    def test_resolve_path_prefers_configured_value_relative_to_config_dir(self, tmp_path):
        """Configured relative paths should resolve from the config file directory."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        resolved = paths.resolve_ml4t_data_path(
            "crypto",
            default_path=paths.default_ml4t_data_path("crypto"),
            configured_path="data/crypto",
            config_dir=config_dir,
        )

        assert resolved == config_dir / "data" / "crypto"

    def test_resolve_path_uses_config_root_before_env(self, tmp_path, monkeypatch):
        """Config root should win over ML4T_DATA_PATH when no explicit path is provided."""
        config_root = tmp_path / "config-root"
        env_root = tmp_path / "env-root"
        monkeypatch.setenv("ML4T_DATA_PATH", str(env_root))

        resolved = paths.resolve_ml4t_data_path(
            "etfs",
            default_path=paths.default_ml4t_data_path("etfs"),
            config={"storage": {"base_path": config_root}},
        )

        assert resolved == config_root / "etfs"

    def test_default_path_is_project_local(self, tmp_path, monkeypatch):
        """Default data paths should resolve under the current working directory."""
        monkeypatch.chdir(tmp_path)

        assert paths.default_ml4t_data_path("futures") == tmp_path / "data" / "futures"
