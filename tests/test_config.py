"""Unit tests for storymesh.config — deep-merge, config discovery, and accessors."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest

import storymesh.config as cfg_module
from storymesh.config import (
    _deep_merge,
    _find_config_files,
    _get_required_env_keys,
    find_config_file,
    get_agent_config,
    get_api_client_config,
    get_cache_dir,
    warn_missing_provider_keys,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


# ---------------------------------------------------------------------------
# TestDeepMerge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_scalar_override_wins(self) -> None:
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_base_key_preserved_when_not_in_override(self) -> None:
        result = _deep_merge({"a": 1, "b": 2}, {"a": 99})
        assert result["b"] == 2

    def test_nested_dict_merged_recursively(self) -> None:
        base = {"api_clients": {"open_library": {"user_agent": None, "max_books": 50}}}
        override = {"api_clients": {"open_library": {"user_agent": "App (a@b.com)"}}}
        result = _deep_merge(base, override)
        assert result["api_clients"]["open_library"]["user_agent"] == "App (a@b.com)"
        assert result["api_clients"]["open_library"]["max_books"] == 50

    def test_list_replaced_not_merged(self) -> None:
        result = _deep_merge({"items": [1, 2, 3]}, {"items": [4, 5]})
        assert result["items"] == [4, 5]

    def test_override_adds_new_key(self) -> None:
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_base_not_mutated(self) -> None:
        base: dict[str, Any] = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 99}})
        assert base["a"]["x"] == 1

    def test_override_not_mutated(self) -> None:
        override: dict[str, Any] = {"a": {"x": 99}}
        _deep_merge({"a": {"x": 1}}, override)
        assert override["a"]["x"] == 99

    def test_empty_base(self) -> None:
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_none_value_in_override_wins(self) -> None:
        result = _deep_merge({"a": "something"}, {"a": None})
        assert result["a"] is None


# ---------------------------------------------------------------------------
# TestFindConfigFiles
# ---------------------------------------------------------------------------


class TestFindConfigFiles:
    def test_returns_list_of_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        _write_yaml(project_cfg, "llm:\n  default_provider: anthropic\n")

        monkeypatch.setattr(cfg_module, "_CONFIG_FILENAME", "storymesh.config.yaml")
        # Point package __file__ to a subdirectory of the project config
        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        # Monkeypatch __file__ in the module so the walk starts from our tmp dir
        original_file = cfg_module.__file__
        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))

        # No user config — should return exactly one path
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

        paths = _find_config_files()
        assert len(paths) == 1
        assert paths[0] == project_cfg

        monkeypatch.setattr(cfg_module, "__file__", original_file)

    def test_user_config_appended_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        _write_yaml(project_cfg, "llm:\n  default_provider: anthropic\n")

        user_cfg = tmp_path / "home" / ".storymesh" / "storymesh.config.yaml"
        _write_yaml(user_cfg, "api_clients:\n  open_library:\n    user_agent: App (a@b.com)\n")

        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        original_file = cfg_module.__file__
        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

        paths = _find_config_files()
        assert len(paths) == 2
        assert paths[0] == project_cfg
        assert paths[1] == user_cfg

        monkeypatch.setattr(cfg_module, "__file__", original_file)

    def test_raises_when_no_config_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pkg_file = tmp_path / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        original_file = cfg_module.__file__
        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty_home")

        with pytest.raises(FileNotFoundError):
            _find_config_files()

        monkeypatch.setattr(cfg_module, "__file__", original_file)


# ---------------------------------------------------------------------------
# TestGetAgentConfig
# ---------------------------------------------------------------------------


class TestGetAgentConfig:
    """Test get_agent_config using the live project config (no mocking needed)."""

    def test_returns_provider_and_model(self) -> None:
        result = get_agent_config("genre_normalizer")
        assert result["provider"] == "anthropic"
        assert "model" in result

    def test_llm_defaults_fill_missing_keys(self) -> None:
        """A minimal agent config should still get provider/model from llm defaults."""
        result = get_agent_config("genre_normalizer")
        assert result.get("provider") is not None
        assert result.get("model") is not None

    def test_agent_specific_keys_preserved(self) -> None:
        """book_ranker has top_n, weights, mmr_lambda — all must survive."""
        result = get_agent_config("book_ranker")
        assert "top_n" in result
        assert "weights" in result
        assert "mmr_lambda" in result

    def test_book_ranker_top_n_is_correct(self) -> None:
        result = get_agent_config("book_ranker")
        assert result["top_n"] == 10

    def test_theme_extractor_max_seeds_present(self) -> None:
        result = get_agent_config("theme_extractor")
        assert "max_seeds" in result

    def test_unknown_agent_returns_llm_defaults(self) -> None:
        result = get_agent_config("nonexistent_agent_xyz")
        # Should not raise; returns LLM defaults
        assert "provider" in result
        assert "model" in result

    def test_result_has_four_llm_keys_at_minimum(self) -> None:
        result = get_agent_config("genre_normalizer")
        for key in ("provider", "model", "temperature", "max_tokens"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TestGetApiClientConfig
# ---------------------------------------------------------------------------


class TestGetApiClientConfig:
    def test_open_library_config_returned(self) -> None:
        result = get_api_client_config("open_library")
        assert isinstance(result, dict)
        assert "max_books" in result

    def test_unknown_client_returns_empty_dict(self) -> None:
        result = get_api_client_config("nonexistent_client_xyz")
        assert result == {}


# ---------------------------------------------------------------------------
# TestConfigMerge (integration: project + user layer)
# ---------------------------------------------------------------------------


class TestConfigMerge:
    """Integration tests verifying that two config files are correctly deep-merged."""

    def test_user_agent_override_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        _write_yaml(
            project_cfg,
            """
            llm:
              default_provider: anthropic
              default_model: claude-haiku-4-5-20251001
              default_temperature: 1.0
              default_max_tokens: 1024
            api_clients:
              open_library:
                user_agent:
                max_books: 50
            cache:
              dir: ~/.cache/storymesh
            logging:
              level: WARNING
            """,
        )
        user_cfg = tmp_path / "home" / ".storymesh" / "storymesh.config.yaml"
        _write_yaml(
            user_cfg,
            """
            api_clients:
              open_library:
                user_agent: "App (dev@example.com)"
            """,
        )

        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        original_file = cfg_module.__file__
        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        # Clear cache so get_config() re-reads from our tmp files
        monkeypatch.setattr(cfg_module, "_config_cache", None)

        from storymesh.config import get_config

        config = get_config()
        assert config["api_clients"]["open_library"]["user_agent"] == "App (dev@example.com)"
        # project-level key not present in user config must survive
        assert config["api_clients"]["open_library"]["max_books"] == 50

        monkeypatch.setattr(cfg_module, "__file__", original_file)
        monkeypatch.setattr(cfg_module, "_config_cache", None)

    def test_project_default_preserved_when_not_overridden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        _write_yaml(
            project_cfg,
            """
            llm:
              default_provider: anthropic
              default_model: claude-haiku-4-5-20251001
              default_temperature: 1.0
              default_max_tokens: 1024
            cache:
              dir: ~/.cache/storymesh
            logging:
              level: WARNING
            """,
        )
        user_cfg = tmp_path / "home" / ".storymesh" / "storymesh.config.yaml"
        _write_yaml(user_cfg, "logging:\n  level: DEBUG\n")

        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        original_file = cfg_module.__file__
        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(cfg_module, "_config_cache", None)

        from storymesh.config import get_config

        config = get_config()
        assert config["logging"]["level"] == "DEBUG"  # user override wins
        assert config["llm"]["default_provider"] == "anthropic"  # project default preserved

        monkeypatch.setattr(cfg_module, "__file__", original_file)
        monkeypatch.setattr(cfg_module, "_config_cache", None)


# ---------------------------------------------------------------------------
# TestFindConfigFileWrapper
# ---------------------------------------------------------------------------


class TestFindConfigFileWrapper:
    """Tests for the single-path ``find_config_file()`` wrapper."""

    def test_returns_only_project_level_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """find_config_file returns the project-level path even when a user-level override exists."""
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        _write_yaml(project_cfg, "llm:\n  default_provider: anthropic\n")
        user_cfg = tmp_path / "home" / ".storymesh" / "storymesh.config.yaml"
        _write_yaml(user_cfg, "logging:\n  level: DEBUG\n")

        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

        assert find_config_file() == project_cfg


# ---------------------------------------------------------------------------
# TestGetRequiredEnvKeys
# ---------------------------------------------------------------------------


class TestGetRequiredEnvKeys:
    """Tests for ``_get_required_env_keys``: provider → env-var-name mapping."""

    def test_single_provider_maps_to_env_key(self) -> None:
        """llm.default_provider = anthropic requires ANTHROPIC_API_KEY."""
        cfg = {"llm": {"default_provider": "anthropic"}}
        assert _get_required_env_keys(cfg) == {"ANTHROPIC_API_KEY"}

    def test_agent_level_provider_adds_env_key(self) -> None:
        """Per-agent provider overrides contribute to the required-keys set."""
        cfg = {
            "llm": {"default_provider": "anthropic"},
            "agents": {"genre_normalizer": {"provider": "openai"}},
        }
        assert _get_required_env_keys(cfg) == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}

    def test_unknown_provider_raises_value_error(self) -> None:
        """An unknown provider name must raise ValueError with the valid list."""
        cfg = {"llm": {"default_provider": "cohere"}}
        with pytest.raises(ValueError, match="Unknown LLM provider.*cohere"):
            _get_required_env_keys(cfg)

    def test_empty_config_returns_empty_set(self) -> None:
        """No providers configured → empty set, not an error."""
        assert _get_required_env_keys({}) == set()

    def test_agent_without_provider_key_is_ignored(self) -> None:
        """An agent entry with no 'provider' key contributes nothing."""
        cfg = {
            "llm": {"default_provider": "anthropic"},
            "agents": {"book_ranker": {"top_n": 10}},
        }
        assert _get_required_env_keys(cfg) == {"ANTHROPIC_API_KEY"}


# ---------------------------------------------------------------------------
# TestWarnMissingProviderKeys
# ---------------------------------------------------------------------------


class TestWarnMissingProviderKeys:
    """Tests for ``warn_missing_provider_keys``."""

    def test_warns_when_env_var_is_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A referenced provider without its env var logs a warning."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = {"llm": {"default_provider": "anthropic"}}

        with caplog.at_level(logging.WARNING, logger="storymesh.config"):
            warn_missing_provider_keys(cfg)

        assert any("ANTHROPIC_API_KEY is not set" in r.message for r in caplog.records)

    def test_no_warning_when_env_var_is_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A referenced provider with its env var set logs no warning."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        cfg = {"llm": {"default_provider": "anthropic"}}

        with caplog.at_level(logging.WARNING, logger="storymesh.config"):
            warn_missing_provider_keys(cfg)

        assert not any(
            "ANTHROPIC_API_KEY is not set" in r.message for r in caplog.records
        )

    def test_unknown_provider_logs_and_returns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unknown provider is surfaced as a single warning (not a raise)."""
        cfg = {"llm": {"default_provider": "cohere"}}

        with caplog.at_level(logging.WARNING, logger="storymesh.config"):
            warn_missing_provider_keys(cfg)

        assert any(
            "Config provider validation warning" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# TestGetConfigErrorPaths
# ---------------------------------------------------------------------------


class TestGetConfigErrorPaths:
    """Tests for ``get_config`` error branches (invalid YAML, non-dict root)."""

    def test_invalid_yaml_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A YAML parse error must be raised as ValueError with the path included."""
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        # Intentional malformed YAML: unbalanced quotes.
        project_cfg.parent.mkdir(parents=True, exist_ok=True)
        project_cfg.write_text('llm:\n  default_provider: "anthropic\n')

        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(cfg_module, "_config_cache", None)

        from storymesh.config import get_config

        with pytest.raises(ValueError, match="Invalid YAML"):
            get_config()

        monkeypatch.setattr(cfg_module, "_config_cache", None)

    def test_non_dict_yaml_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A YAML file whose root is a list (not a mapping) must raise ValueError."""
        project_cfg = tmp_path / "project" / "storymesh.config.yaml"
        project_cfg.parent.mkdir(parents=True, exist_ok=True)
        project_cfg.write_text("- one\n- two\n- three\n")

        pkg_file = tmp_path / "project" / "src" / "storymesh" / "config.py"
        pkg_file.parent.mkdir(parents=True, exist_ok=True)
        pkg_file.touch()

        monkeypatch.setattr(cfg_module, "__file__", str(pkg_file))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(cfg_module, "_config_cache", None)

        from storymesh.config import get_config

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            get_config()

        monkeypatch.setattr(cfg_module, "_config_cache", None)


# ---------------------------------------------------------------------------
# TestGetCacheDir
# ---------------------------------------------------------------------------


class TestGetCacheDir:
    """Tests for ``get_cache_dir``."""

    def test_uses_configured_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_cache_dir appends the name to the configured cache root."""
        cache_root = tmp_path / "cache"
        monkeypatch.setattr(
            cfg_module,
            "get_config",
            lambda: {"cache": {"dir": str(cache_root)}},
        )

        result = get_cache_dir("open_library")

        assert result == cache_root / "open_library"

    def test_expands_home_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``~`` in the configured cache dir is expanded to the user's home.

        Path.expanduser reads ``$HOME`` on POSIX, so that env var is what we
        must override — not ``Path.home``.
        """
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            cfg_module,
            "get_config",
            lambda: {"cache": {"dir": "~/.cache/storymesh"}},
        )

        result = get_cache_dir("book_fetcher")

        assert str(result).startswith(str(fake_home))
        assert result.name == "book_fetcher"

    def test_falls_back_to_default_when_cache_key_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Missing ``cache.dir`` resolves to the documented ~/.cache/storymesh default."""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(cfg_module, "get_config", lambda: {})

        result = get_cache_dir("any")

        assert str(result) == str(fake_home / ".cache" / "storymesh" / "any")


# ---------------------------------------------------------------------------
# TestLoadEnvBestEffort
# ---------------------------------------------------------------------------


class TestLoadEnvBestEffort:
    """Tests for ``_load_env_best_effort`` — silent no-op when no .env files exist."""

    def test_loads_cwd_env_when_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A .env file in the current working directory is loaded."""
        env_file = tmp_path / ".env"
        env_file.write_text("STORYMESH_TEST_ENV_VAR=present\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STORYMESH_TEST_ENV_VAR", raising=False)

        cfg_module._load_env_best_effort()

        import os

        assert os.environ.get("STORYMESH_TEST_ENV_VAR") == "present"

    def test_silent_when_no_env_files_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No .env anywhere → function returns silently without raising."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")

        # Does not raise; the function is documented to be best-effort.
        cfg_module._load_env_best_effort()
