"""StoryMesh configuration loader.

Reads storymesh.config.yaml and .env, provides resolved settings to agents.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_CONFIG_FILENAME = "storymesh.config.yaml"
_config_cache: dict[str, Any] | None = None

logger = logging.getLogger(__name__)

_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _configure_logging(level_name: str) -> None:
    """Configure the storymesh logger hierarchy."""
    storymesh_logger = logging.getLogger("storymesh")
    level = getattr(logging, level_name.upper(), logging.INFO)
    storymesh_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    storymesh_logger.addHandler(handler)


def find_config_file() -> Path:
    """Locate storymesh.config.yaml by walking up from the package directory,
    falling back to ~/.storymesh/.
    """
    current = Path(__file__).resolve().parent  # src/storymesh/
    while True:
        candidate = current / _CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    user_config = Path.home() / ".storymesh" / _CONFIG_FILENAME
    if user_config.is_file():
        return user_config

    raise FileNotFoundError(
        f"Could not find {_CONFIG_FILENAME} in any parent directory "
        f"of {Path(__file__).resolve().parent} or in ~/.storymesh/"
    )


def _get_required_env_keys(config: dict[str, Any]) -> set[str]:
    """Determine which API keys are required based on the config."""
    providers: set[str] = set()

    llm_section = config.get("llm", {})
    default_provider = llm_section.get("default_provider")
    if default_provider:
        providers.add(default_provider)

    for agent_config in config.get("agents", {}).values():
        provider = agent_config.get("provider")
        if provider:
            providers.add(provider)

    unknown = providers - _PROVIDER_KEY_MAP.keys()
    if unknown:
        raise ValueError(
            f"Unknown LLM provider(s) in config: {', '.join(sorted(unknown))}. "
            f"Valid providers: {', '.join(sorted(_PROVIDER_KEY_MAP.keys()))}"
        )

    return {_PROVIDER_KEY_MAP[p] for p in providers}


def _load_env(required_keys: set[str]) -> Path | None:
    """Load .env if any required key is missing from the environment.

    Search order:
      1. If all required keys are already set, skip .env loading.
      2. .env in the current working directory.
      3. ~/.storymesh/.env
    """
    if all(os.environ.get(k) for k in required_keys):
        return None

    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".storymesh" / ".env",
    ]

    for path in candidates:
        if path.is_file():
            load_dotenv(path)
            return path

    return None


def _validate_env(required_keys: set[str]) -> None:
    """Raise if any required API key is missing from the environment."""
    missing = {k for k in required_keys if not os.environ.get(k)}
    if missing:
        raise OSError(
            f"Missing required API key(s): {', '.join(sorted(missing))}. "
            f"Set them in your environment, .env in the current directory, "
            f"or ~/.storymesh/.env"
        )


def get_config() -> dict[str, Any]:
    """Load and cache the StoryMesh configuration.

    On first call: reads the YAML config, determines required API keys,
    loads .env if needed, validates all keys are present, configures
    logging, and caches the result. Subsequent calls return the cached dict.
    """
    global _config_cache  # noqa: PLW0603
    if _config_cache is not None:
        return _config_cache

    # 1. Load YAML
    config_path = find_config_file()
    try:
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected a YAML mapping in {config_path}, got {type(loaded).__name__}"
        )

    # 2. Determine required keys from config
    required_keys = _get_required_env_keys(loaded)

    # 3. Load .env if needed
    env_path = _load_env(required_keys)

    # 4. Validate all required keys are present
    _validate_env(required_keys)

    # 5. Configure logging
    log_level = loaded.get("logging", {}).get("level", "INFO")
    _configure_logging(log_level)

    # 6. Cache and return
    _config_cache = loaded

    logger.debug("Configuration loaded from %s", config_path)
    if env_path is not None:
        logger.debug(".env loaded from %s", env_path)

    return _config_cache


def get_api_client_config(client_name: str) -> dict[str, Any]:
    """Return configuration for a named external API client.

    Reads from the ``api_clients`` section of storymesh.config.yaml.
    Returns an empty dict (with a warning) if no entry exists for the client.

    Args:
        client_name: Key under ``api_clients`` in the config file.

    Returns:
        Dict of client settings. Keys depend on the specific client.
    """
    config = get_config()
    clients_section = config.get("api_clients", {})
    if client_name not in clients_section:
        logger.warning(
            "No config entry for api_client '%s', using defaults", client_name
        )
        return {}
    return dict(clients_section[client_name])


def get_cache_dir(name: str) -> Path:
    """Return the diskcache directory for the given agent or client name.

    Resolves ``cache.dir`` from storymesh.config.yaml (expanding ``~``),
    then appends ``name`` as a subdirectory. The directory is not created
    here; callers (agent constructors) create it via diskcache.

    Args:
        name: Subdirectory name, typically the agent or API client name.

    Returns:
        Absolute Path for the cache directory.
    """
    config = get_config()
    cache_dir_str: str = config.get("cache", {}).get("dir", "~/.cache/storymesh")
    return Path(cache_dir_str).expanduser() / name


def get_agent_config(agent_name: str) -> dict[str, Any]:
    """Return the resolved LLM configuration for a specific agent.

    Merges agent-specific overrides with llm defaults. If no entry exists
    for the given agent, all values fall back to defaults.
    """
    config = get_config()
    llm_defaults = config.get("llm", {})
    agents_section = config.get("agents", {})
    agent_overrides = agents_section.get(agent_name, {})

    if agent_name not in agents_section:
        logger.warning("No config entry for agent '%s', using llm defaults", agent_name)

    return {
        "provider": agent_overrides.get("provider", llm_defaults.get("default_provider")),
        "model": agent_overrides.get("model", llm_defaults.get("default_model")),
        "temperature": agent_overrides.get("temperature", llm_defaults.get("default_temperature")),
        "max_tokens": agent_overrides.get("max_tokens", llm_defaults.get("default_max_tokens")),
    }