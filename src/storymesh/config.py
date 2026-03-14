"""StoryMesh configuration loader.

Reads storymesh.config.yaml and .env, provides resolved settings to agents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_CONFIG_FILENAME = "storymesh.config.yaml"
_config_cache: dict[str, Any] | None = None

logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
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


def get_config() -> dict[str, Any]:
    """Load and cache the StoryMesh configuration.

    On first call: loads .env, reads the YAML config, configures logging,
    and caches the result. Subsequent calls return the cached dict.
    """
    global _config_cache  # noqa: PLW0603 this is a valid global use
    if _config_cache is not None:
        return _config_cache

    load_dotenv()

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

    _config_cache = loaded

    log_level = loaded.get("logging", {}).get("level", "INFO")
    _configure_logging(log_level)

    logger.debug("Configuration loaded from %s", config_path)

    return _config_cache


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