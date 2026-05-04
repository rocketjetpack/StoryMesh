"""StoryMesh configuration loader.

Reads storymesh.config.yaml and .env, provides resolved settings to agents.

Config loading uses a two-layer merge strategy:

1. **Project-level config** — ``storymesh.config.yaml`` found by walking up
   from the package directory.  This file is checked into the repository and
   contains defaults shared by all contributors.

2. **User-level override** — ``~/.storymesh/storymesh.config.yaml``.  When
   present this file is deep-merged on top of the project config, so a
   developer can set personal values (e.g. ``user_agent``, API keys paths,
   preferred models) without modifying the tracked project file.

Deep-merge semantics: nested dicts are merged recursively; scalar values and
lists are replaced wholesale by the override value.
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, with override values winning.

    Nested dicts are merged recursively so that a user-level config that only
    specifies ``api_clients.open_library.user_agent`` does not erase the
    project-level ``api_clients.open_library.max_books`` value.  All other
    value types (strings, ints, floats, booleans, lists, ``None``) are
    replaced wholesale by the override value.

    Neither input dict is mutated.

    Args:
        base: The base config dict (project-level values).
        override: Values to merge on top of *base* (user-level values).

    Returns:
        A new dict with the merged result.
    """
    result: dict[str, Any] = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _find_config_files() -> list[Path]:
    """Return config file paths in merge order: project-level first, then user override.

    The project-level config is discovered by walking up from the package
    directory (``src/storymesh/``) until ``storymesh.config.yaml`` is found or
    the filesystem root is reached.

    The user-level config at ``~/.storymesh/storymesh.config.yaml`` is always
    appended as an additional layer when it exists and is not already the file
    found above.

    Returns:
        List of one or two existing config file paths in load order.

    Raises:
        FileNotFoundError: If no config file is found in the project tree *and*
            no user-level config exists.
    """
    paths: list[Path] = []

    current = Path(__file__).resolve().parent  # src/storymesh/
    while True:
        candidate = current / _CONFIG_FILENAME
        if candidate.is_file():
            paths.append(candidate)
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    user_config = Path.home() / ".storymesh" / _CONFIG_FILENAME
    if user_config.is_file() and user_config not in paths:
        paths.append(user_config)

    if not paths:
        raise FileNotFoundError(
            f"Could not find {_CONFIG_FILENAME} in any parent directory "
            f"of {Path(__file__).resolve().parent} or in ~/.storymesh/"
        )

    return paths


def find_config_file() -> Path:
    """Locate the primary (project-level) storymesh.config.yaml.

    .. note::
        This function returns only the project-level config path.  A
        user-level override at ``~/.storymesh/storymesh.config.yaml`` is
        automatically layered on top by :func:`get_config` but is *not*
        reflected here.  Callers that need the fully merged configuration
        should use :func:`get_config` instead.

    Returns:
        Path to the primary config file.

    Raises:
        FileNotFoundError: If no config file is found anywhere.
    """
    return _find_config_files()[0]


def _get_required_env_keys(config: dict[str, Any]) -> set[str]:
    """Determine which API key environment variable names are needed by the config.

    Args:
        config: Loaded config dict.

    Returns:
        Set of environment variable names (e.g. ``{"ANTHROPIC_API_KEY"}``).

    Raises:
        ValueError: If the config references an unknown provider name.
    """
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


def _load_env_best_effort() -> None:
    """Load the first .env file found, without checking for specific keys.

    Search order:
      1. .env in the current working directory.
      2. ~/.storymesh/.env

    Silently no-ops if neither file exists.
    """
    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".storymesh" / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path)
            logger.debug(".env loaded from %s", path)
            return


def warn_missing_provider_keys(config: dict[str, Any]) -> None:
    """Log a warning for each provider API key referenced in config but absent from the environment.

    This is a non-fatal check intended to be called just before the pipeline
    runs. Agents handle missing keys gracefully (static-only mode), so raising
    here would be overly strict.

    Args:
        config: Loaded config dict, as returned by ``get_config()``.
    """
    try:
        required_keys = _get_required_env_keys(config)
    except ValueError as exc:
        logger.warning("Config provider validation warning: %s", exc)
        return

    for key in sorted(required_keys):
        if not os.environ.get(key):
            logger.warning(
                "%s is not set — agents using this provider will run in static-only mode.",
                key,
            )


def get_config() -> dict[str, Any]:
    """Load and cache the merged StoryMesh configuration.

    On first call: discovers all config files via :func:`_find_config_files`,
    deep-merges them (project-level base, user-level override), loads .env
    (best-effort), configures logging, and caches the result.  Subsequent
    calls return the cached dict.

    Config merge order:

    1. Project-level ``storymesh.config.yaml`` (checked into the repository).
    2. User-level ``~/.storymesh/storymesh.config.yaml`` (personal overrides,
       not in the repository).  Values here win for any key they specify.

    API key presence is intentionally *not* validated here so that CLI
    commands like ``show-config`` and ``show-version`` work without any keys
    configured.  Call ``warn_missing_provider_keys(get_config())`` explicitly
    when the pipeline is about to run.
    """
    global _config_cache  # noqa: PLW0603
    if _config_cache is not None:
        return _config_cache

    # 1. Load and merge all config files
    config_paths = _find_config_files()
    merged: dict[str, Any] = {}
    for path in config_paths:
        try:
            with open(path) as f:
                loaded: Any = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected a YAML mapping in {path}, got {type(loaded).__name__}"
            )

        merged = _deep_merge(merged, loaded)
        logger.debug("Configuration layer loaded from %s", path)

    # 2. Load .env unconditionally (best-effort, no error if absent)
    _load_env_best_effort()

    # 3. Configure logging from the merged result
    log_level = merged.get("logging", {}).get("level", "INFO")
    _configure_logging(log_level)

    # 4. Cache and return
    _config_cache = merged

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
    """Return the fully resolved configuration for a specific agent.

    Builds the result in two steps:

    1. Start with all agent-specific keys from ``agents.<agent_name>`` in the
       merged config (which may include any agent-specific settings such as
       ``top_n``, ``weights``, ``diversity_weight``, ``max_seeds``, etc.).
    2. Fill in missing LLM keys (``provider``, ``model``, ``temperature``,
       ``max_tokens``) from the global ``llm`` defaults section.

    This means agent-specific values always win over LLM defaults, and all
    non-LLM agent keys are preserved rather than silently dropped.

    Args:
        agent_name: Key under ``agents`` in storymesh.config.yaml.

    Returns:
        Dict containing all agent config keys merged with LLM defaults.
    """
    config = get_config()
    llm_defaults = config.get("llm", {})
    agents_section = config.get("agents", {})

    if agent_name not in agents_section:
        logger.warning("No config entry for agent '%s', using llm defaults", agent_name)

    # Start with all agent-specific keys (preserves top_n, weights, etc.)
    result: dict[str, Any] = dict(agents_section.get(agent_name, {}))

    # Fill missing LLM keys from global defaults
    result.setdefault("provider", llm_defaults.get("default_provider"))
    result.setdefault("model", llm_defaults.get("default_model"))
    result.setdefault("temperature", llm_defaults.get("default_temperature"))
    result.setdefault("max_tokens", llm_defaults.get("default_max_tokens"))

    return result


def get_prompt_style() -> str:
    """Return the configured prompt style name."""
    config = get_config()
    prompts_section = config.get("prompts", {})
    style = prompts_section.get("style", "default")
    if not isinstance(style, str) or not style.strip():
        logger.warning(
            "Invalid prompts.style value %r in config; using 'default'.",
            style,
        )
        return "default"
    return style.strip()
