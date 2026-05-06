"""
Prompt loader for StoryMesh agents.

Reads YAML prompt files from the prompts directory and returns
PromptTemplate instances that provide access to the system prompt
and a formattable user prompt template.

Prompt files live under style directories such as
``src/storymesh/prompts/styles/default/<name>.yaml`` and
``src/storymesh/prompts/styles/<style>/<name>.yaml``.

Each agent prompt file has 'system' and 'user' keys. The system prompt is
returned as-is. The user prompt is a template with Python str.format()
placeholders that are populated at runtime.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent
_STYLES_DIR = _PROMPTS_DIR / "styles"
_DEFAULT_PROMPT_STYLE = "default"
_active_prompt_style = _DEFAULT_PROMPT_STYLE

_PREPEND_TOKEN = "{prepend}"
_LEADING_BLANK_LINES_RE = re.compile(r"^[ \t]*\n+")
_prepend_pool: list[str] = []
_prepend_rng: random.Random = random.Random()


class PromptFormattingError(Exception):
    """Raised when a prompt template cannot be formatted with the provided arguments."""


class PromptTemplate:
    """Container for a system prompt and a formattable user prompt template.

    The system template may contain a literal ``{prepend}`` token. When the
    process-wide prepend pool is non-empty, reading ``self.system`` samples a
    random string from the pool and substitutes it. Templates without the token
    are returned unchanged. The most recent sample is exposed via
    ``self.last_prepend`` for logging.
    """

    def __init__(self, *, system: str, user_template: str) -> None:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("System prompt must be a non-empty string.")

        if not isinstance(user_template, str) or not user_template.strip():
            raise ValueError("User prompt template must be a non-empty string.")

        self._system_template = system
        self._user_template = user_template
        self.last_prepend: str | None = None

    @property
    def system(self) -> str:
        """Return the system prompt with prepend token resolved.

        If the template contains ``{prepend}``: samples from the configured pool
        (or uses an empty replacement when the pool is empty). If the template
        has no token, returns it unchanged.
        """
        return self.format_system()

    def format_system(self, prepend: str | None = None) -> str:
        """Format the system template, optionally substituting ``{prepend}``.

        Args:
            prepend: Explicit value to substitute for ``{prepend}``. When
                ``None``, samples from the process-wide prepend pool (an empty
                string is used if the pool is unset).

        Returns:
            The resolved system prompt. When the template contains no
            ``{prepend}`` token, returns it unchanged regardless of *prepend*.
        """
        if _PREPEND_TOKEN not in self._system_template:
            self.last_prepend = None
            return self._system_template

        resolved = sample_prepend() if prepend is None else prepend
        self.last_prepend = resolved or None
        substituted = self._system_template.replace(_PREPEND_TOKEN, resolved)
        if not resolved:
            substituted = _LEADING_BLANK_LINES_RE.sub("", substituted, count=1)
        return substituted

    def format_user(self, **kwargs: object) -> str:
        """Format the user prompt template with the provided keyword arguments.

        Args:
            **kwargs: Values to substitute into the template placeholders.

        Returns:
            The formatted user prompt string.

        Raises:
            PromptFormattingError: If a required placeholder is missing from kwargs.
        """
        try:
            return self._user_template.format(**kwargs)
        except KeyError as exc:
            raise PromptFormattingError(
                f"Missing placeholder {exc} in user prompt template. "
                f"Provided keys: {sorted(kwargs.keys())}"
            ) from exc


def _validate_style_name(style: str) -> str:
    """Validate and normalize a prompt style name."""
    normalized = style.strip()
    if not normalized:
        raise ValueError("Prompt style must be a non-empty string.")
    if any(sep in normalized for sep in ("/", "\\")):
        raise ValueError(
            f"Prompt style {style!r} must not contain path separators."
        )
    return normalized


def set_prompt_style(style: str) -> None:
    """Set the process-wide default prompt style for subsequent loads."""
    global _active_prompt_style  # noqa: PLW0603
    _active_prompt_style = _validate_style_name(style)


def get_prompt_style() -> str:
    """Return the active process-wide prompt style."""
    return _active_prompt_style


def set_prepend_pool(pool: Iterable[str], *, seed: int | None = None) -> None:
    """Set the process-wide prepend pool.

    The pool is sampled by :func:`sample_prepend` and (transitively) by
    :meth:`PromptTemplate.format_system` whenever a system template contains a
    literal ``{prepend}`` token. Empty / whitespace-only entries are dropped.
    Calling with an empty iterable disables sampling.

    Args:
        pool: Strings to sample from.
        seed: Optional RNG seed. When provided, sampling is reproducible —
            useful for tests and audited reruns.
    """
    global _prepend_pool, _prepend_rng  # noqa: PLW0603
    _prepend_pool = [s for s in pool if isinstance(s, str) and s.strip()]
    _prepend_rng = random.Random(seed) if seed is not None else random.Random()


def sample_prepend() -> str:
    """Return a random prepend from the configured pool, or ``""`` if unset."""
    if not _prepend_pool:
        return ""
    return _prepend_rng.choice(_prepend_pool)


def get_prepend_pool() -> list[str]:
    """Return a copy of the currently configured prepend pool."""
    return list(_prepend_pool)


def _resolve_prompt_path(agent_name: str, style: str) -> Path:
    """Resolve the on-disk path for a prompt file, with default-style fallback."""
    candidates = [_STYLES_DIR / style / f"{agent_name}.yaml"]
    if style != _DEFAULT_PROMPT_STYLE:
        candidates.append(
            _STYLES_DIR / _DEFAULT_PROMPT_STYLE / f"{agent_name}.yaml"
        )

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Prompt file not found for agent {agent_name!r} in style {style!r}. "
        f"Checked: {', '.join(str(p) for p in candidates)}"
    )


def load_prompt(agent_name: str, *, style: str | None = None) -> PromptTemplate:
    """Load a prompt YAML file and return a PromptTemplate.

    Expects a file named '{agent_name}.yaml' in the active prompt-style
    directory or, when missing there, in ``styles/default/``, containing
    'system' and 'user' keys with non-empty string values.

    Args:
        agent_name: The agent name, used to locate the YAML file.
        style: Optional prompt style override. When omitted, uses the active
            style set via :func:`set_prompt_style`.

    Returns:
        A PromptTemplate with the loaded system and user prompts.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    resolved_style = _validate_style_name(style or _active_prompt_style)
    path = _resolve_prompt_path(agent_name, resolved_style)

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in prompt file {path}: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        )

    for key in ("system", "user"):
        if key not in data:
            raise ValueError(
                f"Prompt file {path} is missing required key: '{key}'"
            )
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(
                f"Key '{key}' in {path} must be a non-empty string."
            )

    return PromptTemplate(system=data["system"], user_template=data["user"])
