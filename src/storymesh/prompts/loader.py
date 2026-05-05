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

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent
_STYLES_DIR = _PROMPTS_DIR / "styles"
_DEFAULT_PROMPT_STYLE = "default"
_active_prompt_style = _DEFAULT_PROMPT_STYLE


class PromptFormattingError(Exception):
    """Raised when a prompt template cannot be formatted with the provided arguments."""


class PromptTemplate:
    """Container for a system prompt and a formattable user prompt template.

    Attributes:
        system: The static system prompt string, returned as-is.
    """

    def __init__(self, *, system: str, user_template: str) -> None:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("System prompt must be a non-empty string.")

        if not isinstance(user_template, str) or not user_template.strip():
            raise ValueError("User prompt template must be a non-empty string.")

        self.system = system
        self._user_template = user_template

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
