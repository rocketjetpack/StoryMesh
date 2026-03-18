"""
Prompt loader for StoryMesh agents.

Reads YAML prompt files from the prompts directory and returns
PromptTemplate instances that provide access to the system prompt
and a formattable user prompt template.

Each agent has one YAML file with 'system' and 'user' keys.
The system prompt is returned as-is. The user prompt is a template
with Python str.format() placeholders that are populated at runtime.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent


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


def load_prompt(agent_name: str) -> PromptTemplate:
    """Load a prompt YAML file and return a PromptTemplate.

    Expects a file named '{agent_name}.yaml' in the prompts directory
    containing 'system' and 'user' keys with non-empty string values.

    Args:
        agent_name: The agent name, used to locate the YAML file.

    Returns:
        A PromptTemplate with the loaded system and user prompts.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    path = _PROMPTS_DIR / f"{agent_name}.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

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