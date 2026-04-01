"""Abstract class defining the requirements for LLM provider classes."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import orjson

logger = logging.getLogger(__name__)

# This is a defensive regular expression to strip markdown fences from LLM replies.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _traceable[F: Callable[..., object]](fn: F) -> F:
    """Wrap a callable with LangSmith tracing if langsmith is installed.

    When ``langsmith`` is not installed or ``LANGCHAIN_TRACING_V2`` is not
    set, this is a transparent no-op with zero runtime cost. Apply it to
    ``complete()`` implementations so that individual LLM calls appear as
    spans in the LangSmith dashboard alongside the LangGraph node traces.

    Args:
        fn: The callable to wrap.

    Returns:
        The wrapped callable (or the original, if langsmith is absent).
    """
    try:
        from langsmith import traceable

        return traceable(fn)  # type: ignore[return-value]  # langsmith's return type is not F-compatible
    except ImportError:
        return fn

def _strip_markdown_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    return text

class LLMClient(ABC):
    """
    This class provides a vendor agnostic interface for LLM completions.

    Subclasses shall implement complete() for a specific provider.
    complete_json() is a provider-agnostic function and is available to subclasses.
    """

    def __init__(
            self,
            *,
            api_key: str | None = None,
            model: str | None = None,
        ) -> None:

        self.api_key = api_key
        self.model = model

    @abstractmethod
    def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int
        ) -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        Vendor specific provider subclasses *must* implement this method.
        """
        ...

    def complete_json(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int,
            max_retries: int = 1,
        ) -> dict[str, Any]:
        """
        Call the complete() implementation and parse the response as a JSON object.

        Strip markdown fences if present in the response. Retry one time on parse failure
        or if the response is valid JSON but not a dict.
        """

        for attempt in range(max_retries + 1):
            logger.debug(
                "LLM call [attempt %d/%d]\n--- SYSTEM ---\n%s\n--- USER ---\n%s",
                attempt + 1,
                max_retries + 1,
                system_prompt or "(none)",
                prompt,
            )

            raw = self.complete(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.debug(
                "LLM response [attempt %d/%d]\n--- RESPONSE ---\n%s",
                attempt + 1,
                max_retries + 1,
                raw,
            )

            cleaned = _strip_markdown_fences(raw.strip())

            try:
                parsed = orjson.loads(cleaned)
            except orjson.JSONDecodeError:
                logger.warning(
                    "JSON parse failed (attempt %d/%d). Raw response:\n%s",
                    attempt + 1,
                    max_retries + 1,
                    raw,
                )
                if attempt < max_retries:
                    continue
                raise

            if not isinstance(parsed, dict):
                logger.warning(
                    "Expected JSON object, got %s (attempt %d/%d). Raw response:\n%s",
                    type(parsed).__name__,
                    attempt + 1,
                    max_retries + 1,
                    raw,
                )
                if attempt < max_retries:
                    continue
                raise ValueError(
                    f"Expected JSON object as response from the LLM, got {type(parsed).__name__}"
                )
            return parsed

        raise RuntimeError("Reached code that should be unreachable in complete_json()!")

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[LLMClient]] = {}


def register_provider(name: str, cls: type[LLMClient]) -> None:
    """Register an LLMClient subclass for a given provider name.

    Idempotent: registering the same class under the same name twice is
    allowed (e.g., when a module is re-imported). Raises if a *different*
    class is registered under an already-taken name.

    Args:
        name: Provider name as it appears in storymesh.config.yaml (e.g. ``'anthropic'``).
        cls: The concrete LLMClient subclass to instantiate for this provider.

    Raises:
        ValueError: If ``name`` is already registered to a different class.
    """
    if name in _PROVIDER_REGISTRY and _PROVIDER_REGISTRY[name] is not cls:
        raise ValueError(
            f"Provider '{name}' is already registered to "
            f"{_PROVIDER_REGISTRY[name].__name__}, "
            f"cannot re-register to {cls.__name__}."
        )
    _PROVIDER_REGISTRY[name] = cls


def get_provider_class(name: str) -> type[LLMClient]:
    """Return the LLMClient subclass registered for the given provider name.

    Args:
        name: Provider name string.

    Returns:
        The registered LLMClient subclass.

    Raises:
        ValueError: If no provider is registered under ``name``.
    """
    if name not in _PROVIDER_REGISTRY:
        registered = ", ".join(sorted(_PROVIDER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown LLM provider: '{name}'. Registered providers: {registered}"
        )
    return _PROVIDER_REGISTRY[name]


class FakeLLMClient(LLMClient):
    """This implementation exists solely for pytest testing of the base class without any implementation."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(api_key="fake", model="fake")
        self.responses = responses
        self.call_count = 0

    def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int
        ) -> str:
        response = self.responses[self.call_count]
        self.call_count += 1
        return response