"""Abstract class defining the requiements for LLM provider classes."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import orjson

logger = logging.getLogger(__name__)

# This is a defensive regular expression to strip markdown fences from LLM replies.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

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
        Call the complete() implementation and parse the resposne as a JSON object.

        Strip markdown fences if present in the response. Retry one time on parse failure
        or if the response is valid JSON but not a dict.
        """

        for attempt in range(max_retries + 1):
            raw = self.complete(
                prompt,
                system_prompt = system_prompt,
                temperature = temperature,
                max_tokens = max_tokens
            )
            cleaned = _strip_markdown_fences(raw.strip())

            try:
                parsed = orjson.loads(cleaned)
            except orjson.JSONDecodeError:
                if attempt < max_retries:
                    logger.warning("JSON parse failed. Retrying (attempt %d of %d)", attempt + 1, max_retries)
                    continue
                raise
            
            if not isinstance(parsed, dict):
                if attempt < max_retries:
                    logger.warning(
                        "Expected JSON object, got %s. Retrying.", type(parsed).__name__
                    )
                    continue
                raise ValueError(
                    f"Expected JSON object as response from the LLM, got {type(parsed).__name__}"
                )
            return parsed
        
        raise RuntimeError("Reached code that should be unreachable in complete_json()!")

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