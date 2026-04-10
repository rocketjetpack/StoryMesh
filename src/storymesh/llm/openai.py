"""OpenAI LLM client implementation as a subclass of LLMClient."""

from __future__ import annotations

import os
from typing import Any

import openai

from storymesh.llm.base import LLMCallLogger, LLMClient, _traceable, register_provider

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(LLMClient):
    """LLMClient implementation backed by the OpenAI chat completions API.

    Uses the ``openai`` Python SDK. API key is resolved from the ``api_key``
    argument or the ``OPENAI_API_KEY`` environment variable.

    Args:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        model: Model identifier. Defaults to ``gpt-4o-mini``.
        agent_name: Identifies the calling agent in LLM call logs.
        on_call: Optional callback invoked after every ``complete()`` call.

    Raises:
        ValueError: If no API key is found in args or environment.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        agent_name: str = "unknown",
        on_call: LLMCallLogger | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key has not been provided. "
                "Pass api_key or set OPENAI_API_KEY in the environment."
            )

        resolved_model = model or _DEFAULT_MODEL

        super().__init__(
            api_key=resolved_key,
            model=resolved_model,
            agent_name=agent_name,
            on_call=on_call,
        )
        self.client = openai.OpenAI(api_key=resolved_key)

    @_traceable
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send a prompt to the OpenAI chat completions API and return the text response.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system message prepended to the conversation.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate in the response.

        Returns:
            The text content of the model's response.

        Raises:
            ValueError: If the API returns an unexpected response structure.
            openai.OpenAIError: On API-level failures (rate limits, auth errors, etc.).
        """
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # self.model is always a non-None str after __init__ (resolved via
        # _DEFAULT_MODEL), but the base class types it as str | None.
        assert self.model is not None
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,  # type: ignore[arg-type]
        )

        if not response.choices:
            raise ValueError("OpenAI returned an empty choices list.")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                "OpenAI response choice contained a None content field. "
                "This can occur when the model uses tool calls instead of a text response."
            )

        return content


register_provider("openai", OpenAIClient)
