"""Anthropic LLM client implementation as a subclass of LLMClient."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from storymesh.exceptions import LLMOutputTruncatedError
from storymesh.llm.base import LLMCallLogger, LLMClient, _traceable, register_provider

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

class AnthropicClient(LLMClient):
    """LLMClient compatible class for leveraging the Anthropic SDK."""
    
    def __init__(
            self,
            *,
            api_key: str | None = None,
            model: str | None = None,
            agent_name: str = "unknown",
            on_call: LLMCallLogger | None = None,
        ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key has not been provided. Pass api_key or set ANTHROPIC_API_KEY in the environment."
            )

        resolved_model = model or _DEFAULT_MODEL

        super().__init__(api_key=resolved_key, model=resolved_model, agent_name=agent_name, on_call=on_call)
        self.client = anthropic.Anthropic(api_key=resolved_key)

    @_traceable
    def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int
        ) -> str:
        """Send a prompt to the Anthropic API and return the response."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        if len(response.content) != 1:
            raise ValueError(
                f"Expected exactly 1 content block from Anthropic but got {len(response.content)}!"
            )

        block = response.content[0]

        if block.type != "text":
            raise ValueError(
                f"Expected a text content block from Anthropic but got {block.type} instead!"
            )

        if response.stop_reason == "max_tokens":
            raise LLMOutputTruncatedError(
                partial_response=str(block.text),
                token_budget=max_tokens,
            )

        return str(block.text)


register_provider("anthropic", AnthropicClient)