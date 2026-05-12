"""Anthropic LLM client implementation as a subclass of LLMClient."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from storymesh.exceptions import LLMOutputTruncatedError, LLMRefusalError
from storymesh.llm.base import LLMCallLogger, LLMClient, _traceable, register_provider

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Some newer Anthropic models (extended-thinking / reasoning variants) reject
# the ``temperature`` parameter outright with a 400 invalid_request_error
# whose message contains the literal string ``temperature``. This sentinel is
# matched against ``BadRequestError.message`` to detect that case the first
# time a model rejects it, after which we remember the result and stop
# sending the parameter on subsequent calls. Per-class set rather than
# per-instance so the discovery is shared across every AnthropicClient (one
# per agent) that wraps the same model id.
_TEMPERATURE_DEPRECATED_MARKER = "`temperature` is deprecated"
_models_without_temperature: set[str] = set()


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
        """Send a prompt to the Anthropic API and return the response.

        Tolerates models that have deprecated the ``temperature`` parameter:
        the first call against such a model raises a 400 whose message
        contains ``\\`temperature\\` is deprecated``; we catch it, remember
        the model id in a module-level set, and retry without the parameter.
        Subsequent calls against the same model skip ``temperature`` from
        the start.
        """

        send_temperature = self.model not in _models_without_temperature

        def _build_kwargs(include_temperature: bool) -> dict[str, Any]:
            kw: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if include_temperature:
                kw["temperature"] = temperature
            if system_prompt is not None:
                kw["system"] = system_prompt
            return kw

        try:
            response = self.client.messages.create(**_build_kwargs(send_temperature))
        except anthropic.BadRequestError as exc:
            if send_temperature and _TEMPERATURE_DEPRECATED_MARKER in str(exc):
                if self.model:
                    _models_without_temperature.add(self.model)
                response = self.client.messages.create(**_build_kwargs(False))
            else:
                raise

        if response.stop_reason == "refusal":
            raise LLMRefusalError(
                provider="anthropic",
                model=self.model,
                detail=_extract_refusal_detail(response),
            )

        for block in response.content:
            if getattr(block, "type", None) == "refusal":
                raise LLMRefusalError(
                    provider="anthropic",
                    model=self.model,
                    detail=str(getattr(block, "text", "") or "(no detail provided)"),
                )

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


def _extract_refusal_detail(response: Any) -> str:  # noqa: ANN401  # SDK response object is loosely typed
    """Best-effort extraction of refusal text from an Anthropic response.

    The SDK exposes refusals either via ``stop_reason == 'refusal'`` (in which
    case any text-bearing block usually carries the explanation) or via a
    dedicated refusal content block. Return a human-readable string for the
    error message, falling back to a generic placeholder.
    """
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return str(text)
    return "(no detail provided)"


register_provider("anthropic", AnthropicClient)