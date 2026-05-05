"""Abstract class defining the requirements for LLM provider classes."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import orjson

from storymesh.exceptions import LLMOutputTruncatedError

logger = logging.getLogger(__name__)

# A callable that accepts (run_id, record) and writes it somewhere.
# Using a type alias avoids importing ArtifactStore here (circular dep risk).
LLMCallLogger = Callable[[str, dict[str, Any]], None]

# Set by node wrappers before calling agent.run(). Read by complete_json()
# when writing LLM call records. Defaults to empty string so that calls made
# outside a pipeline run (e.g., in tests) do not raise.
current_run_id: ContextVar[str] = ContextVar("current_run_id", default="")

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


def _sanitize_json_strings(text: str) -> str:
    """Escape literal control characters that appear inside JSON string values.

    LLMs sometimes emit multi-paragraph prose with real newlines inside a JSON
    string value, which is invalid JSON. This function fixes that without
    touching structural whitespace that sits outside string delimiters.

    Args:
        text: Raw JSON text, potentially containing unescaped control chars.

    Returns:
        The same text with control characters inside strings properly escaped.
    """
    _CTRL_MAP: dict[str, str] = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    result: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ord(ch) < 0x20:
            result.append(_CTRL_MAP.get(ch, f"\\u{ord(ch):04x}"))
        else:
            result.append(ch)
    return "".join(result)


def _approx_token_count(text: str) -> int:
    """Return a rough token estimate from character length.

    Uses a simple heuristic of ~4 characters per token. This is not intended
    to match provider billing exactly; it exists for lightweight diagnostics
    and CLI progress summaries across providers.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))

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
            agent_name: str = "unknown",
            on_call: LLMCallLogger | None = None,
        ) -> None:

        self.api_key = api_key
        self.model = model
        self.agent_name = agent_name
        self._on_call = on_call

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
        or if the response is valid JSON but not a dict. Each attempt is recorded via
        ``_write_call_record()`` regardless of success or failure.
        """

        effective_max_tokens = max_tokens

        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                raw = self.complete(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
            except LLMOutputTruncatedError as exc:
                latency_ms = round((time.perf_counter() - t0) * 1000)
                logger.warning(
                    "Response truncated at %d tokens (attempt %d/%d). "
                    "Escalating budget to %d and retrying.",
                    effective_max_tokens,
                    attempt + 1,
                    max_retries + 1,
                    effective_max_tokens * 2,
                )
                self._write_call_record(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    raw_response=exc.partial_response,
                    temperature=temperature,
                    attempt=attempt + 1,
                    latency_ms=latency_ms,
                    parse_success=False,
                )
                if attempt < max_retries:
                    effective_max_tokens *= 2
                    continue
                raise
            latency_ms = round((time.perf_counter() - t0) * 1000)

            cleaned = _sanitize_json_strings(_strip_markdown_fences(raw.strip()))

            try:
                parsed = orjson.loads(cleaned)
            except orjson.JSONDecodeError:
                logger.warning(
                    "JSON parse failed (attempt %d/%d). Raw response:\n%s",
                    attempt + 1,
                    max_retries + 1,
                    raw,
                )
                self._write_call_record(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    raw_response=raw,
                    temperature=temperature,
                    attempt=attempt + 1,
                    latency_ms=latency_ms,
                    parse_success=False,
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
                self._write_call_record(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    raw_response=raw,
                    temperature=temperature,
                    attempt=attempt + 1,
                    latency_ms=latency_ms,
                    parse_success=False,
                )
                if attempt < max_retries:
                    continue
                raise ValueError(
                    f"Expected JSON object as response from the LLM, got {type(parsed).__name__}"
                )

            self._write_call_record(
                system_prompt=system_prompt,
                user_prompt=prompt,
                raw_response=raw,
                temperature=temperature,
                attempt=attempt + 1,
                latency_ms=latency_ms,
                parse_success=True,
            )
            return parsed

        raise RuntimeError("Reached code that should be unreachable in complete_json()!")

    def _write_call_record(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        raw_response: str,
        temperature: float,
        attempt: int,
        latency_ms: int,
        parse_success: bool,
    ) -> None:
        """Build and dispatch one LLM call record to self._on_call, if set.

        Never raises — logging errors must not crash the pipeline.

        Args:
            system_prompt: System prompt sent to the model, or None.
            user_prompt: User prompt sent to the model.
            raw_response: Raw string returned by the model.
            temperature: Sampling temperature used for this call.
            attempt: 1-based attempt number within the retry loop.
            latency_ms: Wall-clock milliseconds for the complete() call.
            parse_success: Whether the response parsed as a valid JSON object.
        """
        if self._on_call is None:
            return
        run_id = current_run_id.get()
        system_text = system_prompt or ""
        prompt_tokens = _approx_token_count(system_text) + _approx_token_count(user_prompt)
        response_tokens = _approx_token_count(raw_response)
        record: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "run_id": run_id,
            "agent": self.agent_name,
            "model": self.model,
            "temperature": temperature,
            "attempt": attempt,
            "system_prompt": system_text,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parse_success": parse_success,
            "latency_ms": latency_ms,
            "approx_prompt_tokens": prompt_tokens,
            "approx_response_tokens": response_tokens,
            "approx_total_tokens": prompt_tokens + response_tokens,
        }
        try:
            self._on_call(run_id, record)
        except Exception:
            logger.warning("Failed to write LLM call record", exc_info=True)

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
    """LLMClient stub for unit testing without a real API.

    ``responses`` is a list of values consumed in order on each ``complete()``
    call.  Each element may be:

    - A ``str`` — returned as the raw completion text.
    - An ``LLMOutputTruncatedError`` instance — raised to simulate a
      token-budget truncation, allowing tests to exercise the budget
      escalation path in ``complete_json()``.
    - Any other ``Exception`` instance — raised as-is.
    """

    def __init__(
        self,
        responses: list[str | Exception],
        agent_name: str = "fake",
        on_call: LLMCallLogger | None = None,
    ) -> None:
        super().__init__(api_key="fake", model="fake", agent_name=agent_name, on_call=on_call)
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
        if isinstance(response, Exception):
            raise response
        return response
