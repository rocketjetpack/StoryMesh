from __future__ import annotations

from typing import Any

import orjson
import pytest

from storymesh.exceptions import LLMOutputTruncatedError
from storymesh.llm.base import FakeLLMClient, current_run_id


def test_complete_json_parses_valid_json() -> None:
    client = FakeLLMClient(['{"token": "solarpunk", "type": "genre"}'])
    result = client.complete_json(prompt="test", temperature=0.0, max_tokens=100)
    assert result == {"token": "solarpunk", "type": "genre"}

def test_complete_json_strips_markdown_fences() -> None:
    client = FakeLLMClient(['```json\n{"key": "value"}\n```'])
    result = client.complete_json(prompt="test", temperature=0.0, max_tokens=100)
    assert result == {"key": "value"}

def test_complete_json_retries_on_bad_json() -> None:
    client = FakeLLMClient(["not json", '{"key": "value"}'])
    result = client.complete_json(prompt="test", temperature=0.0, max_tokens=100)
    assert client.call_count == 2
    assert result == {"key": "value"}

def test_complete_json_raises_after_retries_exhausted() -> None:
    client = FakeLLMClient(["bad", "still bad"])
    with pytest.raises(orjson.JSONDecodeError):
        client.complete_json(prompt="test", temperature=0.0, max_tokens=100)


# ---------------------------------------------------------------------------
# current_run_id ContextVar
# ---------------------------------------------------------------------------


def test_current_run_id_defaults_to_empty_string() -> None:
    """current_run_id defaults to '' when not set in this context."""
    assert current_run_id.get() == ""


def test_current_run_id_set_and_reset() -> None:
    """Setting current_run_id is visible via .get() and can be reset."""
    token = current_run_id.set("run-xyz")
    try:
        assert current_run_id.get() == "run-xyz"
    finally:
        current_run_id.reset(token)
    assert current_run_id.get() == ""


# ---------------------------------------------------------------------------
# FakeLLMClient — agent_name and on_call
# ---------------------------------------------------------------------------


def test_fake_llm_client_stores_agent_name() -> None:
    client = FakeLLMClient([], agent_name="my_agent")
    assert client.agent_name == "my_agent"


def test_fake_llm_client_default_agent_name() -> None:
    client = FakeLLMClient([])
    assert client.agent_name == "fake"


def test_fake_llm_client_on_call_none_by_default() -> None:
    client = FakeLLMClient([])
    assert client._on_call is None


# ---------------------------------------------------------------------------
# _write_call_record / on_call integration
# ---------------------------------------------------------------------------


def test_on_call_not_invoked_when_none() -> None:
    """_write_call_record is a no-op when on_call is None."""
    client = FakeLLMClient(['{"x": 1}'])
    client.complete_json(prompt="p", temperature=0.0, max_tokens=10)
    # No assertion needed — just confirm no AttributeError is raised


def test_on_call_invoked_on_successful_parse() -> None:
    """on_call receives a record with parse_success=True on a clean response."""
    records: list[dict[str, Any]] = []
    client = FakeLLMClient(['{"x": 1}'], agent_name="tester", on_call=lambda run_id, rec: records.append(rec))

    client.complete_json(prompt="p", temperature=0.5, max_tokens=10)

    assert len(records) == 1
    assert records[0]["parse_success"] is True
    assert records[0]["agent"] == "tester"
    assert records[0]["temperature"] == 0.5


def test_on_call_invoked_on_parse_failure() -> None:
    """on_call receives a record with parse_success=False before re-raising."""
    records: list[dict[str, Any]] = []
    client = FakeLLMClient(["bad json"], on_call=lambda run_id, rec: records.append(rec))

    with pytest.raises(orjson.JSONDecodeError):
        client.complete_json(prompt="p", temperature=0.0, max_tokens=10, max_retries=0)

    assert len(records) == 1
    assert records[0]["parse_success"] is False


def test_on_call_receives_run_id_from_context() -> None:
    """The run_id in the record comes from current_run_id ContextVar."""
    records: list[dict[str, Any]] = []
    client = FakeLLMClient(['{"x": 1}'], on_call=lambda run_id, rec: records.append(rec))

    token = current_run_id.set("run-test-123")
    try:
        client.complete_json(prompt="p", temperature=0.0, max_tokens=10)
    finally:
        current_run_id.reset(token)

    assert records[0]["run_id"] == "run-test-123"


def test_on_call_exception_does_not_propagate() -> None:
    """An exception raised inside on_call must not crash the pipeline."""
    def bad_logger(run_id: str, rec: dict[str, Any]) -> None:
        raise RuntimeError("logger exploded")

    client = FakeLLMClient(['{"x": 1}'], on_call=bad_logger)
    result = client.complete_json(prompt="p", temperature=0.0, max_tokens=10)
    assert result == {"x": 1}


def test_latency_ms_is_non_negative_integer() -> None:
    """latency_ms in the record must be a non-negative integer."""
    records: list[dict[str, Any]] = []
    client = FakeLLMClient(['{"x": 1}'], on_call=lambda run_id, rec: records.append(rec))

    client.complete_json(prompt="p", temperature=0.0, max_tokens=10)

    assert isinstance(records[0]["latency_ms"], int)
    assert records[0]["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# complete_json non-dict JSON handling
# ---------------------------------------------------------------------------


def test_complete_json_retries_when_response_is_non_dict_json() -> None:
    """Valid JSON that is not an object must retry, then accept a dict on retry."""
    # Attempt 1: JSON array → triggers the non-dict retry path.
    # Attempt 2: JSON object → accepted.
    client = FakeLLMClient(["[1, 2, 3]", '{"key": "ok"}'])

    result = client.complete_json(prompt="p", temperature=0.0, max_tokens=10)

    assert result == {"key": "ok"}
    assert client.call_count == 2


def test_complete_json_raises_value_error_when_non_dict_json_exhausts_retries() -> None:
    """After max_retries all non-dict JSON responses, complete_json raises ValueError."""
    client = FakeLLMClient(["[1, 2]", '"just a string"'])

    with pytest.raises(ValueError, match="Expected JSON object"):
        client.complete_json(
            prompt="p", temperature=0.0, max_tokens=10, max_retries=1
        )

    assert client.call_count == 2


def test_complete_json_records_non_dict_retry_with_parse_success_false() -> None:
    """Each non-dict attempt must be logged with parse_success=False."""
    records: list[dict[str, Any]] = []
    client = FakeLLMClient(
        ["[1, 2]", '{"ok": true}'],
        on_call=lambda run_id, rec: records.append(rec),
    )

    client.complete_json(prompt="p", temperature=0.0, max_tokens=10)

    assert len(records) == 2
    assert records[0]["parse_success"] is False
    assert records[1]["parse_success"] is True


# ---------------------------------------------------------------------------
# Budget escalation — LLMOutputTruncatedError handling
# ---------------------------------------------------------------------------


def test_complete_json_escalates_budget_on_truncation() -> None:
    """complete_json retries with doubled max_tokens when complete() is truncated."""
    truncation = LLMOutputTruncatedError(partial_response='{"incomplete":', token_budget=100)
    client = FakeLLMClient([truncation, '{"key": "value"}'])

    result = client.complete_json(prompt="test", temperature=0.0, max_tokens=100)

    assert result == {"key": "value"}
    assert client.call_count == 2


def test_complete_json_raises_truncation_when_budget_exhausted() -> None:
    """complete_json re-raises LLMOutputTruncatedError after max_retries with no budget left."""
    truncation = LLMOutputTruncatedError(partial_response='{"incomplete":', token_budget=100)
    client = FakeLLMClient([truncation])

    with pytest.raises(LLMOutputTruncatedError):
        client.complete_json(prompt="test", temperature=0.0, max_tokens=100, max_retries=0)

    assert client.call_count == 1


def test_complete_json_records_truncated_attempt() -> None:
    """The partial response from a truncated attempt is recorded in the call log."""
    records: list[dict[str, Any]] = []
    truncation = LLMOutputTruncatedError(partial_response='{"incomplete":', token_budget=100)
    client = FakeLLMClient(
        [truncation, '{"key": "value"}'],
        on_call=lambda run_id, rec: records.append(rec),
    )

    client.complete_json(prompt="test", temperature=0.0, max_tokens=100)

    assert len(records) == 2
    assert records[0]["parse_success"] is False
    assert records[0]["raw_response"] == '{"incomplete":'
    assert records[1]["parse_success"] is True


def test_fake_llm_client_raises_exception_instances() -> None:
    """FakeLLMClient raises any Exception instance found in its responses list."""
    err = LLMOutputTruncatedError(partial_response="partial", token_budget=50)
    client = FakeLLMClient([err])

    with pytest.raises(LLMOutputTruncatedError):
        client.complete(prompt="test", temperature=0.0, max_tokens=50)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def test_register_provider_rejects_different_class_for_same_name() -> None:
    """Registering a *different* class under an existing name raises ValueError."""
    from storymesh.llm.base import LLMClient, register_provider

    class ProviderA(LLMClient):
        def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return ""

    class ProviderB(LLMClient):
        def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return ""

    register_provider("__test_collide__", ProviderA)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_provider("__test_collide__", ProviderB)
    finally:
        # Clean up so later tests don't see this name.
        from storymesh.llm.base import _PROVIDER_REGISTRY

        _PROVIDER_REGISTRY.pop("__test_collide__", None)


def test_register_provider_idempotent_for_same_class() -> None:
    """Re-registering the *same* class under the same name must not raise."""
    from storymesh.llm.base import LLMClient, register_provider

    class ProviderIdem(LLMClient):
        def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return ""

    register_provider("__test_idem__", ProviderIdem)
    try:
        register_provider("__test_idem__", ProviderIdem)  # no raise
    finally:
        from storymesh.llm.base import _PROVIDER_REGISTRY

        _PROVIDER_REGISTRY.pop("__test_idem__", None)


def test_get_provider_class_unknown_name_raises_value_error() -> None:
    """get_provider_class must raise with a list of registered providers."""
    from storymesh.llm.base import get_provider_class

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider_class("does_not_exist_anywhere")


def test_get_provider_class_returns_registered_class() -> None:
    """get_provider_class returns the previously registered class."""
    from storymesh.llm.base import LLMClient, get_provider_class, register_provider

    class ProviderGet(LLMClient):
        def complete(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return ""

    register_provider("__test_get__", ProviderGet)
    try:
        assert get_provider_class("__test_get__") is ProviderGet
    finally:
        from storymesh.llm.base import _PROVIDER_REGISTRY

        _PROVIDER_REGISTRY.pop("__test_get__", None)