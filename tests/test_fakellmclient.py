from __future__ import annotations

from typing import Any

import orjson
import pytest

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