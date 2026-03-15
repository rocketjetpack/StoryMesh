import orjson
import pytest

from storymesh.llm.base import FakeLLMClient


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