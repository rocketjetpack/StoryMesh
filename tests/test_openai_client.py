"""Unit tests for storymesh.llm.openai."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from storymesh.exceptions import LLMOutputTruncatedError
from storymesh.llm.base import get_provider_class
from storymesh.llm.openai import _DEFAULT_MODEL, OpenAIClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(content: str = "hello", finish_reason: str = "stop") -> MagicMock:
    """Build a mock OpenAI chat completions response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Constructor — API key resolution
# ---------------------------------------------------------------------------


class TestConstructorApiKey:
    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_explicit_key(self, mock_openai: MagicMock) -> None:
        client = OpenAIClient(api_key="sk-test-123")
        assert client.api_key == "sk-test-123"

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_key_from_env(self, mock_openai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-456")
        client = OpenAIClient()
        assert client.api_key == "sk-env-456"

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_no_key_raises(self, mock_openai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            OpenAIClient()


# ---------------------------------------------------------------------------
# Constructor — model resolution
# ---------------------------------------------------------------------------


class TestConstructorModel:
    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_explicit_model(self, mock_openai: MagicMock) -> None:
        client = OpenAIClient(api_key="sk-test", model="gpt-4o")
        assert client.model == "gpt-4o"

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_default_model(self, mock_openai: MagicMock) -> None:
        client = OpenAIClient(api_key="sk-test")
        assert client.model == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# complete() — response handling
# ---------------------------------------------------------------------------


class TestComplete:
    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_returns_text(self, mock_openai: MagicMock) -> None:
        mock_openai.return_value.chat.completions.create.return_value = _mock_response("test output")
        client = OpenAIClient(api_key="sk-test")
        result = client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        assert result == "test output"

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_system_prompt_passed(self, mock_openai: MagicMock) -> None:
        mock_openai.return_value.chat.completions.create.return_value = _mock_response("ok")
        client = OpenAIClient(api_key="sk-test")
        client.complete(prompt="hello", system_prompt="be helpful", temperature=0.0, max_tokens=100)
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be helpful"}

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_no_system_prompt_omitted(self, mock_openai: MagicMock) -> None:
        mock_openai.return_value.chat.completions.create.return_value = _mock_response("ok")
        client = OpenAIClient(api_key="sk-test")
        client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_empty_choices_raises(self, mock_openai: MagicMock) -> None:
        response = _mock_response()
        response.choices = []
        mock_openai.return_value.chat.completions.create.return_value = response
        client = OpenAIClient(api_key="sk-test")
        with pytest.raises(ValueError, match="empty choices"):
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_none_content_raises(self, mock_openai: MagicMock) -> None:
        response = _mock_response()
        response.choices[0].message.content = None
        mock_openai.return_value.chat.completions.create.return_value = response
        client = OpenAIClient(api_key="sk-test")
        with pytest.raises(ValueError, match="None content"):
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_truncated_response_raises_llm_output_truncated_error(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.return_value.chat.completions.create.return_value = _mock_response(
            content='{"incomplete":', finish_reason="length"
        )
        client = OpenAIClient(api_key="sk-test")
        with pytest.raises(LLMOutputTruncatedError) as exc_info:
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        assert exc_info.value.token_budget == 100
        assert exc_info.value.partial_response == '{"incomplete":'

    @patch("storymesh.llm.openai.openai.OpenAI")
    def test_stop_finish_reason_does_not_raise_truncation(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.return_value.chat.completions.create.return_value = _mock_response(
            content="complete response", finish_reason="stop"
        )
        client = OpenAIClient(api_key="sk-test")
        result = client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        assert result == "complete response"


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    def test_registers_on_import(self) -> None:
        assert get_provider_class("openai") is OpenAIClient
