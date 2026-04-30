"""Unit tests for storymesh.llm.anthropic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from storymesh.exceptions import LLMOutputTruncatedError
from storymesh.llm.anthropic import _DEFAULT_MODEL, AnthropicClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(
    text: str = "hello",
    block_type: str = "text",
    num_blocks: int = 1,
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a mock Anthropic API response."""
    block = MagicMock()
    block.type = block_type
    block.text = text
    response = MagicMock()
    response.content = [block] * num_blocks
    response.stop_reason = stop_reason
    return response


# ---------------------------------------------------------------------------
# Constructor — API key resolution
# ---------------------------------------------------------------------------

class TestConstructorApiKey:
    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_explicit_key(self, mock_anthropic: MagicMock) -> None:
        client = AnthropicClient(api_key="sk-test-123")
        assert client.api_key == "sk-test-123"

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_key_from_env(self, mock_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-456")
        client = AnthropicClient()
        assert client.api_key == "sk-env-456"

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_no_key_raises(self, mock_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            AnthropicClient()


# ---------------------------------------------------------------------------
# Constructor — model resolution
# ---------------------------------------------------------------------------

class TestConstructorModel:
    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_explicit_model(self, mock_anthropic: MagicMock) -> None:
        client = AnthropicClient(api_key="sk-test", model="claude-sonnet-4-6")
        assert client.model == "claude-sonnet-4-6"

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_default_model(self, mock_anthropic: MagicMock) -> None:
        client = AnthropicClient(api_key="sk-test")
        assert client.model == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# complete() — response handling
# ---------------------------------------------------------------------------

class TestComplete:
    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_returns_text(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response("test output")
        client = AnthropicClient(api_key="sk-test")
        result = client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        assert result == "test output"

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_system_prompt_passed(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response("ok")
        client = AnthropicClient(api_key="sk-test")
        client.complete(prompt="hello", system_prompt="be helpful", temperature=0.0, max_tokens=100)
        call_kwargs = mock_anthropic.return_value.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "be helpful"

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_no_system_prompt_omitted(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response("ok")
        client = AnthropicClient(api_key="sk-test")
        client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        call_kwargs = mock_anthropic.return_value.messages.create.call_args
        assert "system" not in call_kwargs.kwargs

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_multiple_blocks_raises(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response(num_blocks=2)
        client = AnthropicClient(api_key="sk-test")
        with pytest.raises(ValueError, match="Expected exactly 1 content block"):
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_zero_blocks_raises(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response(num_blocks=0)
        client = AnthropicClient(api_key="sk-test")
        with pytest.raises(ValueError, match="Expected exactly 1 content block"):
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_non_text_block_raises(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response(block_type="tool_use")
        client = AnthropicClient(api_key="sk-test")
        with pytest.raises(ValueError, match="Expected a text content block"):
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_truncated_response_raises_llm_output_truncated_error(
        self, mock_anthropic: MagicMock
    ) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response(
            text='{"incomplete":', stop_reason="max_tokens"
        )
        client = AnthropicClient(api_key="sk-test")
        with pytest.raises(LLMOutputTruncatedError) as exc_info:
            client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        assert exc_info.value.token_budget == 100
        assert exc_info.value.partial_response == '{"incomplete":'

    @patch("storymesh.llm.anthropic.anthropic.Anthropic")
    def test_end_turn_does_not_raise_truncation(self, mock_anthropic: MagicMock) -> None:
        mock_anthropic.return_value.messages.create.return_value = _mock_response(
            text="complete response", stop_reason="end_turn"
        )
        client = AnthropicClient(api_key="sk-test")
        result = client.complete(prompt="hello", temperature=0.0, max_tokens=100)
        assert result == "complete response"

