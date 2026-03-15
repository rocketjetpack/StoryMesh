import pytest

from storymesh.llm.anthropic import AnthropicClient


@pytest.mark.real_api
def test_anthropic_real_api() -> None:
    """Requires ANTHROPIC_API_KEY in environment. Run with: pytest -m integration"""
    client = AnthropicClient()
    result = client.complete(
        prompt="Reply with exactly: hello",
        temperature=0.0,
        max_tokens=10,
    )
    assert "hello" in result.lower()