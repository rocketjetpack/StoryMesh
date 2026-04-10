"""Real API integration test for OpenAIClient.

Run with: pytest -m real_api --real-apis
Requires OPENAI_API_KEY to be set in the environment.
"""

from __future__ import annotations

import pytest

from storymesh.llm.openai import OpenAIClient


@pytest.mark.real_api
def test_openai_real_api() -> None:
    """Requires OPENAI_API_KEY in environment."""
    client = OpenAIClient()
    result = client.complete(
        prompt="Reply with exactly: hello",
        temperature=0.0,
        max_tokens=10,
    )
    assert "hello" in result.lower()
