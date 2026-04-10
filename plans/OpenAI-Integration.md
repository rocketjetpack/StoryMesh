# StoryMesh Implementation Plan — OpenAI Provider

**Date:** 2026-04-10
**Version:** 0.6.0 → 0.7.0
**Scope:** Add `OpenAIClient` as a second registered LLM provider

---

## Context

The LLM abstraction layer (`src/storymesh/llm/`) is fully in place. The provider registry (`register_provider` / `get_provider_class` in `base.py`), the dynamic import mechanism (`_ensure_provider_imported` in `graph.py`), and the `_PROVIDER_KEY_MAP` in `config.py` all already reference `"openai"`. The only missing piece is the concrete `OpenAIClient` class and its tests.

The `openai` SDK is currently an optional extra in `pyproject.toml`. Per the decision made during planning, it stays optional (users install `storymesh[openai]` to use it). The dynamic import in `_ensure_provider_imported` already handles a missing SDK gracefully — it logs a warning and the client is not instantiated.

**Default model:** `gpt-4o-mini`

---

## Work Item Ordering

```
WI-1: Create src/storymesh/llm/openai.py
  │
WI-2: Update src/storymesh/llm/__init__.py
  │
WI-3: Update storymesh.config.yaml and storymesh.config.yaml.example
  │
WI-4: Create tests/test_openai_client.py
  │
WI-5: Create tests/llm_clients/test_openai_real_api.py
  │
WI-6: Version bump + README update
```

---

## WI-1: Create `src/storymesh/llm/openai.py`

**This is the primary deliverable.** Model it directly on `src/storymesh/llm/anthropic.py`.

### Key differences from `AnthropicClient`

| Concern | `AnthropicClient` | `OpenAIClient` |
|---|---|---|
| SDK import | `import anthropic` | `import openai` |
| API key env var | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| Default model constant | `claude-haiku-4-5-20251001` | `gpt-4o-mini` |
| SDK client class | `anthropic.Anthropic(api_key=...)` | `openai.OpenAI(api_key=...)` |
| API call | `client.messages.create(...)` | `client.chat.completions.create(...)` |
| System prompt injection | Top-level `system=` kwarg | Prepend `{"role": "system", "content": system_prompt}` to the messages list |
| Response extraction | `response.content[0].text` | `response.choices[0].message.content` |
| Response guard | Check `len(response.content) == 1` and `block.type == "text"` | Check `len(response.choices) >= 1` and `response.choices[0].message.content is not None` |

### File structure

```python
"""OpenAI LLM client implementation as a subclass of LLMClient."""

from __future__ import annotations

import os
from typing import Any

import openai

from storymesh.llm.base import LLMCallLogger, LLMClient, _traceable, register_provider

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(LLMClient):
    """LLMClient implementation backed by the OpenAI chat completions API.

    Uses the ``openai`` Python SDK. API key is resolved from the ``api_key``
    argument or the ``OPENAI_API_KEY`` environment variable.

    Args:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        model: Model identifier. Defaults to ``gpt-4o-mini``.
        agent_name: Identifies the calling agent in LLM call logs.
        on_call: Optional callback invoked after every ``complete()`` call.

    Raises:
        ValueError: If no API key is found in args or environment.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        agent_name: str = "unknown",
        on_call: LLMCallLogger | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key has not been provided. "
                "Pass api_key or set OPENAI_API_KEY in the environment."
            )

        resolved_model = model or _DEFAULT_MODEL

        super().__init__(
            api_key=resolved_key,
            model=resolved_model,
            agent_name=agent_name,
            on_call=on_call,
        )
        self.client = openai.OpenAI(api_key=resolved_key)

    @_traceable
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send a prompt to the OpenAI chat completions API and return the text response.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system message prepended to the conversation.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate in the response.

        Returns:
            The text content of the model's response.

        Raises:
            ValueError: If the API returns an unexpected response structure.
            openai.OpenAIError: On API-level failures (rate limits, auth errors, etc.).
        """
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,  # type: ignore[arg-type]
        )

        if not response.choices:
            raise ValueError("OpenAI returned an empty choices list.")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                "OpenAI response choice contained a None content field. "
                "This can occur when the model uses tool calls instead of a text response."
            )

        return content


register_provider("openai", OpenAIClient)
```

### Notes for Claude Code

- The `@_traceable` decorator is imported from `base.py` — same as in `anthropic.py`. Do not redefine it.
- The `messages` list type annotation uses `list[dict[str, Any]]` to avoid importing OpenAI's internal types. The `# type: ignore[arg-type]` comment suppresses the mypy complaint about the dict not being the SDK's `ChatCompletionMessageParam` type. This is acceptable because the dict structure matches exactly.
- `register_provider("openai", OpenAIClient)` must be the last line in the file (module-level, after the class definition), same as in `anthropic.py`.

---

## WI-2: Update `src/storymesh/llm/__init__.py`

Add `OpenAIClient` to the public exports. The current file exports `AnthropicClient` and the base class symbols. Add `OpenAIClient` in the same pattern.

**Exact change:** Add `OpenAIClient` to the `__all__` list and add the corresponding import. The import must be conditional on the SDK being installed, matching the existing pattern for `AnthropicClient` if one exists — otherwise import directly.

Check the current `__init__.py` first. If `AnthropicClient` is imported unconditionally, do the same for `OpenAIClient`. If it uses a `try/except ImportError` guard, apply the same guard.

---

## WI-3: Update `storymesh.config.yaml` and `storymesh.config.yaml.example`

Add a commented-out block under the `agents:` section in both files to document how to switch an agent to OpenAI. Place this after the last real agent config block and before any closing comments.

```yaml
  # To use OpenAI for any agent, set provider and model:
  # proposal_draft:
  #   provider: openai
  #   model: gpt-4o
  #   temperature: 0.7
  #   max_tokens: 2048
```

No functional change to the config — this is documentation only. Both files must receive the identical comment block.

---

## WI-4: Create `tests/test_openai_client.py`

Mirror the structure of `tests/test_anthropic.py` exactly. Use `unittest.mock.patch` to mock `storymesh.llm.openai.openai.OpenAI` so no real API calls are made.

### Helper

```python
def _mock_response(content: str = "hello") -> MagicMock:
    """Build a mock OpenAI chat completions response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response
```

### Test classes required

**`TestConstructorApiKey`**
- `test_explicit_key` — pass `api_key="sk-test-123"`, assert `client.api_key == "sk-test-123"`
- `test_key_from_env` — `monkeypatch.setenv("OPENAI_API_KEY", "sk-env-456")`, assert `client.api_key == "sk-env-456"`
- `test_no_key_raises` — `monkeypatch.delenv("OPENAI_API_KEY", raising=False)`, assert `ValueError` with message matching `"API key"`

**`TestConstructorModel`**
- `test_explicit_model` — pass `model="gpt-4o"`, assert `client.model == "gpt-4o"`
- `test_default_model` — no model arg, assert `client.model == _DEFAULT_MODEL`

**`TestComplete`**
- `test_returns_text` — mock returns content `"test output"`, assert `complete(...)` returns `"test output"`
- `test_system_prompt_passed` — call with `system_prompt="be helpful"`, assert the mock was called with a messages list where `messages[0] == {"role": "system", "content": "be helpful"}`
- `test_no_system_prompt_omitted` — call without `system_prompt`, assert no system message appears in the messages list
- `test_empty_choices_raises` — set `response.choices = []`, assert `ValueError`
- `test_none_content_raises` — set `response.choices[0].message.content = None`, assert `ValueError`

**`TestProviderRegistration`**
- `test_registers_on_import` — import `OpenAIClient`, call `get_provider_class("openai")`, assert the result `is OpenAIClient`

### Patch path

The patch target must be `"storymesh.llm.openai.openai.OpenAI"` — patching at the point of use in the module, not in the `openai` package itself.

---

## WI-5: Create `tests/llm_clients/test_openai_real_api.py`

```python
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
```

---

## WI-6: Version Bump and README Update

### `pyproject.toml`

Bump version from `0.6.0` to `0.7.0`.

```toml
version = "0.7.0"
```

### `README.md`

In the **Roadmap** section, replace or mark complete:

```
- ~~Expand provider support beyond the current Anthropic implementation~~
- OpenAI (`gpt-4o-mini` default) now supported; configure via `provider: openai` in `storymesh.config.yaml`
```

In the **Known Gaps** section, remove any bullet referencing OpenAI as unimplemented if one exists.

---

## Validation Checklist

Run these after all work items are complete, in order.

```bash
# 1. Linting
ruff check src/ tests/

# 2. Type checking
mypy src/storymesh/

# 3. Unit tests (no API keys required)
pytest tests/test_openai_client.py -v

# 4. Full test suite
pytest

# 5. Verify registry at the REPL
python -c "
from storymesh.llm.openai import OpenAIClient
from storymesh.llm.base import get_provider_class
assert get_provider_class('openai') is OpenAIClient
print('Registry OK')
"

# 6. Verify show-agent-config still works (no API keys needed)
storymesh show-config
storymesh show-agent-config genre_normalizer

# 7. Real API test (requires OPENAI_API_KEY)
pytest tests/llm_clients/test_openai_real_api.py -m real_api --real-apis -v
```

---

## What Is NOT in Scope

- Streaming responses (not part of the `LLMClient` contract)
- OpenAI function/tool calling
- Switching any existing agent to OpenAI (that is a config-only change, not a code change)
- Google/Gemini provider (deferred)
- Removing the `openai` optional extra — it stays optional; users install `storymesh[openai]` if they want it