"""Tests for the LLM provider registry (storymesh.llm.base).

Covers register_provider(), get_provider_class(), and the effect on
_build_llm_client() in graph.py. Tests that touch the registry restore
its pre-test state via the ``clean_registry`` fixture.
"""

from __future__ import annotations

import pytest

import storymesh.llm.base as _base_module
import storymesh.llm.image_base as _image_base_module
from storymesh.llm.base import LLMClient, get_provider_class, register_provider
from storymesh.llm.image_base import (
    GeneratedImage,
    ImageClient,
    get_image_provider_class,
    register_image_provider,
)

# ---------------------------------------------------------------------------
# Minimal concrete subclasses for testing (no real LLM calls)
# ---------------------------------------------------------------------------


class _ClientA(LLMClient):
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return ""


class _ClientB(LLMClient):
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return ""


# ---------------------------------------------------------------------------
# Fixture: restore registry state after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry() -> None:  # type: ignore[return]
    """Snapshot and restore _PROVIDER_REGISTRY around each test."""
    snapshot = dict(_base_module._PROVIDER_REGISTRY)
    yield
    _base_module._PROVIDER_REGISTRY.clear()
    _base_module._PROVIDER_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------


class TestRegisterProvider:
    def test_registers_class(self) -> None:
        """Registered class is retrievable by name."""
        register_provider("_test_a", _ClientA)
        assert get_provider_class("_test_a") is _ClientA

    def test_idempotent_same_class(self) -> None:
        """Registering the same class twice under the same name does not raise."""
        register_provider("_test_b", _ClientA)
        register_provider("_test_b", _ClientA)  # should be a no-op

    def test_raises_on_conflicting_class(self) -> None:
        """Registering a different class under an already-taken name raises ValueError."""
        register_provider("_test_c", _ClientA)
        with pytest.raises(ValueError, match="already registered"):
            register_provider("_test_c", _ClientB)


# ---------------------------------------------------------------------------
# get_provider_class
# ---------------------------------------------------------------------------


class TestGetProviderClass:
    def test_returns_registered_class(self) -> None:
        register_provider("_test_d", _ClientA)
        assert get_provider_class("_test_d") is _ClientA

    def test_raises_for_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider_class("_nonexistent_provider_xyz")

    def test_error_message_lists_registered_providers(self) -> None:
        register_provider("_test_e", _ClientA)
        with pytest.raises(ValueError, match="_test_e"):
            get_provider_class("_nonexistent_provider_xyz")


# ---------------------------------------------------------------------------
# AnthropicClient auto-registration
# ---------------------------------------------------------------------------


class TestAnthropicRegistration:
    def test_anthropic_registered_after_import(self) -> None:
        """AnthropicClient self-registers when its module is imported."""
        import storymesh.llm.anthropic  # noqa: F401
        from storymesh.llm.anthropic import AnthropicClient

        assert get_provider_class("anthropic") is AnthropicClient


# ---------------------------------------------------------------------------
# _build_llm_client integration
# ---------------------------------------------------------------------------


class TestBuildLLMClient:
    def test_returns_none_when_no_provider(self) -> None:
        from storymesh.orchestration.graph import _build_llm_client

        assert _build_llm_client({}) is None

    def test_returns_none_when_provider_key_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing API key causes static-only mode (None returned)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from storymesh.orchestration.graph import _build_llm_client

        result = _build_llm_client({"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})
        assert result is None

    def test_returns_correct_type_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid API key causes the registered client class to be instantiated."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from storymesh.llm.anthropic import AnthropicClient
        from storymesh.orchestration.graph import _build_llm_client

        result = _build_llm_client({"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})
        assert isinstance(result, AnthropicClient)

    def test_raises_for_unregistered_provider_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A provider that has a key set but is not in the registry raises ValueError."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        from storymesh.orchestration.graph import _build_llm_client

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _build_llm_client({"provider": "google", "model": "gemini-pro"})


# ---------------------------------------------------------------------------
# Image client registry
# ---------------------------------------------------------------------------


class _FakeImageClientA(ImageClient):
    def generate(self, prompt: str, *, size: str, quality: str, style: str) -> GeneratedImage:
        return GeneratedImage(image_bytes=b"PNG", revised_prompt=None)


class _FakeImageClientB(ImageClient):
    def generate(self, prompt: str, *, size: str, quality: str, style: str) -> GeneratedImage:
        return GeneratedImage(image_bytes=b"PNG", revised_prompt=None)


@pytest.fixture(autouse=True)
def clean_image_registry() -> None:  # type: ignore[return]
    """Snapshot and restore _IMAGE_REGISTRY around each test."""
    snapshot = dict(_image_base_module._IMAGE_REGISTRY)
    yield
    _image_base_module._IMAGE_REGISTRY.clear()
    _image_base_module._IMAGE_REGISTRY.update(snapshot)


class TestImageRegistry:
    def test_register_and_retrieve(self) -> None:
        """Registered class is retrievable by name."""
        register_image_provider("_test_img_a", _FakeImageClientA)
        assert get_image_provider_class("_test_img_a") is _FakeImageClientA

    def test_duplicate_registration_same_class_ok(self) -> None:
        """Registering the same class twice under the same name does not raise."""
        register_image_provider("_test_img_b", _FakeImageClientA)
        register_image_provider("_test_img_b", _FakeImageClientA)  # idempotent

    def test_duplicate_registration_different_class_raises(self) -> None:
        """Registering a different class under an already-taken name raises ValueError."""
        register_image_provider("_test_img_c", _FakeImageClientA)
        with pytest.raises(ValueError, match="already registered"):
            register_image_provider("_test_img_c", _FakeImageClientB)

    def test_unknown_provider_raises(self) -> None:
        """Requesting an unregistered provider raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown image provider"):
            get_image_provider_class("_nonexistent_image_provider_xyz")
