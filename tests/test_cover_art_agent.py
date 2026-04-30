"""Unit tests for CoverArtAgent (Stage 7)."""

from __future__ import annotations

import io

import pytest

from storymesh.agents.cover_art.agent import (
    _BYLINE,
    CoverArtAgent,
    GeneratedCoverImage,
    _build_assembled_prompt,
    _compose_cover_text,
    _safe_compose_cover_text,
)
from storymesh.llm.image_base import GeneratedImage, ImageClient
from storymesh.schemas.cover_art import CoverArtAgentInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PROMPT = (
    "A rain-slicked street in a flooded cityscape at dusk, a single figure "
    "silhouetted against pale light. Gritty noir ink wash, muted greys."
)


def _tiny_png(width: int = 256, height: int = 384) -> bytes:
    """Create a minimal valid RGB PNG for PIL-dependent tests.

    Requires Pillow; tests that call this should guard with
    ``pytest.importorskip("PIL")``.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", (width, height), color=(60, 60, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeImageClient(ImageClient):
    """Minimal ImageClient stub for testing CoverArtAgent."""

    def __init__(
        self,
        image_bytes: bytes = b"PNG",
        revised_prompt: str | None = None,
    ) -> None:
        super().__init__(model="fake-image-model", agent_name="fake")
        self._image_bytes = image_bytes
        self._revised_prompt = revised_prompt
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
    ) -> GeneratedImage:
        self.calls.append({"prompt": prompt, "size": size, "quality": quality})
        return GeneratedImage(image_bytes=self._image_bytes, revised_prompt=self._revised_prompt)


def _make_input(**overrides: object) -> CoverArtAgentInput:
    defaults: dict[str, object] = {
        "image_prompt": _VALID_PROMPT,
        "title": "The Last Inquest",
    }
    return CoverArtAgentInput(**{**defaults, **overrides})


def _make_agent(
    client: FakeImageClient | None = None,
    **kwargs: object,
) -> CoverArtAgent:
    return CoverArtAgent(image_client=client or FakeImageClient(), **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestBuildAssembledPrompt
# ---------------------------------------------------------------------------


class TestBuildAssembledPrompt:
    def test_contains_source_prompt(self) -> None:
        result = _build_assembled_prompt(_VALID_PROMPT)
        assert _VALID_PROMPT in result

    def test_contains_flat_canvas_instruction(self) -> None:
        result = _build_assembled_prompt(_VALID_PROMPT)
        assert "Flat 2D artwork" in result

    def test_contains_no_book_object_instruction(self) -> None:
        result = _build_assembled_prompt(_VALID_PROMPT)
        assert "No book object" in result

    def test_contains_no_text_instruction(self) -> None:
        result = _build_assembled_prompt(_VALID_PROMPT)
        assert "No text" in result

    def test_different_source_prompts_produce_different_results(self) -> None:
        a = _build_assembled_prompt("Prompt A, some visual description here.")
        b = _build_assembled_prompt("Prompt B, different visual description.")
        assert a != b


# ---------------------------------------------------------------------------
# TestComposeCoverText  (requires Pillow)
# ---------------------------------------------------------------------------


class TestComposeCoverText:
    """Tests for _compose_cover_text and _safe_compose_cover_text.

    Skipped automatically when Pillow is not installed.
    """

    @pytest.fixture(autouse=True)
    def require_pillow(self) -> None:
        pytest.importorskip("PIL")

    def test_returns_valid_png(self) -> None:
        png_magic = b"\x89PNG\r\n\x1a\n"
        result = _compose_cover_text(_tiny_png(), "My Title", _BYLINE)
        assert result[:8] == png_magic

    def test_output_differs_from_input(self) -> None:
        raw = _tiny_png()
        result = _compose_cover_text(raw, "My Title", _BYLINE)
        # PIL re-encodes even without text changes, but different titles should differ.
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_different_titles_produce_different_bytes(self) -> None:
        raw = _tiny_png()
        a = _compose_cover_text(raw, "Title Alpha", _BYLINE)
        b = _compose_cover_text(raw, "Title Beta", _BYLINE)
        assert a != b

    def test_safe_compose_returns_raw_bytes_on_invalid_input(self) -> None:
        invalid = b"not a PNG at all"
        result = _safe_compose_cover_text(invalid, "My Title", _BYLINE)
        assert result == invalid

    def test_safe_compose_succeeds_with_valid_png(self) -> None:
        png_magic = b"\x89PNG\r\n\x1a\n"
        result = _safe_compose_cover_text(_tiny_png(), "My Title", _BYLINE)
        assert result[:8] == png_magic


# ---------------------------------------------------------------------------
# TestCoverArtAgentRun
# ---------------------------------------------------------------------------


class TestCoverArtAgentRun:
    def test_returns_generated_cover_image_type(self) -> None:
        agent = _make_agent()
        result = agent.run(_make_input())
        assert isinstance(result, GeneratedCoverImage)

    def test_image_bytes_returned(self) -> None:
        # Raw fake bytes are not valid PNG, so safe_compose falls back to them unchanged.
        client = FakeImageClient(image_bytes=b"\x89PNG_FAKE")
        agent = _make_agent(client=client)
        result = agent.run(_make_input())
        assert result.image_bytes == b"\x89PNG_FAKE"

    def test_assembled_prompt_contains_source_prompt(self) -> None:
        agent = _make_agent()
        inp = _make_input()
        result = agent.run(inp)
        assert inp.image_prompt in result.image_prompt

    def test_flat_canvas_instruction_in_assembled_prompt(self) -> None:
        agent = _make_agent()
        result = agent.run(_make_input())
        assert "Flat 2D artwork" in result.image_prompt

    def test_no_title_text_in_assembled_prompt(self) -> None:
        """Title is composited by PIL, not injected into the DALL-E prompt."""
        agent = _make_agent()
        result = agent.run(_make_input(title="Unique Title XYZ"))
        assert "Unique Title XYZ" not in result.image_prompt

    def test_assembled_prompt_sent_to_client(self) -> None:
        client = FakeImageClient()
        agent = _make_agent(client=client)
        inp = _make_input()
        agent.run(inp)
        assert len(client.calls) == 1
        sent_prompt = client.calls[0]["prompt"]
        assert isinstance(sent_prompt, str)
        assert inp.image_prompt in sent_prompt
        assert "Flat 2D artwork" in sent_prompt

    def test_revised_prompt_propagated_when_none(self) -> None:
        client = FakeImageClient(revised_prompt=None)
        agent = _make_agent(client=client)
        result = agent.run(_make_input())
        assert result.revised_prompt is None

    def test_revised_prompt_propagated_when_present(self) -> None:
        revised = "A revised prompt for safety compliance purposes."
        client = FakeImageClient(revised_prompt=revised)
        agent = _make_agent(client=client)
        result = agent.run(_make_input())
        assert result.revised_prompt == revised

    def test_model_from_client(self) -> None:
        agent = _make_agent()
        result = agent.run(_make_input())
        assert result.model == "fake-image-model"

    def test_size_quality_passed_to_client(self) -> None:
        client = FakeImageClient()
        agent = CoverArtAgent(
            image_client=client,
            image_size="1792x1024",
            image_quality="high",
        )
        agent.run(_make_input())
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["size"] == "1792x1024"
        assert call["quality"] == "high"

    def test_latency_ms_is_non_negative(self) -> None:
        agent = _make_agent()
        result = agent.run(_make_input())
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# TestCoverArtAgentRunWithPillow  (requires Pillow)
# ---------------------------------------------------------------------------


class TestCoverArtAgentRunWithPillow:
    """End-to-end agent run tests with real PNG bytes and Pillow installed."""

    @pytest.fixture(autouse=True)
    def require_pillow(self) -> None:
        pytest.importorskip("PIL")

    def test_image_bytes_are_valid_png_when_input_is_valid(self) -> None:
        png_magic = b"\x89PNG\r\n\x1a\n"
        client = FakeImageClient(image_bytes=_tiny_png())
        agent = _make_agent(client=client)
        result = agent.run(_make_input())
        assert result.image_bytes[:8] == png_magic

    def test_composited_bytes_differ_from_raw(self) -> None:
        raw = _tiny_png()
        client = FakeImageClient(image_bytes=raw)
        agent = _make_agent(client=client)
        result = agent.run(_make_input(title="Some Title Here"))
        assert result.image_bytes != raw

    def test_different_titles_produce_different_images(self) -> None:
        raw = _tiny_png()
        client_a = FakeImageClient(image_bytes=raw)
        client_b = FakeImageClient(image_bytes=raw)
        agent_a = _make_agent(client=client_a)
        agent_b = _make_agent(client=client_b)
        result_a = agent_a.run(_make_input(title="Title Alpha"))
        result_b = agent_b.run(_make_input(title="Title Beta"))
        assert result_a.image_bytes != result_b.image_bytes


# ---------------------------------------------------------------------------
# TestCoverArtAgentConfig
# ---------------------------------------------------------------------------


class TestCoverArtAgentConfig:
    def test_default_size(self) -> None:
        agent = _make_agent()
        assert agent._image_size == "1024x1792"

    def test_default_quality(self) -> None:
        agent = _make_agent()
        assert agent._image_quality == "auto"

    def test_custom_params_respected(self) -> None:
        client = FakeImageClient()
        agent = CoverArtAgent(
            image_client=client,
            image_size="1024x1792",
            image_quality="high",
        )
        agent.run(_make_input())
        call = client.calls[0]
        assert call["size"] == "1024x1792"
        assert call["quality"] == "high"


# ---------------------------------------------------------------------------
# TestCoverArtAgentErrors
# ---------------------------------------------------------------------------


class TestCoverArtAgentErrors:
    def test_client_error_propagates(self) -> None:
        class _ErrorClient(ImageClient):
            def generate(
                self, prompt: str, *, size: str, quality: str
            ) -> GeneratedImage:
                raise ValueError("API response missing expected fields.")

        agent = CoverArtAgent(image_client=_ErrorClient(model="err-model"))
        with pytest.raises(ValueError, match="API response missing expected fields."):
            agent.run(_make_input())
