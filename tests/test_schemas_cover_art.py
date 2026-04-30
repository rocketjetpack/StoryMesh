"""Unit tests for storymesh.schemas.cover_art."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.cover_art import CoverArtAgentInput, CoverArtAgentOutput
from storymesh.versioning.schemas import COVER_ART_SCHEMA_VERSION

_VALID_PROMPT = (
    "A rain-slicked street in a flooded cityscape at dusk, a single figure "
    "silhouetted against pale light. Gritty noir ink wash style, muted greys."
)


def _valid_input(**overrides: object) -> CoverArtAgentInput:
    defaults: dict[str, object] = {
        "image_prompt": _VALID_PROMPT,
        "title": "The Last Inquest",
    }
    return CoverArtAgentInput(**{**defaults, **overrides})


def _valid_output(**overrides: object) -> CoverArtAgentOutput:
    defaults: dict[str, object] = {
        "image_path": "/runs/abc123/cover_art.png",
        "image_prompt": _VALID_PROMPT,
        "revised_prompt": None,
        "model": "dall-e-3",
        "image_size": "1024x1024",
        "image_quality": "standard",
        "image_style": "vivid",
    }
    return CoverArtAgentOutput(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# TestCoverArtAgentInput
# ---------------------------------------------------------------------------


class TestCoverArtAgentInput:
    def test_valid_construction(self) -> None:
        inp = _valid_input()
        assert inp.title == "The Last Inquest"
        assert inp.image_prompt == _VALID_PROMPT

    def test_image_prompt_min_length(self) -> None:
        with pytest.raises(ValidationError):
            _valid_input(image_prompt="Too short prompt here.")

    def test_title_min_length(self) -> None:
        with pytest.raises(ValidationError):
            _valid_input(title="")


# ---------------------------------------------------------------------------
# TestCoverArtAgentOutput
# ---------------------------------------------------------------------------


class TestCoverArtAgentOutput:
    def test_valid_construction(self) -> None:
        out = _valid_output()
        assert out.model == "dall-e-3"
        assert out.image_size == "1024x1024"

    def test_frozen(self) -> None:
        out = _valid_output()
        with pytest.raises(ValidationError):
            out.model = "dall-e-2"  # type: ignore[misc]

    def test_revised_prompt_defaults_to_none(self) -> None:
        out = _valid_output()
        assert out.revised_prompt is None

    def test_revised_prompt_can_be_set(self) -> None:
        revised = "A revised version of the prompt for safety compliance."
        out = _valid_output(revised_prompt=revised)
        assert out.revised_prompt == revised

    def test_debug_defaults_to_empty_dict(self) -> None:
        out = _valid_output()
        assert out.debug == {}

    def test_debug_accepts_metadata(self) -> None:
        out = _valid_output(debug={"latency_ms": 4200, "title": "The Last Inquest"})
        assert out.debug["latency_ms"] == 4200

    def test_schema_version_matches(self) -> None:
        out = _valid_output()
        assert out.schema_version == COVER_ART_SCHEMA_VERSION

    def test_image_path_can_be_empty_string(self) -> None:
        out = _valid_output(image_path="")
        assert out.image_path == ""
