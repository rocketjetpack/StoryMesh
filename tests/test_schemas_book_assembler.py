"""Tests for BookAssemblerAgent Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.book_assembler import (
    BookAssemblerAgentInput,
    BookAssemblerAgentOutput,
)
from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.story_writer import SceneOutline, StoryWriterAgentOutput
from storymesh.versioning.schemas import (
    BOOK_ASSEMBLER_SCHEMA_VERSION,
    STORY_WRITER_SCHEMA_VERSION,
)


def _make_minimal_input(**overrides: object) -> BookAssemblerAgentInput:
    """Construct a minimal valid BookAssemblerAgentInput for schema tests."""
    scenes = [
        SceneOutline(
            scene_id=f"scene_{i:02d}",
            title=f"Scene {i}",
            summary=f"Something important happens in scene {i}.",
            narrative_pressure=(
                f"Scene {i} inhabits the central tension without resolving it."
            ),
            observational_anchor=f"A concrete detail grounding scene {i}.",
            opens_with="The light changed in a way that made her feel exposed.",
        )
        for i in range(1, 4)
    ]
    story = StoryWriterAgentOutput(
        back_cover_summary=(
            "A back-cover summary that comfortably exceeds the minimum length "
            "required by the StoryWriterAgentOutput schema for these tests."
        ),
        scene_list=scenes,
        full_draft="Some draft prose. " * 60,
        word_count=120,
        schema_version=STORY_WRITER_SCHEMA_VERSION,
    )
    proposal = StoryProposal.model_construct(
        seed_id="seed_01",
        title="Title",
        protagonist="A protagonist.",
        setting="A setting.",
        plot_arc="A plot arc.",
        thematic_thesis="A thematic thesis.",
        key_scenes=["Scene one."],
        tensions_addressed=["t_01"],
        tone=["dark"],
        genre_blend=["Mystery"],
        image_prompt="A mood-evoking image.",
    )
    kwargs: dict[str, object] = {
        "story_writer_output": story,
        "proposal": proposal,
        "run_id": "abc123",
    }
    kwargs.update(overrides)
    return BookAssemblerAgentInput(**kwargs)  # type: ignore[arg-type]

# ── BookAssemblerAgentOutput ───────────────────────────────────────────────────


class TestBookAssemblerAgentOutput:
    def test_valid_with_both_paths(self) -> None:
        output = BookAssemblerAgentOutput(
            pdf_path="/tmp/run/output.pdf",
            epub_path="/tmp/run/output.epub",
            title="The Dark Case",
            word_count=3000,
        )
        assert output.pdf_path == "/tmp/run/output.pdf"
        assert output.epub_path == "/tmp/run/output.epub"
        assert output.title == "The Dark Case"
        assert output.word_count == 3000
        assert output.schema_version == BOOK_ASSEMBLER_SCHEMA_VERSION

    def test_valid_with_empty_paths(self) -> None:
        output = BookAssemblerAgentOutput(
            pdf_path="",
            epub_path="",
            title="No Libraries",
            word_count=0,
        )
        assert output.pdf_path == ""
        assert output.epub_path == ""
        assert output.word_count == 0

    def test_debug_defaults_to_empty_dict(self) -> None:
        output = BookAssemblerAgentOutput(
            pdf_path="",
            epub_path="",
            title="Test",
            word_count=100,
        )
        assert output.debug == {}

    def test_word_count_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            BookAssemblerAgentOutput(
                pdf_path="",
                epub_path="",
                title="Test",
                word_count=-1,
            )

    def test_frozen(self) -> None:
        output = BookAssemblerAgentOutput(
            pdf_path="/a.pdf",
            epub_path="/a.epub",
            title="Frozen",
            word_count=500,
        )
        with pytest.raises(ValidationError):
            output.pdf_path = "/b.pdf"  # type: ignore[misc]

    def test_schema_version_constant(self) -> None:
        output = BookAssemblerAgentOutput(
            pdf_path="",
            epub_path="",
            title="Version Check",
            word_count=100,
        )
        assert output.schema_version == BOOK_ASSEMBLER_SCHEMA_VERSION


# ── BookAssemblerAgentInput ────────────────────────────────────────────────────


class TestBookAssemblerAgentInputRunInfoFields:
    def test_defaults_for_run_info_fields(self) -> None:
        inp = _make_minimal_input()
        assert inp.user_prompt is None
        assert inp.runtime_seconds is None
        assert inp.token_usage is None

    def test_accepts_run_info_fields(self) -> None:
        inp = _make_minimal_input(
            user_prompt="Write a noir mystery in a flooded city.",
            runtime_seconds=42.5,
            token_usage={
                "calls": 5,
                "approx_prompt_tokens": 100,
                "approx_response_tokens": 50,
                "approx_total_tokens": 150,
            },
        )
        assert inp.user_prompt == "Write a noir mystery in a flooded city."
        assert inp.runtime_seconds == 42.5
        assert inp.token_usage is not None
        assert inp.token_usage["approx_total_tokens"] == 150

    def test_runtime_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            _make_minimal_input(runtime_seconds=-0.1)
