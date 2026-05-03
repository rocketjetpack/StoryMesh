"""Unit tests for BookAssemblerAgent (Stage 8)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from storymesh.agents.book_assembler.agent import (
    BookAssemblerAgent,
    BookRawOutput,
    _prose_to_html,
    _tag_line,
)
from storymesh.schemas.book_assembler import BookAssemblerAgentInput
from storymesh.schemas.cover_art import CoverArtAgentOutput
from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.story_writer import SCENE_BREAK, SceneOutline, StoryWriterAgentOutput
from storymesh.versioning.schemas import STORY_WRITER_SCHEMA_VERSION

# ── Factories ──────────────────────────────────────────────────────────────────


def _make_scene_outline(n: int = 1) -> SceneOutline:
    return SceneOutline(
        scene_id=f"scene_{n:02d}",
        title=f"Scene {n} Title",
        summary=f"Something important happens in scene {n}.",
        narrative_pressure=f"Scene {n} inhabits the central tension without resolving it.",
        observational_anchor=f"A concrete detail grounding scene {n}.",
        opens_with="The light changed in a way that made her feel exposed.",
    )


def _make_story_writer_output(scene_count: int = 3) -> StoryWriterAgentOutput:
    scenes = [_make_scene_outline(i + 1) for i in range(scene_count)]
    # Build draft with SCENE_BREAK delimiters matching the scene outlines.
    prose_blocks = [
        f"Scene {i + 1} opens with something vivid. The protagonist moved forward. "
        f"The tension built slowly. Nothing resolved. The moment passed unfulfilled."
        for i in range(scene_count)
    ]
    full_draft = SCENE_BREAK.join(prose_blocks)
    # Pad to meet min_length=500 if needed.
    if len(full_draft) < 500:
        full_draft += " " + ("x " * 300)
    return StoryWriterAgentOutput(
        back_cover_summary=(
            "In a world where truth is buried beneath rubble, one detective refuses "
            "to stop asking questions. A story about justice, memory, and the cost "
            "of caring when no one else does. The past never stays buried."
        ),
        scene_list=scenes,
        full_draft=full_draft,
        word_count=max(100, len(full_draft.split())),
        schema_version=STORY_WRITER_SCHEMA_VERSION,
    )


def _make_proposal() -> StoryProposal:
    return StoryProposal.model_construct(
        seed_id="seed_01",
        title="The Last Signal",
        protagonist="A burned-out detective with nothing left to lose.",
        setting="A flooded post-collapse city where old cases never close.",
        plot_arc=(
            "Act one: the detective takes a missing-person case no one else will touch. "
            "Act two: every clue leads deeper into a conspiracy the city wants buried. "
            "Act three: the truth surfaces but justice remains impossible."
        ),
        thematic_thesis="Justice and order are incompatible in a system that has collapsed.",
        key_scenes=["The detective finds the first clue.", "The confrontation in the archive."],
        tensions_addressed=["t_01"],
        tone=["dark", "cerebral"],
        genre_blend=["Post-Apocalyptic", "Mystery"],
        image_prompt="A rain-slicked flooded street at dusk, lone figure silhouetted. Noir.",
    )


def _make_input(
    *,
    cover_art_output: CoverArtAgentOutput | None = None,
) -> BookAssemblerAgentInput:
    return BookAssemblerAgentInput(
        story_writer_output=_make_story_writer_output(),
        proposal=_make_proposal(),
        cover_art_output=cover_art_output,
        run_id="abc123",
    )


def _make_agent(**kwargs: Any) -> BookAssemblerAgent:  # noqa: ANN401
    return BookAssemblerAgent(**kwargs)


# ── Helpers ────────────────────────────────────────────────────────────────────


class TestProseToHtml:
    def test_single_paragraph(self) -> None:
        result = _prose_to_html("Hello world.")
        assert result == "<p>Hello world.</p>"

    def test_multiple_paragraphs(self) -> None:
        result = _prose_to_html("First.\n\nSecond.")
        assert "<p>First.</p>" in result
        assert "<p>Second.</p>" in result

    def test_single_newlines_joined(self) -> None:
        result = _prose_to_html("Line one.\nLine two.")
        assert result == "<p>Line one. Line two.</p>"

    def test_html_escaping(self) -> None:
        result = _prose_to_html("A & B < C > D")
        assert "&amp;" in result
        assert "&lt;" in result
        assert "&gt;" in result

    def test_empty_string_returns_empty(self) -> None:
        result = _prose_to_html("")
        assert result == ""

    def test_blank_line_blocks_skipped(self) -> None:
        result = _prose_to_html("First.\n\n\n\nSecond.")
        assert result.count("<p>") == 2


class TestTagLine:
    def test_combines_genres_and_tones(self) -> None:
        result = _tag_line(["Mystery", "noir"], ["Dark"])
        assert "Mystery" in result
        assert "Dark" in result

    def test_empty_lists_produce_empty_string(self) -> None:
        result = _tag_line([], [])
        assert result == ""

    def test_only_genre(self) -> None:
        result = _tag_line(["Mystery"], [])
        assert result == "Mystery"

    def test_separator_present(self) -> None:
        result = _tag_line(["A"], ["B"])
        assert "·" in result


# ── HTML building ──────────────────────────────────────────────────────────────


class TestBuildHtml:
    def test_contains_title(self) -> None:
        agent = _make_agent()
        html = agent._build_html(
            title="The Last Signal",
            genre_tags="Mystery · Dark",
            back_cover_summary="A great synopsis about nothing resolved.",
            scene_pairs=[("Scene One", "Some prose here about things.")],
            cover_b64=None,
        )
        assert "The Last Signal" in html

    def test_typographic_cover_when_no_image(self) -> None:
        agent = _make_agent()
        html = agent._build_html(
            title="Title",
            genre_tags="",
            back_cover_summary="Synopsis text.",
            scene_pairs=[("S", "Prose.")],
            cover_b64=None,
        )
        assert "cover-text-page" in html
        assert '<div class="cover-image-page">' not in html

    def test_image_cover_when_b64_provided(self) -> None:
        agent = _make_agent()
        html = agent._build_html(
            title="Title",
            genre_tags="",
            back_cover_summary="Synopsis text.",
            scene_pairs=[("S", "Prose.")],
            cover_b64="ZmFrZWRhdGE=",  # base64 of "fakedata"
        )
        assert "cover-image-page" in html
        assert "data:image/png;base64,ZmFrZWRhdGE=" in html

    def test_scene_titles_appear_in_output(self) -> None:
        agent = _make_agent()
        html = agent._build_html(
            title="T",
            genre_tags="",
            back_cover_summary="S",
            scene_pairs=[("The First Scene", "Prose."), ("The Second Scene", "More prose.")],
            cover_b64=None,
        )
        assert "The First Scene" in html
        assert "The Second Scene" in html

    def test_html5_doctype(self) -> None:
        agent = _make_agent()
        html = agent._build_html(
            title="T", genre_tags="", back_cover_summary="S",
            scene_pairs=[("S", "P")], cover_b64=None,
        )
        assert html.startswith("<!DOCTYPE html>")

    def test_special_chars_escaped_in_title(self) -> None:
        agent = _make_agent()
        html = agent._build_html(
            title="A & B < C",
            genre_tags="",
            back_cover_summary="Synopsis.",
            scene_pairs=[("S", "Prose.")],
            cover_b64=None,
        )
        assert "&amp;" in html
        assert "&lt;" in html


# ── PDF generation ─────────────────────────────────────────────────────────────


class TestBuildPdf:
    def test_returns_bytes_when_weasyprint_available(self) -> None:
        fake_pdf = b"%PDF-1.4 fake content"
        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML.return_value.write_pdf.return_value = fake_pdf

        agent = _make_agent()
        with patch.dict(sys.modules, {"weasyprint": mock_weasyprint}):
            result = agent._build_pdf(
                title="T",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_b64=None,
            )
        assert result == fake_pdf

    def test_returns_none_when_weasyprint_missing(self) -> None:
        agent = _make_agent()
        with patch.dict(sys.modules, {"weasyprint": None}):  # type: ignore[dict-item]
            result = agent._build_pdf(
                title="T",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_b64=None,
            )
        assert result is None

    def test_html_string_passed_to_weasyprint(self) -> None:
        mock_weasyprint = MagicMock()
        mock_weasyprint.HTML.return_value.write_pdf.return_value = b"pdf"

        agent = _make_agent()
        with patch.dict(sys.modules, {"weasyprint": mock_weasyprint}):
            agent._build_pdf(
                title="My Title",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_b64=None,
            )

        call_kwargs = mock_weasyprint.HTML.call_args
        html_arg = call_kwargs[1].get("string") or call_kwargs[0][0]
        assert "My Title" in html_arg


# ── EPUB generation ────────────────────────────────────────────────────────────


class TestBuildEpub:
    def _make_fake_epub_module(self) -> MagicMock:
        """Return a mock ebooklib.epub module that writes real bytes to disk."""
        mock_epub = MagicMock()

        def _fake_write_epub(path: str, book: Any, options: Any = None) -> None:  # noqa: ANN401
            Path(path).write_bytes(b"PK\x03\x04fake-epub-content")

        mock_epub.write_epub.side_effect = _fake_write_epub
        return mock_epub

    def test_returns_bytes_when_ebooklib_available(self) -> None:
        mock_epub = self._make_fake_epub_module()
        mock_ebooklib = MagicMock()
        mock_ebooklib.epub = mock_epub

        agent = _make_agent()
        with patch.dict(sys.modules, {"ebooklib": mock_ebooklib}):
            result = agent._build_epub(
                title="T",
                run_id="run123",
                genre_tags="",
                back_cover_summary="Synopsis.",
                scene_pairs=[("Chapter One", "Some prose here.")],
                cover_bytes=None,
            )
        assert result == b"PK\x03\x04fake-epub-content"

    def test_returns_none_when_ebooklib_missing(self) -> None:
        agent = _make_agent()
        with patch.dict(sys.modules, {"ebooklib": None}):  # type: ignore[dict-item]
            result = agent._build_epub(
                title="T",
                run_id="run123",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_bytes=None,
            )
        assert result is None

    def test_temp_file_cleaned_up_on_success(self) -> None:
        mock_epub = self._make_fake_epub_module()
        mock_ebooklib = MagicMock()
        mock_ebooklib.epub = mock_epub

        captured_paths: list[str] = []
        original_side_effect = mock_epub.write_epub.side_effect

        def _capture_and_write(path: str, book: Any, options: Any = None) -> None:  # noqa: ANN401
            captured_paths.append(path)
            original_side_effect(path, book, options)

        mock_epub.write_epub.side_effect = _capture_and_write

        agent = _make_agent()
        with patch.dict(sys.modules, {"ebooklib": mock_ebooklib}):
            agent._build_epub(
                title="T",
                run_id="r",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_bytes=None,
            )

        assert len(captured_paths) == 1
        assert not Path(captured_paths[0]).exists()

    def test_cover_image_set_when_provided(self) -> None:
        mock_epub = self._make_fake_epub_module()
        mock_ebooklib = MagicMock()
        mock_ebooklib.epub = mock_epub

        agent = _make_agent()
        with patch.dict(sys.modules, {"ebooklib": mock_ebooklib}):
            agent._build_epub(
                title="T",
                run_id="r",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_bytes=b"\x89PNG fake image",
            )

        mock_epub.EpubBook.return_value.set_cover.assert_called_once()

    def test_cover_image_not_set_when_none(self) -> None:
        mock_epub = self._make_fake_epub_module()
        mock_ebooklib = MagicMock()
        mock_ebooklib.epub = mock_epub

        agent = _make_agent()
        with patch.dict(sys.modules, {"ebooklib": mock_ebooklib}):
            agent._build_epub(
                title="T",
                run_id="r",
                genre_tags="",
                back_cover_summary="S",
                scene_pairs=[("S", "P")],
                cover_bytes=None,
            )

        mock_epub.EpubBook.return_value.set_cover.assert_not_called()


# ── Full agent.run() ───────────────────────────────────────────────────────────


class TestBookAssemblerAgentRun:
    def _mock_modules(self) -> tuple[MagicMock, MagicMock]:
        """Return (mock_weasyprint, mock_ebooklib) configured for run tests."""
        mock_wp = MagicMock()
        mock_wp.HTML.return_value.write_pdf.return_value = b"%PDF fake"

        mock_epub_mod = MagicMock()

        def _fake_write_epub(path: str, book: Any, options: Any = None) -> None:  # noqa: ANN401
            Path(path).write_bytes(b"PK fake epub")

        mock_epub_mod.write_epub.side_effect = _fake_write_epub

        mock_ebl = MagicMock()
        mock_ebl.epub = mock_epub_mod

        return mock_wp, mock_ebl

    def test_returns_book_raw_output(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            agent = _make_agent()
            result = agent.run(_make_input())
        assert isinstance(result, BookRawOutput)

    def test_pdf_bytes_populated_when_weasyprint_available(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(_make_input())
        assert result.pdf_bytes == b"%PDF fake"

    def test_epub_bytes_populated_when_ebooklib_available(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(_make_input())
        assert result.epub_bytes == b"PK fake epub"

    def test_pdf_bytes_none_when_weasyprint_missing(self) -> None:
        _, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": None, "ebooklib": mock_ebl}):  # type: ignore[dict-item]
            result = _make_agent().run(_make_input())
        assert result.pdf_bytes is None

    def test_epub_bytes_none_when_ebooklib_missing(self) -> None:
        mock_wp, _ = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": None}):  # type: ignore[dict-item]
            result = _make_agent().run(_make_input())
        assert result.epub_bytes is None

    def test_title_from_proposal(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(_make_input())
        assert result.title == "The Last Signal"

    def test_word_count_from_story_output(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        story = _make_story_writer_output()
        inp = BookAssemblerAgentInput(
            story_writer_output=story,
            proposal=_make_proposal(),
            run_id="r",
        )
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(inp)
        assert result.word_count == story.word_count

    def test_debug_contains_scene_count(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(_make_input())
        assert "scene_count" in result.debug

    def test_pdf_only_format(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent(output_formats=["pdf"]).run(_make_input())
        assert result.pdf_bytes is not None
        assert result.epub_bytes is None

    def test_epub_only_format(self) -> None:
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent(output_formats=["epub"]).run(_make_input())
        assert result.epub_bytes is not None
        assert result.pdf_bytes is None

    def test_cover_image_loaded_from_file(self, tmp_path: Path) -> None:
        cover_file = tmp_path / "cover.png"
        cover_file.write_bytes(b"\x89PNG fake image data")

        mock_wp, mock_ebl = self._mock_modules()
        cover_output = CoverArtAgentOutput(
            image_path=str(cover_file),
            image_prompt="A scenic view.",
            model="gpt-image-1",
            image_size="1024x1536",
            image_quality="auto",
        )
        inp = BookAssemblerAgentInput(
            story_writer_output=_make_story_writer_output(),
            proposal=_make_proposal(),
            cover_art_output=cover_output,
            run_id="r",
        )
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(inp)

        assert result.debug.get("has_cover_image") is True

    def test_missing_cover_file_handled_gracefully(self, tmp_path: Path) -> None:
        cover_output = CoverArtAgentOutput(
            image_path=str(tmp_path / "nonexistent.png"),
            image_prompt="A scenic view.",
            model="gpt-image-1",
            image_size="1024x1536",
            image_quality="auto",
        )
        inp = BookAssemblerAgentInput(
            story_writer_output=_make_story_writer_output(),
            proposal=_make_proposal(),
            cover_art_output=cover_output,
            run_id="r",
        )
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(inp)

        assert result.debug.get("has_cover_image") is False

    def test_scene_split_on_scene_break(self) -> None:
        """Verify that full_draft is correctly split into per-scene texts."""
        story = _make_story_writer_output(scene_count=3)
        inp = BookAssemblerAgentInput(
            story_writer_output=story,
            proposal=_make_proposal(),
            run_id="r",
        )
        mock_wp, mock_ebl = self._mock_modules()
        with patch.dict(sys.modules, {"weasyprint": mock_wp, "ebooklib": mock_ebl}):
            result = _make_agent().run(inp)

        assert result.debug.get("scene_count") == 3


# ── Constructor ────────────────────────────────────────────────────────────────


class TestBookAssemblerAgentConstructor:
    def test_default_output_formats(self) -> None:
        agent = BookAssemblerAgent()
        assert set(agent._output_formats) == {"pdf", "epub"}

    def test_custom_output_formats(self) -> None:
        agent = BookAssemblerAgent(output_formats=["pdf"])
        assert agent._output_formats == ["pdf"]

    def test_empty_formats_defaults_to_both(self) -> None:
        agent = BookAssemblerAgent(output_formats=None)
        assert set(agent._output_formats) == {"pdf", "epub"}
