"""Tests for BookAssemblerAgent Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.book_assembler import (
    BookAssemblerAgentOutput,
)
from storymesh.versioning.schemas import BOOK_ASSEMBLER_SCHEMA_VERSION

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
