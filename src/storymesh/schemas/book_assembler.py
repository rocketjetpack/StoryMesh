"""Pydantic schemas for BookAssemblerAgent (Stage 8).

Defines the input/output contracts for the final stage of the StoryMesh
pipeline. BookAssemblerAgent receives story_writer_output and cover_art_output
and produces formatted PDF and EPUB deliverables from the completed prose.

No LLM is required — this stage is purely deterministic rendering.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.cover_art import CoverArtAgentOutput
from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.story_writer import StoryWriterAgentOutput
from storymesh.versioning.schemas import BOOK_ASSEMBLER_SCHEMA_VERSION


class BookAssemblerAgentInput(BaseModel):
    """Input contract for BookAssemblerAgent (Stage 8).

    Assembled by the node wrapper from pipeline state. The agent itself
    has no knowledge of the pipeline or artifact store.
    """

    model_config = {"frozen": True}

    story_writer_output: StoryWriterAgentOutput = Field(
        description="Completed story output from StoryWriterAgent.",
    )
    proposal: StoryProposal = Field(
        description="The selected story proposal carrying title, genre_blend, and tone.",
    )
    cover_art_output: CoverArtAgentOutput | None = Field(
        default=None,
        description=(
            "Cover art output from CoverArtAgent. When present, the cover image "
            "is embedded as the PDF and EPUB cover page. When absent, a typographic "
            "cover is generated from the title."
        ),
    )
    run_id: str = Field(
        description="Unique run identifier. Used as the EPUB identifier.",
    )


class BookAssemblerAgentOutput(BaseModel):
    """Output contract for BookAssemblerAgent (Stage 8).

    pdf_path and epub_path are absolute paths to the generated files within
    the run directory. Either may be an empty string if the corresponding
    format was not generated (e.g. library not installed, format not requested).
    """

    model_config = {"frozen": True}

    pdf_path: str = Field(
        description=(
            "Absolute path to output.pdf in the run directory. "
            "Empty string if PDF was not generated."
        ),
    )
    epub_path: str = Field(
        description=(
            "Absolute path to output.epub in the run directory. "
            "Empty string if EPUB was not generated."
        ),
    )
    title: str = Field(
        description="Story title, mirrored from the proposal for convenient access.",
    )
    word_count: int = Field(
        ge=0,
        description="Word count of the full prose draft, carried from StoryWriterAgentOutput.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Assembly metadata: scene_count, has_cover_image, "
            "pdf_generated, epub_generated."
        ),
    )
    schema_version: str = BOOK_ASSEMBLER_SCHEMA_VERSION
