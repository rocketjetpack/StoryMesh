"""Pydantic schemas for CoverArtAgent (Stage 7)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.versioning.schemas import COVER_ART_SCHEMA_VERSION


class CoverArtAgentInput(BaseModel):
    """Input contract for CoverArtAgent (Stage 7).

    Assembled by the node wrapper from ProposalDraftAgentOutput.
    The agent has no knowledge of the pipeline.
    """

    image_prompt: str = Field(
        min_length=30,
        description="Image generation prompt from StoryProposal.",
    )
    title: str = Field(
        min_length=1,
        description="Story title — incorporated into the assembled DALL-E prompt.",
    )


class CoverArtAgentOutput(BaseModel):
    """Output contract for CoverArtAgent (Stage 7).

    image_path points to the PNG saved in the run artifact directory.
    The JSON artifact (cover_art_output.json) holds everything except
    the raw image bytes, which live only in the PNG file.
    """

    model_config = {"frozen": True}

    image_path: str = Field(
        description=(
            "Absolute path to the generated cover PNG in the run directory. "
            "Empty string when running without an artifact store (e.g. tests)."
        ),
    )
    image_prompt: str = Field(
        description="The prompt submitted to the image generation API.",
    )
    revised_prompt: str | None = Field(
        default=None,
        description=(
            "Provider-rewritten prompt, if returned by the model. "
            "None when the model does not return a revised prompt."
        ),
    )
    model: str = Field(description="Image model identifier (e.g. 'gpt-image-2').")
    image_size: str = Field(description="Image dimensions (e.g. '1024x1792').")
    image_quality: str = Field(description="Quality setting ('auto', 'low', 'medium', or 'high').")
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Generation metadata: latency_ms, title, provider.",
    )
    schema_version: str = COVER_ART_SCHEMA_VERSION
