"""Pydantic schemas for VoiceProfileSelectorAgent.

Defines the input/output contracts for Stage 0.5 of the StoryMesh pipeline.
VoiceProfileSelectorAgent receives normalized genres and tone keywords and
selects the best-fit VoiceProfile for the run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.voice_profile import VoiceProfile
from storymesh.versioning.schemas import VOICE_PROFILE_SELECTOR_SCHEMA_VERSION


class VoiceProfileSelectorAgentInput(BaseModel):
    """Input contract for VoiceProfileSelectorAgent (Stage 0.5).

    Assembled by the node wrapper from the pipeline state after genre normalization.
    """

    model_config = {"frozen": True}

    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string.",
    )
    normalized_genres: list[str] = Field(
        min_length=1,
        description="Canonical genre names from GenreNormalizerAgent.",
    )
    user_tones: list[str] = Field(
        default_factory=list,
        description="User-specified tone words from GenreNormalizerAgent.",
    )
    available_profile_ids: list[str] = Field(
        min_length=1,
        description="Profile IDs available for selection (the full built-in set).",
    )


class VoiceProfileSelectorAgentOutput(BaseModel):
    """Output contract for VoiceProfileSelectorAgent (Stage 0.5)."""

    model_config = {"frozen": True}

    selected_profile_id: str = Field(
        min_length=1,
        description="The snake_case ID of the selected voice profile.",
    )
    rationale: str = Field(
        min_length=1,
        description="1–2 sentence explanation of why this profile was selected.",
    )
    voice_profile: VoiceProfile = Field(
        description="The fully loaded VoiceProfile corresponding to selected_profile_id.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Selection metadata: temperature, llm_used, defaulted_to_fallback.",
    )
    schema_version: str = VOICE_PROFILE_SELECTOR_SCHEMA_VERSION
