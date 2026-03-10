"""
Pydantic schemas for the GenreNormalizerAgent.

Defines input/output contracts, resolution tracking models,
and mapping file schemas for the process of taking user provided
genre names and mapping them to normalized genre names along
with tonal information associated with genres.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from storymesh.versioning.schemas import GENRE_CONSTRAINT_SCHEMA_VERSION


# -------- Mapping File Schemas --------
class GenreMapEntry(BaseModel):
    """Schema for an entry in the genre_map.json local mapping file."""

    genres: list[str] = Field(
        default_factory=list, 
        description = "List of genre names that map to some normalized genre."
    )

    subgenres: list[str] = Field(
        default_factory=list,
        description="List of subgenre names that map to some normalized genre."
    )

    default_tones: list[str] = Field(
        default_factory=list,
        description="List of default tones associated with the normalized genre."
    )

    alternates: list[str] = Field(
        default_factory=list,
        description="List of alternate names for the normalized genre."
    )

    @model_validator(mode="after")
    def check_at_least_one_genre_or_subgenre(self) -> GenreMapEntry:
        """
        Ensures that at least one normalized genre and/or subgenre exists in
        the mapping entry.
        """

        if not self.genres and not self.subgenres:
            raise ValueError(
                "GenreMapEntry records must map to at least one genre or subgenre."
            )

        return self
    
class ToneMapEntry(BaseModel):
    """Schema for an entry in the tone_map.json local mapping file."""

    normalized_tones: list[str] = Field(
        min_length=1,
        description="List of normalized tone names that map to some tone descriptor."
    )

    alternates: list[str] = Field(
        default_factory=list,
        description="List of alternate names for the normalized tone."
    )

# -------- Resolution Tracking Schema --------
class ResolutionMethod(StrEnum):
    """Document how a token was resolved during genre normalization."""

    STATIC_EXACT = "static_exact" # Came from an exact match in a local mapping file.
    STATIC_FUZZY = "static_fuzzy" # Came from a fuzzy match in a local mapping file.
    LLM_LIVE = "llm_live" # Came from a LLM query run in this pass.
    LLM_CACHED = "llm_cached" # Came from a cached LLM response from a previous run.

class GenreResolution(BaseModel):
    """
    Record for one genre token that is resolved in Pass 1 or Pass 3 of the
    GenreNormalizerAgent.
    """

    model_config = { "frozen": True }

    input_token: str = Field(
        min_length=1,
        description="The original token provided by the user."
    )

    canonical_genres: list[str] = Field(
        min_length=1,
        description="The normalized genre names that the input token maps to."
    )

    default_tones: list[str] = Field(
        default_factory=list,
        description="The default tones associated with the resolved genres."
    )

    method: ResolutionMethod

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="A confidence score between 0 and 1 for how well the input token maps to the canonical genres." # noqa E501
    )

class ToneResolution(BaseModel):
    """
    Record for one tone modifier that is resolved during Pass 2 or Pass 3 of the
    GenreNormalizerAgent.
    """

    model_config = { "frozen": True }

    input_token: str = Field(
        min_length=1,
        description="The original tone token provided by the user."
    )

    normalized_tones: list[str] = Field(
        min_length=1,
        description="The normalized tone names that the input token maps to."
    )

    method: ResolutionMethod

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="A confidence score between 0 and 1 for how well the input token maps to the tones." # noqa E501
    )

    is_override: bool = True

# -------- GenreNormalizerAgent Input Schema --------
class GenreNormalizerAgentInput(BaseModel):
    """Input contract for the GenreNormalizerAgent."""

    raw_genre: str = Field(
        min_length=1,
        description="The raw genre string provided by the user to the agent."
    )

    allow_llm_fallback: bool = True # Allow LLM calls if True

# -------- GenreNormalizerAgent Output Schema --------
class GenreNormalizerAgentOutput(BaseModel):
    """Output contract for the GenreNormalizerAgent.
    This output is frozen and versioned. It represents the complete result
    of genre normalization including thorough audit trails.
    """

    model_config = { "frozen": True }

    raw_input: str = Field(
        min_length=1,
        description="The original raw input string provided by the user."
    )

    normalized_genres: list[str] = Field(
        min_length=1,
        description="The list of normalized genres that the agent resolved from the raw input." # noqa E501
    )

    subgenres: list[str] = Field(
        default_factory=list,
        description="The list of normalized subgenres that the agent resolved from the raw input." # noqa E501
    )

    default_tones: list[str] = Field(
        default_factory=list,
        description="The list of default tones associated with the normalized genres."
    )

    explicit_tones: list[str] = Field(
        default_factory=list,
        description="The list of explicit tones that the agent resolved from the raw input." # noqa E501
    )

    effective_tone: str = Field(
        min_length=1,
        description="The single effective tone that results from a combination of default and explicit tones." # noqa E501
    )

    tone_conflicts: list[str] | None = None # Document any tone conflicts.

    genre_resolutions: list[GenreResolution] = Field(min_length=1)

    tone_resolutions: list[ToneResolution] = Field(
        default_factory=list
    )

    unresolved_tokens: list[str] = Field(
        default_factory=list,
        description="Any tokens from the raw input that failed normalization."
    )

    tone_profile: list[str] = Field (
        min_length = 1,
        description="Ordered list of all applicable tones, ranked by priority."
    )

    schema_version: str = GENRE_CONSTRAINT_SCHEMA_VERSION

    @model_validator(mode="after")
    def check_effective_tone_has_source(self) -> GenreNormalizerAgentOutput:
        """
        Validate that the effective tone is derived from at least one default
        or explicit tone.
        """
        
        all_tones = set(self.default_tones) | set(self.explicit_tones)
        
        if self.effective_tone not in all_tones:
            raise ValueError(
                "Effective tone must be derived from at least one default or explicit tone." # noqa E501
            )

        return self

    @model_validator(mode="after")
    def check_effective_tone_matches_profile(self) -> GenreNormalizerAgentOutput:
        if self.tone_profile[0] != self.effective_tone:
            raise ValueError(
                "effective_tone must match the first element of tone_profile."
            )
        return self

    @model_validator(mode="after")
    def check_tone_conflicts_present_when_expected(self) -> GenreNormalizerAgentOutput:
        if self.explicit_tones and self.default_tones:
            explicit_set = set(self.explicit_tones)
            default_set = set(self.default_tones)

            if not explicit_set.issubset(default_set) and self.tone_conflicts is None: 
                raise ValueError(
                    "Tone conflicts must not be None when explicit tones differ from default tones." # noqa E501
                )
        return self
