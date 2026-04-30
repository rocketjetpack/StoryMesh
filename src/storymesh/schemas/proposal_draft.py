"""Pydantic schemas for ProposalDraftAgent.

Defines the input/output contracts for Stage 4 of the StoryMesh pipeline.
ProposalDraftAgent receives a ThemePack from ThemeExtractorAgent and generates
a fully developed story proposal using a multi-sample with self-selection
architecture: N candidates are drafted at elevated temperature, then a
low-temperature critic call selects the strongest one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.theme_extractor import (
    GenreCluster,
    NarrativeSeed,
    ThematicTension,
)
from storymesh.versioning.schemas import PROPOSAL_SCHEMA_VERSION


class ProposalDraftAgentInput(BaseModel):
    """Input contract for ProposalDraftAgent (Stage 4).

    Assembled by the node wrapper from ThemeExtractorAgentOutput,
    GenreNormalizerAgentOutput, and pipeline state. The agent itself
    has no knowledge of the pipeline.
    """

    narrative_seeds: list[NarrativeSeed] = Field(
        min_length=1,
        description=(
            "Narrative seeds from ThemeExtractorAgent. Each candidate is "
            "steered toward a different seed."
        ),
    )
    tensions: list[ThematicTension] = Field(
        min_length=1,
        description=(
            "Thematic tensions with clichéd resolutions. Used for candidate "
            "evaluation in the selection step."
        ),
    )
    genre_clusters: list[GenreCluster] = Field(
        min_length=1,
        description=(
            "Genre clusters with thematic assumptions. Provides genre context "
            "for proposals."
        ),
    )
    normalized_genres: list[str] = Field(
        min_length=1,
        description="Canonical genre names from GenreNormalizerAgent.",
    )
    user_tones: list[str] = Field(
        default_factory=list,
        description="User-specified tone words carried through from earlier stages.",
    )
    narrative_context: list[str] = Field(
        default_factory=list,
        description=(
            "Narrative tokens (settings, time periods, character archetypes) "
            "from GenreNormalizerAgent."
        ),
    )
    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string.",
    )


class StoryProposal(BaseModel):
    """A fully developed story proposal generated from a narrative seed.

    Contains enough structural detail for RubricJudgeAgent to evaluate
    and SynopsisWriterAgent to expand into a full synopsis.
    """

    model_config = {"frozen": True}

    seed_id: str = Field(
        min_length=1,
        description="Which narrative seed this proposal was primarily developed from.",
    )
    title: str = Field(
        min_length=1,
        description="Working title for the story.",
    )
    protagonist: str = Field(
        min_length=10,
        description=(
            "The main character: name, defining trait, internal conflict, "
            "and what they want vs. what they need."
        ),
    )
    setting: str = Field(
        min_length=10,
        description=(
            "Where and when the story takes place. Must reflect the "
            "narrative context tokens and genre traditions."
        ),
    )
    plot_arc: str = Field(
        min_length=50,
        description=(
            "A 3-act plot summary (setup, confrontation, resolution) "
            "with specific story beats. 150-300 words."
        ),
    )
    thematic_thesis: str = Field(
        min_length=10,
        description=(
            "The central thematic argument the story makes — what it says "
            "about the tensions it explores. Not a moral or lesson, but the "
            "story's philosophical stance."
        ),
    )
    key_scenes: list[str] = Field(
        min_length=2,
        description=(
            "3-5 pivotal scenes described in 1-2 sentences each. These are "
            "the moments where the thematic tensions become visible."
        ),
    )
    tensions_addressed: list[str] = Field(
        min_length=1,
        description="Which tension_ids from the ThemePack this proposal explores.",
    )
    tone: list[str] = Field(
        min_length=1,
        description=(
            "The tonal qualities of this proposal "
            "(e.g., 'dark', 'cerebral', 'hopeful')."
        ),
    )
    genre_blend: list[str] = Field(
        min_length=1,
        description="Which genres from the input this proposal blends.",
    )
    image_prompt: str = Field(
        min_length=30,
        description=(
            "An image generation prompt for the book cover (consumed by CoverArtAgent "
            "via gpt-image-1). Describes the dominant visual (a scene, object, or "
            "atmosphere from the story world — not a named character portrait), art "
            "style, mood, color palette, and period or setting details. Contains no "
            "character names, text, or readable symbols."
        ),
    )


class SelectionRationale(BaseModel):
    """The critic's reasoning for selecting the winning proposal.

    Persisted in the debug artifacts so the selection decision is
    auditable and inspectable.
    """

    model_config = {"frozen": True}

    selected_index: int = Field(
        ge=0,
        description="0-based index of the winning candidate in the candidates list.",
    )
    rationale: str = Field(
        min_length=10,
        description="Why the critic selected this candidate over the others.",
    )
    cliche_violations: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of candidate index (as string) to list of clichéd resolutions "
            "the critic identified in that candidate. Empty list means no violations."
        ),
    )
    runner_up_index: int | None = Field(
        default=None,
        description="Index of the second-best candidate, if applicable.",
    )


class ProposalDraftAgentOutput(BaseModel):
    """Output contract for ProposalDraftAgent (Stage 4).

    Contains the selected proposal as the primary output, plus all
    candidates and selection rationale for artifact inspection and
    debugging.
    """

    model_config = {"frozen": True}

    proposal: StoryProposal = Field(
        description="The selected (winning) story proposal.",
    )
    all_candidates: list[StoryProposal] = Field(
        min_length=1,
        description=(
            "All valid candidate proposals, including the winner. "
            "Persisted for artifact inspection and debugging."
        ),
    )
    selection_rationale: SelectionRationale = Field(
        description="The critic's reasoning for the selection.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Generation metadata: num_candidates_requested, num_valid_candidates, "
            "num_parse_failures, draft_temperature, selection_temperature, "
            "seed_assignments, total_llm_calls."
        ),
    )
    schema_version: str = PROPOSAL_SCHEMA_VERSION
