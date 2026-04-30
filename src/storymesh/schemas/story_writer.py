"""Pydantic schemas for StoryWriterAgent.

Defines the input/output contracts for Stage 6 of the StoryMesh pipeline.
StoryWriterAgent receives a StoryProposal and rubric feedback and produces
a back-cover summary, a structured scene list, and a full prose draft.

The back-cover summary populates the public-API ``GenerationResult.final_synopsis``
field. The full draft and scene list are persisted as run artifacts and
consumed by the book assembly layer to produce PDF and EPUB outputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput
from storymesh.schemas.theme_extractor import ThematicTension
from storymesh.versioning.schemas import STORY_WRITER_SCHEMA_VERSION

# Delimiter used between scenes in ``StoryWriterAgentOutput.full_draft``.
# The book assembler splits on this marker to place decorative section breaks
# in the PDF and EPUB outputs.
SCENE_BREAK = "\n\n---\n\n"


class StoryWriterAgentInput(BaseModel):
    """Input contract for StoryWriterAgent (Stage 6).

    Assembled by the node wrapper from the pipeline state. The agent itself
    has no knowledge of the pipeline.
    """

    model_config = {"frozen": True}

    proposal: StoryProposal = Field(
        description="The selected story proposal from ProposalDraftAgent.",
    )
    tensions: list[ThematicTension] = Field(
        min_length=1,
        description=(
            "Thematic tensions with clichéd resolutions. The writer uses these "
            "to ensure the prose inhabits the tensions rather than resolving them."
        ),
    )
    rubric_feedback: RubricJudgeAgentOutput | None = Field(
        default=None,
        description=(
            "Rubric evaluation from Stage 5. When present, the writer uses "
            "dimension-level feedback as direct craft notes for the draft."
        ),
    )
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
        description="User-specified tone words carried through from earlier stages.",
    )


class SceneOutline(BaseModel):
    """Structured outline for one scene in the story.

    Produced in the first LLM pass of StoryWriterAgent before prose generation.
    Each outline becomes one section of the final draft, separated by
    ``SCENE_BREAK`` delimiters.
    """

    model_config = {"frozen": True}

    scene_id: str = Field(
        min_length=1,
        description="Sequential identifier, e.g. 'scene_01', 'scene_02'.",
    )
    title: str = Field(
        min_length=1,
        description="Scene heading as it will appear in the formatted book output.",
    )
    summary: str = Field(
        min_length=10,
        description="2–3 sentence description of what happens in this scene.",
    )
    thematic_function: str = Field(
        min_length=10,
        description=(
            "Which thematic tension this scene inhabits and how — without "
            "resolving it. Guides the prose pass toward sustained tension "
            "rather than premature resolution."
        ),
    )
    opens_with: str = Field(
        min_length=10,
        description=(
            "The first sentence or image that anchors this scene's tone. "
            "Passed verbatim to the prose generation pass as the scene's "
            "entry point to prevent generic AI opening lines."
        ),
    )


class StoryWriterAgentOutput(BaseModel):
    """Output contract for StoryWriterAgent (Stage 6).

    Contains the back-cover summary, structured scene list, and full prose draft.
    The book assembly layer uses all three to produce the final PDF and EPUB.
    """

    model_config = {"frozen": True}

    back_cover_summary: str = Field(
        min_length=50,
        description=(
            "~300-word back-cover marketing copy. Present-tense, hook-driven. "
            "Does not reveal the ending. Populates GenerationResult.final_synopsis."
        ),
    )
    scene_list: list[SceneOutline] = Field(
        min_length=3,
        description=(
            "6–10 structured scene outlines. The auditable intermediate between "
            "the proposal's key_scenes and the final prose draft."
        ),
    )
    full_draft: str = Field(
        min_length=500,
        description=(
            f"Complete prose draft. Scenes are separated by '{SCENE_BREAK.strip()}' "
            "delimiters, which the book assembler uses to place decorative "
            "section breaks in the PDF and EPUB outputs."
        ),
    )
    word_count: int = Field(
        ge=100,
        description="Approximate word count of full_draft.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Generation metadata: outline_temperature, draft_temperature, "
            "total_llm_calls, scene_count."
        ),
    )
    schema_version: str = STORY_WRITER_SCHEMA_VERSION
