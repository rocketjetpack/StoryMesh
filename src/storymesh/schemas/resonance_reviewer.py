"""Pydantic schemas for ResonanceReviewerAgent.

Defines the input/output contracts for Stage 6b of the StoryMesh pipeline.
ResonanceReviewerAgent reads a completed prose draft, identifies 0-3
near-miss moments where the story implies depth but does not engage with it,
and produces a revised draft with targeted expansions.

A near-miss moment is a point where the narrative brushes up against deeper
meaning — emotional, relational, philosophical — but retreats before fully
engaging. The reviewer distinguishes between *restraint* (silence that says
the thing) and *avoidance* (silence that replaces the thing), and only
expands moments classified as avoidance.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from storymesh.versioning.schemas import RESONANCE_REVIEWER_SCHEMA_VERSION


class NearMissMoment(BaseModel):
    """A moment where the story implies depth but does not engage with it."""

    model_config = {"frozen": True}

    passage_ref: str = Field(
        min_length=10,
        description=(
            "Direct quote or close paraphrase (2-3 sentences) from the draft "
            "that constitutes the near-miss moment. Must be specific enough "
            "to locate in the text."
        ),
    )
    what_it_implies: str = Field(
        min_length=20,
        description=(
            "What the moment implies in human terms — not literary analysis, "
            "but what a reader would feel pulling at them. Written as felt "
            "experience, not academic observation."
        ),
    )
    what_the_reader_wanted: str = Field(
        min_length=20,
        description=(
            "What the reader wanted to happen next — not plot, but emotional "
            "or relational follow-through. Described as desire, not critique."
        ),
    )
    what_the_story_did: str = Field(
        min_length=10,
        description=(
            "How the story retreated: deflected to action, cut to a new scene, "
            "moved to procedure, ended the paragraph."
        ),
    )
    expansion_directive: str = Field(
        min_length=20,
        description=(
            "Specific instruction for the revision pass. Framed as 'stay, "
            "don't add': extend the moment, let a silence land, let the "
            "character's reaction show before the story moves on. Must not "
            "request new scenes, characters, or plot events."
        ),
    )
    classification: Literal["avoidance", "restraint"] = Field(
        description=(
            "Either 'avoidance' or 'restraint'. Avoidance: the silence "
            "replaces the thing. Restraint: the silence says the thing. "
            "Only 'avoidance' moments should be expanded."
        ),
    )


class ResonanceReviewerAgentInput(BaseModel):
    """Input contract for ResonanceReviewerAgent (Stage 6b).

    Assembled by the node wrapper from the pipeline state. The reviewer
    receives only the draft and minimal structural context — no rubric
    feedback or upstream evaluation — to ensure a fresh, unbiased read.
    """

    model_config = {"frozen": True}

    full_draft: str = Field(
        min_length=500,
        description="Complete prose draft from StoryWriterAgent.",
    )
    proposal_title: str = Field(
        min_length=1,
        description="Title of the story, for context.",
    )
    thematic_thesis: str = Field(
        min_length=1,
        description=(
            "The story's central pressure/tension — what it circles around. "
            "Helps the reviewer distinguish thematically relevant near-misses "
            "from incidental ones."
        ),
    )
    scene_list_summary: str = Field(
        description=(
            "Brief scene-by-scene summary for structural context. The reviewer "
            "needs to understand the story's shape to judge whether restraint "
            "is earned at a given point."
        ),
    )
    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string (for summary re-run).",
    )


class ResonanceReviewerAgentOutput(BaseModel):
    """Output contract for ResonanceReviewerAgent (Stage 6b).

    Contains the identified near-miss moments, the revised draft with
    targeted expansions, and optionally a re-generated back-cover summary.
    """

    model_config = {"frozen": True}

    near_miss_moments: list[NearMissMoment] = Field(
        max_length=3,
        description=(
            "0-3 identified near-miss moments classified as 'avoidance', "
            "ordered by significance. Restraint moments are identified "
            "during review but filtered out before output."
        ),
    )
    revised_draft: str = Field(
        min_length=500,
        description=(
            "The full prose draft with targeted expansions applied to the "
            "identified avoidance moments. Untouched passages remain "
            "exactly as they were in the original."
        ),
    )
    revised_summary: str | None = Field(
        default=None,
        description=(
            "Re-generated back-cover summary reflecting the revised draft. "
            "None when no moments were expanded (draft unchanged)."
        ),
    )
    revision_word_delta: int = Field(
        description="Word count change: revised minus original.",
    )
    moments_found: int = Field(
        ge=0,
        description=(
            "Total near-miss moments identified in the review pass "
            "(both avoidance and restraint, before filtering)."
        ),
    )
    moments_expanded: int = Field(
        ge=0,
        le=3,
        description="Number of avoidance moments actually expanded in the revised draft.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Review metadata: temperatures, token counts, provider info.",
    )
    schema_version: str = RESONANCE_REVIEWER_SCHEMA_VERSION
