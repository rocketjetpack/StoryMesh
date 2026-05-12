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

from storymesh.schemas.voice_profile import VoiceProfile
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
            "request new scenes, characters, or plot events. Budget: roughly "
            "100-250 words of expansion (raised from 50-150 in schema 1.0)."
        ),
    )
    classification: Literal["avoidance", "restraint"] = Field(
        description=(
            "Either 'avoidance' or 'restraint'. Avoidance: the silence "
            "replaces the thing. Restraint: the silence says the thing. "
            "Only 'avoidance' moments should be expanded."
        ),
    )


class ToneDriftFinding(BaseModel):
    """A passage where the prose register drifts from the user's requested tones.

    A "silly" prompt that comes back as restrained-literary is the canonical
    failure mode this lens watches for. The lens only emits findings when the
    drift is *unearned* — a deliberate quiet moment inside a kinetic story
    is not flagged.
    """

    model_config = {"frozen": True}

    passage_ref: str = Field(
        min_length=10,
        description=(
            "Direct quote or close paraphrase (2-3 sentences) where the prose "
            "register diverges from the requested tones."
        ),
    )
    requested_tones: list[str] = Field(
        min_length=1,
        description=(
            "Echo of the user's requested tones the prose is failing to honour "
            "(e.g. ['silly', 'high energy']). Quoted so the revision pass has "
            "the contract in front of it."
        ),
    )
    observed_register: str = Field(
        min_length=10,
        description=(
            "Plain description of how the prose actually reads at this passage "
            "(e.g. 'contemplative literary restraint with hushed interiority')."
        ),
    )
    why_unearned: str = Field(
        min_length=20,
        description=(
            "Why this divergence is not earned by the story's needs at this "
            "moment. Distinguishes drift from intentional tonal modulation."
        ),
    )
    rewrite_directive: str = Field(
        min_length=20,
        description=(
            "Specific instruction for the revision pass — what the rewritten "
            "passage should *do* tonally. Framed as 'rewrite to match', not "
            "'add tone'."
        ),
    )


class EndingVerdictFinding(BaseModel):
    """A finding that the story's ending collapses unresolved tension into a verdict.

    Singular per draft (a story has one ending). The lens reads only the final
    200-400 words and asks whether they preserve the unresolved pressure named
    in the thematic_thesis or settle it into a moral, lesson, or summary.
    """

    model_config = {"frozen": True}

    final_passage: str = Field(
        min_length=10,
        description=(
            "Direct quote of the closing paragraph(s) where the verdict appears."
        ),
    )
    verdict_named: str = Field(
        min_length=10,
        description=(
            "Plain description of the verdict the ending delivers — the moral, "
            "lesson, or settled answer the prose collapses the story into."
        ),
    )
    tension_lost: str = Field(
        min_length=20,
        description=(
            "Which unresolved tension from the thematic_thesis the verdict "
            "smoothes over. Describes what the story should have kept open."
        ),
    )
    cut_directive: str = Field(
        min_length=20,
        description=(
            "Specific instruction for the revision pass. Usually 'cut N lines' "
            "or 'end one beat earlier' — net-negative or net-neutral word "
            "delta. Must NOT request new ending material."
        ),
    )


class SlopMarker(BaseModel):
    """An AI-tell phrase or passage that exhibits canonical LLM-prose hallmarks.

    High-confidence only: emitted only when a specific phrase can be quoted
    verbatim as evidence. The lens is opinionated and prone to over-triggering,
    so the prompt requires explicit evidence and caps the count.
    """

    model_config = {"frozen": True}

    quoted_phrase: str = Field(
        min_length=4,
        description=(
            "The exact phrase from the draft that exhibits the AI-tell. Verbatim "
            "quote — the revision pass uses this to locate and replace."
        ),
    )
    tell_category: Literal[
        "hedging_adverb",
        "mixed_emotion_abstraction",
        "couldnt_help_but",
        "in_that_moment",
        "the_kind_of_x_that_y",
        "named_emotion_over_body",
        "throat_clearing",
        "other",
    ] = Field(
        description=(
            "Which slop category this phrase belongs to. Closed enum so we "
            "can measure category frequencies across runs."
        ),
    )
    why_slop: str = Field(
        min_length=20,
        description=(
            "Plain explanation of why this phrase is a slop marker — what work "
            "it is doing in place of concrete bodied prose."
        ),
    )
    replacement_directive: str = Field(
        min_length=20,
        description=(
            "Specific instruction for the revision pass: what concrete, bodied "
            "prose should take this phrase's place. Replacement, not addition "
            "— net-neutral word delta."
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
    requested_tones: list[str] = Field(
        default_factory=list,
        description=(
            "Tones the user asked for (e.g. ['silly', 'high energy']). Used by "
            "the tone-drift review lens to flag prose that drifts from the "
            "requested register. Pulled from StoryProposal.tone."
        ),
    )
    voice_profile: VoiceProfile | None = Field(
        default=None,
        description=(
            "Voice profile selected by VoiceProfileSelectorAgent. When present, "
            "voice_register_note is applied to the revision pass and summary_overlay "
            "to the summary pass. When None, base behavior is used."
        ),
    )


class ResonanceReviewerAgentOutput(BaseModel):
    """Output contract for ResonanceReviewerAgent (Stage 6b).

    Contains the identified near-miss moments, the revised draft with
    targeted expansions, and optionally a re-generated back-cover summary.
    """

    model_config = {"frozen": True}

    near_miss_moments: list[NearMissMoment] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "0-3 identified near-miss moments classified as 'avoidance', "
            "ordered by significance. Restraint moments are identified "
            "during review but filtered out before output."
        ),
    )
    tone_drift_findings: list[ToneDriftFinding] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "0-3 passages where the prose register drifts from requested tones. "
            "Empty when requested_tones is empty or the prose holds the contract."
        ),
    )
    ending_verdict_finding: EndingVerdictFinding | None = Field(
        default=None,
        description=(
            "Singular: at most one verdict-ending finding per draft, or None "
            "when the ending preserves the unresolved pressure."
        ),
    )
    slop_markers: list[SlopMarker] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "0-5 high-confidence AI-tell phrases flagged for replacement. "
            "Each carries a verbatim quote so the revision pass can locate it."
        ),
    )
    revised_draft: str = Field(
        min_length=500,
        description=(
            "The full prose draft with targeted revisions applied to all "
            "actionable findings (avoidance moments + tone drifts + verdict "
            "ending + slop markers). Untouched passages remain exactly as "
            "they were in the original."
        ),
    )
    revised_summary: str | None = Field(
        default=None,
        description=(
            "Re-generated back-cover summary reflecting the revised draft. "
            "None when no findings were actionable (draft unchanged)."
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
    findings_total: int = Field(
        default=0,
        ge=0,
        description=(
            "Total count of actionable findings across all four review lenses "
            "(near-miss avoidance + tone drifts + verdict ending + slop markers) "
            "that drove the single revision pass."
        ),
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Review metadata: temperatures, token counts, provider info, per-lens diagnostics.",
    )
    schema_version: str = RESONANCE_REVIEWER_SCHEMA_VERSION
