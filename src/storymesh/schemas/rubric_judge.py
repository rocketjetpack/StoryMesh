"""Pydantic schemas for RubricJudgeAgent.

Defines the input/output contracts for Stage 5 of the StoryMesh pipeline.
RubricJudgeAgent evaluates a StoryProposal against a craft-quality rubric
whose dimensions align with three creative principles: restraint, convention
then departure, and specificity without performance.

Pass/fail is determined by the agent in Python from the weighted composite
score — the LLM returns only scores and feedback, keeping the decision
deterministic and auditable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.theme_extractor import ThematicTension
from storymesh.versioning.schemas import RUBRIC_SCHEMA_VERSION

EXPECTED_DIMENSIONS: frozenset[str] = frozenset({
    "restraint",
    "convention_departure",
    "specificity",
    "protagonist_interiority",
    "user_intent_fidelity",
})


class RubricJudgeAgentInput(BaseModel):
    """Input contract for RubricJudgeAgent (Stage 5).

    Assembled by the node wrapper from upstream pipeline state.
    """

    model_config = {"frozen": True}

    proposal: StoryProposal = Field(
        description="The story proposal to evaluate.",
    )
    tensions: list[ThematicTension] = Field(
        min_length=1,
        description=(
            "Thematic tensions from ThemeExtractorAgent, including familiar "
            "resolutions that form the genre contract."
        ),
    )
    cliched_resolutions: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of tension_id to list of familiar resolution patterns. "
            "Derived from tensions for convenient access during evaluation."
        ),
    )
    user_tones: list[str] = Field(
        default_factory=list,
        description="User-specified tone words for intent-fidelity evaluation.",
    )
    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string.",
    )
    normalized_genres: list[str] = Field(
        min_length=1,
        description="Canonical genre names from GenreNormalizerAgent.",
    )
    attempt_number: int = Field(
        ge=1,
        default=1,
        description="Which attempt this evaluation covers (1 = initial, 2+ = retry).",
    )


class DimensionResult(BaseModel):
    """Score and feedback for a single rubric dimension."""

    model_config = {"frozen": True}

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Dimension score from 0.0 (fail) to 1.0 (excellent).",
    )
    feedback: str = Field(
        min_length=10,
        description="Specific, actionable feedback referencing the relevant creative principle.",
    )
    principle_ref: str = Field(
        min_length=1,
        description=(
            "Which creative principle this dimension evaluates "
            "(e.g., 'restraint', 'convention_departure', 'specificity')."
        ),
    )


class RubricJudgeAgentOutput(BaseModel):
    """Complete rubric evaluation of a story proposal.

    The ``passed`` field and ``composite_score`` are computed by the agent in
    Python from the LLM's per-dimension scores — the LLM does not determine
    pass/fail. This keeps the decision deterministic and auditable.
    """

    model_config = {"frozen": True}

    passed: bool = Field(
        description="True when composite_score >= pass_threshold.",
    )
    composite_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted average of dimension scores.",
    )
    pass_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="The threshold used for pass/fail determination.",
    )
    dimensions: dict[str, DimensionResult] = Field(
        min_length=1,
        description=(
            "Mapping of dimension name to score and feedback. "
            "Expected keys: restraint, convention_departure, "
            "specificity, protagonist_interiority, user_intent_fidelity."
        ),
    )
    convention_departures: list[str] = Field(
        default_factory=list,
        description=(
            "Specific genre conventions the proposal follows and the "
            "departure moment(s) identified, if any."
        ),
    )
    overall_feedback: str = Field(
        min_length=10,
        description=(
            "Holistic editorial assessment: strongest element, weakest element, "
            "one specific suggestion for improvement."
        ),
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Evaluation metadata: weights_used, threshold, attempt_number, "
            "raw_scores, total_llm_calls."
        ),
    )
    schema_version: str = RUBRIC_SCHEMA_VERSION
