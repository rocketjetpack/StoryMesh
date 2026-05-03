"""Pydantic schemas for RubricJudgeAgent.

Defines the input/output contracts for Stage 5 of the StoryMesh pipeline.
RubricJudgeAgent evaluates a StoryProposal against a craft-quality rubric
using three-tier scoring (0=fail, 1=acceptable, 2=strong) across five
dimensions aligned with creative principles: story-serving restraint,
story-serving choices, specificity with texture, protagonist interiority,
and user intent fidelity.

Pass/fail is determined by the agent in Python from the sum of tier scores
— the LLM returns only scores and feedback, keeping the decision
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
    "story_serving_choices",
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
    """Score and feedback for a single rubric dimension.

    Uses three-tier scoring: 0 (fail), 1 (acceptable), 2 (strong).
    """

    model_config = {"frozen": True}

    score: int = Field(
        ge=0,
        le=2,
        description="Dimension tier: 0 (fail), 1 (acceptable), 2 (strong).",
    )
    feedback: str = Field(
        min_length=10,
        description="Specific, actionable feedback referencing the relevant creative principle.",
    )
    principle_ref: str = Field(
        min_length=1,
        description=(
            "Which creative principle this dimension evaluates "
            "(e.g., 'restraint', 'story_serving_choices', 'specificity')."
        ),
    )


class RubricJudgeAgentOutput(BaseModel):
    """Complete rubric evaluation of a story proposal.

    Uses three-tier scoring (0/1/2) per dimension with a sum composite
    (max 10). The ``passed`` field and ``composite_score`` are computed
    by the agent in Python — the LLM does not determine pass/fail.
    This keeps the decision deterministic and auditable.
    """

    model_config = {"frozen": True}

    passed: bool = Field(
        description="True when composite_score >= pass_threshold.",
    )
    composite_score: int = Field(
        ge=0,
        le=10,
        description="Sum of all dimension tier scores (max 10).",
    )
    pass_threshold: int = Field(
        ge=0,
        le=10,
        description="The threshold used for pass/fail determination.",
    )
    dimensions: dict[str, DimensionResult] = Field(
        min_length=1,
        description=(
            "Mapping of dimension name to score and feedback. "
            "Expected keys: restraint, story_serving_choices, "
            "specificity, protagonist_interiority, user_intent_fidelity."
        ),
    )
    creative_direction: str = Field(
        default="",
        description=(
            "One specific, actionable editorial instruction for how the proposal "
            "should be revised. Names the element and describes the change. "
            "Empty only on LLM failure."
        ),
    )
    overall_feedback: str = Field(
        min_length=10,
        description=(
            "Holistic editorial assessment: strongest element (with quote), "
            "weakest element (with quote), and whether the proposal explains "
            "itself too much."
        ),
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Evaluation metadata: threshold, attempt_number, "
            "raw_scores, total_llm_calls."
        ),
    )
    schema_version: str = RUBRIC_SCHEMA_VERSION
