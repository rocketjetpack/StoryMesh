"""Pydantic schemas for ProposalReaderAgent (Stage 4.5 — retry path only).

Defines the input/output contracts for reader-perspective evaluation of a
story proposal. This agent runs on the retry path between RubricJudgeAgent
and ProposalDraftAgent, providing non-technical reader reactions that
complement the rubric's craft evaluation.

Unlike the rubric, reader feedback is intentionally non-technical: it
captures engagement, character interest, and premise believability in the
vocabulary of a reader, not an editor.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.versioning.schemas import PROPOSAL_READER_SCHEMA_VERSION


class ProposalReaderFeedback(BaseModel):
    """Reader-perspective reactions to a story proposal.

    Written from the perspective of an experienced fiction reader, not a
    craft editor. Vocabulary is intentionally non-technical — reactions
    and impressions rather than craft principles.
    """

    model_config = {"frozen": True}

    what_engaged_me: str = Field(
        min_length=10,
        description="What made the reader curious or want to read further.",
    )
    what_fell_flat: str = Field(
        min_length=10,
        description="What felt familiar, generic, or emotionally uninvolving.",
    )
    protagonist_gap: str = Field(
        min_length=10,
        description="What felt missing or underdeveloped about the main character.",
    )
    premise_question: str = Field(
        min_length=10,
        description="Something the reader does not quite believe or cannot follow.",
    )
    reader_direction: str = Field(
        min_length=10,
        description="One specific change the reader would want — in reader terms, not editor terms.",
    )


class ProposalReaderAgentInput(BaseModel):
    """Input contract for ProposalReaderAgent (Stage 4.5).

    Assembled by the node wrapper from the best-scoring proposal in
    proposal_history plus genre and tone context from upstream stages.
    """

    model_config = {"frozen": True}

    proposal: StoryProposal = Field(
        description="The best-scoring story proposal to evaluate.",
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
        description="User-specified tone words for context.",
    )


class ProposalReaderAgentOutput(BaseModel):
    """Complete reader-perspective evaluation of a story proposal."""

    model_config = {"frozen": True}

    feedback: ProposalReaderFeedback = Field(
        description="Structured reader reactions to the proposal.",
    )
    schema_version: str = PROPOSAL_READER_SCHEMA_VERSION
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation metadata: temperature, model, etc.",
    )
