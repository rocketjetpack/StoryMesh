"""LangGraph state definition for the StoryMesh pipeline.

StoryMeshState is the single mutable accumulator that flows through every
node in the graph. Each node receives the full state and returns a partial
dict containing only the keys it updates — LangGraph merges these back in
automatically.

All fields are optional (``total=False``) so nodes can return partial dicts.
The pipeline populates ``user_prompt`` and ``pipeline_version`` before
invocation; all other fields start as ``None`` and are filled as stages run.
"""

from __future__ import annotations

from typing import TypedDict

from storymesh.schemas.book_assembler import BookAssemblerAgentOutput
from storymesh.schemas.book_fetcher import BookFetcherAgentOutput
from storymesh.schemas.book_ranker import BookRankerAgentOutput
from storymesh.schemas.cover_art import CoverArtAgentOutput
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentOutput
from storymesh.schemas.proposal_draft import ProposalDraftAgentOutput
from storymesh.schemas.proposal_reader import ProposalReaderAgentOutput
from storymesh.schemas.resonance_reviewer import ResonanceReviewerAgentOutput
from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput
from storymesh.schemas.story_writer import StoryWriterAgentOutput
from storymesh.schemas.theme_extractor import ThemeExtractorAgentOutput


class StoryMeshState(TypedDict, total=False):
    """Shared state accumulator for the StoryMesh LangGraph pipeline.

    All fields are optional (``total=False``) so that nodes can return
    partial dicts without needing to repeat unchanged keys. The pipeline
    populates ``user_prompt`` and ``pipeline_version`` before invocation;
    all other fields start as ``None`` and are filled in as stages run.
    """

    # ── Pipeline bookkeeping ───────────────────────────────────────────────
    user_prompt: str
    """Raw user input string describing the desired fiction synopsis."""

    pipeline_version: str
    """Package version string, set by StoryMeshPipeline before invocation."""

    run_id: str
    """Unique run identifier (UUID hex), generated before graph invocation."""

    # ── Stage 0: GenreNormalizerAgent ──────────────────────────────────────
    genre_normalizer_output: GenreNormalizerAgentOutput | None

    # ── Stage 1: BookFetcherAgent ──────────────────────────────────────────
    book_fetcher_output: BookFetcherAgentOutput | None

    # ── Stage 2: BookRankerAgent ───────────────────────────────────────────
    book_ranker_output: BookRankerAgentOutput | None

    # ── Stage 3: ThemeExtractorAgent (LLM) ────────────────────────────────
    theme_extractor_output: ThemeExtractorAgentOutput | None

    # ── Stage 4: ProposalDraftAgent (LLM) ─────────────────────────────────
    proposal_draft_output: ProposalDraftAgentOutput | None

    # ── Stage 4.5: ProposalReaderAgent (LLM, retry path only) ─────────────
    proposal_reader_output: ProposalReaderAgentOutput | None

    # ── Stage 5: RubricJudgeAgent (LLM, conditional retry edge) ───────────
    rubric_judge_output: RubricJudgeAgentOutput | None

    # ── Stage 6: StoryWriterAgent (LLM) ───────────────────────────────────
    story_writer_output: StoryWriterAgentOutput | None

    # ── Stage 6b: ResonanceReviewerAgent ────────────────────────────────────
    resonance_reviewer_output: ResonanceReviewerAgentOutput | None

    # ── Stage 7: CoverArtAgent ─────────────────────────────────────────────
    cover_art_output: CoverArtAgentOutput | None

    # ── Stage 8: BookAssemblerAgent ────────────────────────────────────────
    book_assembler_output: BookAssemblerAgentOutput | None

    # ── Rubric retry tracking ──────────────────────────────────────────────
    rubric_retry_count: int
    """Number of times rubric_judge has routed back to proposal_draft. Starts at 0."""

    # ── Attempt history (for SynopsisWriter synthesis) ─────────────────────
    proposal_history: list[ProposalDraftAgentOutput]
    """All proposal attempts in order. Appended by proposal_draft node on each run."""

    rubric_history: list[RubricJudgeAgentOutput]
    """All rubric evaluations in order. Appended by rubric_judge node on each run."""

    best_proposal_index: int
    """Index into proposal_history of the highest-scoring attempt across all rounds."""

    # ── Error tracking ─────────────────────────────────────────────────────
    errors: list[str]
    """Non-fatal errors logged by any node during execution."""
