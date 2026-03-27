"""LangGraph state definition for the StoryMesh pipeline.

StoryMeshState is the single mutable accumulator that flows through every
node in the graph. Each node receives the full state and returns a partial
dict containing only the keys it updates — LangGraph merges these back in
automatically.

Fields for stages 2–6 are typed as ``object | None`` until those agents are
implemented and their schemas are added to storymesh.schemas. Tighten each
field's type annotation as the corresponding schema is introduced.
"""

from __future__ import annotations

from typing import TypedDict

from storymesh.schemas.book_fetcher import BookFetcherAgentOutput
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentOutput


class StoryMeshState(TypedDict, total=False):
    """Shared state accumulator for the StoryMesh LangGraph pipeline.

    All fields are optional (``total=False``) so that nodes can return
    partial dicts without needing to repeat unchanged keys. The pipeline
    populates ``input_genre`` and ``pipeline_version`` before invocation;
    all other fields start as ``None`` and are filled in as stages run.
    """

    # ── Pipeline bookkeeping ───────────────────────────────────────────────
    input_genre: str
    """Raw genre string supplied by the caller."""

    pipeline_version: str
    """Package version string, set by StoryMeshPipeline before invocation."""

    # ── Stage 0: GenreNormalizerAgent ──────────────────────────────────────
    genre_normalizer_output: GenreNormalizerAgentOutput | None

    # ── Stage 1: BookFetcherAgent ──────────────────────────────────────────
    book_fetcher_output: BookFetcherAgentOutput | None

    # ── Stage 2: BookRankerAgent ───────────────────────────────────────────
    # TODO: Replace object with BookRankerAgentOutput once implemented.
    book_ranker_output: object | None

    # ── Stage 3: ThemeExtractorAgent (LLM) ────────────────────────────────
    # TODO: Replace object with ThemePack once implemented.
    theme_extractor_output: object | None

    # ── Stage 4: ProposalDraftAgent (LLM) ─────────────────────────────────
    # TODO: Replace object with ProposalDraftOutput once implemented.
    proposal_draft_output: object | None

    # ── Stage 5: RubricJudgeAgent (LLM, conditional retry edge) ───────────
    # TODO: Replace object with RubricResult once implemented.
    rubric_judge_output: object | None

    # ── Stage 6: SynopsisWriterAgent (LLM) ────────────────────────────────
    # TODO: Replace object with SynopsisWriterOutput once implemented.
    synopsis_writer_output: object | None

    # ── Error tracking ─────────────────────────────────────────────────────
    errors: list[str]
    """Non-fatal errors logged by any node during execution."""
