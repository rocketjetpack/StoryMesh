"""LangGraph state definition for the StoryMesh pipeline.

StoryMeshState is the single mutable accumulator that flows through every
node in the graph. Each node receives the full state and returns a partial
dict containing only the keys it updates — LangGraph merges these back in
automatically.

Fields for stages 1–7 are typed as ``object | None`` until those agents are
implemented and their schemas are added to storymesh.schemas. Tighten each
field's type annotation as the corresponding schema is introduced.
"""

from __future__ import annotations

from typing import TypedDict

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

    # ── Stage 1: GenreSeedFetcherAgent ─────────────────────────────────────
    # TODO: Replace ``object`` with GenreSeedFetcherAgentOutput once implemented.
    genre_seed_fetcher_output: object | None

    # ── Stage 2: SeedRankerAgent ───────────────────────────────────────────
    # TODO: Replace ``object`` with SeedRankerAgentOutput once implemented.
    seed_ranker_output: object | None

    # ── Stage 3: BookProfileSynthesizerAgent (fan-out / parallel) ──────────
    # TODO: Replace ``object`` with list[BookProfileOutput] once implemented.
    #       The fan-out pattern will require an Annotated[list, operator.add]
    #       reducer on this field.
    book_profile_synthesizer_output: object | None

    # ── Stage 4: ThemeAggregatorAgent ─────────────────────────────────────
    # TODO: Replace ``object`` with ThemePack once implemented.
    theme_aggregator_output: object | None

    # ── Stage 5: ProposalAgent ────────────────────────────────────────────
    # TODO: Replace ``object`` with ProposalOutput once implemented.
    proposal_output: object | None

    # ── Stage 6: RubricJudgeAgent (conditional retry edge) ────────────────
    # TODO: Replace ``object`` with RubricResult once implemented.
    #       The retry loop will require a conditional edge in graph.py.
    rubric_judge_output: object | None

    # ── Stage 7: SynthesisWriterAgent ─────────────────────────────────────
    # TODO: Replace ``object`` with SynthesisWriterOutput once implemented.
    synthesis_writer_output: object | None

    # ── Error tracking ─────────────────────────────────────────────────────
    errors: list[str]
    """Non-fatal errors logged by any node during execution."""
