"""Pydantic schemas for the BookRankerAgent.

Defines the input and output contracts for Stage 2 of the StoryMesh pipeline.
The agent receives enriched BookRecord objects from the BookFetcherAgent and
returns a ranked list with full scoring detail (for artifacts) and a slim
summary list (for downstream LLM token efficiency).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.book_fetcher import BookRecord
from storymesh.versioning.schemas import BOOK_RANKER_SCHEMA_VERSION


class BookRankerAgentInput(BaseModel):
    """Input contract for the BookRankerAgent."""

    books: list[BookRecord] = Field(
        min_length=1,
        description="Deduplicated book records from the BookFetcherAgent.",
    )
    user_prompt: str = Field(
        min_length=1,
        description=(
            "Original user input string. Passed through for the optional "
            "LLM re-rank prompt, which assesses narrative potential."
        ),
    )
    total_genres_queried: int = Field(
        ge=1,
        description=(
            "Number of genres that were queried by the BookFetcherAgent. "
            "Used as the denominator for genre overlap scoring."
        ),
    )


class ScoreBreakdown(BaseModel):
    """Individual scoring components for a ranked book."""

    model_config = {"frozen": True}

    genre_overlap: float = Field(ge=0.0, le=1.0)
    reader_engagement: float = Field(ge=0.0, le=1.0)
    rating_quality: float = Field(ge=0.0, le=1.0)
    rating_volume: float = Field(ge=0.0, le=1.0)


class RankedBook(BaseModel):
    """A book with its computed ranking data. Full detail version for artifacts."""

    model_config = {"frozen": True}

    book: BookRecord
    composite_score: float = Field(
        ge=0.0,
        description="Final weighted composite score.",
    )
    score_breakdown: ScoreBreakdown
    rank: int = Field(ge=1, description="1-indexed rank position.")


class RankedBookSummary(BaseModel):
    """Slim book representation for downstream LLM consumption.

    Strips scoring internals and fields not needed by ThemeExtractor
    or ProposalDraft to minimize LLM token usage.
    """

    model_config = {"frozen": True}

    work_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    first_publish_year: int | None = None
    source_genres: list[str]
    composite_score: float
    rank: int = Field(ge=1)


class BookRankerAgentOutput(BaseModel):
    """Output contract for the BookRankerAgent."""

    model_config = {"frozen": True}

    ranked_books: list[RankedBook] = Field(
        description="Full-detail ranked books, ordered by rank. Persisted in artifacts.",
    )
    ranked_summaries: list[RankedBookSummary] = Field(
        description=(
            "Slim ranked book summaries for downstream LLM agents. "
            "Same ordering as ranked_books."
        ),
    )
    dropped_count: int = Field(
        ge=0,
        description="Number of books that fell below the top_n cutoff.",
    )
    llm_reranked: bool = Field(
        default=False,
        description="Whether the LLM re-rank path was applied.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Scoring metadata: weights used, score distribution stats, "
            "LLM re-rank details if applicable."
        ),
    )
    schema_version: str = BOOK_RANKER_SCHEMA_VERSION
