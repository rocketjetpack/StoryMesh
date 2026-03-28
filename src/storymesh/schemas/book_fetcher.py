"""Pydantic schemas for the BookFetcherAgent.

Defines the input and output contracts for Stage 1 of the StoryMesh pipeline.
The agent receives normalized genre names and returns a deduplicated list of book
metadata records sourced from the Open Library Search API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.versioning.schemas import BOOK_FETCHER_SCHEMA_VERSION


class BookRecord(BaseModel):
    """Metadata for a single book returned by the Open Library Search API.

    Each record is unique by ``work_key``. When a book is found under multiple
    genre queries, all matched genres are accumulated in ``source_genres``.
    """

    model_config = {"frozen": True}

    work_key: str = Field(
        description="Open Library work key (e.g., '/works/OL27448W'). Unique identifier.",
    )
    title: str = Field(
        description="Title of the work.",
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author name(s). Empty list if not present in the API response.",
    )
    first_publish_year: int | None = Field(
        default=None,
        description="Year of first publication. None if unavailable.",
    )
    edition_count: int = Field(
        default=0,
        description="Number of editions. Primary popularity proxy for ranking.",
    )
    ratings_average: float | None = Field(
        default=None,
        description="Average rating on a 0–5 scale. None if no ratings exist.",
    )
    ratings_count: int = Field(
        default=0,
        description="Number of ratings recorded on Open Library.",
    )
    subjects: list[str] = Field(
        default_factory=list,
        description="Subject tags from Open Library.",
    )
    cover_id: int | None = Field(
        default=None,
        description=(
            "Open Library cover image ID. None if no cover is available. "
            "Not used downstream but retained for future UI use."
        ),
    )
    readinglog_count: int = Field(
        default=0,
        description=(
            "Total number of Open Library users who have this book on any "
            "reading shelf (want to read + currently reading + already read). "
            "Direct popularity signal for ranking."
        ),
    )
    want_to_read_count: int = Field(
        default=0,
        description="Number of users with this book on their 'want to read' shelf.",
    )
    already_read_count: int = Field(
        default=0,
        description="Number of users who have marked this book as 'already read'.",
    )
    currently_reading_count: int = Field(
        default=0,
        description="Number of users currently reading this book.",
    )
    number_of_pages_median: int | None = Field(
        default=None,
        description=(
            "Median page count across all editions. None if unavailable. "
            "Not used for ranking but useful for downstream synopsis calibration."
        ),
    )
    source_genres: list[str] = Field(
        description=(
            "Genre subject strings whose queries returned this book "
            "(e.g., ['post apocalyptic', 'mystery']). "
            "Lets the BookRankerAgent compute cross-genre alignment scores."
        ),
    )


class BookFetcherAgentInput(BaseModel):
    """Input contract for the BookFetcherAgent."""

    normalized_genres: list[str] = Field(
        min_length=1,
        description="Normalized genre names from the GenreNormalizerAgent.",
    )
    limit_per_genre: int = Field(
        default=30,
        description="Maximum number of books to fetch per genre query.",
    )


class BookFetcherAgentOutput(BaseModel):
    """Output contract for the BookFetcherAgent.

    This output is frozen and versioned. It represents the downstream contract
    consumed by the BookRankerAgent (Stage 2). Books are deduplicated by
    ``work_key`` within this agent; each record's ``source_genres`` field
    records every genre query that returned it.
    """

    model_config = {"frozen": True}

    books: list[BookRecord] = Field(
        description=(
            "Deduplicated book records from all genre queries combined. "
            "Each work_key appears at most once; source_genres lists all "
            "genres that returned the book."
        ),
    )
    queries_executed: list[str] = Field(
        description="Genre subject strings actually sent to the API. Useful for debugging.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-run audit data: per-genre fetch counts, cache hit/miss status, "
            "and deduplication summary. Captured in the artifact automatically."
        ),
    )
    schema_version: str = BOOK_FETCHER_SCHEMA_VERSION
