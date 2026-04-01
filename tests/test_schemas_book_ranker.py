"""Unit tests for storymesh.schemas.book_ranker."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.book_fetcher import BookRecord
from storymesh.schemas.book_ranker import (
    BookRankerAgentInput,
    BookRankerAgentOutput,
    RankedBook,
    RankedBookSummary,
    ScoreBreakdown,
)
from storymesh.versioning.schemas import BOOK_RANKER_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _book_record(**overrides: object) -> BookRecord:
    defaults: dict[str, object] = dict(
        work_key="/works/OL1W",
        title="Test Book",
        source_genres=["mystery"],
    )
    return BookRecord(**(defaults | overrides))


def _score_breakdown(**overrides: object) -> ScoreBreakdown:
    defaults: dict[str, object] = dict(
        genre_overlap=0.5,
        reader_engagement=0.5,
        rating_quality=0.5,
        rating_volume=0.5,
    )
    return ScoreBreakdown(**(defaults | overrides))


def _ranked_book(**overrides: object) -> RankedBook:
    defaults: dict[str, object] = dict(
        book=_book_record(),
        composite_score=0.5,
        score_breakdown=_score_breakdown(),
        rank=1,
    )
    return RankedBook(**(defaults | overrides))


def _ranked_summary(**overrides: object) -> RankedBookSummary:
    defaults: dict[str, object] = dict(
        work_key="/works/OL1W",
        title="Test Book",
        source_genres=["mystery"],
        composite_score=0.5,
        rank=1,
    )
    return RankedBookSummary(**(defaults | overrides))


def _ranker_output(**overrides: object) -> BookRankerAgentOutput:
    defaults: dict[str, object] = dict(
        ranked_books=[],
        ranked_summaries=[],
        dropped_count=0,
    )
    return BookRankerAgentOutput(**(defaults | overrides))


# ---------------------------------------------------------------------------
# ScoreBreakdown
# ---------------------------------------------------------------------------


class TestScoreBreakdown:
    def test_valid_construction(self) -> None:
        bd = _score_breakdown()
        assert bd.genre_overlap == 0.5
        assert bd.reader_engagement == 0.5
        assert bd.rating_quality == 0.5
        assert bd.rating_volume == 0.5

    def test_boundary_values_zero(self) -> None:
        bd = ScoreBreakdown(
            genre_overlap=0.0,
            reader_engagement=0.0,
            rating_quality=0.0,
            rating_volume=0.0,
        )
        assert bd.genre_overlap == 0.0

    def test_boundary_values_one(self) -> None:
        bd = ScoreBreakdown(
            genre_overlap=1.0,
            reader_engagement=1.0,
            rating_quality=1.0,
            rating_volume=1.0,
        )
        assert bd.genre_overlap == 1.0

    def test_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoreBreakdown(
                genre_overlap=-0.1,
                reader_engagement=0.5,
                rating_quality=0.5,
                rating_volume=0.5,
            )

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoreBreakdown(
                genre_overlap=1.1,
                reader_engagement=0.5,
                rating_quality=0.5,
                rating_volume=0.5,
            )

    def test_frozen(self) -> None:
        bd = _score_breakdown()
        with pytest.raises(ValidationError):
            bd.genre_overlap = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RankedBook
# ---------------------------------------------------------------------------


class TestRankedBook:
    def test_valid_construction(self) -> None:
        rb = _ranked_book()
        assert rb.rank == 1
        assert rb.composite_score == 0.5
        assert isinstance(rb.book, BookRecord)
        assert isinstance(rb.score_breakdown, ScoreBreakdown)

    def test_rank_below_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            RankedBook(
                book=_book_record(),
                composite_score=0.5,
                score_breakdown=_score_breakdown(),
                rank=0,
            )

    def test_composite_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RankedBook(
                book=_book_record(),
                composite_score=-0.1,
                score_breakdown=_score_breakdown(),
                rank=1,
            )

    def test_frozen(self) -> None:
        rb = _ranked_book()
        with pytest.raises(ValidationError):
            rb.rank = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RankedBookSummary
# ---------------------------------------------------------------------------


class TestRankedBookSummary:
    def test_valid_construction(self) -> None:
        s = _ranked_summary()
        assert s.work_key == "/works/OL1W"
        assert s.rank == 1

    def test_authors_defaults_to_empty_list(self) -> None:
        s = _ranked_summary()
        assert s.authors == []

    def test_first_publish_year_defaults_to_none(self) -> None:
        s = _ranked_summary()
        assert s.first_publish_year is None

    def test_rank_below_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            RankedBookSummary(
                work_key="/works/OL1W",
                title="T",
                source_genres=["mystery"],
                composite_score=0.5,
                rank=0,
            )

    def test_frozen(self) -> None:
        s = _ranked_summary()
        with pytest.raises(ValidationError):
            s.rank = 2  # type: ignore[misc]

    def test_constructed_from_ranked_book(self) -> None:
        rb = _ranked_book(
            book=_book_record(
                authors=["Author A"],
                first_publish_year=2000,
                source_genres=["mystery", "thriller"],
            ),
            composite_score=0.75,
            rank=3,
        )
        summary = RankedBookSummary(
            work_key=rb.book.work_key,
            title=rb.book.title,
            authors=rb.book.authors,
            first_publish_year=rb.book.first_publish_year,
            source_genres=rb.book.source_genres,
            composite_score=rb.composite_score,
            rank=rb.rank,
        )
        assert summary.work_key == "/works/OL1W"
        assert summary.authors == ["Author A"]
        assert summary.first_publish_year == 2000
        assert summary.source_genres == ["mystery", "thriller"]
        assert summary.composite_score == 0.75
        assert summary.rank == 3


# ---------------------------------------------------------------------------
# BookRankerAgentInput
# ---------------------------------------------------------------------------


class TestBookRankerAgentInput:
    def test_valid_construction(self) -> None:
        inp = BookRankerAgentInput(
            books=[_book_record()],
            user_prompt="dark mystery",
            total_genres_queried=2,
        )
        assert len(inp.books) == 1
        assert inp.user_prompt == "dark mystery"
        assert inp.total_genres_queried == 2

    def test_empty_books_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRankerAgentInput(
                books=[],
                user_prompt="dark mystery",
                total_genres_queried=2,
            )

    def test_empty_user_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRankerAgentInput(
                books=[_book_record()],
                user_prompt="",
                total_genres_queried=2,
            )

    def test_total_genres_below_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRankerAgentInput(
                books=[_book_record()],
                user_prompt="dark mystery",
                total_genres_queried=0,
            )


# ---------------------------------------------------------------------------
# BookRankerAgentOutput
# ---------------------------------------------------------------------------


class TestBookRankerAgentOutput:
    def test_valid_empty_output(self) -> None:
        out = _ranker_output()
        assert out.ranked_books == []
        assert out.ranked_summaries == []
        assert out.dropped_count == 0
        assert out.llm_reranked is False
        assert out.debug == {}

    def test_valid_with_ranked_books(self) -> None:
        rb = _ranked_book()
        rs = _ranked_summary()
        out = _ranker_output(ranked_books=[rb], ranked_summaries=[rs], dropped_count=5)
        assert len(out.ranked_books) == 1
        assert len(out.ranked_summaries) == 1
        assert out.dropped_count == 5

    def test_dropped_count_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            BookRankerAgentOutput(
                ranked_books=[],
                ranked_summaries=[],
                dropped_count=-1,
            )

    def test_llm_reranked_default_false(self) -> None:
        out = _ranker_output()
        assert out.llm_reranked is False

    def test_llm_reranked_can_be_true(self) -> None:
        out = _ranker_output(llm_reranked=True)
        assert out.llm_reranked is True

    def test_schema_version(self) -> None:
        out = _ranker_output()
        assert out.schema_version == BOOK_RANKER_SCHEMA_VERSION

    def test_schema_version_value(self) -> None:
        assert BOOK_RANKER_SCHEMA_VERSION == "1.1"

    def test_frozen(self) -> None:
        out = _ranker_output()
        with pytest.raises(ValidationError):
            out.dropped_count = 5  # type: ignore[misc]

    def test_debug_accepts_arbitrary_dict(self) -> None:
        out = _ranker_output(
            debug={"weights": {"genre_overlap": 0.4}, "total_scored": 20}
        )
        assert out.debug["total_scored"] == 20
