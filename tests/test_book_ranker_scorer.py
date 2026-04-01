"""Unit tests for storymesh.agents.book_ranker.scorer.

All functions are pure and stateless, so tests use exact float assertions
with pytest.approx where floating-point arithmetic applies.
"""

from __future__ import annotations

import pytest

from storymesh.agents.book_ranker.scorer import (
    DEFAULT_RATING_CONFIDENCE_THRESHOLD,
    DEFAULT_WEIGHTS,
    _jaccard_similarity,
    compute_scores,
    score_genre_overlap,
    score_rating_quality,
    score_rating_volume,
    score_reader_engagement,
    select_diverse,
)
from storymesh.schemas.book_fetcher import BookRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _book(**overrides: object) -> BookRecord:
    defaults: dict[str, object] = dict(
        work_key="/works/OL1W",
        title="Test Book",
        source_genres=["mystery"],
        readinglog_count=100,
        ratings_count=100,
        ratings_average=4.0,
    )
    return BookRecord(**(defaults | overrides))


# ---------------------------------------------------------------------------
# score_genre_overlap
# ---------------------------------------------------------------------------


class TestScoreGenreOverlap:
    def test_one_of_one(self) -> None:
        assert score_genre_overlap(1, 1) == 1.0

    def test_one_of_three(self) -> None:
        assert score_genre_overlap(1, 3) == pytest.approx(1 / 3)

    def test_two_of_three(self) -> None:
        assert score_genre_overlap(2, 3) == pytest.approx(2 / 3)

    def test_three_of_three(self) -> None:
        assert score_genre_overlap(3, 3) == 1.0

    def test_capped_at_one(self) -> None:
        # Should never happen in practice but guard against it.
        assert score_genre_overlap(5, 3) == 1.0

    def test_zero_total_genres_returns_zero(self) -> None:
        assert score_genre_overlap(2, 0) == 0.0

    def test_zero_source_genres(self) -> None:
        assert score_genre_overlap(0, 3) == 0.0


# ---------------------------------------------------------------------------
# score_reader_engagement
# ---------------------------------------------------------------------------


class TestScoreReaderEngagement:
    def test_uniform_data_returns_half(self) -> None:
        assert score_reader_engagement(500, 500, 500) == 0.5

    def test_minimum_of_range(self) -> None:
        assert score_reader_engagement(0, 0, 1000) == 0.0

    def test_maximum_of_range(self) -> None:
        assert score_reader_engagement(1000, 0, 1000) == 1.0

    def test_midpoint_of_range(self) -> None:
        assert score_reader_engagement(500, 0, 1000) == pytest.approx(0.5)

    def test_arbitrary_value(self) -> None:
        # (200 - 100) / (600 - 100) = 100/500 = 0.2
        assert score_reader_engagement(200, 100, 600) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# score_rating_quality
# ---------------------------------------------------------------------------


class TestScoreRatingQuality:
    def test_none_rating_returns_zero(self) -> None:
        assert score_rating_quality(None, 100, DEFAULT_RATING_CONFIDENCE_THRESHOLD) == 0.0

    def test_zero_rating_returns_zero(self) -> None:
        assert score_rating_quality(0.0, 100, DEFAULT_RATING_CONFIDENCE_THRESHOLD) == 0.0

    def test_negative_rating_returns_zero(self) -> None:
        assert score_rating_quality(-1.0, 100, DEFAULT_RATING_CONFIDENCE_THRESHOLD) == 0.0

    def test_full_confidence_perfect_rating(self) -> None:
        # 5.0 / 5.0 * 1.0 = 1.0
        assert score_rating_quality(5.0, 500, 50) == pytest.approx(1.0)

    def test_full_confidence_average_rating(self) -> None:
        # 4.0 / 5.0 * 1.0 = 0.8
        assert score_rating_quality(4.0, 500, 50) == pytest.approx(0.8)

    def test_low_confidence_discounts_rating(self) -> None:
        # confidence = 5/50 = 0.1; 4.0/5.0 * 0.1 = 0.08
        assert score_rating_quality(4.0, 5, 50) == pytest.approx(0.08)

    def test_confidence_capped_at_one(self) -> None:
        # ratings_count (1000) >> threshold (50); confidence = 1.0
        assert score_rating_quality(4.0, 1000, 50) == pytest.approx(0.8)

    def test_exact_threshold_reaches_full_confidence(self) -> None:
        # confidence = 50/50 = 1.0; 4.0/5.0 = 0.8
        assert score_rating_quality(4.0, 50, 50) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# score_rating_volume
# ---------------------------------------------------------------------------


class TestScoreRatingVolume:
    def test_uniform_data_returns_half(self) -> None:
        assert score_rating_volume(100, 100, 100) == 0.5

    def test_minimum_of_range(self) -> None:
        assert score_rating_volume(0, 0, 1000) == 0.0

    def test_maximum_of_range(self) -> None:
        assert score_rating_volume(1000, 0, 1000) == 1.0

    def test_arbitrary_value(self) -> None:
        # (300 - 100) / (500 - 100) = 200/400 = 0.5
        assert score_rating_volume(300, 100, 500) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------


class TestComputeScores:
    def test_returns_all_books(self) -> None:
        books = [_book(work_key=f"/works/OL{i}W") for i in range(5)]
        results = compute_scores(books, total_genres_queried=1)
        assert len(results) == 5

    def test_sorted_descending_by_composite_score(self) -> None:
        # Book A: high readinglog, high ratings → should rank first
        book_a = _book(
            work_key="/works/OLA",
            readinglog_count=1000,
            ratings_count=1000,
            ratings_average=5.0,
            source_genres=["mystery", "thriller"],
        )
        # Book B: low readinglog, low ratings
        book_b = _book(
            work_key="/works/OLB",
            readinglog_count=10,
            ratings_count=10,
            ratings_average=2.0,
            source_genres=["mystery"],
        )
        results = compute_scores([book_a, book_b], total_genres_queried=2)
        assert results[0][0].work_key == "/works/OLA"
        assert results[1][0].work_key == "/works/OLB"

    def test_composite_score_in_expected_range(self) -> None:
        books = [_book()]
        results = compute_scores(books, total_genres_queried=1)
        _, score, _ = results[0]
        assert 0.0 <= score <= 1.0

    def test_single_book_edge_case(self) -> None:
        book = _book(readinglog_count=500, ratings_count=500, ratings_average=4.0)
        results = compute_scores([book], total_genres_queried=1)
        assert len(results) == 1
        # With a single book, min == max for both engagement and volume → 0.5 each
        _, _, breakdown = results[0]
        assert breakdown.reader_engagement == 0.5
        assert breakdown.rating_volume == 0.5

    def test_custom_weights_applied(self) -> None:
        # All weight on genre_overlap, which is 1.0 (1 source genre / 1 queried)
        custom_weights = {
            "genre_overlap": 1.0,
            "reader_engagement": 0.0,
            "rating_quality": 0.0,
            "rating_volume": 0.0,
        }
        book = _book(source_genres=["mystery"])
        results = compute_scores([book], total_genres_queried=1, weights=custom_weights)
        _, score, _ = results[0]
        assert score == pytest.approx(1.0)

    def test_score_breakdown_present(self) -> None:
        books = [_book()]
        results = compute_scores(books, total_genres_queried=1)
        _, _, breakdown = results[0]
        assert 0.0 <= breakdown.genre_overlap <= 1.0
        assert 0.0 <= breakdown.reader_engagement <= 1.0
        assert 0.0 <= breakdown.rating_quality <= 1.0
        assert 0.0 <= breakdown.rating_volume <= 1.0

    def test_default_weights_used_when_none(self) -> None:
        book_a = _book(
            work_key="/works/OLA",
            readinglog_count=1000,
            ratings_count=1000,
            ratings_average=5.0,
            source_genres=["mystery", "thriller"],
        )
        book_b = _book(
            work_key="/works/OLB",
            readinglog_count=0,
            ratings_count=0,
            ratings_average=None,
            source_genres=["mystery"],
        )
        results_default = compute_scores([book_a, book_b], total_genres_queried=2, weights=None)
        results_explicit = compute_scores(
            [book_a, book_b], total_genres_queried=2, weights=DEFAULT_WEIGHTS
        )
        assert results_default[0][1] == pytest.approx(results_explicit[0][1])
        assert results_default[1][1] == pytest.approx(results_explicit[1][1])

    def test_uniform_scores_maintains_all_books(self) -> None:
        # Identical books: stable sort should keep all entries.
        books = [
            _book(
                work_key=f"/works/OL{i}W",
                readinglog_count=100,
                ratings_count=100,
                ratings_average=4.0,
                source_genres=["mystery"],
            )
            for i in range(3)
        ]
        results = compute_scores(books, total_genres_queried=1)
        assert len(results) == 3
        scores = [r[1] for r in results]
        assert all(s == pytest.approx(scores[0]) for s in scores)


# ---------------------------------------------------------------------------
# _jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    """_jaccard_similarity operates on Open Library subject-tag lists (case-insensitive)."""

    def test_identical_subjects(self) -> None:
        assert _jaccard_similarity(["Mystery", "Victorian"], ["Mystery", "Victorian"]) == pytest.approx(1.0)

    def test_disjoint_subjects(self) -> None:
        assert _jaccard_similarity(["Fantasy"], ["Mystery"]) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # intersection={"mystery"}, union={"fantasy","mystery","thriller"} → 1/3
        result = _jaccard_similarity(["Fantasy", "Mystery"], ["Mystery", "Thriller"])
        assert result == pytest.approx(1 / 3)

    def test_both_empty_returns_zero(self) -> None:
        """Both empty → 0.0 (no evidence of similarity, unlike old genre-based version)."""
        assert _jaccard_similarity([], []) == pytest.approx(0.0)

    def test_one_empty_returns_zero(self) -> None:
        assert _jaccard_similarity(["Fantasy"], []) == pytest.approx(0.0)

    def test_other_empty_returns_zero(self) -> None:
        assert _jaccard_similarity([], ["Mystery"]) == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        """Tags "MYSTERY" and "mystery" are the same subject."""
        assert _jaccard_similarity(["MYSTERY"], ["mystery"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_diverse
# ---------------------------------------------------------------------------


def _scored(
    work_key: str,
    score: float,
    subjects: list[str] | None = None,
) -> tuple:
    book = _book(work_key=work_key, subjects=subjects or [])
    from storymesh.schemas.book_ranker import ScoreBreakdown

    breakdown = ScoreBreakdown(
        genre_overlap=score,
        reader_engagement=score,
        rating_quality=score,
        rating_volume=score,
    )
    return (book, score, breakdown)


class TestSelectDiverse:
    def test_lambda_one_matches_pure_relevance(self) -> None:
        """mmr_lambda=1.0 must produce the same order as sorted-by-score."""
        books = [
            _scored("/works/OL1W", 0.9, ["Mystery", "Victorian"]),
            _scored("/works/OL2W", 0.8, ["Mystery", "Victorian"]),
            _scored("/works/OL3W", 0.7, ["Fantasy", "Epic"]),
        ]
        result = select_diverse(books, top_n=2, mmr_lambda=1.0)
        assert [b[0].work_key for b in result] == ["/works/OL1W", "/works/OL2W"]

    def test_low_lambda_promotes_diverse_book(self) -> None:
        """With low mmr_lambda, a lower-scoring but subject-diverse book
        is preferred over a higher-scoring subject-redundant book."""
        # OL1W and OL2W share all subjects (redundant). OL3W has disjoint subjects.
        books = [
            _scored("/works/OL1W", 0.9, ["Mystery", "Victorian"]),
            _scored("/works/OL2W", 0.85, ["Mystery", "Victorian"]),
            _scored("/works/OL3W", 0.7, ["Fantasy", "Epic"]),
        ]
        result = select_diverse(books, top_n=2, mmr_lambda=0.1)
        selected_keys = {b[0].work_key for b in result}
        # OL1W is always first. OL3W should beat OL2W due to diversity.
        assert "/works/OL1W" in selected_keys
        assert "/works/OL3W" in selected_keys

    def test_first_selection_is_always_highest_scoring(self) -> None:
        """First pick is always the highest-scoring candidate regardless of lambda."""
        books = [
            _scored("/works/OL2W", 0.9, ["Fantasy"]),
            _scored("/works/OL3W", 0.7, ["Thriller"]),
            _scored("/works/OL1W", 0.5, ["Mystery"]),
        ]
        result = select_diverse(books, top_n=2, mmr_lambda=0.5)
        assert result[0][0].work_key == "/works/OL2W"

    def test_top_n_gte_candidates_returns_all(self) -> None:
        """When top_n >= len(candidates), all candidates are returned."""
        books = [_scored(f"/works/OL{i}W", 1.0 - i * 0.1) for i in range(3)]
        result = select_diverse(books, top_n=10, mmr_lambda=0.5)
        assert len(result) == 3

    def test_single_book_input(self) -> None:
        books = [_scored("/works/OL1W", 0.9, ["Mystery"])]
        result = select_diverse(books, top_n=1, mmr_lambda=0.5)
        assert len(result) == 1
        assert result[0][0].work_key == "/works/OL1W"

    def test_empty_subjects_no_crash(self) -> None:
        """Books with empty subjects produce 0.0 similarity and do not crash."""
        books = [
            _scored("/works/OL1W", 0.9, []),
            _scored("/works/OL2W", 0.8, []),
            _scored("/works/OL3W", 0.7, []),
        ]
        result = select_diverse(books, top_n=2, mmr_lambda=0.5)
        assert len(result) == 2

    def test_top_n_zero_returns_empty(self) -> None:
        books = [_scored("/works/OL1W", 0.9)]
        result = select_diverse(books, top_n=0, mmr_lambda=0.5)
        assert result == []

    def test_top_n_respected(self) -> None:
        books = [_scored(f"/works/OL{i}W", 1.0 - i * 0.1) for i in range(5)]
        result = select_diverse(books, top_n=3, mmr_lambda=0.6)
        assert len(result) == 3

    def test_mmr_candidates_limits_pool(self) -> None:
        """mmr_candidates truncates the input pool before MMR selection."""
        # 5 books but only consider top 2 candidates.
        books = [_scored(f"/works/OL{i}W", 1.0 - i * 0.1) for i in range(5)]
        result = select_diverse(books, top_n=3, mmr_lambda=0.6, mmr_candidates=2)
        # Can only return up to mmr_candidates books.
        assert len(result) == 2

    def test_integration_diverse_books_promoted(self) -> None:
        """Integration: with 5 near-identical and 5 unique-subject books,
        unique books appear in top_n even if scored slightly lower."""
        similar = [
            _scored(f"/works/OLsim{i}W", 0.85, ["Mystery", "Victorian", "Crime"])
            for i in range(5)
        ]
        unique = [
            _scored(f"/works/OLuniq{i}W", 0.75, [f"Genre{i}", f"Subject{i}"])
            for i in range(5)
        ]
        # Sort by score descending as compute_scores guarantees.
        all_books = sorted(similar + unique, key=lambda t: t[1], reverse=True)
        result = select_diverse(all_books, top_n=6, mmr_lambda=0.4, mmr_candidates=30)
        selected_keys = {b[0].work_key for b in result}
        unique_keys = {b[0].work_key for b in unique}
        # At least some unique books should be promoted over similar ones.
        assert len(selected_keys & unique_keys) >= 2
