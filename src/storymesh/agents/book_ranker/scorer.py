"""Pure scoring functions for the BookRankerAgent.

All functions are stateless — they accept only the values they need and return
floats in [0.0, 1.0]. This separation from agent.py makes the scoring math
independently testable with exact float assertions.
"""

from __future__ import annotations

from storymesh.schemas.book_fetcher import BookRecord
from storymesh.schemas.book_ranker import ScoreBreakdown

DEFAULT_WEIGHTS: dict[str, float] = {
    "genre_overlap": 0.40,
    "reader_engagement": 0.25,
    "rating_quality": 0.20,
    "rating_volume": 0.15,
}
DEFAULT_RATING_CONFIDENCE_THRESHOLD: int = 50


def score_genre_overlap(source_genres_count: int, total_genres_queried: int) -> float:
    """Compute the fraction of queried genres that returned this book.

    Args:
        source_genres_count: Number of genres whose queries included this book.
        total_genres_queried: Total number of genres queried in this pipeline run.

    Returns:
        Score in [0.0, 1.0]. Capped at 1.0 if source_genres_count somehow
        exceeds total_genres_queried.
    """
    if total_genres_queried <= 0:
        return 0.0
    return min(1.0, source_genres_count / total_genres_queried)


def score_reader_engagement(
    readinglog_count: int,
    min_readinglog: int,
    max_readinglog: int,
) -> float:
    """Compute min-max normalized readinglog_count across the current batch.

    Args:
        readinglog_count: Total reading-shelf count for this book.
        min_readinglog: Minimum readinglog_count in the batch.
        max_readinglog: Maximum readinglog_count in the batch.

    Returns:
        Score in [0.0, 1.0]. Returns 0.5 when all books have identical counts
        (uniform data — treat neutrally).
    """
    if max_readinglog == min_readinglog:
        return 0.5
    return (readinglog_count - min_readinglog) / (max_readinglog - min_readinglog)


def score_rating_quality(
    ratings_average: float | None,
    ratings_count: int,
    confidence_threshold: int,
) -> float:
    """Compute confidence-adjusted rating quality.

    Discounts ratings with low sample sizes by multiplying the normalised
    rating by a confidence factor that reaches 1.0 at ``confidence_threshold``
    ratings.

    Args:
        ratings_average: Average rating on a 0–5 scale, or None if absent.
        ratings_count: Number of ratings contributing to the average.
        confidence_threshold: ratings_count at which full confidence is reached.

    Returns:
        Score in [0.0, 1.0]. Returns 0.0 for absent or non-positive ratings.
    """
    if ratings_average is None or ratings_average <= 0:
        return 0.0
    confidence = min(1.0, ratings_count / confidence_threshold)
    return (ratings_average / 5.0) * confidence


def score_rating_volume(
    ratings_count: int,
    min_ratings: int,
    max_ratings: int,
) -> float:
    """Compute min-max normalized ratings_count across the current batch.

    Args:
        ratings_count: Number of ratings for this book.
        min_ratings: Minimum ratings_count in the batch.
        max_ratings: Maximum ratings_count in the batch.

    Returns:
        Score in [0.0, 1.0]. Returns 0.5 when all books have identical counts.
    """
    if max_ratings == min_ratings:
        return 0.5
    return (ratings_count - min_ratings) / (max_ratings - min_ratings)


def compute_scores(
    books: list[BookRecord],
    total_genres_queried: int,
    weights: dict[str, float] | None = None,
    confidence_threshold: int = DEFAULT_RATING_CONFIDENCE_THRESHOLD,
) -> list[tuple[BookRecord, float, ScoreBreakdown]]:
    """Score all books and return sorted (book, composite_score, breakdown) tuples.

    Computes batch-level min/max statistics for normalization, then scores
    each book using the four component functions. Results are sorted by
    composite_score descending. The caller handles top_n truncation.

    Args:
        books: Book records to score. Must be non-empty.
        total_genres_queried: Used as the denominator for genre overlap scoring.
        weights: Scoring component weights. Defaults to DEFAULT_WEIGHTS if None.
            Keys: genre_overlap, reader_engagement, rating_quality, rating_volume.
        confidence_threshold: ratings_count at which full confidence is reached
            for the rating_quality component.

    Returns:
        List of (book, composite_score, ScoreBreakdown) tuples, sorted by
        composite_score descending.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS

    # Compute batch-level statistics for normalisation.
    readinglog_counts = [b.readinglog_count for b in books]
    ratings_counts = [b.ratings_count for b in books]
    min_readinglog = min(readinglog_counts)
    max_readinglog = max(readinglog_counts)
    min_ratings = min(ratings_counts)
    max_ratings = max(ratings_counts)

    results: list[tuple[BookRecord, float, ScoreBreakdown]] = []

    for book in books:
        go = score_genre_overlap(len(book.source_genres), total_genres_queried)
        re = score_reader_engagement(book.readinglog_count, min_readinglog, max_readinglog)
        rq = score_rating_quality(book.ratings_average, book.ratings_count, confidence_threshold)
        rv = score_rating_volume(book.ratings_count, min_ratings, max_ratings)

        composite = (
            w.get("genre_overlap", 0.0) * go
            + w.get("reader_engagement", 0.0) * re
            + w.get("rating_quality", 0.0) * rq
            + w.get("rating_volume", 0.0) * rv
        )

        breakdown = ScoreBreakdown(
            genre_overlap=go,
            reader_engagement=re,
            rating_quality=rq,
            rating_volume=rv,
        )
        results.append((book, composite, breakdown))

    results.sort(key=lambda t: t[1], reverse=True)
    return results
