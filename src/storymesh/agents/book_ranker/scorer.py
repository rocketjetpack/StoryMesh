"""Pure scoring functions for the BookRankerAgent.

All functions are stateless — they accept only the values they need and return
floats in [0.0, 1.0]. This separation from agent.py makes the scoring math
independently testable with exact float assertions.
"""

from __future__ import annotations

from storymesh.schemas.book_fetcher import BookRecord
from storymesh.schemas.book_ranker import RankedBookSummary, ScoreBreakdown

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


def _jaccard_similarity(genres_a: list[str], genres_b: list[str]) -> float:
    """Compute Jaccard similarity between two genre lists.

    Args:
        genres_a: First list of genre strings.
        genres_b: Second list of genre strings.

    Returns:
        Similarity score in [0.0, 1.0]. Returns 1.0 when both lists are empty
        (identical empty sets), 0.0 when one is empty and the other is not.
    """
    set_a, set_b = set(genres_a), set(genres_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def select_with_diversity(
    scored_books: list[tuple[BookRecord, float, ScoreBreakdown]],
    top_n: int,
    diversity_weight: float = 0.3,
) -> list[tuple[BookRecord, float, ScoreBreakdown]]:
    """Select top_n books balancing relevance against redundancy.

    Uses Maximal Marginal Relevance (MMR): each selection maximizes
    ``(1 - diversity_weight) * composite_score - diversity_weight * max_similarity_to_selected``.
    Similarity is Jaccard similarity over source_genres sets.

    When ``diversity_weight=0.0`` the result is identical to taking the first
    ``top_n`` entries from ``scored_books`` (pure relevance). The output is
    returned in MMR selection order so that downstream LLM agents see diverse
    genre traditions early in the list.

    Args:
        scored_books: All books with computed composite scores, sorted by
            composite_score descending. Must be non-empty if top_n > 0.
        top_n: Number of books to select.
        diversity_weight: 0.0 = pure relevance (current behavior),
            1.0 = maximum diversity.

    Returns:
        Selected books in MMR selection order (most marginal-relevant first).
    """
    if diversity_weight == 0.0 or top_n <= 0:
        return scored_books[:top_n]

    remaining = list(scored_books)
    selected: list[tuple[BookRecord, float, ScoreBreakdown]] = []

    while remaining and len(selected) < top_n:
        if not selected:
            # First selection is always the highest-scoring book.
            selected.append(remaining.pop(0))
            continue

        best_idx = 0
        best_mmr = float("-inf")

        for i, (book, score, _) in enumerate(remaining):
            max_sim = max(
                _jaccard_similarity(book.source_genres, sel[0].source_genres)
                for sel in selected
            )
            mmr = (1.0 - diversity_weight) * score - diversity_weight * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _summaries_from_scored(
    scored: list[tuple[BookRecord, float, ScoreBreakdown]],
) -> list[RankedBookSummary]:
    """Build RankedBookSummary list from MMR-ordered scored tuples.

    Rank reflects MMR selection order, not composite score order, so that
    downstream LLM agents weight diverse genre traditions by their position.

    Args:
        scored: MMR-ordered (book, composite_score, breakdown) tuples.

    Returns:
        List of RankedBookSummary with rank assigned by MMR order (1-indexed).
    """
    return [
        RankedBookSummary(
            work_key=book.work_key,
            title=book.title,
            authors=book.authors,
            first_publish_year=book.first_publish_year,
            source_genres=book.source_genres,
            composite_score=score,
            rank=idx + 1,
        )
        for idx, (book, score, _) in enumerate(scored)
    ]
