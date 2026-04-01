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


def _jaccard_similarity(subjects_a: list[str], subjects_b: list[str]) -> float:
    """Compute Jaccard similarity between two Open Library subject-tag lists.

    Comparison is case-insensitive. Both empty → 0.0 (no evidence of similarity).
    One empty → 0.0.

    Args:
        subjects_a: First list of subject tag strings.
        subjects_b: Second list of subject tag strings.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    set_a = {s.lower() for s in subjects_a}
    set_b = {s.lower() for s in subjects_b}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def select_diverse(
    scored: list[tuple[BookRecord, float, ScoreBreakdown]],
    top_n: int,
    mmr_lambda: float = 0.6,
    mmr_candidates: int = 30,
) -> list[tuple[BookRecord, float, ScoreBreakdown]]:
    """MMR-style diversity selection using Open Library subject tags.

    Scores each candidate with:
        mmr_lambda * composite_score - (1 - mmr_lambda) * max_similarity_to_selected

    Similarity is Jaccard similarity over the books' ``subjects`` lists (Open
    Library subject tags), not source_genres. Subject tags carry much richer
    thematic signal, so similar books are penalized more precisely.

    ``mmr_lambda=1.0`` degenerates to pure relevance (identical to simple
    truncation). ``mmr_lambda=0.0`` is pure diversity. The first selection is
    always the highest-scoring candidate regardless of lambda.

    Args:
        scored: (book, composite_score, breakdown) tuples sorted by
            composite_score descending. Must be non-empty if top_n > 0.
        top_n: Number of books to select.
        mmr_lambda: Relevance/diversity trade-off in [0.0, 1.0]. Default 0.6.
        mmr_candidates: Maximum number of top-scored books to consider as
            candidates. Limits quadratic similarity computation on large pools.

    Returns:
        Selected books in MMR selection order (most marginal-relevant first).
    """
    if top_n <= 0:
        return []

    candidates = scored[:mmr_candidates]

    if mmr_lambda == 1.0 or top_n >= len(candidates):
        return candidates[:top_n]

    remaining = list(candidates)
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
                _jaccard_similarity(book.subjects, sel[0].subjects)
                for sel in selected
            )
            mmr_score = mmr_lambda * score - (1.0 - mmr_lambda) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
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
