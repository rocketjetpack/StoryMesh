"""BookRankerAgent — Stage 2 of the StoryMesh pipeline.

Receives enriched BookRecord objects from the BookFetcherAgent, scores them
using a deterministic weighted composite, optionally invokes an LLM re-rank
for narrative potential assessment, and returns a dual-representation output:
full detail in artifacts, slim summaries for downstream LLM token efficiency.
"""

from __future__ import annotations

import logging
from typing import Any

from storymesh.agents.book_ranker.scorer import (
    DEFAULT_RATING_CONFIDENCE_THRESHOLD,
    DEFAULT_WEIGHTS,
    compute_scores,
    select_with_diversity,
)
from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.book_fetcher import BookRecord
from storymesh.schemas.book_ranker import (
    BookRankerAgentInput,
    BookRankerAgentOutput,
    RankedBook,
    RankedBookSummary,
    ScoreBreakdown,
)

logger = logging.getLogger(__name__)


class BookRankerAgent:
    """Ranks books by composite scoring with optional LLM re-ranking (Stage 2).

    Deterministic scoring runs first, applying weighted combination of four
    signals: genre overlap, reader engagement, rating quality, and rating volume.
    If ``llm_rerank=True`` and a valid ``llm_client`` is provided, the top_n
    shortlist is passed to an LLM which re-orders it by narrative potential for
    the user's creative brief. LLM failures fall back gracefully to the
    deterministic order.
    """

    def __init__(
        self,
        *,
        top_n: int = 10,
        weights: dict[str, float] | None = None,
        rating_confidence_threshold: int = DEFAULT_RATING_CONFIDENCE_THRESHOLD,
        llm_rerank: bool = False,
        llm_client: LLMClient | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        diversity_weight: float = 0.0,
    ) -> None:
        """Construct the agent.

        Args:
            top_n: Maximum number of books to return after scoring.
            weights: Scoring component weights. If None, DEFAULT_WEIGHTS is used.
                Keys: genre_overlap, reader_engagement, rating_quality, rating_volume.
            rating_confidence_threshold: ratings_count at which full confidence
                is granted for the rating_quality score component.
            llm_rerank: Whether to invoke the LLM re-rank pass after deterministic
                scoring. Requires a valid ``llm_client``.
            llm_client: LLM client instance for re-ranking. If None when
                ``llm_rerank=True``, a warning is logged and the agent falls back
                to deterministic ordering.
            temperature: LLM temperature for the re-rank call. Default 0.0 for
                deterministic output.
            max_tokens: Maximum tokens for the re-rank LLM call.
            diversity_weight: MMR diversity weight in [0.0, 1.0]. 0.0 preserves
                pure relevance ordering (default, backward-compatible). Higher
                values penalize genre-redundant books, ensuring the shortlist
                covers the thematic space for ThemeExtractorAgent.
        """
        self._top_n = top_n
        self._weights = weights
        self._rating_confidence_threshold = rating_confidence_threshold
        self._llm_rerank = llm_rerank
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._diversity_weight = diversity_weight

        if llm_rerank and llm_client is None:
            logger.warning(
                "BookRankerAgent: llm_rerank=True but no llm_client provided. "
                "Falling back to deterministic ordering."
            )

        # Load the prompt eagerly so misconfiguration is caught at construction.
        self._prompt_template = load_prompt("book_ranker") if llm_rerank else None

    def run(self, input_data: BookRankerAgentInput) -> BookRankerAgentOutput:
        """Score books and return a ranked output with dual representations.

        Args:
            input_data: Validated input from the BookFetcherAgent node.

        Returns:
            A frozen BookRankerAgentOutput with ranked_books (full detail) and
            ranked_summaries (slim), plus scoring metadata in the debug dict.
        """
        logger.info(
            "BookRankerAgent starting | books=%d total_genres_queried=%d top_n=%d",
            len(input_data.books),
            input_data.total_genres_queried,
            self._top_n,
        )

        # ── Deterministic scoring pass ─────────────────────────────────────
        scored = compute_scores(
            books=input_data.books,
            total_genres_queried=input_data.total_genres_queried,
            weights=self._weights,
            confidence_threshold=self._rating_confidence_threshold,
        )

        # ── MMR diversity selection pass ───────────────────────────────────
        top_scored = select_with_diversity(
            scored_books=scored,
            top_n=self._top_n,
            diversity_weight=self._diversity_weight,
        )
        dropped_count = len(scored) - len(top_scored)
        selection_order = [book.work_key for book, _, _ in top_scored]

        # ── Optional LLM re-rank pass ──────────────────────────────────────
        llm_reranked = False
        llm_debug: dict[str, Any] = {}

        if self._llm_rerank and self._llm_client is not None:
            top_scored, llm_reranked, llm_debug = self._apply_llm_rerank(
                top_scored, input_data.user_prompt
            )

        # ── Build output ───────────────────────────────────────────────────
        ranked_books: list[RankedBook] = [
            RankedBook(
                book=book,
                composite_score=score,
                score_breakdown=breakdown,
                rank=idx + 1,
            )
            for idx, (book, score, breakdown) in enumerate(top_scored)
        ]

        ranked_summaries: list[RankedBookSummary] = [
            self._to_summary(rb) for rb in ranked_books
        ]

        weights_used = self._weights if self._weights is not None else DEFAULT_WEIGHTS
        debug: dict[str, Any] = {
            "weights_used": weights_used,
            "total_scored": len(scored),
            "top_n": self._top_n,
            "dropped_count": dropped_count,
            "diversity_weight": self._diversity_weight,
            "diversity_applied": self._diversity_weight > 0.0,
            "selection_order": selection_order,
        }
        if llm_debug:
            debug["llm_rerank"] = llm_debug

        logger.info(
            "BookRankerAgent complete | returned=%d dropped=%d llm_reranked=%s",
            len(ranked_books),
            dropped_count,
            llm_reranked,
        )

        return BookRankerAgentOutput(
            ranked_books=ranked_books,
            ranked_summaries=ranked_summaries,
            dropped_count=dropped_count,
            llm_reranked=llm_reranked,
            debug=debug,
        )

    def _apply_llm_rerank(
        self,
        top_scored: list[tuple[BookRecord, float, ScoreBreakdown]],
        user_prompt: str,
    ) -> tuple[list[tuple[BookRecord, float, ScoreBreakdown]], bool, dict[str, Any]]:
        """Invoke the LLM re-rank pass and reorder top_scored accordingly.

        If the LLM call fails or returns unresolvable data, logs a warning and
        returns the original deterministic order.

        Args:
            top_scored: Deterministically ranked (book, score, breakdown) tuples.
            user_prompt: Original user creative brief.

        Returns:
            (reordered_list, llm_reranked_flag, debug_dict)
        """
        assert self._llm_client is not None  # Caller guarantees this.
        assert self._prompt_template is not None

        book_lines = "\n".join(
            f"{i + 1}. {b.title} by {', '.join(b.authors) or 'Unknown'} "
            f"[{b.work_key}] (genres: {', '.join(b.source_genres)})"
            for i, (b, _, _) in enumerate(top_scored)
        )
        formatted_user = self._prompt_template.format_user(
            user_prompt=user_prompt,
            book_list=book_lines,
            count=len(top_scored),
        )

        try:
            response = self._llm_client.complete_json(
                formatted_user,
                system_prompt=self._prompt_template.system,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            llm_keys: list[str] = response.get("ranked_work_keys", [])
        except Exception:
            logger.warning(
                "BookRankerAgent: LLM re-rank call failed. "
                "Keeping deterministic order.",
                exc_info=True,
            )
            return top_scored, False, {"error": "llm_call_failed"}

        # Build a lookup from work_key → scored tuple.
        key_to_tuple: dict[str, tuple[BookRecord, float, ScoreBreakdown]] = {
            book.work_key: (book, score, breakdown)
            for book, score, breakdown in top_scored
        }

        # Validate: every key from the LLM must exist in our set.
        our_keys = set(key_to_tuple.keys())
        llm_key_set = set(llm_keys)

        if not llm_key_set or not llm_key_set.issubset(our_keys):
            logger.warning(
                "BookRankerAgent: LLM returned unrecognized or incomplete work_keys. "
                "Keeping deterministic order. llm_keys=%s our_keys=%s",
                llm_keys,
                our_keys,
            )
            return top_scored, False, {"error": "invalid_llm_keys", "llm_keys": llm_keys}

        # Reorder per LLM preference; append any books the LLM omitted at the end.
        reordered = [key_to_tuple[k] for k in llm_keys if k in key_to_tuple]
        omitted_keys = our_keys - llm_key_set
        if omitted_keys:
            reordered.extend(key_to_tuple[k] for k in omitted_keys)

        return reordered, True, {"llm_ranked_work_keys": llm_keys}

    @staticmethod
    def _to_summary(ranked: RankedBook) -> RankedBookSummary:
        """Project a RankedBook to its slim RankedBookSummary representation.

        Args:
            ranked: Full-detail ranked book.

        Returns:
            Slim summary with only the fields needed by downstream LLM agents.
        """
        return RankedBookSummary(
            work_key=ranked.book.work_key,
            title=ranked.book.title,
            authors=ranked.book.authors,
            first_publish_year=ranked.book.first_publish_year,
            source_genres=ranked.book.source_genres,
            composite_score=ranked.composite_score,
            rank=ranked.rank,
        )
