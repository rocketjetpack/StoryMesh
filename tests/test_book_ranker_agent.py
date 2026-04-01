"""Unit tests for storymesh.agents.book_ranker.agent."""

from __future__ import annotations

import json
import logging

import pytest

from storymesh.agents.book_ranker.agent import BookRankerAgent
from storymesh.llm.base import FakeLLMClient
from storymesh.schemas.book_fetcher import BookRecord
from storymesh.schemas.book_ranker import (
    BookRankerAgentInput,
    BookRankerAgentOutput,
    RankedBook,
    RankedBookSummary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _book(work_key: str = "/works/OL1W", **overrides: object) -> BookRecord:
    defaults: dict[str, object] = dict(
        work_key=work_key,
        title=f"Book {work_key}",
        source_genres=["mystery"],
        readinglog_count=100,
        ratings_count=100,
        ratings_average=4.0,
    )
    return BookRecord(**(defaults | overrides))


def _input(books: list[BookRecord], user_prompt: str = "dark mystery") -> BookRankerAgentInput:
    return BookRankerAgentInput(
        books=books,
        user_prompt=user_prompt,
        total_genres_queried=len({g for b in books for g in b.source_genres}),
    )


def _agent(**kwargs: object) -> BookRankerAgent:
    return BookRankerAgent(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        agent = _agent()
        assert agent is not None

    def test_llm_rerank_false_no_client_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="storymesh.agents.book_ranker.agent"):
            _agent(llm_rerank=False, llm_client=None)
        assert not any("Falling back" in r.message for r in caplog.records)

    def test_llm_rerank_true_no_client_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="storymesh.agents.book_ranker.agent"):
            _agent(llm_rerank=True, llm_client=None)
        assert any("Falling back" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Basic ranking
# ---------------------------------------------------------------------------


class TestBasicRanking:
    def test_returns_book_ranker_output_type(self) -> None:
        books = [_book("/works/OL1W")]
        output = _agent().run(_input(books))
        assert isinstance(output, BookRankerAgentOutput)

    def test_ranked_books_match_input_count(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(5)]
        output = _agent().run(_input(books))
        assert len(output.ranked_books) == 5

    def test_ranks_are_one_indexed(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(3)]
        output = _agent().run(_input(books))
        ranks = [rb.rank for rb in output.ranked_books]
        assert ranks == [1, 2, 3]

    def test_ranked_books_are_ranked_book_instances(self) -> None:
        books = [_book("/works/OL1W")]
        output = _agent().run(_input(books))
        assert all(isinstance(rb, RankedBook) for rb in output.ranked_books)

    def test_ranked_summaries_are_ranked_book_summary_instances(self) -> None:
        books = [_book("/works/OL1W")]
        output = _agent().run(_input(books))
        assert all(isinstance(rs, RankedBookSummary) for rs in output.ranked_summaries)

    def test_higher_score_book_ranked_first(self) -> None:
        # Book A: all signals high
        book_a = _book(
            "/works/OLA",
            readinglog_count=1000,
            ratings_count=1000,
            ratings_average=5.0,
            source_genres=["mystery", "thriller"],
        )
        # Book B: all signals low
        book_b = _book(
            "/works/OLB",
            readinglog_count=0,
            ratings_count=0,
            ratings_average=None,
            source_genres=["mystery"],
        )
        output = _agent().run(
            BookRankerAgentInput(
                books=[book_a, book_b],
                user_prompt="dark mystery thriller",
                total_genres_queried=2,
            )
        )
        assert output.ranked_books[0].book.work_key == "/works/OLA"
        assert output.ranked_books[1].book.work_key == "/works/OLB"


# ---------------------------------------------------------------------------
# top_n truncation
# ---------------------------------------------------------------------------


class TestTopNTruncation:
    def test_top_n_limits_output(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(20)]
        output = _agent(top_n=5).run(_input(books))
        assert len(output.ranked_books) == 5
        assert len(output.ranked_summaries) == 5

    def test_dropped_count_correct(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(20)]
        output = _agent(top_n=5).run(_input(books))
        assert output.dropped_count == 15

    def test_no_truncation_when_under_top_n(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(3)]
        output = _agent(top_n=10).run(_input(books))
        assert len(output.ranked_books) == 3
        assert output.dropped_count == 0

    def test_single_book_input(self) -> None:
        books = [_book("/works/OL1W")]
        output = _agent().run(_input(books))
        assert len(output.ranked_books) == 1
        assert output.dropped_count == 0


# ---------------------------------------------------------------------------
# Summaries match ranked_books
# ---------------------------------------------------------------------------


class TestSummariesMatchRankedBooks:
    def test_summaries_and_books_same_length(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(5)]
        output = _agent().run(_input(books))
        assert len(output.ranked_summaries) == len(output.ranked_books)

    def test_summary_work_keys_match_books(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(5)]
        output = _agent().run(_input(books))
        book_keys = [rb.book.work_key for rb in output.ranked_books]
        summary_keys = [rs.work_key for rs in output.ranked_summaries]
        assert book_keys == summary_keys

    def test_summary_composite_scores_match_books(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(5)]
        output = _agent().run(_input(books))
        for rb, rs in zip(output.ranked_books, output.ranked_summaries, strict=True):
            assert rb.composite_score == rs.composite_score
            assert rb.rank == rs.rank


# ---------------------------------------------------------------------------
# LLM re-rank disabled
# ---------------------------------------------------------------------------


class TestLLMRerankedFalse:
    def test_llm_reranked_false_by_default(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(3)]
        output = _agent().run(_input(books))
        assert output.llm_reranked is False

    def test_llm_rerank_true_no_client_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(3)]
        agent = _agent(llm_rerank=True, llm_client=None)
        output = agent.run(_input(books))
        assert output.llm_reranked is False


# ---------------------------------------------------------------------------
# LLM re-rank enabled (mock client)
# ---------------------------------------------------------------------------


class TestLLMRerank:
    def _make_books(self) -> list[BookRecord]:
        return [
            _book("/works/OL1W"),
            _book("/works/OL2W"),
            _book("/works/OL3W"),
        ]

    def test_llm_rerank_reorders_books(self) -> None:
        books = self._make_books()
        # LLM prefers OL3W, OL1W, OL2W
        fake_llm = FakeLLMClient(
            responses=['{"ranked_work_keys": ["/works/OL3W", "/works/OL1W", "/works/OL2W"]}']
        )
        agent = _agent(llm_rerank=True, llm_client=fake_llm)
        output = agent.run(_input(books))
        assert output.llm_reranked is True
        assert output.ranked_books[0].book.work_key == "/works/OL3W"
        assert output.ranked_books[1].book.work_key == "/works/OL1W"
        assert output.ranked_books[2].book.work_key == "/works/OL2W"

    def test_llm_rerank_bad_json_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        books = self._make_books()
        fake_llm = FakeLLMClient(responses=["not valid json {{{"])
        agent = _agent(llm_rerank=True, llm_client=fake_llm)
        with caplog.at_level(logging.WARNING, logger="storymesh.agents.book_ranker.agent"):
            output = agent.run(_input(books))
        assert output.llm_reranked is False

    def test_llm_rerank_unknown_keys_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        books = self._make_books()
        fake_llm = FakeLLMClient(
            responses=['{"ranked_work_keys": ["/works/UNKNOWN1", "/works/UNKNOWN2"]}']
        )
        agent = _agent(llm_rerank=True, llm_client=fake_llm)
        with caplog.at_level(logging.WARNING, logger="storymesh.agents.book_ranker.agent"):
            output = agent.run(_input(books))
        assert output.llm_reranked is False

    def test_llm_reranked_sets_flag_in_output(self) -> None:
        books = self._make_books()
        work_keys = [b.work_key for b in books]
        fake_llm = FakeLLMClient(
            responses=[json.dumps({"ranked_work_keys": work_keys})]
        )
        agent = _agent(llm_rerank=True, llm_client=fake_llm)
        output = agent.run(_input(books))
        assert output.llm_reranked is True


# ---------------------------------------------------------------------------
# Debug dict
# ---------------------------------------------------------------------------


class TestDebugDict:
    def test_debug_contains_weights_used(self) -> None:
        books = [_book("/works/OL1W")]
        output = _agent().run(_input(books))
        assert "weights_used" in output.debug

    def test_debug_contains_total_scored(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(7)]
        output = _agent(top_n=3).run(_input(books))
        assert output.debug["total_scored"] == 7

    def test_debug_contains_dropped_count(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(7)]
        output = _agent(top_n=3).run(_input(books))
        assert output.debug["dropped_count"] == 4


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_info_logged_at_start(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        books = [_book("/works/OL1W")]
        with caplog.at_level(logging.INFO, logger="storymesh.agents.book_ranker.agent"):
            _agent().run(_input(books))
        assert any("BookRankerAgent starting" in r.message for r in caplog.records)

    def test_info_logged_at_completion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        books = [_book("/works/OL1W")]
        with caplog.at_level(logging.INFO, logger="storymesh.agents.book_ranker.agent"):
            _agent().run(_input(books))
        assert any("BookRankerAgent complete" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# MMR diversity selection
# ---------------------------------------------------------------------------


class TestBookRankerAgentMMR:
    def test_mmr_lambda_from_constructor(self) -> None:
        """Agent stores the mmr_lambda passed to the constructor."""
        agent = BookRankerAgent(mmr_lambda=0.4)
        assert agent._mmr_lambda == pytest.approx(0.4)

    def test_mmr_candidates_from_constructor(self) -> None:
        """Agent stores the mmr_candidates passed to the constructor."""
        agent = BookRankerAgent(mmr_candidates=20)
        assert agent._mmr_candidates == 20

    def test_default_mmr_lambda(self) -> None:
        """Default mmr_lambda is 0.6."""
        agent = BookRankerAgent()
        assert agent._mmr_lambda == pytest.approx(0.6)

    def test_default_mmr_candidates(self) -> None:
        """Default mmr_candidates is 30."""
        agent = BookRankerAgent()
        assert agent._mmr_candidates == 30

    def test_debug_records_mmr_metadata(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(3)]
        output = _agent(mmr_lambda=0.4).run(_input(books))
        assert "mmr" in output.debug
        assert output.debug["mmr"]["lambda"] == pytest.approx(0.4)
        assert "candidates_considered" in output.debug["mmr"]
        assert "selection_order" in output.debug["mmr"]

    def test_debug_selection_order_length(self) -> None:
        books = [_book(f"/works/OL{i}W") for i in range(3)]
        output = _agent(mmr_lambda=0.6).run(_input(books))
        assert len(output.debug["mmr"]["selection_order"]) == len(output.ranked_books)

    def test_mmr_promotes_diverse_subjects(self) -> None:
        """With low mmr_lambda, a book with unique subject tags is promoted
        over subject-redundant higher-scoring books."""
        mystery_books = [
            _book(f"/works/OLM{i}W", subjects=["Mystery", "Crime", "Victorian"])
            for i in range(5)
        ]
        unique_book = _book("/works/OLUNIQ", subjects=["Fantasy", "Epic", "Dragons"])
        all_books = mystery_books + [unique_book]
        output = _agent(top_n=3, mmr_lambda=0.1).run(_input(all_books))
        selected_keys = {rb.book.work_key for rb in output.ranked_books}
        assert "/works/OLUNIQ" in selected_keys
