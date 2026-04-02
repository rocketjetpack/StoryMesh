"""Unit tests for storymesh.agents.book_fetcher.agent.

All tests inject a mock OpenLibraryClient and use tmp_path for the diskcache
directory so no real HTTP calls or config loading are required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from storymesh.agents.book_fetcher.agent import BookFetcherAgent
from storymesh.agents.book_fetcher.client import OpenLibraryAPIError
from storymesh.schemas.book_fetcher import BookFetcherAgentInput, BookRecord

# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------

class _StubClient:
    """Minimal OpenLibraryClient stub for unit tests."""

    def __init__(
        self,
        responses: dict[str, list[dict[str, Any]]] | None = None,
        rate_limit_delay: float = 0.4,
        subject_info_responses: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._responses = responses or {}
        self.rate_limit_delay = rate_limit_delay
        self.fetch_calls: list[str] = []
        # Full (subject, sort) pair log for multi-sort assertions.
        self.fetch_call_args: list[tuple[str, str]] = []
        self.subject_info_responses: dict[str, dict[str, Any]] = subject_info_responses or {}
        self.subject_info_calls: list[str] = []

    def fetch_books_by_subject(
        self,
        subject: str,
        limit: int = 30,
        sort: str = "editions",
    ) -> list[dict[str, Any]]:
        self.fetch_calls.append(subject)
        self.fetch_call_args.append((subject, sort))
        return self._responses.get(subject, [])

    def fetch_subject_info(self, subject: str) -> dict[str, Any]:
        self.subject_info_calls.append(subject)
        return self.subject_info_responses.get(subject, {"work_count": 1})

    def close(self) -> None:
        pass


class _FailingClient(_StubClient):
    """Stub that raises OpenLibraryAPIError for specified subjects."""

    def __init__(
        self,
        fail_on: set[str],
        responses: dict[str, list[dict[str, Any]]] | None = None,
        rate_limit_delay: float = 0.4,
    ) -> None:
        super().__init__(responses=responses, rate_limit_delay=rate_limit_delay)
        self._fail_on = fail_on

    def fetch_books_by_subject(
        self,
        subject: str,
        limit: int = 30,
        sort: str = "editions",
    ) -> list[dict[str, Any]]:
        if subject in self._fail_on:
            msg = f"Simulated API failure for '{subject}'"
            raise OpenLibraryAPIError(msg)
        return super().fetch_books_by_subject(subject, limit, sort)


_MYSTERY_DOC: dict[str, Any] = {
    "key": "/works/OL1W",
    "title": "Sherlock Holmes",
    "author_name": ["Arthur Conan Doyle"],
    "edition_count": 50,
    "ratings_average": 4.5,
    "ratings_count": 1000,
    "readinglog_count": 500,
    "want_to_read_count": 200,
    "already_read_count": 250,
    "currently_reading_count": 50,
    "number_of_pages_median": 307,
    "subject": ["Mystery", "Victorian"],
    "cover_i": 123,
    "first_publish_year": 1892,
}

_FANTASY_DOC: dict[str, Any] = {
    "key": "/works/OL2W",
    "title": "The Hobbit",
    "author_name": ["J. R. R. Tolkien"],
    "edition_count": 200,
}

# A doc whose work_key appears in both mystery and fantasy responses (cross-genre book).
_CROSSGENRE_DOC: dict[str, Any] = {
    "key": "/works/OL99W",
    "title": "The Name of the Rose",
    "author_name": ["Umberto Eco"],
    "edition_count": 75,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub_client() -> _StubClient:
    return _StubClient(
        responses={
            "mystery": [_MYSTERY_DOC],
            "fantasy": [_FANTASY_DOC],
        }
    )


@pytest.fixture()
def agent(tmp_path: Path, stub_client: _StubClient) -> BookFetcherAgent:
    with patch(
        "storymesh.agents.book_fetcher.agent.get_cache_dir",
        return_value=tmp_path,
    ):
        return BookFetcherAgent(client=stub_client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construct_with_client(self, tmp_path: Path) -> None:
        stub = _StubClient()
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        assert a is not None

    def test_default_cache_ttl(self, tmp_path: Path) -> None:
        stub = _StubClient()
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        assert a._cache_ttl == 86400

    def test_custom_cache_ttl(self, tmp_path: Path) -> None:
        stub = _StubClient()
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub, cache_ttl=3600)  # type: ignore[arg-type]
        assert a._cache_ttl == 3600


# ---------------------------------------------------------------------------
# Genre name conversion
# ---------------------------------------------------------------------------

class TestGenreConversion:
    def test_underscores_replaced_with_spaces(self, tmp_path: Path) -> None:
        stub = _StubClient(responses={"post apocalyptic": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        a.run(BookFetcherAgentInput(normalized_genres=["post_apocalyptic"]))

        assert stub.fetch_calls == ["post apocalyptic"]

    def test_genre_lowercased(self, tmp_path: Path) -> None:
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        a.run(BookFetcherAgentInput(normalized_genres=["Mystery"]))

        assert stub.fetch_calls == ["mystery"]

    def test_genre_already_clean_unchanged(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert stub_client.fetch_calls == ["mystery"]


# ---------------------------------------------------------------------------
# Single-genre results
# ---------------------------------------------------------------------------

class TestSingleGenre:
    def test_returns_book_records(self, agent: BookFetcherAgent) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert len(output.books) == 1
        assert isinstance(output.books[0], BookRecord)

    def test_book_record_fields_populated(self, agent: BookFetcherAgent) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        record = output.books[0]
        assert record.work_key == "/works/OL1W"
        assert record.title == "Sherlock Holmes"
        assert record.authors == ["Arthur Conan Doyle"]
        assert record.edition_count == 50
        assert record.ratings_average == 4.5
        assert record.first_publish_year == 1892
        assert "Mystery" in record.subjects
        assert record.cover_id == 123
        assert record.readinglog_count == 500
        assert record.want_to_read_count == 200
        assert record.already_read_count == 250
        assert record.currently_reading_count == 50
        assert record.number_of_pages_median == 307

    def test_source_genres_set_correctly(self, agent: BookFetcherAgent) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.books[0].source_genres == ["mystery"]

    def test_queries_executed_populated(self, agent: BookFetcherAgent) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.queries_executed == ["mystery"]

    def test_empty_api_response_returns_empty_books(self, tmp_path: Path) -> None:
        stub = _StubClient(responses={"mystery": []})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.books == []
        assert output.queries_executed == ["mystery"]


# ---------------------------------------------------------------------------
# Multi-genre results
# ---------------------------------------------------------------------------

class TestMultiGenre:
    def test_two_unique_genres_combined(self, agent: BookFetcherAgent) -> None:
        output = agent.run(
            BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
        )
        assert len(output.books) == 2

    def test_source_genres_per_unique_book(self, agent: BookFetcherAgent) -> None:
        output = agent.run(
            BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
        )
        by_key = {r.work_key: r for r in output.books}
        assert by_key["/works/OL1W"].source_genres == ["mystery"]
        assert by_key["/works/OL2W"].source_genres == ["fantasy"]

    def test_queries_executed_contains_all_subjects(
        self, agent: BookFetcherAgent
    ) -> None:
        output = agent.run(
            BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
        )
        assert output.queries_executed == ["mystery", "fantasy"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_work_key_in_two_genres_merged(self, tmp_path: Path) -> None:
        stub = _StubClient(
            responses={
                "mystery": [_MYSTERY_DOC, _CROSSGENRE_DOC],
                "fantasy": [_FANTASY_DOC, _CROSSGENRE_DOC],
            }
        )
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )

        work_keys = [r.work_key for r in output.books]
        assert len(work_keys) == len(set(work_keys)), "Duplicate work_keys found"

    def test_merged_record_has_both_source_genres(self, tmp_path: Path) -> None:
        stub = _StubClient(
            responses={
                "mystery": [_CROSSGENRE_DOC],
                "fantasy": [_CROSSGENRE_DOC],
            }
        )
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )

        assert len(output.books) == 1
        assert set(output.books[0].source_genres) == {"mystery", "fantasy"}

    def test_unique_books_not_affected(self, agent: BookFetcherAgent) -> None:
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = agent.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )
        assert len(output.books) == 2
        for record in output.books:
            assert len(record.source_genres) == 1


# ---------------------------------------------------------------------------
# Debug dict
# ---------------------------------------------------------------------------

class TestDebugDict:
    def test_debug_per_genre_keys_match_queries_executed(
        self, agent: BookFetcherAgent
    ) -> None:
        output = agent.run(
            BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
        )
        assert set(output.debug["per_genre"].keys()) == set(output.queries_executed)

    def test_debug_total_raw_correct(self, agent: BookFetcherAgent) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.debug["total_raw"] == 1

    def test_debug_after_dedup_with_no_duplicates(
        self, agent: BookFetcherAgent
    ) -> None:
        output = agent.run(
            BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
        )
        assert output.debug["total_raw"] == 2
        assert output.debug["total_after_dedup"] == 2
        assert output.debug["duplicates_merged"] == 0

    def test_debug_after_dedup_with_duplicates(self, tmp_path: Path) -> None:
        stub = _StubClient(
            responses={
                "mystery": [_CROSSGENRE_DOC],
                "fantasy": [_CROSSGENRE_DOC],
            }
        )
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )

        assert output.debug["total_raw"] == 2
        assert output.debug["total_after_dedup"] == 1
        assert output.debug["duplicates_merged"] == 1

    def test_debug_per_genre_contains_cache_status(
        self, agent: BookFetcherAgent
    ) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        genre_debug = output.debug["per_genre"]["mystery"]
        assert genre_debug["cache"] in ("hit", "miss")

    def test_debug_first_call_is_cache_miss(self, agent: BookFetcherAgent) -> None:
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.debug["per_genre"]["mystery"]["cache"] == "miss"

    def test_debug_second_call_is_cache_hit(self, agent: BookFetcherAgent) -> None:
        agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.debug["per_genre"]["mystery"]["cache"] == "hit"


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_hit_skips_client(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        input_data = BookFetcherAgentInput(normalized_genres=["mystery"])
        agent.run(input_data)
        agent.run(input_data)
        assert stub_client.fetch_calls.count("mystery") == 1

    def test_cache_miss_calls_client(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert "mystery" in stub_client.fetch_calls

    def test_different_genres_cached_independently(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        agent.run(BookFetcherAgentInput(normalized_genres=["fantasy"]))
        assert stub_client.fetch_calls.count("mystery") == 1
        assert stub_client.fetch_calls.count("fantasy") == 1

    def test_second_run_uses_cache(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        first = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        second = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert first.books == second.books
        assert stub_client.fetch_calls.count("mystery") == 1


# ---------------------------------------------------------------------------
# Doc parsing
# ---------------------------------------------------------------------------

class TestDocParsing:
    def test_doc_missing_work_key_is_skipped(self, tmp_path: Path) -> None:
        bad_doc: dict[str, Any] = {"title": "No Key Book"}
        stub = _StubClient(responses={"mystery": [bad_doc]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.books == []

    def test_doc_missing_title_is_skipped(self, tmp_path: Path) -> None:
        bad_doc: dict[str, Any] = {"key": "/works/OL99W"}
        stub = _StubClient(responses={"mystery": [bad_doc]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.books == []

    def test_optional_fields_default_when_absent(self, tmp_path: Path) -> None:
        minimal_doc: dict[str, Any] = {"key": "/works/OL99W", "title": "Minimal Book"}
        stub = _StubClient(responses={"mystery": [minimal_doc]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        record = output.books[0]
        assert record.authors == []
        assert record.first_publish_year is None
        assert record.edition_count == 0
        assert record.ratings_average is None
        assert record.ratings_count == 0
        assert record.readinglog_count == 0
        assert record.want_to_read_count == 0
        assert record.already_read_count == 0
        assert record.currently_reading_count == 0
        assert record.number_of_pages_median is None
        assert record.subjects == []
        assert record.cover_id is None

    def test_mixed_valid_and_invalid_docs(self, tmp_path: Path) -> None:
        docs: list[dict[str, Any]] = [
            {"key": "/works/OL1W", "title": "Valid Book"},
            {"title": "No Key"},
            {"key": "/works/OL3W"},
        ]
        stub = _StubClient(responses={"mystery": docs})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]
        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert len(output.books) == 1
        assert output.books[0].title == "Valid Book"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_sleep_called_between_genres(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        with patch("storymesh.agents.book_fetcher.agent.time.sleep") as mock_sleep:
            agent.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )
        mock_sleep.assert_called_once_with(stub_client.rate_limit_delay)

    def test_no_sleep_for_single_genre(self, agent: BookFetcherAgent) -> None:
        with patch("storymesh.agents.book_fetcher.agent.time.sleep") as mock_sleep:
            agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        mock_sleep.assert_not_called()

    def test_no_sleep_on_cache_hit(
        self, agent: BookFetcherAgent, stub_client: _StubClient
    ) -> None:
        input_data = BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
        agent.run(input_data)  # populates cache, sleeps once

        with patch("storymesh.agents.book_fetcher.agent.time.sleep") as mock_sleep:
            agent.run(input_data)  # all cache hits — no sleep

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_info_logged_at_entry(
        self, agent: BookFetcherAgent, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="storymesh.agents.book_fetcher.agent"):
            agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert any("BookFetcherAgent starting" in r.message for r in caplog.records)

    def test_info_logged_per_genre(
        self, agent: BookFetcherAgent, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="storymesh.agents.book_fetcher.agent"):
            agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert any("mystery" in r.message for r in caplog.records)

    def test_info_logged_at_completion(
        self, agent: BookFetcherAgent, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="storymesh.agents.book_fetcher.agent"):
            agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert any("BookFetcherAgent complete" in r.message for r in caplog.records)

    def test_info_logged_for_each_genre(
        self, agent: BookFetcherAgent, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"), caplog.at_level(
            logging.INFO, logger="storymesh.agents.book_fetcher.agent"
        ):
            agent.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )
        genre_logs = [
            r for r in caplog.records
            if "mystery" in r.message or "fantasy" in r.message
        ]
        subjects_logged = {
            s for r in genre_logs for s in ("mystery", "fantasy") if s in r.message
        }
        assert "mystery" in subjects_logged
        assert "fantasy" in subjects_logged


# ---------------------------------------------------------------------------
# API error graceful degradation
# ---------------------------------------------------------------------------

class TestAPIErrorGracefulDegradation:
    def test_single_genre_api_failure_returns_empty(self, tmp_path: Path) -> None:
        """When the only genre's API call fails, return empty books list."""
        stub = _FailingClient(fail_on={"mystery"})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.books == []
        assert output.debug["per_genre"]["mystery"]["cache"] == "error"
        assert output.queries_executed == ["mystery"]

    def test_partial_failure_returns_successful_genres(
        self, tmp_path: Path
    ) -> None:
        """When one genre fails but another succeeds, return partial results."""
        stub = _FailingClient(
            fail_on={"fantasy"},
            responses={"mystery": [_MYSTERY_DOC]},
        )
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(
                BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"])
            )

        assert len(output.books) == 1
        assert output.books[0].title == "Sherlock Holmes"
        assert output.debug["per_genre"]["mystery"]["cache"] == "miss"
        assert output.debug["per_genre"]["fantasy"]["cache"] == "error"

    def test_api_failure_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """API failures are logged at WARNING level (no traceback)."""
        stub = _FailingClient(fail_on={"mystery"})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        with caplog.at_level(logging.WARNING, logger="storymesh.agents.book_fetcher.agent"):
            a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))

        assert any("API call failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# max_books truncation
# ---------------------------------------------------------------------------


def _make_doc(work_key: str, title: str, edition_count: int = 1) -> dict[str, Any]:
    """Build a minimal Open Library doc dict for testing."""
    return {"key": work_key, "title": title, "edition_count": edition_count}


class TestMaxBooks:
    def test_truncates_to_max_books(self, tmp_path: Path) -> None:
        """Output is capped at max_books when more books are fetched."""
        docs = [_make_doc(f"/works/OL{i}W", f"Book {i}", edition_count=i) for i in range(1, 21)]
        stub = _StubClient(responses={"fantasy": docs})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub, max_books=5)  # type: ignore[arg-type]

        output = a.run(BookFetcherAgentInput(normalized_genres=["fantasy"]))

        assert len(output.books) == 5

    def test_no_truncation_when_under_limit(self, tmp_path: Path) -> None:
        """All books are returned when the total is within the max_books limit."""
        docs = [_make_doc(f"/works/OL{i}W", f"Book {i}") for i in range(1, 4)]
        stub = _StubClient(responses={"fantasy": docs})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub, max_books=10)  # type: ignore[arg-type]

        output = a.run(BookFetcherAgentInput(normalized_genres=["fantasy"]))

        assert len(output.books) == 3

    def test_truncation_prioritises_cross_genre_books(self, tmp_path: Path) -> None:
        """Books found under more genre queries are retained over single-genre books."""
        mystery_and_fantasy = _make_doc("/works/OLXW", "Cross-genre Book", edition_count=1)
        fantasy_only = [
            _make_doc(f"/works/OL{i}W", f"Fantasy Only {i}", edition_count=100)
            for i in range(1, 10)
        ]
        stub = _StubClient(
            responses={
                "mystery": [mystery_and_fantasy],
                "fantasy": [mystery_and_fantasy, *fantasy_only],
            }
        )
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub, max_books=3)  # type: ignore[arg-type]

        output = a.run(BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"]))

        retained_keys = {b.work_key for b in output.books}
        assert "/works/OLXW" in retained_keys

    def test_debug_records_truncation_flag(self, tmp_path: Path) -> None:
        """debug dict reports max_books_applied=True when truncation occurred."""
        docs = [_make_doc(f"/works/OL{i}W", f"Book {i}") for i in range(1, 11)]
        stub = _StubClient(responses={"fantasy": docs})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub, max_books=3)  # type: ignore[arg-type]

        output = a.run(BookFetcherAgentInput(normalized_genres=["fantasy"]))

        assert output.debug["max_books_applied"] is True
        assert output.debug["max_books_limit"] == 3

    def test_debug_records_no_truncation_flag(self, tmp_path: Path) -> None:
        """debug dict reports max_books_applied=False when no truncation occurred."""
        docs = [_make_doc(f"/works/OL{i}W", f"Book {i}") for i in range(1, 4)]
        stub = _StubClient(responses={"fantasy": docs})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub, max_books=50)  # type: ignore[arg-type]

        output = a.run(BookFetcherAgentInput(normalized_genres=["fantasy"]))

        assert output.debug["max_books_applied"] is False
        assert output.debug["max_books_limit"] == 50

    def test_default_max_books(self, tmp_path: Path) -> None:
        """Default max_books is 50 when client is provided directly."""
        stub = _StubClient()
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(client=stub)  # type: ignore[arg-type]

        assert a._max_books == 50


# ---------------------------------------------------------------------------
# Multi-sort strategy
# ---------------------------------------------------------------------------


class TestMultiSortStrategies:
    """sort_strategies with more than one entry doubles the API call count per genre."""

    def test_single_sort_makes_one_call_per_genre(self, tmp_path: Path) -> None:
        """With one sort strategy, each genre produces exactly one API call."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions"],
            )
        a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert stub.fetch_call_args == [("mystery", "editions")]

    def test_two_sorts_make_two_calls_per_genre(self, tmp_path: Path) -> None:
        """With two sort strategies, each genre produces two API calls."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions", "rating"],
            )
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert stub.fetch_call_args == [("mystery", "editions"), ("mystery", "rating")]

    def test_deduplication_across_sorts(self, tmp_path: Path) -> None:
        """A book returned by both sort passes appears only once in the output."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions", "rating"],
            )
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        # OL1W comes back from both editions and rating — must be deduplicated.
        assert len(output.books) == 1
        assert output.books[0].work_key == "/works/OL1W"

    def test_source_genres_not_duplicated_across_sorts(self, tmp_path: Path) -> None:
        """source_genres lists each genre at most once even across multiple sorts."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions", "rating"],
            )
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.books[0].source_genres == ["mystery"]

    def test_queries_executed_deduplicates_subjects(self, tmp_path: Path) -> None:
        """queries_executed contains each subject once regardless of sort count."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions", "rating"],
            )
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        assert output.queries_executed == ["mystery"]

    def test_sleep_between_multi_sort_calls(self, tmp_path: Path) -> None:
        """Sleep fires between every consecutive API miss, including across sorts."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC], "fantasy": [_FANTASY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions", "rating"],
            )
        with patch("storymesh.agents.book_fetcher.agent.time.sleep") as mock_sleep:
            a.run(BookFetcherAgentInput(normalized_genres=["mystery", "fantasy"]))
        # Calls: mystery:editions, mystery:rating, fantasy:editions, fantasy:rating
        # Sleep fires after all but the last → 3 sleeps.
        assert mock_sleep.call_count == 3

    def test_per_genre_debug_has_per_sort_detail(self, tmp_path: Path) -> None:
        """debug['per_genre'][subject]['sorts'] maps each sort to its own stats."""
        stub = _StubClient(responses={"mystery": [_MYSTERY_DOC]})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            a = BookFetcherAgent(
                client=stub,  # type: ignore[arg-type]
                sort_strategies=["editions", "rating"],
            )
        with patch("storymesh.agents.book_fetcher.agent.time.sleep"):
            output = a.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
        sorts = output.debug["per_genre"]["mystery"]["sorts"]
        assert "editions" in sorts
        assert "rating" in sorts
        assert sorts["editions"]["cache"] == "miss"
        assert sorts["rating"]["cache"] == "miss"


# ---------------------------------------------------------------------------
# validate_subjects
# ---------------------------------------------------------------------------


class TestValidateSubjects:
    """Tests for BookFetcherAgent.validate_subjects()."""

    def _make_agent(
        self,
        tmp_path: Path,
        subject_info_responses: dict[str, dict[str, Any]] | None = None,
    ) -> BookFetcherAgent:
        client = _StubClient(subject_info_responses=subject_info_responses)
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            return BookFetcherAgent(client=client)  # type: ignore[arg-type]

    def test_subject_with_works_included(self, tmp_path: Path) -> None:
        """Subject with work_count > 0 is included."""
        agent = self._make_agent(
            tmp_path, {"adventure": {"work_count": 500}}
        )
        result = agent.validate_subjects(["adventure"])
        assert result == ["adventure"]

    def test_subject_with_zero_works_excluded(self, tmp_path: Path) -> None:
        """Subject with work_count == 0 is dropped."""
        agent = self._make_agent(
            tmp_path, {"middle grade": {"work_count": 0}}
        )
        result = agent.validate_subjects(["middle grade"])
        assert result == []

    def test_empty_input_returns_empty(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        assert agent.validate_subjects([]) == []

    def test_api_error_includes_subject_by_default(self, tmp_path: Path) -> None:
        """When the Subjects API call fails, the subject is included (benefit of the doubt)."""
        client = _StubClient()

        def _raise(subject: str) -> dict[str, Any]:
            raise OpenLibraryAPIError("connection error")

        client.fetch_subject_info = _raise  # type: ignore[method-assign]

        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            agent = BookFetcherAgent(client=client)  # type: ignore[arg-type]

        result = agent.validate_subjects(["unknown"])
        assert result == ["unknown"]

    def test_positive_result_cached(self, tmp_path: Path) -> None:
        """A work_count > 0 result is cached; second call skips the API."""
        client = _StubClient(subject_info_responses={"fantasy": {"work_count": 100}})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            agent = BookFetcherAgent(client=client)  # type: ignore[arg-type]

        agent.validate_subjects(["fantasy"])
        agent.validate_subjects(["fantasy"])

        assert client.subject_info_calls.count("fantasy") == 1

    def test_negative_result_cached(self, tmp_path: Path) -> None:
        """A work_count == 0 result is cached; second call skips the API."""
        client = _StubClient(subject_info_responses={"middle grade": {"work_count": 0}})
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            agent = BookFetcherAgent(client=client)  # type: ignore[arg-type]

        agent.validate_subjects(["middle grade"])
        agent.validate_subjects(["middle grade"])

        assert client.subject_info_calls.count("middle grade") == 1

    def test_mixed_subjects_filtered_correctly(self, tmp_path: Path) -> None:
        """Valid subjects kept, zero-work-count subjects dropped."""
        agent = self._make_agent(
            tmp_path,
            {
                "adventure": {"work_count": 200},
                "middle grade": {"work_count": 0},
                "mystery": {"work_count": 1500},
            },
        )
        result = agent.validate_subjects(["adventure", "middle grade", "mystery"])
        assert result == ["adventure", "mystery"]

    def test_order_preserved(self, tmp_path: Path) -> None:
        """Output order matches input order for passing subjects."""
        agent = self._make_agent(
            tmp_path,
            {
                "fantasy": {"work_count": 300},
                "thriller": {"work_count": 400},
                "horror": {"work_count": 200},
            },
        )
        result = agent.validate_subjects(["fantasy", "thriller", "horror"])
        assert result == ["fantasy", "thriller", "horror"]


# ---------------------------------------------------------------------------
# Node wrapper
# ---------------------------------------------------------------------------

class TestNodeWrapper:
    def test_none_genre_output_raises_runtime_error(self) -> None:
        """Node wrapper raises RuntimeError when genre_normalizer_output is None."""
        from storymesh.orchestration.nodes.book_fetcher import make_book_fetcher_node

        stub = _StubClient()
        node = make_book_fetcher_node(stub)  # type: ignore[arg-type]
        state = {"genre_normalizer_output": None}

        with pytest.raises(RuntimeError, match="genre_normalizer_output.*None"):
            node(state)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration test (real API — skipped in normal CI)
# ---------------------------------------------------------------------------

@pytest.mark.real_api
def test_real_open_library_mystery(tmp_path: Path) -> None:
    """Fetch real mystery books from Open Library and verify schema compliance."""
    with patch(
        "storymesh.agents.book_fetcher.agent.get_cache_dir",
        return_value=tmp_path,
    ), patch(
        "storymesh.agents.book_fetcher.agent.get_api_client_config",
        return_value={},
    ):
        agent = BookFetcherAgent()

    output = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))

    assert len(output.books) > 0
    assert all(isinstance(b, BookRecord) for b in output.books)
    assert output.queries_executed == ["mystery"]
    assert output.debug["total_raw"] >= output.debug["total_after_dedup"]
