"""Unit tests for storymesh.agents.book_fetcher.client.

All tests mock httpx so no real network calls are made.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from storymesh.agents.book_fetcher.client import OpenLibraryAPIError, OpenLibraryClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int,
    json_data: dict[str, Any] | None = None,
    text: str = "",
) -> MagicMock:
    """Build a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.side_effect = Exception("Invalid JSON")
    return response


def _docs_response(docs: list[dict[str, Any]]) -> MagicMock:
    """Build a successful (200) mock response containing a docs list."""
    return _mock_response(200, json_data={"docs": docs, "numFound": len(docs)})


def _subject_response(work_count: int, name: str = "Test Subject") -> MagicMock:
    """Build a successful (200) mock Subjects API response."""
    return _mock_response(200, json_data={"work_count": work_count, "name": name})


_SAMPLE_DOCS: list[dict[str, Any]] = [
    {
        "key": "/works/OL1W",
        "title": "Sherlock Holmes",
        "author_name": ["Arthur Conan Doyle"],
        "edition_count": 50,
    },
    {
        "key": "/works/OL2W",
        "title": "And Then There Were None",
        "author_name": ["Agatha Christie"],
        "edition_count": 40,
    },
]


# ---------------------------------------------------------------------------
# Construction / rate limit delay
# ---------------------------------------------------------------------------


class TestOpenLibraryClientInit:
    def test_no_user_agent_sets_slow_rate_limit(self) -> None:
        client = OpenLibraryClient(user_agent=None)
        assert client.rate_limit_delay == 1.0
        client.close()

    def test_user_agent_sets_fast_rate_limit(self) -> None:
        client = OpenLibraryClient(user_agent="StoryMesh (test@example.com)")
        assert client.rate_limit_delay == 0.4
        client.close()

    def test_user_agent_header_sent(self) -> None:
        ua = "StoryMesh (test@example.com)"
        with patch("httpx.Client") as mock_class:
            OpenLibraryClient(user_agent=ua)
            mock_class.assert_called_once_with(
                headers={"User-Agent": ua},
                timeout=30.0,
            )

    def test_no_user_agent_no_header(self) -> None:
        with patch("httpx.Client") as mock_class:
            OpenLibraryClient(user_agent=None)
            mock_class.assert_called_once_with(headers={}, timeout=30.0)

    def test_custom_timeout(self) -> None:
        with patch("httpx.Client") as mock_class:
            OpenLibraryClient(user_agent=None, timeout=30.0)
            mock_class.assert_called_once_with(headers={}, timeout=30.0)

    def test_default_max_retries(self) -> None:
        client = OpenLibraryClient()
        assert client._max_retries == 8
        client.close()

    def test_custom_max_retries(self) -> None:
        client = OpenLibraryClient(max_retries=3)
        assert client._max_retries == 3
        client.close()


# ---------------------------------------------------------------------------
# Successful responses — fetch_books_by_subject
# ---------------------------------------------------------------------------


class TestFetchBooksBySubject:
    def test_successful_response_returns_docs(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(return_value=_docs_response(_SAMPLE_DOCS))  # type: ignore[method-assign]

        result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        assert result[0]["title"] == "Sherlock Holmes"

    def test_empty_docs_key_returns_empty_list(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_response(200, json_data={"numFound": 0})
        )

        result = client.fetch_books_by_subject("mystery")

        assert result == []

    def test_subject_passed_in_params(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_books_by_subject("post apocalyptic")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["subject"] == "post apocalyptic"

    def test_limit_passed_in_params(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_books_by_subject("mystery", limit=10)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 10

    def test_sort_editions_in_params(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_books_by_subject("mystery")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["sort"] == "editions"

    def test_fields_parameter_included(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_books_by_subject("mystery")

        _, kwargs = mock_get.call_args
        fields = kwargs["params"]["fields"]
        assert "key" in fields
        assert "title" in fields
        assert "author_name" in fields
        assert "edition_count" in fields
        assert "ratings_average" in fields
        assert "readinglog_count" in fields
        assert "want_to_read_count" in fields
        assert "already_read_count" in fields
        assert "currently_reading_count" in fields
        assert "number_of_pages_median" in fields


# ---------------------------------------------------------------------------
# fetch_subject_info
# ---------------------------------------------------------------------------


class TestFetchSubjectInfo:
    def test_returns_work_count_for_known_subject(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(return_value=_subject_response(1234))  # type: ignore[method-assign]

        result = client.fetch_subject_info("mystery")

        assert result["work_count"] == 1234

    def test_url_encodes_spaces(self) -> None:
        """Spaces in the subject must be percent-encoded in the URL path."""
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_subject_response(500))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_subject_info("science fiction")

        url_called = mock_get.call_args[0][0]
        assert "science%20fiction" in url_called
        assert " " not in url_called

    def test_url_encodes_special_characters(self) -> None:
        """Characters like '&' and '/' in subject names must be encoded."""
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_subject_response(10))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_subject_info("science & technology")

        url_called = mock_get.call_args[0][0]
        assert "&" not in url_called.split("subjects/")[1]

    def test_404_returns_zero_work_count(self) -> None:
        """404 means the subject does not exist — treat as work_count=0."""
        client = OpenLibraryClient()
        client._client.get = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_response(404, text="Not found")
        )

        result = client.fetch_subject_info("nonexistent_subject_xyz")

        assert result == {"work_count": 0}

    def test_404_does_not_raise(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_response(404, text="Not found")
        )

        # Should not raise
        result = client.fetch_subject_info("nonexistent")
        assert "work_count" in result

    def test_5xx_retries_then_raises(self) -> None:
        client = OpenLibraryClient(max_retries=2)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _mock_response(503, text="Service unavailable"),
                _mock_response(503, text="Service unavailable"),
            ]
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError):
            client.fetch_subject_info("mystery")

    def test_subjects_url_contains_json_extension(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_subject_response(100))
        client._client.get = mock_get  # type: ignore[method-assign]

        client.fetch_subject_info("fantasy")

        url_called = mock_get.call_args[0][0]
        assert url_called.endswith(".json")
        assert "/subjects/" in url_called


# ---------------------------------------------------------------------------
# Retry logic — exponential backoff
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_5xx_retries_up_to_max_then_raises(self) -> None:
        """After max_retries 503s, OpenLibraryAPIError is raised."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[_mock_response(503)] * 3
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError, match="503"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 3

    def test_5xx_succeeds_after_multiple_retries(self) -> None:
        """Succeeds on the third attempt after two 503s."""
        client = OpenLibraryClient(max_retries=5)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _mock_response(503),
                _mock_response(503),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        assert mock_sleep.call_count == 2

    def test_429_retries_up_to_max_then_raises(self) -> None:
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[_mock_response(429)] * 3
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError, match="429"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 3

    def test_429_succeeds_on_second_attempt(self) -> None:
        client = OpenLibraryClient(max_retries=5)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _mock_response(429),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        mock_sleep.assert_called_once()

    def test_5xx_first_retry_wait_is_one_second(self) -> None:
        """First 5xx retry waits base * 2^0 = 1.0s."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[_mock_response(503), _docs_response(_SAMPLE_DOCS)]
        )

        with patch("time.sleep") as mock_sleep:
            client.fetch_books_by_subject("mystery")

        mock_sleep.assert_called_once_with(1.0)

    def test_429_first_retry_wait_is_two_seconds(self) -> None:
        """First 429 retry waits base * 2^0 = 2.0s."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[_mock_response(429), _docs_response(_SAMPLE_DOCS)]
        )

        with patch("time.sleep") as mock_sleep:
            client.fetch_books_by_subject("mystery")

        mock_sleep.assert_called_once_with(2.0)

    def test_exponential_backoff_doubles_each_retry(self) -> None:
        """5xx wait doubles: 1s, 2s, 4s for attempts 0, 1, 2."""
        client = OpenLibraryClient(max_retries=4)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _mock_response(503),
                _mock_response(503),
                _mock_response(503),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            client.fetch_books_by_subject("mystery")

        assert mock_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]

    def test_backoff_capped_at_thirty_seconds(self) -> None:
        """Wait never exceeds 30s regardless of attempt count."""
        client = OpenLibraryClient(max_retries=10)
        # 9 failures then success
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[_mock_response(503)] * 9 + [_docs_response(_SAMPLE_DOCS)]
        )

        with patch("time.sleep") as mock_sleep:
            client.fetch_books_by_subject("mystery")

        for actual_call in mock_sleep.call_args_list:
            assert actual_call[0][0] <= 30.0

    def test_retry_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each retry attempt is logged at WARNING level."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _mock_response(503),
                _mock_response(503),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep"), caplog.at_level(
            logging.WARNING, logger="storymesh.agents.book_fetcher.client"
        ):
            client.fetch_books_by_subject("mystery")

        retry_logs = [r for r in caplog.records if "retrying" in r.message.lower()]
        assert len(retry_logs) == 2

    def test_4xx_raises_immediately_without_retry(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_response(403, text="Forbidden")
        )

        with pytest.raises(OpenLibraryAPIError, match="403"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 1

    def test_timeout_retries_then_raises(self) -> None:
        """Timeouts are retried like 5xx errors, then raise after max_retries."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=httpx.TimeoutException("timed out")
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError, match="timed out"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 3

    def test_timeout_succeeds_after_retry(self) -> None:
        """A timeout followed by a successful response returns data normally."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                httpx.TimeoutException("timed out"),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_timeout_first_retry_wait_is_one_second(self) -> None:
        """Timeout retries use the same 1 s base backoff as 5xx."""
        client = OpenLibraryClient(max_retries=3)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                httpx.TimeoutException("timed out"),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            client.fetch_books_by_subject("mystery")

        mock_sleep.assert_called_once_with(1.0)

    def test_json_parse_failure_raises(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(return_value=_mock_response(200, json_data=None))  # type: ignore[method-assign]

        with pytest.raises(OpenLibraryAPIError, match="JSON"):
            client.fetch_books_by_subject("mystery")

    def test_non_dict_json_raises(self) -> None:
        client = OpenLibraryClient()
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = ["not", "a", "dict"]
        client._client.get = MagicMock(return_value=response)  # type: ignore[method-assign]

        with pytest.raises(OpenLibraryAPIError, match="JSON object"):
            client.fetch_books_by_subject("mystery")

    def test_error_has_status_code_attribute(self) -> None:
        """OpenLibraryAPIError carries the HTTP status code for callers."""
        client = OpenLibraryClient(max_retries=1)
        client._client.get = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_response(503, text="unavailable")
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError) as exc_info:
            client.fetch_books_by_subject("mystery")

        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_calls_close(self) -> None:
        with patch("httpx.Client"):
            client = OpenLibraryClient()
            client._client = MagicMock()

            with client:
                pass

            client._client.close.assert_called_once()

    def test_context_manager_returns_self(self) -> None:
        with OpenLibraryClient() as client:
            assert isinstance(client, OpenLibraryClient)
