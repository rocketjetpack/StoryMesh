"""Unit tests for storymesh.agents.book_fetcher.client.

All tests mock httpx so no real network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

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
                timeout=10.0,
            )

    def test_no_user_agent_no_header(self) -> None:
        with patch("httpx.Client") as mock_class:
            OpenLibraryClient(user_agent=None)
            mock_class.assert_called_once_with(headers={}, timeout=10.0)

    def test_custom_timeout(self) -> None:
        with patch("httpx.Client") as mock_class:
            OpenLibraryClient(user_agent=None, timeout=30.0)
            mock_class.assert_called_once_with(headers={}, timeout=30.0)


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------

class TestFetchBooksBySubject:
    def test_successful_response_returns_docs(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(return_value=_docs_response(_SAMPLE_DOCS))

        result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        assert result[0]["title"] == "Sherlock Holmes"

    def test_empty_docs_key_returns_empty_list(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            return_value=_mock_response(200, json_data={"numFound": 0})
        )

        result = client.fetch_books_by_subject("mystery")

        assert result == []

    def test_subject_passed_in_params(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get

        client.fetch_books_by_subject("post apocalyptic")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["subject"] == "post apocalyptic"

    def test_limit_passed_in_params(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get

        client.fetch_books_by_subject("mystery", limit=10)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 10

    def test_sort_editions_in_params(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get

        client.fetch_books_by_subject("mystery")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["sort"] == "editions"

    def test_fields_parameter_included(self) -> None:
        client = OpenLibraryClient()
        mock_get = MagicMock(return_value=_docs_response([]))
        client._client.get = mock_get

        client.fetch_books_by_subject("mystery")

        _, kwargs = mock_get.call_args
        fields = kwargs["params"]["fields"]
        assert "key" in fields
        assert "title" in fields
        assert "author_name" in fields
        assert "edition_count" in fields
        assert "ratings_average" in fields


# ---------------------------------------------------------------------------
# Error handling and retry logic
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_429_retries_once_then_raises(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            side_effect=[
                _mock_response(429, text="Rate limited"),
                _mock_response(429, text="Rate limited"),
            ]
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError, match="429"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 2

    def test_429_retries_once_succeeds(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            side_effect=[
                _mock_response(429, text="Rate limited"),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        mock_sleep.assert_called_once_with(2.0)

    def test_5xx_retries_once_then_raises(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            side_effect=[
                _mock_response(503, text="Service unavailable"),
                _mock_response(503, text="Service unavailable"),
            ]
        )

        with patch("time.sleep"), pytest.raises(OpenLibraryAPIError, match="503"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 2

    def test_5xx_retries_once_succeeds(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            side_effect=[
                _mock_response(500, text="Internal server error"),
                _docs_response(_SAMPLE_DOCS),
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = client.fetch_books_by_subject("mystery")

        assert len(result) == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_4xx_raises_immediately_without_retry(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            return_value=_mock_response(404, text="Not found")
        )

        with pytest.raises(OpenLibraryAPIError, match="404"):
            client.fetch_books_by_subject("mystery")

        assert client._client.get.call_count == 1

    def test_timeout_raises(self) -> None:
        client = OpenLibraryClient()
        client._client.get = MagicMock(
            side_effect=httpx.TimeoutException("timed out")
        )

        with pytest.raises(OpenLibraryAPIError, match="timed out"):
            client.fetch_books_by_subject("mystery")

    def test_json_parse_failure_raises(self) -> None:
        client = OpenLibraryClient()
        bad_response = _mock_response(200, json_data=None)
        client._client.get = MagicMock(return_value=bad_response)

        with pytest.raises(OpenLibraryAPIError, match="JSON"):
            client.fetch_books_by_subject("mystery")

    def test_non_dict_json_raises(self) -> None:
        client = OpenLibraryClient()
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = ["not", "a", "dict"]
        client._client.get = MagicMock(return_value=response)

        with pytest.raises(OpenLibraryAPIError, match="JSON object"):
            client.fetch_books_by_subject("mystery")


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
