"""OpenLibraryClient — thin HTTP wrapper for the Open Library Search API.

Handles all network communication for the BookFetcherAgent. The agent
handles business logic; this module handles only API communication.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenLibraryAPIError(Exception):
    """Raised when the Open Library API returns an error or is unreachable."""


class OpenLibraryClient:
    """Thin synchronous HTTP wrapper for the Open Library Search API.

    Rate limiting is enforced by the caller (BookFetcherAgent) via the
    ``rate_limit_delay`` attribute. This class exposes the appropriate delay
    value based on whether a User-Agent is configured.
    """

    _BASE_URL = "https://openlibrary.org/search.json"
    _FIELDS = (
        "key,title,author_name,first_publish_year,"
        "edition_count,ratings_average,ratings_count,subject,cover_i"
    )

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Construct the client.

        Args:
            user_agent: Identifies this application to Open Library for a higher
                rate limit (3 req/sec). If None, the client operates anonymously
                at 1 req/sec. Open Library requires the format:
                "AppName (contact@email.com)".
            timeout: HTTP request timeout in seconds.
        """
        self._user_agent = user_agent
        self.rate_limit_delay: float = 0.4 if user_agent else 1.0
        headers: dict[str, str] = {}
        if user_agent:
            headers["User-Agent"] = user_agent
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def fetch_books_by_subject(
        self,
        subject: str,
        limit: int = 30,
        sort: str = "editions",
    ) -> list[dict[str, Any]]:
        """Query the Open Library Search API for books by subject.

        Args:
            subject: Subject/genre string. Use spaces, not underscores
                (e.g., "post apocalyptic", not "post_apocalyptic").
            limit: Maximum number of results to return.
            sort: Sort order. "editions" returns the most widely published
                books first, used as a proxy for cultural significance.

        Returns:
            The raw ``docs`` list from the API response.

        Raises:
            OpenLibraryAPIError: On HTTP errors (after retries), timeouts,
                or JSON parse failures.
        """
        params: dict[str, str | int] = {
            "subject": subject,
            "fields": self._FIELDS,
            "limit": limit,
            "sort": sort,
        }
        return self._get_with_retry(params)

    def _get_with_retry(
        self,
        params: dict[str, str | int],
    ) -> list[dict[str, Any]]:
        """Make a GET request with retry logic for transient errors.

        Retry policy:
            - HTTP 429 (rate limited): wait 2 seconds, retry once.
            - HTTP 5xx (server error): wait 1 second, retry once.
            - Other 4xx: raise ``OpenLibraryAPIError`` immediately (no retry).
            - Timeout: raise ``OpenLibraryAPIError`` immediately.
            - JSON parse error: raise ``OpenLibraryAPIError`` immediately.

        Args:
            params: Query parameters to include in the GET request.

        Returns:
            The ``docs`` list extracted from the API JSON response.

        Raises:
            OpenLibraryAPIError: On unrecoverable errors or exhausted retries.
        """
        for attempt in range(2):
            try:
                response = self._client.get(self._BASE_URL, params=params)
            except httpx.TimeoutException as exc:
                raise OpenLibraryAPIError(
                    f"Request to Open Library timed out: {exc}"
                ) from exc

            if response.status_code == 200:
                return self._extract_docs(response)

            if response.status_code == 429:
                if attempt == 0:
                    logger.warning(
                        "Open Library rate limit hit (429). Retrying in 2s."
                    )
                    time.sleep(2.0)
                    continue
                raise OpenLibraryAPIError(
                    "Open Library rate limit hit (429) and retry also failed."
                )

            if response.status_code >= 500:
                if attempt == 0:
                    logger.warning(
                        "Open Library server error (%s). Retrying in 1s.",
                        response.status_code,
                    )
                    time.sleep(1.0)
                    continue
                raise OpenLibraryAPIError(
                    f"Open Library server error after retry: HTTP {response.status_code}"
                )

            # Other 4xx — no retry warranted
            raise OpenLibraryAPIError(
                f"Open Library returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        # Unreachable: the loop always returns or raises before exhausting two
        # attempts without a continue. Satisfies the type checker.
        raise OpenLibraryAPIError(  # pragma: no cover
            "Unexpected state in retry loop."
        )

    def _extract_docs(self, response: httpx.Response) -> list[dict[str, Any]]:
        """Extract the docs list from a successful API response.

        Args:
            response: A successful (HTTP 200) httpx Response object.

        Returns:
            The ``docs`` list from the response body, or ``[]`` if absent.

        Raises:
            OpenLibraryAPIError: If the response body is not valid JSON or
                is not a JSON object.
        """
        try:
            data: Any = response.json()
        except Exception as exc:
            raise OpenLibraryAPIError(
                f"Failed to parse Open Library response as JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise OpenLibraryAPIError(
                "Open Library response was not a JSON object."
            )

        docs = data.get("docs", [])
        if not isinstance(docs, list):
            return []
        return docs

    def close(self) -> None:
        """Close the underlying httpx.Client."""
        self._client.close()

    def __enter__(self) -> OpenLibraryClient:
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the client on context manager exit."""
        self.close()
