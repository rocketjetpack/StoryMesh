"""OpenLibraryClient — thin HTTP wrapper for the Open Library Search and Subjects APIs.

Handles all network communication for the BookFetcherAgent. The agent
handles business logic; this module handles only API communication.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenLibraryAPIError(Exception):
    """Raised when the Open Library API returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenLibraryClient:
    """Thin synchronous HTTP wrapper for the Open Library Search and Subjects APIs.

    Rate limiting is enforced by the caller (BookFetcherAgent) via the
    ``rate_limit_delay`` attribute. This class exposes the appropriate delay
    value based on whether a User-Agent is configured.

    Both endpoints share connection pool, headers, and retry logic. The retry
    policy uses exponential backoff (base × 2^attempt, capped at 30 s) for
    transient errors (429 and 5xx), with a WARNING logged on each retry.
    """

    _SEARCH_URL = "https://openlibrary.org/search.json"
    _SUBJECTS_URL = "https://openlibrary.org/subjects"
    _FIELDS = (
        "key,title,author_name,first_publish_year,"
        "edition_count,ratings_average,ratings_count,"
        "readinglog_count,want_to_read_count,already_read_count,"
        "currently_reading_count,number_of_pages_median,"
        "subject,cover_i"
    )

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 8,
    ) -> None:
        """Construct the client.

        Args:
            user_agent: Identifies this application to Open Library for a higher
                rate limit (3 req/sec). If None, the client operates anonymously
                at 1 req/sec. Open Library requires the format:
                "AppName (contact@email.com)".
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of attempts per request before raising
                ``OpenLibraryAPIError``. Applies to both the search and subjects
                endpoints. Includes the initial attempt, so ``max_retries=8``
                means up to 7 retries.
        """
        self._user_agent = user_agent
        self._max_retries = max_retries
        self.rate_limit_delay: float = 0.4 if user_agent else 1.0
        headers: dict[str, str] = {}
        if user_agent:
            headers["User-Agent"] = user_agent
        self._client = httpx.Client(headers=headers, timeout=timeout)
        logger.warning("OpenLibraryClient initialized with User-Agent '%s'. ", user_agent)

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
        data = self._fetch_with_retry(self._SEARCH_URL, params=params)
        docs = data.get("docs", [])
        return docs if isinstance(docs, list) else []

    def fetch_subject_info(self, subject: str) -> dict[str, Any]:
        """Query the Open Library Subjects API for metadata about a subject.

        The subject string is URL-encoded so that spaces and special characters
        are handled correctly in the path segment (e.g. "science fiction"
        becomes ``/subjects/science%20fiction.json``).

        A 404 response is treated as "subject not found" and returns
        ``{"work_count": 0}`` rather than raising, since a missing subject
        is a definitive signal that no works exist — not a transient error.

        Args:
            subject: Subject string (may contain spaces).

        Returns:
            Parsed JSON dict from the Subjects API, guaranteed to contain at
            least ``{"work_count": N}``. Returns ``{"work_count": 0}`` for
            subjects that do not exist in Open Library.

        Raises:
            OpenLibraryAPIError: On transient HTTP errors after all retries,
                timeouts, or JSON parse failures.
        """
        encoded = urllib.parse.quote(subject, safe="")
        url = f"{self._SUBJECTS_URL}/{encoded}.json"
        try:
            return self._fetch_with_retry(url)
        except OpenLibraryAPIError as exc:
            if exc.status_code == 404:
                logger.debug(
                    "fetch_subject_info: subject '%s' not found (404) — work_count=0.",
                    subject,
                )
                return {"work_count": 0}
            raise

    def _fetch_with_retry(
        self,
        url: str,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        """Make a GET request with exponential-backoff retry for transient errors.

        Retry policy:
            - HTTP 429 (rate limited): base wait 2 s, doubles each attempt,
              capped at 30 s.
            - HTTP 5xx (server error): base wait 1 s, doubles each attempt,
              capped at 30 s.
            - Other 4xx (inc. 404): raise ``OpenLibraryAPIError`` immediately,
              no retry.
            - Timeout: raise ``OpenLibraryAPIError`` immediately, no retry.
            - JSON parse error: raise ``OpenLibraryAPIError`` immediately.

        Each retry is logged at WARNING level with attempt count and wait time.

        Args:
            url: Full URL to GET.
            params: Optional query parameters.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            OpenLibraryAPIError: On unrecoverable errors or exhausted retries.
        """
        logger.debug("OL API GET %s params=%s", url, params)

        for attempt in range(self._max_retries):
            try:
                response = self._client.get(url, params=params or {})
            except httpx.TimeoutException as exc:
                raise OpenLibraryAPIError(
                    f"Request to Open Library timed out: {exc}"
                ) from exc

            if response.status_code == 200:
                return self._parse_json(response)

            if response.status_code == 429:
                wait = min(2.0 * (2.0 ** attempt), 30.0)
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "OL API [attempt %d/%d]: HTTP 429 (rate limited)"
                        " — retrying in %.1fs.",
                        attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise OpenLibraryAPIError(
                    f"Open Library rate limit (429) after {self._max_retries} attempts.",
                    status_code=429,
                )

            if response.status_code >= 500:
                wait = min(1.0 * (2.0 ** attempt), 30.0)
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "OL API [attempt %d/%d]: HTTP %d"
                        " — retrying in %.1fs.",
                        attempt + 1,
                        self._max_retries,
                        response.status_code,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise OpenLibraryAPIError(
                    f"Open Library server error after {self._max_retries} attempts:"
                    f" HTTP {response.status_code}",
                    status_code=response.status_code,
                )

            # Other 4xx — no retry warranted.
            raise OpenLibraryAPIError(
                f"Open Library returned HTTP {response.status_code}:"
                f" {response.text[:200]}",
                status_code=response.status_code,
            )

        # Unreachable: every loop iteration either returns, raises, or continues.
        raise OpenLibraryAPIError(  # pragma: no cover
            "Unexpected state in retry loop."
        )

    def _parse_json(self, response: httpx.Response) -> dict[str, Any]:
        """Parse a successful (HTTP 200) response body as a JSON object.

        Args:
            response: A successful httpx Response.

        Returns:
            The parsed JSON dict.

        Raises:
            OpenLibraryAPIError: If the body is not valid JSON or not a dict.
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

        return data

    def close(self) -> None:
        """Close the underlying httpx.Client."""
        self._client.close()

    def __enter__(self) -> OpenLibraryClient:
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the client on context manager exit."""
        self.close()
