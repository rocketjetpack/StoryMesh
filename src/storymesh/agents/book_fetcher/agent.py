"""BookFetcherAgent — Stage 1 of the StoryMesh pipeline.

Receives normalized genre names from the GenreNormalizerAgent, queries
the Open Library Search API for each genre, caches results to disk, and
returns a deduplicated list of BookRecord objects for the BookRankerAgent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import diskcache  # type: ignore[import-untyped]
import orjson

from storymesh.agents.book_fetcher.client import OpenLibraryAPIError, OpenLibraryClient
from storymesh.config import get_api_client_config, get_cache_dir
from storymesh.schemas.book_fetcher import (
    BookFetcherAgentInput,
    BookFetcherAgentOutput,
    BookRecord,
)

logger = logging.getLogger(__name__)


class BookFetcherAgent:
    """Fetches genre-relevant books from the Open Library Search API (Stage 1).

    Accepts normalized genre names from the GenreNormalizerAgent, queries the
    Open Library Search API for each genre, caches results to disk, and returns
    a deduplicated flat list of BookRecord objects keyed by work_key.

    Books found under multiple genre queries are merged into a single record
    with all matched genres listed in ``source_genres``. This agent makes no
    LLM calls — it is a thin wrapper around the Open Library Search API.
    """

    def __init__(
        self,
        client: OpenLibraryClient | None = None,
        cache_ttl: int = 86400,
        max_books: int = 50,
    ) -> None:
        """Construct the agent.

        Args:
            client: Pre-built OpenLibraryClient. If None, one is constructed
                from the ``api_clients.open_library`` section of config.
                Providing a client directly is useful for testing.
            cache_ttl: Cache time-to-live in seconds. Default 86400 (24 hours).
                Book metadata changes infrequently; aggressive caching is
                appropriate and respectful to Open Library's infrastructure.
            max_books: Maximum number of books to return after deduplication.
                When ``client`` is None (normal runtime), the value from
                ``api_clients.open_library.max_books`` in config takes
                precedence over this parameter. Default 50.
        """
        self._cache_ttl = cache_ttl

        if client is None:
            ol_cfg = get_api_client_config("open_library")
            user_agent: str | None = ol_cfg.get("user_agent") or None
            self._max_books: int = ol_cfg.get("max_books", max_books)
            self._client = OpenLibraryClient(user_agent=user_agent)
            self._owns_client = True
        else:
            self._max_books = max_books
            self._client = client
            self._owns_client = False

        cache_dir = get_cache_dir("open_library")
        self._cache: diskcache.Cache = diskcache.Cache(str(cache_dir))

    def run(self, input_data: BookFetcherAgentInput) -> BookFetcherAgentOutput:
        """Fetch books for each normalized genre and return deduplicated results.

        For each genre:
          1. Convert underscores to spaces for the Open Library subject format.
          2. Check the disk cache. Return cached results without an API call.
          3. On a cache miss, query the API and store the result.
          4. Merge results by work_key: books found under multiple genres
             accumulate all matched genres in their ``source_genres`` list.
          5. Sleep between consecutive API calls to respect the rate limit.
             The sleep is skipped on cache hits and after the final genre.

        Args:
            input_data: Validated input contract from the GenreNormalizerAgent.

        Returns:
            A frozen BookFetcherAgentOutput with deduplicated book records and
            a debug dict containing per-genre counts and deduplication summary.
        """
        genres = input_data.normalized_genres
        limit = input_data.limit_per_genre

        logger.info("BookFetcherAgent starting | genres=%s", genres)

        # Accumulator keyed by work_key for deduplication.
        seen: dict[str, BookRecord] = {}
        queries_executed: list[str] = []
        per_genre_debug: dict[str, dict[str, Any]] = {}
        total_raw = 0

        for index, genre in enumerate(genres):
            subject = genre.replace("_", " ").lower()
            cache_key = f"ol_search:{subject}:{limit}:editions"

            cached_bytes: bytes | None = self._cache.get(cache_key)
            cache_status: str
            if cached_bytes is not None:
                cache_status = "hit"
                raw_docs: list[dict[str, Any]] = orjson.loads(cached_bytes)
            else:
                cache_status = "miss"
                try:
                    raw_docs = self._client.fetch_books_by_subject(
                        subject=subject, limit=limit
                    )
                except OpenLibraryAPIError:
                    logger.exception("Genre '%s': API call failed, skipping", subject)
                    per_genre_debug[subject] = {"books_fetched": 0, "cache": "error"}
                    queries_executed.append(subject)
                    continue
                self._cache.set(
                    cache_key,
                    orjson.dumps(raw_docs),
                    expire=self._cache_ttl,
                )
                # Sleep between API calls, but not after the last genre.
                if index < len(genres) - 1:
                    time.sleep(self._client.rate_limit_delay)

            queries_executed.append(subject)
            books_fetched = 0

            for doc in raw_docs:
                record = self._parse_book_record(doc, source_genres=[subject])
                if record is None:
                    continue
                books_fetched += 1
                total_raw += 1
                work_key = record.work_key
                if work_key in seen:
                    existing = seen[work_key]
                    seen[work_key] = existing.model_copy(
                        update={"source_genres": [*existing.source_genres, subject]}
                    )
                else:
                    seen[work_key] = record

            per_genre_debug[subject] = {
                "books_fetched": books_fetched,
                "cache": cache_status,
            }
            logger.info(
                "Genre '%s': fetched %d books | cache=%s",
                subject,
                books_fetched,
                cache_status,
            )

        all_books = list(seen.values())
        total_dedup = len(all_books)
        duplicates_merged = total_raw - total_dedup

        # Apply max_books cap: prioritise books found across more genres, then
        # by edition count as a popularity proxy.
        truncated = len(all_books) > self._max_books
        if truncated:
            all_books.sort(
                key=lambda b: (len(b.source_genres), b.edition_count),
                reverse=True,
            )
            all_books = all_books[: self._max_books]

        logger.info(
            "BookFetcherAgent complete | %d books (%d after dedup, %d merged, %d returned)",
            total_raw,
            total_dedup,
            duplicates_merged,
            len(all_books),
        )

        debug: dict[str, Any] = {
            "per_genre": per_genre_debug,
            "total_raw": total_raw,
            "total_after_dedup": total_dedup,
            "duplicates_merged": duplicates_merged,
            "max_books_limit": self._max_books,
            "max_books_applied": truncated,
        }

        return BookFetcherAgentOutput(
            books=all_books,
            queries_executed=queries_executed,
            debug=debug,
        )

    def _parse_book_record(
        self,
        doc: dict[str, Any],
        source_genres: list[str],
    ) -> BookRecord | None:
        """Parse a raw Open Library doc dict into a BookRecord.

        Docs missing a work key or title are silently skipped — an incomplete
        API response should not crash the pipeline.

        Args:
            doc: Raw dict from the Open Library ``docs`` array.
            source_genres: Genre subject strings that produced this result.

        Returns:
            A validated BookRecord, or None if minimum required fields are absent.
        """
        work_key = doc.get("key")
        title = doc.get("title")
        if not work_key or not title:
            return None

        return BookRecord(
            work_key=str(work_key),
            title=str(title),
            authors=list(doc.get("author_name", [])),
            first_publish_year=doc.get("first_publish_year"),
            edition_count=int(doc.get("edition_count", 0)),
            ratings_average=doc.get("ratings_average"),
            ratings_count=int(doc.get("ratings_count", 0)),
            subjects=list(doc.get("subject", [])),
            cover_id=doc.get("cover_i"),
            source_genres=source_genres,
        )

    def close(self) -> None:
        """Close the cache and, if owned by this instance, the HTTP client."""
        self._cache.close()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> BookFetcherAgent:
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close resources on context manager exit."""
        self.close()
