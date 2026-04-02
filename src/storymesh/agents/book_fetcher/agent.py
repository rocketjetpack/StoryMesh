"""BookFetcherAgent — Stage 1 of the StoryMesh pipeline.

Receives normalized genre names from the GenreNormalizerAgent, queries
the Open Library Search API for each genre using one or more sort strategies,
caches results to disk, and returns a deduplicated list of BookRecord objects
for the BookRankerAgent.
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

# Subjects API results are stable: a subject with zero works today will almost
# certainly still have zero works tomorrow. Cache validation results for 7 days.
_SUBJECT_VALIDATE_TTL: int = 7 * 24 * 3600


class BookFetcherAgent:
    """Fetches genre-relevant books from the Open Library Search API (Stage 1).

    Accepts normalized genre names from the GenreNormalizerAgent, queries the
    Open Library Search API for each genre using one or more sort strategies,
    caches results to disk, and returns a deduplicated flat list of BookRecord
    objects keyed by work_key.

    Books found under multiple genre queries or sort passes are merged into a
    single record with all matched genres listed in ``source_genres``. Each
    genre name appears at most once in ``source_genres`` regardless of how many
    sort passes returned it. This agent makes no LLM calls — it is a thin
    wrapper around the Open Library Search API.
    """

    def __init__(
        self,
        client: OpenLibraryClient | None = None,
        cache_ttl: int = 86400,
        max_books: int = 50,
        sort_strategies: list[str] | None = None,
        limit_per_sort: int = 30,
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
                precedence. Default 50.
            sort_strategies: Sort orders to use per genre. Each strategy is a
                separate API call, broadening the candidate pool. When ``client``
                is None, the value from config takes precedence. Default
                ``["editions"]`` (single pass, identical to previous behaviour).
            limit_per_sort: Books to fetch per (genre, sort) pair. When
                ``client`` is None, the value from config takes precedence.
                Default 30.
        """
        self._cache_ttl = cache_ttl

        if client is None:
            ol_cfg = get_api_client_config("open_library")
            user_agent: str | None = ol_cfg.get("user_agent") or None
            self._max_books: int = ol_cfg.get("max_books", max_books)
            self._sort_strategies: list[str] = ol_cfg.get(
                "sort_strategies", sort_strategies or ["editions"]
            )
            self._limit_per_sort: int = ol_cfg.get("limit_per_sort", limit_per_sort)
            self._client = OpenLibraryClient(
                user_agent=user_agent,
                timeout=float(ol_cfg.get("timeout", 30.0)),
                max_retries=ol_cfg.get("max_retries", 8),
            )
            self._owns_client = True
        else:
            self._max_books = max_books
            self._sort_strategies = sort_strategies or ["editions"]
            self._limit_per_sort = limit_per_sort
            self._client = client
            self._owns_client = False

        cache_dir = get_cache_dir("open_library")
        self._cache: diskcache.Cache = diskcache.Cache(str(cache_dir))

    def run(self, input_data: BookFetcherAgentInput) -> BookFetcherAgentOutput:
        """Fetch books for each normalized genre across all sort strategies.

        For each (genre, sort) pair:
          1. Convert underscores to spaces for the Open Library subject format.
          2. Check the disk cache using a key that encodes subject, limit, and
             sort. Return cached results without an API call.
          3. On a cache miss, query the API and store the result.
          4. Merge results by work_key: books found under multiple genres or
             sort passes accumulate matched genres in ``source_genres`` (each
             genre name appears at most once per book).
          5. Sleep between consecutive API calls to respect the rate limit.
             The sleep is skipped on cache hits, on errors, and after the
             final (genre, sort) call.

        Args:
            input_data: Validated input contract from the GenreNormalizerAgent.

        Returns:
            A frozen BookFetcherAgentOutput with deduplicated book records and
            a debug dict containing per-genre counts and deduplication summary.
        """
        genres = input_data.normalized_genres

        logger.info(
            "BookFetcherAgent starting | genres=%s | sorts=%s",
            genres,
            self._sort_strategies,
        )

        # Accumulator keyed by work_key for deduplication.
        seen: dict[str, BookRecord] = {}
        queries_executed: list[str] = []
        per_genre_debug: dict[str, dict[str, Any]] = {}
        total_raw = 0

        # Flat call list enables clean "is this the last API call?" check.
        calls = [
            (genre, sort)
            for genre in genres
            for sort in self._sort_strategies
        ]
        last_idx = len(calls) - 1

        for idx, (genre, sort) in enumerate(calls):
            subject = genre.replace("_", " ").lower()

            # Initialise per-genre tracking on first sort for this genre.
            if subject not in per_genre_debug:
                per_genre_debug[subject] = {"books_fetched": 0, "sorts": {}}
                queries_executed.append(subject)

            cache_key = f"ol_search:{subject}:{self._limit_per_sort}:{sort}"
            cached_bytes: bytes | None = self._cache.get(cache_key)
            cache_status: str

            if cached_bytes is not None:
                cache_status = "hit"
                raw_docs: list[dict[str, Any]] = orjson.loads(cached_bytes)
            else:
                cache_status = "miss"
                try:
                    raw_docs = self._client.fetch_books_by_subject(
                        subject=subject,
                        limit=self._limit_per_sort,
                        sort=sort,
                    )
                except OpenLibraryAPIError as exc:
                    logger.warning(
                        "Genre '%s' sort '%s': API call failed, skipping: %s",
                        subject,
                        sort,
                        exc,
                    )
                    per_genre_debug[subject]["sorts"][sort] = {
                        "books_fetched": 0,
                        "cache": "error",
                    }
                    continue

                self._cache.set(
                    cache_key,
                    orjson.dumps(raw_docs),
                    expire=self._cache_ttl,
                )
                # Sleep between API calls, but not after the last one.
                if idx < last_idx:
                    time.sleep(self._client.rate_limit_delay)

            sort_books_fetched = 0
            for doc in raw_docs:
                record = self._parse_book_record(doc, source_genres=[subject])
                if record is None:
                    continue
                sort_books_fetched += 1
                total_raw += 1
                work_key = record.work_key
                if work_key in seen:
                    existing = seen[work_key]
                    # Add genre only if it hasn't been recorded for this book.
                    if subject not in existing.source_genres:
                        seen[work_key] = existing.model_copy(
                            update={
                                "source_genres": [*existing.source_genres, subject]
                            }
                        )
                else:
                    seen[work_key] = record

            per_genre_debug[subject]["sorts"][sort] = {
                "books_fetched": sort_books_fetched,
                "cache": cache_status,
            }
            per_genre_debug[subject]["books_fetched"] += sort_books_fetched

            logger.info(
                "Genre '%s' sort '%s': fetched %d books | cache=%s",
                subject,
                sort,
                sort_books_fetched,
                cache_status,
            )

        # Compute aggregate cache status per genre:
        # "hit"   → all sort passes were cache hits
        # "error" → all sort passes errored (no books at all)
        # "miss"  → any pass missed the cache (API was called)
        for gdata in per_genre_debug.values():
            sort_statuses = [s["cache"] for s in gdata["sorts"].values()]
            if all(s == "hit" for s in sort_statuses):
                gdata["cache"] = "hit"
            elif sort_statuses and all(s == "error" for s in sort_statuses):
                gdata["cache"] = "error"
            else:
                gdata["cache"] = "miss"

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
            readinglog_count=int(doc.get("readinglog_count", 0)),
            want_to_read_count=int(doc.get("want_to_read_count", 0)),
            already_read_count=int(doc.get("already_read_count", 0)),
            currently_reading_count=int(doc.get("currently_reading_count", 0)),
            number_of_pages_median=doc.get("number_of_pages_median"),
            subjects=list(doc.get("subject", [])),
            cover_id=doc.get("cover_i"),
            source_genres=source_genres,
        )

    def validate_subjects(self, subjects: list[str]) -> list[str]:
        """Return only subjects that Open Library has at least one work for.

        Checks the disk cache before hitting the network. Both positive
        (``work_count > 0``) and negative (``work_count == 0``) results are
        cached so that repeated pipeline runs avoid redundant API calls.

        Subjects that cannot be validated due to a transient API error are
        included in the output (benefit of the doubt — only confirmed zeros
        are dropped).

        Args:
            subjects: Resolved Open Library subject strings to validate.

        Returns:
            Filtered list preserving original order, containing only subjects
            with a confirmed ``work_count > 0`` or that could not be checked.
        """
        valid: list[str] = []

        for subject in subjects:
            cache_key = f"ol_subject_wc:{subject}"
            cached: int | None = self._cache.get(cache_key)

            if cached is not None:
                if cached > 0:
                    valid.append(subject)
                else:
                    logger.debug(
                        "validate_subjects: '%s' cached with work_count=0 — skipping.",
                        subject,
                    )
                continue

            # Cache miss — probe the Subjects API.
            try:
                info = self._client.fetch_subject_info(subject)
                work_count = int(info.get("work_count", 0))
            except OpenLibraryAPIError as exc:
                logger.warning(
                    "validate_subjects: failed to probe '%s' — including by default: %s",
                    subject,
                    exc,
                )
                valid.append(subject)
                continue

            self._cache.set(cache_key, work_count, expire=_SUBJECT_VALIDATE_TTL)

            if work_count > 0:
                logger.debug(
                    "validate_subjects: '%s' has work_count=%d — including.",
                    subject,
                    work_count,
                )
                valid.append(subject)
            else:
                logger.warning(
                    "validate_subjects: '%s' has work_count=0 — dropping.",
                    subject,
                )

        return valid

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
