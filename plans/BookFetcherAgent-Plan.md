# BookFetcherAgent Implementation Plan

## Context

This document provides the complete, final implementation plan for the BookFetcherAgent
(Stage 1 of the StoryMesh pipeline). It supersedes any earlier draft of this plan.

This agent takes `normalized_genres` from the GenreNormalizerAgent (Stage 0) and queries
the Open Library Search API to discover genre-relevant books. Its output feeds directly
into the BookRankerAgent (Stage 2).

The BookFetcherAgent follows the One-Agent-One-Tool philosophy: it wraps exactly one tool
(the Open Library API), has strict Pydantic input/output schemas, and is independently
testable and cacheable.

Review README.md for overall pipeline context. Review the GenreNormalizerAgent
implementation for patterns to follow — it is the model for how agents are structured.

---

## What This Agent Does

1. Receives `normalized_genres` from the GenreNormalizerAgent output (e.g., `["mystery", "post_apocalyptic"]`)
2. For each normalized genre, queries the Open Library Search API to fetch popular books tagged with that subject
3. Collects the raw results into a unified list of book metadata objects
4. Returns the list as a validated Pydantic output contract

This agent does NOT rank, deduplicate, or score books — that is the BookRankerAgent's job.
This agent's sole responsibility is fetching raw data from Open Library.

---

## Open Library API Details

### Primary Endpoint: Search API

Use the Search API rather than the Subjects API. The Search API provides richer metadata
per result (including `first_publish_year`, `ratings_average`, `ratings_count`,
`number_of_pages_median`, and `subject`) which the BookRankerAgent will need downstream.

**URL format:**
```
https://openlibrary.org/search.json?subject={genre}&fields={field_list}&limit={limit}&sort=editions
```

**Example request:**
```
https://openlibrary.org/search.json?subject=mystery&fields=key,title,author_name,author_key,first_publish_year,edition_count,ratings_average,ratings_count,subject,cover_i&limit=30&sort=editions
```

**Key query parameters:**
- `subject` — the genre to search. Use the normalized genre name from GenreNormalizerAgent.
  Replace underscores with spaces (e.g., `post_apocalyptic` → `post apocalyptic`).
- `fields` — comma-separated list of fields to return. Only request fields the BookRecord
  schema needs.
- `limit` — number of results per query. Use 30 per genre as a reasonable ceiling.
- `sort` — sort by `editions` to get the most widely published (culturally significant)
  books first. This is the closest proxy to "bestseller" available without the NYT API.

**Response format (per doc in `docs` array):**
```json
{
  "key": "/works/OL27448W",
  "title": "The Lord of the Rings",
  "author_name": ["J. R. R. Tolkien"],
  "author_key": ["OL26320A"],
  "first_publish_year": 1954,
  "edition_count": 120,
  "ratings_average": 4.2,
  "ratings_count": 850,
  "subject": ["Fantasy fiction", "Middle Earth", "..."],
  "cover_i": 258027
}
```

Not all fields are guaranteed present on every work. The schema must handle optional
fields gracefully.

### Rate Limits and User-Agent

Open Library enforces two rate tiers:
- **Anonymous** (no User-Agent identification): 1 request per second
- **Identified** (User-Agent with app name and contact email): 3 requests per second

The User-Agent string is configured in `storymesh.config.yaml` under
`api_clients.open_library.user_agent`. If this key is absent or empty, the client
operates in anonymous mode at the slower rate.

**Rate limit implementation:**
- User-Agent configured → `time.sleep(0.4)` between requests (3 req/sec identified limit)
- User-Agent absent → `time.sleep(1.0)` between requests (1 req/sec anonymous limit)
- Sleep is skipped entirely on cache hits (no API call was made)

### Caching

Use `diskcache.Cache` for response caching. The cache directory is derived from
`cache.dir` in `storymesh.config.yaml` with `open_library` appended as a subdirectory.

Example: if `cache.dir` is `~/.cache/storymesh`, the Open Library cache lives at
`~/.cache/storymesh/open_library/`.

- **Cache key:** `f"ol_search:{subject}:{limit}:{sort}"` — a simple deterministic string
- **Cache value:** The raw `docs` list serialized as `orjson` bytes
- **Cache TTL:** Configurable via `BookFetcherAgent.__init__(cache_ttl)`, default 86400s (24h)
- **Cache hits:** Skip the API call and the inter-request sleep entirely

---

## Configuration Changes

### `storymesh.config.yaml`

Add an `api_clients` section at the top level:

```yaml
# ------
# External API Clients
# ------
# Configuration for external API integrations.
# user_agent: Identifies this application to Open Library for a higher rate limit
#   (3 req/sec vs 1 req/sec anonymous). Format required by Open Library:
#   "AppName (contact@email.com)". Remove or leave blank to use anonymous rate limit.
api_clients:
  open_library:
    user_agent: "StoryMesh (kali@rocketjetpack.site)"
```

### `src/storymesh/config.py`

Add two new public functions. Both use the existing `get_config()` machinery.

**`get_api_client_config(client_name: str) -> dict[str, Any]`**

Mirrors `get_agent_config`. Reads from the `api_clients` section. Returns a dict of
settings for the named client, or an empty dict if the client has no config entry.
Log a warning (same pattern as `get_agent_config`) if the client name is not found.

```python
def get_api_client_config(client_name: str) -> dict[str, Any]:
    """Return configuration for a named external API client.

    Reads from the ``api_clients`` section of storymesh.config.yaml.
    Returns an empty dict (with a warning) if no entry exists for the client.

    Args:
        client_name: Key under ``api_clients`` in the config file.

    Returns:
        Dict of client settings. Keys depend on the specific client.
    """
    config = get_config()
    clients_section = config.get("api_clients", {})
    if client_name not in clients_section:
        logger.warning(
            "No config entry for api_client '%s', using defaults", client_name
        )
        return {}
    return dict(clients_section[client_name])
```

**`get_cache_dir(name: str) -> Path`**

Reads `cache.dir` from config, expands `~`, and appends `name` as a subdirectory.
All agents that use `diskcache` call this function so the cache root is always
consistent with the configured value.

```python
def get_cache_dir(name: str) -> Path:
    """Return the diskcache directory for the given agent or client name.

    Resolves ``cache.dir`` from storymesh.config.yaml (expanding ``~``),
    then appends ``name`` as a subdirectory. The directory is not created
    here; callers (agent constructors) create it via diskcache.

    Args:
        name: Subdirectory name, typically the agent or API client name.

    Returns:
        Absolute Path for the cache directory.
    """
    config = get_config()
    cache_dir_str: str = config.get("cache", {}).get("dir", "~/.cache/storymesh")
    return Path(cache_dir_str).expanduser() / name
```

---

## File Structure

```
src/storymesh/agents/book_fetcher/
    __init__.py
    agent.py              # BookFetcherAgent class with run() method
    client.py             # OpenLibraryClient — thin HTTP wrapper

src/storymesh/schemas/
    book_fetcher.py       # Pydantic input/output schemas

src/storymesh/orchestration/nodes/
    book_fetcher.py       # LangGraph node wrapper

tests/
    test_schemas_book_fetcher.py
    test_book_fetcher_client.py
    test_book_fetcher_agent.py
```

---

## Pydantic Schemas (`src/storymesh/schemas/book_fetcher.py`)

Follow the patterns in `src/storymesh/schemas/genre_normalizer.py`.

### Input Schema

```python
class BookFetcherAgentInput(BaseModel):
    """Input contract for the BookFetcherAgent."""

    normalized_genres: list[str] = Field(
        min_length=1,
        description="Normalized genre names from the GenreNormalizerAgent.",
    )
    limit_per_genre: int = Field(
        default=30,
        description="Maximum number of books to fetch per genre query.",
    )
```

### Output Schema

```python
class BookRecord(BaseModel):
    """Metadata for a single book returned by the Open Library Search API."""

    model_config = {"frozen": True}

    work_key: str = Field(
        description="Open Library work key (e.g., '/works/OL27448W'). Unique identifier.",
    )
    title: str
    authors: list[str] = Field(
        default_factory=list,
        description="Author name(s). Empty list if not present in API response.",
    )
    first_publish_year: int | None = Field(
        default=None,
        description="Year of first publication. None if unavailable.",
    )
    edition_count: int = Field(
        default=0,
        description="Number of editions. Primary popularity proxy.",
    )
    ratings_average: float | None = Field(
        default=None,
        description="Average rating (0–5 scale). None if no ratings exist.",
    )
    ratings_count: int = Field(
        default=0,
        description="Number of ratings.",
    )
    subjects: list[str] = Field(
        default_factory=list,
        description="Subject tags from Open Library.",
    )
    cover_id: int | None = Field(
        default=None,
        description="Open Library cover image ID. None if no cover.",
    )
    source_query: str = Field(
        description=(
            "The genre string that produced this result. "
            "Lets the BookRankerAgent know which genre each book matched."
        ),
    )


class BookFetcherAgentOutput(BaseModel):
    """Output contract for the BookFetcherAgent."""

    model_config = {"frozen": True}

    books: list[BookRecord] = Field(
        description=(
            "Unranked book records from all genre queries combined. "
            "May contain duplicates (same work under multiple genres). "
            "Deduplication is the BookRankerAgent's responsibility."
        ),
    )
    queries_executed: list[str] = Field(
        description="Genre query strings actually sent to the API. Useful for debugging.",
    )
    schema_version: str = BOOK_FETCHER_SCHEMA_VERSION
```

**`source_query` rationale:** When the user requests "post-apocalyptic mystery," the agent
queries both genres independently. A book might appear under "post apocalyptic" but not
under "mystery." The BookRankerAgent needs this field to compute cross-genre alignment
scores. Without it, that information is lost.

---

## Versioning

### `src/storymesh/versioning/schemas.py`

Add:
```python
BOOK_FETCHER_SCHEMA_VERSION = "1.0"
```

Register in `SCHEMA_VERSIONS`:
```python
"Book Fetcher": BOOK_FETCHER_SCHEMA_VERSION,
```

### `src/storymesh/versioning/agents.py`

Add:
```python
BOOK_FETCHER_AGENT_VERSION = "1.0"
```

Register in `AGENT_VERSIONS`:
```python
"Book Fetcher": BOOK_FETCHER_AGENT_VERSION,
```

---

## OpenLibraryClient (`src/storymesh/agents/book_fetcher/client.py`)

A thin, synchronous HTTP client wrapping `httpx`. Handles all API communication;
the agent handles business logic.

### Custom Exception

```python
class OpenLibraryAPIError(Exception):
    """Raised when the Open Library API returns an error or is unreachable."""
```

### Class Design

```python
class OpenLibraryClient:
    """Thin synchronous HTTP wrapper for the Open Library Search API."""

    _BASE_URL = "https://openlibrary.org/search.json"
    _FIELDS = "key,title,author_name,first_publish_year,edition_count,ratings_average,ratings_count,subject,cover_i"

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            user_agent: Identifies this app to Open Library for a higher rate limit.
                If None, the client operates anonymously at 1 req/sec.
                If set, the client may make up to 3 req/sec.
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
    ) -> list[dict[str, object]]:
        """Query the Open Library Search API for books by subject.

        Args:
            subject: Subject/genre string (spaces, not underscores).
            limit: Maximum number of results to return.
            sort: Sort order. "editions" returns most widely published first.

        Returns:
            The raw ``docs`` list from the API response.

        Raises:
            OpenLibraryAPIError: On HTTP errors (after retries), timeouts,
                or JSON parse failures.
        """
        params = {
            "subject": subject,
            "fields": self._FIELDS,
            "limit": limit,
            "sort": sort,
        }
        return self._get_with_retry(params)

    def close(self) -> None:
        """Close the underlying httpx.Client."""
        self._client.close()
```

### Error Handling

- **HTTP 429:** Wait 2 seconds and retry once. If still 429, raise `OpenLibraryAPIError`.
- **HTTP 5xx:** Retry once after 1 second. If still failing, raise `OpenLibraryAPIError`.
- **HTTP 4xx (other):** Raise `OpenLibraryAPIError` immediately.
- **Timeout (`httpx.TimeoutException`):** Raise `OpenLibraryAPIError`.
- **JSON parse failure:** Raise `OpenLibraryAPIError`.

Implement a private `_get_with_retry(params)` method that encapsulates this logic.
Extract the `docs` list from the response JSON (default to `[]` if `docs` key is absent).

### Context Manager Support

Implement `__enter__` / `__exit__` so callers can use `with OpenLibraryClient(...) as client:`.
`__exit__` calls `self.close()`.

---

## BookFetcherAgent (`src/storymesh/agents/book_fetcher/agent.py`)

### Constructor

```python
class BookFetcherAgent:
    """Fetches genre-relevant books from the Open Library Search API (Stage 1)."""

    def __init__(
        self,
        client: OpenLibraryClient | None = None,
        cache_ttl: int = 86400,
    ) -> None:
        """
        Args:
            client: Pre-built OpenLibraryClient. If None, one is constructed
                from config (reads user_agent from get_api_client_config("open_library")).
            cache_ttl: Cache TTL in seconds. Default 86400 (24 hours).
        """
        if client is None:
            ol_cfg = get_api_client_config("open_library")
            user_agent: str | None = ol_cfg.get("user_agent") or None
            self._client = OpenLibraryClient(user_agent=user_agent)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

        cache_dir = get_cache_dir("open_library")
        self._cache: diskcache.Cache = diskcache.Cache(str(cache_dir), timeout=cache_ttl)
```

Note: `self._owns_client` tracks whether this instance created the client (and thus is
responsible for closing it).

### run() Method

```python
    def run(self, input_data: BookFetcherAgentInput) -> BookFetcherAgentOutput:
        """Fetch books for each normalized genre and return the combined results.

        Args:
            input_data: Validated input contract from the GenreNormalizerAgent.

        Returns:
            A frozen BookFetcherAgentOutput with all book records combined.
        """
```

Algorithm:
1. For each genre in `input_data.normalized_genres`:
   a. Convert to Open Library subject format: replace underscores with spaces, lowercase.
   b. Build cache key: `f"ol_search:{subject}:{input_data.limit_per_genre}:editions"`
   c. Check `self._cache` for this key.
   d. **Cache hit:** Load the raw docs list from cache (orjson-decode the stored bytes).
      Skip the sleep.
   e. **Cache miss:** Call `self._client.fetch_books_by_subject(subject, limit)`.
      Store the result in cache as orjson-encoded bytes with the configured TTL.
      Sleep `self._client.rate_limit_delay` seconds after the call
      (except after the last genre to avoid an unnecessary delay).
   f. Parse each raw dict into a `BookRecord`, setting `source_query` to the genre string.
   g. Append all `BookRecord` objects to the combined results list.
   h. Append the subject string to `queries_executed`.
2. Return `BookFetcherAgentOutput(books=all_records, queries_executed=queries_executed)`.

### BookRecord Parsing

Parse raw dicts defensively. Each field that could be missing must default gracefully:

```python
def _parse_book_record(self, doc: dict[str, object], source_query: str) -> BookRecord | None:
    """Parse a raw Open Library doc dict into a BookRecord.

    Returns None if the doc lacks a work key or title (minimum required fields).
    """
    work_key = doc.get("key")
    title = doc.get("title")
    if not work_key or not title:
        return None

    return BookRecord(
        work_key=str(work_key),
        title=str(title),
        authors=list(doc.get("author_name", [])),          # type: ignore[arg-type]
        first_publish_year=doc.get("first_publish_year"),  # type: ignore[arg-type]
        edition_count=int(doc.get("edition_count", 0)),
        ratings_average=doc.get("ratings_average"),         # type: ignore[arg-type]
        ratings_count=int(doc.get("ratings_count", 0)),
        subjects=list(doc.get("subject", [])),              # type: ignore[arg-type]
        cover_id=doc.get("cover_i"),                        # type: ignore[arg-type]
        source_query=source_query,
    )
```

Docs that fail to parse (missing work_key or title) are silently skipped — an incomplete
response from Open Library should not crash the pipeline.

### Cleanup

Implement `__enter__` / `__exit__` on `BookFetcherAgent`. `__exit__` closes
`self._cache` and calls `self._client.close()` if `self._owns_client`.

---

## LangGraph Node Wrapper (`src/storymesh/orchestration/nodes/book_fetcher.py`)

Follow the exact factory pattern from `genre_normalizer.py`:

```python
def make_book_fetcher_node(
    agent: BookFetcherAgent,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for BookFetcherAgent (Stage 1).

    Args:
        agent: A fully constructed BookFetcherAgent instance.

    Returns:
        A node callable with signature StoryMeshState -> dict[str, Any].
    """

    def book_fetcher_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 1 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``genre_normalizer_output``.

        Returns:
            Partial state update dict with ``book_fetcher_output`` set.
        """
        genre_output = state["genre_normalizer_output"]
        input_data = BookFetcherAgentInput(
            normalized_genres=genre_output.normalized_genres,
        )
        output = agent.run(input_data)
        return {"book_fetcher_output": output}

    return book_fetcher_node
```

---

## Pipeline State Changes

### `src/storymesh/orchestration/state.py`

Replace all stage 1–7 fields. The pipeline now has 7 stages (0–6), down from 8.
`BookProfileSynthesizerAgent` and `ThemeAggregatorAgent` have been consolidated into
`ThemeExtractorAgent` (a single LLM call).

**Old fields → New fields:**

| Old field name | New field name | Type |
|---|---|---|
| `genre_seed_fetcher_output` | `book_fetcher_output` | `BookFetcherAgentOutput \| None` |
| `seed_ranker_output` | `book_ranker_output` | `object \| None` |
| `book_profile_synthesizer_output` | *(removed)* | — |
| `theme_aggregator_output` | `theme_extractor_output` | `object \| None` |
| `proposal_output` | `proposal_draft_output` | `object \| None` |
| `rubric_judge_output` | `rubric_judge_output` | `object \| None` *(unchanged)* |
| `synthesis_writer_output` | `synopsis_writer_output` | `object \| None` |

Add the import:
```python
from storymesh.schemas.book_fetcher import BookFetcherAgentOutput
```

Full updated stage block:
```python
# ── Stage 1: BookFetcherAgent ──────────────────────────────────────
book_fetcher_output: BookFetcherAgentOutput | None

# ── Stage 2: BookRankerAgent ───────────────────────────────────────
# TODO: Replace object with BookRankerAgentOutput once implemented.
book_ranker_output: object | None

# ── Stage 3: ThemeExtractorAgent (LLM) ────────────────────────────
# TODO: Replace object with ThemePack once implemented.
theme_extractor_output: object | None

# ── Stage 4: ProposalDraftAgent (LLM) ─────────────────────────────
# TODO: Replace object with ProposalDraftOutput once implemented.
proposal_draft_output: object | None

# ── Stage 5: RubricJudgeAgent (LLM, conditional retry edge) ───────
# TODO: Replace object with RubricResult once implemented.
rubric_judge_output: object | None

# ── Stage 6: SynopsisWriterAgent (LLM) ────────────────────────────
# TODO: Replace object with SynopsisWriterOutput once implemented.
synopsis_writer_output: object | None
```

### `src/storymesh/orchestration/pipeline.py`

Update `initial_state` to use the new field names:

```python
initial_state: StoryMeshState = {
    "input_genre": genre,
    "pipeline_version": storymesh_version,
    "genre_normalizer_output": None,
    "book_fetcher_output": None,
    "book_ranker_output": None,
    "theme_extractor_output": None,
    "proposal_draft_output": None,
    "rubric_judge_output": None,
    "synopsis_writer_output": None,
    "errors": [],
}
```

Update `stage_outputs` to use the new field names and new node name keys:

```python
stage_outputs: dict[str, object | None] = {
    "genre_normalizer": final_state.get("genre_normalizer_output"),
    "book_fetcher": final_state.get("book_fetcher_output"),
    "book_ranker": final_state.get("book_ranker_output"),
    "theme_extractor": final_state.get("theme_extractor_output"),
    "proposal_draft": final_state.get("proposal_draft_output"),
    "rubric_judge": final_state.get("rubric_judge_output"),
    "synopsis_writer": final_state.get("synopsis_writer_output"),
}
```

### `src/storymesh/orchestration/graph.py`

**Node renames (old → new):**

| Old node name | New node name | Status |
|---|---|---|
| `seed_fetcher` | `book_fetcher` | **Real node** (wire in `make_book_fetcher_node`) |
| `seed_ranker` | `book_ranker` | `_noop_node` placeholder |
| `book_profile_synthesizer` | *(removed)* | — |
| `theme_aggregator` | `theme_extractor` | `_noop_node` placeholder |
| `proposal` | `proposal_draft` | `_noop_node` placeholder |
| `rubric_judge` | `rubric_judge` | `_noop_node` placeholder *(name unchanged)* |
| `synthesis_writer` | `synopsis_writer` | `_noop_node` placeholder |

**New graph topology (linear, 7 stages):**
```
START → genre_normalizer → book_fetcher → book_ranker → theme_extractor
      → proposal_draft → rubric_judge → synopsis_writer → END
```

**`build_graph()` changes:**

Add `BookFetcherAgent` construction under the Stage 1 comment. The agent requires no LLM
client — pass no provider arguments:

```python
# ── Stage 1: BookFetcherAgent ──────────────────────────────────────
from storymesh.agents.book_fetcher.agent import BookFetcherAgent          # noqa: PLC0415
from storymesh.orchestration.nodes.book_fetcher import make_book_fetcher_node  # noqa: PLC0415

book_fetcher_agent = BookFetcherAgent()
book_fetcher_node = make_book_fetcher_node(book_fetcher_agent)
```

Move the `# Future node imports` comment block to reflect new names.

Update the docstring's topology description in `build_graph()` to match the new 7-node
linear graph.

---

## Testing Strategy

### `tests/test_schemas_book_fetcher.py`

```
TestBookRecord:
  - test_valid_all_fields: construct with every field populated
  - test_valid_optional_fields_absent: None/empty defaults work
  - test_missing_work_key_raises: work_key is required
  - test_missing_title_raises: title is required
  - test_frozen: mutation raises ValidationError

TestBookFetcherAgentInput:
  - test_valid: normalized_genres=["mystery"], limit_per_genre defaults to 30
  - test_custom_limit: limit_per_genre=10 is accepted
  - test_empty_genres_raises: min_length=1 enforced

TestBookFetcherAgentOutput:
  - test_valid_with_books: books list with real BookRecord objects
  - test_valid_empty_books: empty list is valid (a genre may return nothing)
  - test_frozen: mutation raises ValidationError
  - test_schema_version: value matches BOOK_FETCHER_SCHEMA_VERSION

TestRoundTrip:
  - test_output_roundtrip: model_dump_json → model_validate_json produces equal object
```

### `tests/test_book_fetcher_client.py`

Use `unittest.mock.patch` or `pytest-mock` to mock `httpx.Client.get`. Do not hit the
real API.

```
TestOpenLibraryClientInit:
  - test_no_user_agent_sets_slow_rate_limit: rate_limit_delay == 1.0
  - test_user_agent_sets_fast_rate_limit: rate_limit_delay == 0.4
  - test_user_agent_header_set: httpx.Client constructed with correct User-Agent header
  - test_no_user_agent_no_header: no User-Agent header when user_agent=None

TestFetchBooksBySubject:
  - test_successful_response: returns docs list from mocked response
  - test_empty_docs: returns [] when docs key missing in response
  - test_fields_parameter_correct: verifies the fields query param is constructed correctly
  - test_subject_passed_correctly: subject appears in request params

TestErrorHandling:
  - test_429_retries_once_then_raises: mock 429 → 429 raises OpenLibraryAPIError
  - test_429_retries_once_succeeds: mock 429 → 200 returns docs
  - test_5xx_retries_once_then_raises: mock 500 → 500 raises OpenLibraryAPIError
  - test_5xx_retries_once_succeeds: mock 500 → 200 returns docs
  - test_4xx_raises_immediately: mock 404 raises OpenLibraryAPIError without retry
  - test_timeout_raises: httpx.TimeoutException raises OpenLibraryAPIError
  - test_json_parse_failure_raises: invalid JSON body raises OpenLibraryAPIError

TestContextManager:
  - test_context_manager_closes_client: __exit__ calls close()
```

### `tests/test_book_fetcher_agent.py`

Use a mock `OpenLibraryClient` (constructed manually with a spec or simple stub) to avoid
any real HTTP calls. Use `tmp_path` pytest fixture for the diskcache directory — pass the
client directly to the constructor so config is not loaded.

```
TestBookFetcherAgentConstruction:
  - test_construct_with_client: accepts a pre-built client
  - test_default_cache_ttl: cache_ttl defaults to 86400

TestGenreConversion:
  - test_underscore_to_space: "post_apocalyptic" → "post apocalyptic" in the API call
  - test_already_no_underscores: "mystery" remains "mystery"
  - test_lowercased: genre names are lowercased before querying

TestSingleGenre:
  - test_single_genre_returns_book_records: mock returns 2 docs → 2 BookRecords
  - test_source_query_set_correctly: BookRecord.source_query matches the genre string
  - test_empty_api_response_returns_empty: mock returns [] → books is []

TestMultiGenre:
  - test_two_genres_combined: mock returns 2+3 docs → 5 BookRecords in output
  - test_source_query_per_genre: each BookRecord has the correct source_query
  - test_queries_executed_contains_all_subjects: queries_executed has both subjects

TestCaching:
  - test_cache_hit_skips_client: second call for same genre does not call client again
  - test_cache_miss_calls_client: first call for a genre calls client exactly once
  - test_different_genres_cached_independently: "mystery" and "fantasy" cached separately

TestDocParsing:
  - test_doc_missing_work_key_skipped: doc without "key" is silently skipped
  - test_doc_missing_title_skipped: doc without "title" is silently skipped
  - test_optional_fields_default: doc with only key+title gets defaults for all other fields

TestRateLimiting:
  - test_sleep_called_between_genres: sleep is called between the first and second genre
  - test_sleep_not_called_on_cache_hit: no sleep when result came from cache
```

### Integration Test (optional)

```python
@pytest.mark.real_api
def test_real_open_library_mystery() -> None:
    """Fetch real mystery books from Open Library and verify schema compliance."""
    agent = BookFetcherAgent()
    result = agent.run(BookFetcherAgentInput(normalized_genres=["mystery"]))
    assert len(result.books) > 0
    assert all(isinstance(b, BookRecord) for b in result.books)
    assert result.queries_executed == ["mystery"]
```

### `tests/test_graph.py`

Update `test_full_initial_state_is_valid` to use the new state field names:

```python
def test_full_initial_state_is_valid(self) -> None:
    state: StoryMeshState = {
        "input_genre": "cozy mystery",
        "pipeline_version": "0.4.0",
        "genre_normalizer_output": None,
        "book_fetcher_output": None,
        "book_ranker_output": None,
        "theme_extractor_output": None,
        "proposal_draft_output": None,
        "rubric_judge_output": None,
        "synopsis_writer_output": None,
        "errors": [],
    }
    assert state["input_genre"] == "cozy mystery"
    assert state["errors"] == []
    assert state["genre_normalizer_output"] is None
```

---

## Build Order

Follow this sequence. Run tests at each step before proceeding.

1. Update `storymesh.config.yaml` — add `api_clients.open_library.user_agent`
2. Update `src/storymesh/config.py` — add `get_api_client_config()` and `get_cache_dir()`
3. Update `src/storymesh/versioning/schemas.py` — add `BOOK_FETCHER_SCHEMA_VERSION`
4. Update `src/storymesh/versioning/agents.py` — add `BOOK_FETCHER_AGENT_VERSION`
5. Create `src/storymesh/schemas/book_fetcher.py` — Pydantic schemas
6. Create `tests/test_schemas_book_fetcher.py` — run and verify
7. Create `src/storymesh/agents/book_fetcher/__init__.py`
8. Create `src/storymesh/agents/book_fetcher/client.py` — OpenLibraryClient
9. Create `tests/test_book_fetcher_client.py` — run and verify
10. Create `src/storymesh/agents/book_fetcher/agent.py` — BookFetcherAgent
11. Create `tests/test_book_fetcher_agent.py` — run and verify
12. Create `src/storymesh/orchestration/nodes/book_fetcher.py` — LangGraph node
13. Update `src/storymesh/orchestration/state.py` — rename fields, add import
14. Update `src/storymesh/orchestration/pipeline.py` — rename fields
15. Update `src/storymesh/orchestration/graph.py` — rename nodes, wire real node
16. Update `tests/test_graph.py` — fix `test_full_initial_state_is_valid`
17. Run full test suite: `pytest`
18. Run `ruff check src/ tests/` and `mypy src/` — verify lint and type compliance

---

## File Summary

| File | Action |
|------|--------|
| `storymesh.config.yaml` | Update — add `api_clients.open_library.user_agent` |
| `src/storymesh/config.py` | Update — add `get_api_client_config()`, `get_cache_dir()` |
| `src/storymesh/versioning/schemas.py` | Update — add `BOOK_FETCHER_SCHEMA_VERSION` |
| `src/storymesh/versioning/agents.py` | Update — add `BOOK_FETCHER_AGENT_VERSION` |
| `src/storymesh/schemas/book_fetcher.py` | Create — `BookRecord`, `BookFetcherAgentInput`, `BookFetcherAgentOutput` |
| `src/storymesh/agents/book_fetcher/__init__.py` | Create — package init |
| `src/storymesh/agents/book_fetcher/client.py` | Create — `OpenLibraryClient`, `OpenLibraryAPIError` |
| `src/storymesh/agents/book_fetcher/agent.py` | Create — `BookFetcherAgent` |
| `src/storymesh/orchestration/nodes/book_fetcher.py` | Create — `make_book_fetcher_node` |
| `src/storymesh/orchestration/state.py` | Update — rename all stage 1–6 fields |
| `src/storymesh/orchestration/pipeline.py` | Update — `initial_state` and `stage_outputs` |
| `src/storymesh/orchestration/graph.py` | Update — wire real node, rename placeholder nodes |
| `tests/test_schemas_book_fetcher.py` | Create — schema validation tests |
| `tests/test_book_fetcher_client.py` | Create — client tests with mocked `httpx` |
| `tests/test_book_fetcher_agent.py` | Create — agent tests with mock client |
| `tests/test_graph.py` | Update — fix state field names in one test |

---

## What This Plan Does NOT Cover

- Deduplication of books found under multiple genre queries — BookRankerAgent's job
- Scoring or ranking — BookRankerAgent's job
- Any LLM calls — this agent is purely an API client
- Subgenre queries — initial implementation queries by normalized genre only
