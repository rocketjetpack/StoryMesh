# StoryMesh Implementation Plan — BookFetcher Enrichment & BookRankerAgent

**Date:** 2026-03-27
**Version:** 0.5.0 → 0.6.0
**Scope:** Enrich BookFetcher with Open Library reader engagement fields, implement BookRankerAgent (Stage 2)

This document is the authoritative implementation plan for two tightly coupled work items: enriching the BookFetcherAgent's data capture from the Open Library API, and implementing the BookRankerAgent that consumes the enriched data. It follows the format established by Refine-Pass-1 and the BookFetcherAgent plan.

---

## Table of Contents

1. [Work Item Ordering and Dependencies](#1-work-item-ordering-and-dependencies)
2. [WI-1: BookFetcher Schema Enrichment](#2-wi-1-bookfetcher-schema-enrichment)
3. [WI-2: BookRankerAgent Schemas](#3-wi-2-bookrankeragent-schemas)
4. [WI-3: BookRankerAgent Scoring Engine](#4-wi-3-bookrankeragent-scoring-engine)
5. [WI-4: BookRankerAgent Core](#5-wi-4-bookrankeragent-core)
6. [WI-5: BookRanker LLM Re-rank Prompt](#6-wi-5-bookranker-llm-re-rank-prompt)
7. [WI-6: BookRanker LangGraph Node](#7-wi-6-bookranker-langgraph-node)
8. [WI-7: Graph Wiring and State Update](#8-wi-7-graph-wiring-and-state-update)
9. [WI-8: Configuration](#9-wi-8-configuration)
10. [WI-9: README Update](#10-wi-9-readme-update)
11. [Open Library API Compliance](#11-open-library-api-compliance)
12. [Version Bump Strategy](#12-version-bump-strategy)
13. [Validation Checklist](#13-validation-checklist)

---

## 1. Work Item Ordering and Dependencies

```text
WI-1 (BookFetcher enrichment)  — no dependencies, do first
  │
WI-2 (BookRanker schemas)      — depends on WI-1 (enriched BookRecord)
  │
WI-3 (scoring engine)          — depends on WI-2 (schema types)
  │
WI-4 (agent core)              — depends on WI-2, WI-3
  │
WI-5 (LLM re-rank prompt)      — depends on WI-2 (schema fields for prompt placeholders)
  │
WI-6 (LangGraph node)          — depends on WI-4
  │
WI-7 (graph wiring + state)    — depends on WI-6
  │
WI-8 (configuration)           — depends on WI-4 (default values must be settled)
  │
WI-9 (README update)           — LAST
```

**Recommended execution order:** WI-1 → WI-2 → WI-3 → WI-5 → WI-4 → WI-6 → WI-7 → WI-8 → WI-9

Note: WI-5 (prompt file) is placed before WI-4 because the agent's `__init__` will reference the prompt loader, and having the YAML file present avoids import-time errors.

---

## 2. WI-1: BookFetcher Schema Enrichment

### Rationale

The Open Library Search API provides reader engagement fields (`readinglog_count`, `want_to_read_count`, `already_read_count`, `currently_reading_count`) and `number_of_pages_median` that are available at zero additional API cost — they are additional Solr fields returned in the same search response. These provide direct popularity and engagement signals that are superior to proxying through `edition_count`. The BookRankerAgent needs these fields for accurate scoring.

### Open Library API Compliance

This change requests more fields in the same `fields=` parameter of the existing Search API call. It does NOT add extra API calls, increase request frequency, or change the caching behavior. Per Open Library's guidance, applications should "set the fields parameter on their requests with the specific fields their application requires." We are following this guidance precisely.

### Files Affected

- `src/storymesh/agents/book_fetcher/client.py` — update `_FIELDS` constant
- `src/storymesh/schemas/book_fetcher.py` — add new fields to `BookRecord`
- `src/storymesh/agents/book_fetcher/agent.py` — update `_parse_book_record()`
- `src/storymesh/versioning/schemas.py` — bump `BOOK_FETCHER_SCHEMA_VERSION` to `"1.2"`
- `tests/test_book_fetcher_client.py` — update field assertion tests
- `tests/test_book_fetcher_agent.py` — update stub data and parsing tests
- `tests/test_schemas_book_fetcher.py` — test new fields

### Changes

#### `src/storymesh/agents/book_fetcher/client.py`

Update the `_FIELDS` constant to include the new fields:

```python
# BEFORE
_FIELDS = (
    "key,title,author_name,first_publish_year,"
    "edition_count,ratings_average,ratings_count,subject,cover_i"
)

# AFTER
_FIELDS = (
    "key,title,author_name,first_publish_year,"
    "edition_count,ratings_average,ratings_count,"
    "readinglog_count,want_to_read_count,already_read_count,"
    "currently_reading_count,number_of_pages_median,"
    "subject,cover_i"
)
```

#### `src/storymesh/schemas/book_fetcher.py`

Add five new fields to `BookRecord`. All are optional with safe defaults so existing cached data and tests degrade gracefully:

```python
readinglog_count: int = Field(
    default=0,
    description=(
        "Total number of Open Library users who have this book on any "
        "reading shelf (want to read + currently reading + already read). "
        "Direct popularity signal for ranking."
    ),
)
want_to_read_count: int = Field(
    default=0,
    description="Number of users with this book on their 'want to read' shelf.",
)
already_read_count: int = Field(
    default=0,
    description="Number of users who have marked this book as 'already read'.",
)
currently_reading_count: int = Field(
    default=0,
    description="Number of users currently reading this book.",
)
number_of_pages_median: int | None = Field(
    default=None,
    description=(
        "Median page count across all editions. None if unavailable. "
        "Not used for ranking but useful for downstream synopsis calibration."
    ),
)
```

#### `src/storymesh/agents/book_fetcher/agent.py`

Update `_parse_book_record()` to populate the new fields:

```python
# Add to the BookRecord constructor call in _parse_book_record():
readinglog_count=int(doc.get("readinglog_count", 0)),
want_to_read_count=int(doc.get("want_to_read_count", 0)),
already_read_count=int(doc.get("already_read_count", 0)),
currently_reading_count=int(doc.get("currently_reading_count", 0)),
number_of_pages_median=doc.get("number_of_pages_median"),  # type: ignore[arg-type]
```

#### `src/storymesh/versioning/schemas.py`

```python
# BEFORE
BOOK_FETCHER_SCHEMA_VERSION = "1.1"

# AFTER
BOOK_FETCHER_SCHEMA_VERSION = "1.2"
```

Add version history comment:

```python
# 2026-03-27: Increment Book Fetcher schema to 1.2. Added readinglog_count,
#             want_to_read_count, already_read_count, currently_reading_count,
#             and number_of_pages_median fields to BookRecord for richer
#             downstream ranking signals.
```

### Cache Invalidation Note

Existing cached responses will not contain the new fields. Because all new fields have `default=0` or `default=None`, cached data will parse correctly — books fetched before this change simply have zero engagement counts, which is factually accurate (we don't know their values from the old cache). The 24-hour cache TTL means fresh data with the new fields will flow in naturally. No forced cache purge is needed, but the plan should document that a `storymesh cache clear` (or deleting `~/.cache/storymesh/open_library`) would force immediate refresh.

### Testing

- Test that `_FIELDS` contains all new field names (`readinglog_count`, etc.)
- Test that `_parse_book_record` correctly populates new fields from API docs
- Test that `_parse_book_record` defaults to 0 / None when new fields are absent (backward compat with cached data)
- Test that `BookRecord` validates with all new fields provided
- Test that `BookRecord` validates with new fields absent (defaults apply)
- Run `ruff check` and `mypy` to verify type compliance

---

## 3. WI-2: BookRankerAgent Schemas

### Rationale

The BookRankerAgent needs its own input, output, and intermediate Pydantic schemas. Following the dual-output pattern: full detail in artifacts, slim summaries in state for downstream LLM token efficiency.

### Files Affected

- `src/storymesh/schemas/book_ranker.py` — CREATE
- `src/storymesh/versioning/schemas.py` — add `BOOK_RANKER_SCHEMA_VERSION`
- `src/storymesh/versioning/agents.py` — add `BOOK_RANKER_AGENT_VERSION`
- `tests/test_schemas_book_ranker.py` — CREATE

### Schemas

#### `BookRankerAgentInput`

```python
class BookRankerAgentInput(BaseModel):
    """Input contract for the BookRankerAgent."""

    books: list[BookRecord] = Field(
        min_length=1,
        description="Deduplicated book records from the BookFetcherAgent.",
    )
    user_prompt: str = Field(
        min_length=1,
        description=(
            "Original user input string. Passed through for the optional "
            "LLM re-rank prompt, which assesses narrative potential."
        ),
    )
    total_genres_queried: int = Field(
        ge=1,
        description=(
            "Number of genres that were queried by the BookFetcherAgent. "
            "Used as the denominator for genre overlap scoring."
        ),
    )
```

#### `RankedBook` (full detail — stored in artifacts)

```python
class ScoreBreakdown(BaseModel):
    """Individual scoring components for a ranked book."""

    model_config = {"frozen": True}

    genre_overlap: float = Field(ge=0.0, le=1.0)
    reader_engagement: float = Field(ge=0.0, le=1.0)
    rating_quality: float = Field(ge=0.0, le=1.0)
    rating_volume: float = Field(ge=0.0, le=1.0)


class RankedBook(BaseModel):
    """A book with its computed ranking data. Full detail version for artifacts."""

    model_config = {"frozen": True}

    book: BookRecord
    composite_score: float = Field(
        ge=0.0,
        description="Final weighted composite score.",
    )
    score_breakdown: ScoreBreakdown
    rank: int = Field(ge=1, description="1-indexed rank position.")
```

#### `RankedBookSummary` (slim — passed downstream in state)

```python
class RankedBookSummary(BaseModel):
    """Slim book representation for downstream LLM consumption.

    Strips scoring internals and fields not needed by ThemeExtractor
    or ProposalDraft to minimize LLM token usage.
    """

    model_config = {"frozen": True}

    work_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    first_publish_year: int | None = None
    source_genres: list[str]
    composite_score: float
    rank: int = Field(ge=1)
```

#### `BookRankerAgentOutput`

```python
class BookRankerAgentOutput(BaseModel):
    """Output contract for the BookRankerAgent."""

    model_config = {"frozen": True}

    ranked_books: list[RankedBook] = Field(
        description="Full-detail ranked books, ordered by rank. Persisted in artifacts.",
    )
    ranked_summaries: list[RankedBookSummary] = Field(
        description=(
            "Slim ranked book summaries for downstream LLM agents. "
            "Same ordering as ranked_books."
        ),
    )
    dropped_count: int = Field(
        ge=0,
        description="Number of books that fell below the top_n cutoff.",
    )
    llm_reranked: bool = Field(
        default=False,
        description="Whether the LLM re-rank path was applied.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Scoring metadata: weights used, score distribution stats, "
            "LLM re-rank details if applicable."
        ),
    )
    schema_version: str = BOOK_RANKER_SCHEMA_VERSION
```

#### Versioning

```python
# schemas.py
BOOK_RANKER_SCHEMA_VERSION = "1.0"

# agents.py
BOOK_RANKER_AGENT_VERSION = "1.0"
```

### Testing

- Test that all schema models construct correctly with valid data
- Test frozen model enforcement (cannot mutate after construction)
- Test that `RankedBookSummary` can be constructed from a `RankedBook`
- Test validation failures on invalid data (negative scores, rank < 1, etc.)
- Test that `BookRankerAgentInput` rejects empty books list

---

## 4. WI-3: BookRankerAgent Scoring Engine

### Rationale

Scoring math is pure-function territory — no state, no side effects. Separating it into `scorer.py` makes it independently testable with exact float assertions. This follows the same separation as GenreNormalizerAgent's `resolver.py` / `agent.py` split.

### File

- `src/storymesh/agents/book_ranker/scorer.py` — CREATE
- `tests/test_book_ranker_scorer.py` — CREATE

### Scoring Algorithm

#### Default Weights

```python
DEFAULT_WEIGHTS: dict[str, float] = {
    "genre_overlap": 0.40,
    "reader_engagement": 0.25,
    "rating_quality": 0.20,
    "rating_volume": 0.15,
}
DEFAULT_RATING_CONFIDENCE_THRESHOLD: int = 50
```

#### Component Functions

All component functions return a float in `[0.0, 1.0]`.

**1. Genre Overlap Score (weight: 0.40)**

```python
def score_genre_overlap(source_genres_count: int, total_genres_queried: int) -> float:
    """Fraction of queried genres that returned this book."""
    if total_genres_queried <= 0:
        return 0.0
    return min(1.0, source_genres_count / total_genres_queried)
```

**2. Reader Engagement Score (weight: 0.25)**

```python
def score_reader_engagement(
    readinglog_count: int,
    min_readinglog: int,
    max_readinglog: int,
) -> float:
    """Min-max normalized readinglog_count across the current batch."""
    if max_readinglog == min_readinglog:
        return 0.5  # Uniform data: treat neutrally
    return (readinglog_count - min_readinglog) / (max_readinglog - min_readinglog)
```

**3. Rating Quality Score (weight: 0.20)**

```python
def score_rating_quality(
    ratings_average: float | None,
    ratings_count: int,
    confidence_threshold: int,
) -> float:
    """Confidence-adjusted rating: discounts ratings with low sample sizes."""
    if ratings_average is None or ratings_average <= 0:
        return 0.0
    confidence = min(1.0, ratings_count / confidence_threshold)
    return (ratings_average / 5.0) * confidence
```

**4. Rating Volume Score (weight: 0.15)**

```python
def score_rating_volume(
    ratings_count: int,
    min_ratings: int,
    max_ratings: int,
) -> float:
    """Min-max normalized ratings_count across the current batch."""
    if max_ratings == min_ratings:
        return 0.5
    return (ratings_count - min_ratings) / (max_ratings - min_ratings)
```

#### Composite Scoring Function

```python
def compute_scores(
    books: list[BookRecord],
    total_genres_queried: int,
    weights: dict[str, float] | None = None,
    confidence_threshold: int = DEFAULT_RATING_CONFIDENCE_THRESHOLD,
) -> list[tuple[BookRecord, float, ScoreBreakdown]]:
    """Score all books and return (book, composite_score, breakdown) tuples.

    Sorts by composite_score descending. The caller handles top_n truncation.
    """
```

This function:
1. Computes min/max values for `readinglog_count` and `ratings_count` across the full batch (needed for normalization)
2. Calls each component function per book
3. Computes the weighted composite
4. Returns sorted results (highest score first)

### Testing

- Test each component function independently with known inputs and exact expected floats
- Test `score_genre_overlap` with 1/3 genres, 2/3, 3/3
- Test `score_reader_engagement` with uniform data (all same readinglog → 0.5)
- Test `score_reader_engagement` with varied data (min/max normalization)
- Test `score_rating_quality` with None ratings_average → 0.0
- Test `score_rating_quality` with low confidence (5 ratings, threshold 50)
- Test `score_rating_quality` with full confidence (500 ratings, threshold 50)
- Test `score_rating_volume` with uniform data → 0.5
- Test `compute_scores` returns results sorted by composite_score descending
- Test `compute_scores` with custom weights overriding defaults
- Test `compute_scores` with a single book (edge case: min == max for everything)

---

## 5. WI-4: BookRankerAgent Core

### Rationale

The agent orchestrates: receive input, run deterministic scoring, optionally invoke LLM re-rank, assemble output with dual representations.

### Files

- `src/storymesh/agents/book_ranker/__init__.py` — CREATE
- `src/storymesh/agents/book_ranker/agent.py` — CREATE
- `tests/test_book_ranker_agent.py` — CREATE

### Constructor

```python
class BookRankerAgent:
    """Ranks books by composite scoring with optional LLM re-ranking (Stage 2)."""

    def __init__(
        self,
        *,
        top_n: int = 10,
        weights: dict[str, float] | None = None,
        rating_confidence_threshold: int = 50,
        llm_rerank: bool = False,
        llm_client: LLMClient | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
```

- `top_n`: configurable cutoff, defaults to 10
- `weights`: scoring weights, defaults to `DEFAULT_WEIGHTS` from scorer.py
- `llm_rerank`: whether to invoke LLM re-ranking after deterministic pass
- If `llm_rerank=True` but `llm_client=None`, log a warning and fall back to deterministic only

### `run()` Method

```python
def run(self, input_data: BookRankerAgentInput) -> BookRankerAgentOutput:
```

Algorithm:
1. Call `compute_scores()` from scorer.py with input books and weights
2. Truncate to `top_n`
3. If `llm_rerank` and `llm_client` is available:
   a. Load the prompt via `load_prompt("book_ranker")`
   b. Format user prompt with book list and user's original prompt
   c. Call `llm_client.complete_json()` at the configured temperature
   d. Parse the LLM response (expected: ordered list of work_keys)
   e. Re-order the top_n books according to the LLM's preference
   f. If the LLM call fails or returns unparseable data, log warning and keep deterministic order
4. Build `RankedBook` objects with 1-indexed ranks
5. Build `RankedBookSummary` objects (slim projection)
6. Return `BookRankerAgentOutput`

### Building the Slim Summary

```python
def _to_summary(ranked: RankedBook) -> RankedBookSummary:
    return RankedBookSummary(
        work_key=ranked.book.work_key,
        title=ranked.book.title,
        authors=ranked.book.authors,
        first_publish_year=ranked.book.first_publish_year,
        source_genres=ranked.book.source_genres,
        composite_score=ranked.composite_score,
        rank=ranked.rank,
    )
```

### Testing

- Test basic ranking with known books and expected order
- Test `top_n` truncation (pass 20 books, top_n=5, get 5 back)
- Test `dropped_count` is correct (total - top_n)
- Test that `ranked_summaries` matches `ranked_books` in order and content
- Test that `llm_reranked` is False when LLM is disabled
- Test that `llm_rerank=True` with `llm_client=None` falls back gracefully
- Test with mock LLM client returning valid re-ordering
- Test with mock LLM client returning bad JSON (should fall back to deterministic)
- Test single-book input (edge case)
- Test all books having identical scores (stable sort behavior)

---

## 6. WI-5: BookRanker LLM Re-rank Prompt

### File

- `src/storymesh/prompts/book_ranker.yaml` — CREATE

### Prompt Design

The prompt uses the existing `load_prompt()` / `PromptTemplate` system with `system` and `user` keys. Placeholders use Python `str.format()` syntax.

**System prompt:** Instructs the LLM to act as a fiction editorial advisor, assessing which books from a shortlist have the most potential for generating an interesting, original fiction synopsis given the user's creative brief.

**User prompt placeholders:**
- `{user_prompt}` — the original user input
- `{book_list}` — formatted list of books with titles, authors, source genres
- `{count}` — number of books in the list

**Expected response format:** JSON array of work_keys in the LLM's preferred order:

```json
{"ranked_work_keys": ["/works/OL1W", "/works/OL2W", ...]}
```

### Testing

- Test that `load_prompt("book_ranker")` loads without error
- Test that `format_user()` accepts the expected placeholders
- Test that `format_user()` raises `PromptFormattingError` if a placeholder is missing

---

## 7. WI-6: BookRanker LangGraph Node

### File

- `src/storymesh/orchestration/nodes/book_ranker.py` — CREATE

### Pattern

Follow the exact factory pattern from `genre_normalizer.py` and `book_fetcher.py`:

```python
def make_book_ranker_node(
    agent: BookRankerAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
```

The node:
1. Reads `book_fetcher_output` and `user_prompt` from state
2. Constructs `BookRankerAgentInput`
3. Calls `agent.run()`
4. Persists the full output artifact (if artifact_store is provided)
5. Returns `{"book_ranker_output": output}`

The `total_genres_queried` input is derived from `len(book_fetcher_output.queries_executed)`.

### Testing

- Test node with mock agent and artifact_store=None
- Test that the node reads the correct state keys
- Test artifact persistence when store is provided

---

## 8. WI-7: Graph Wiring and State Update

### Files Affected

- `src/storymesh/orchestration/state.py` — tighten type annotation
- `src/storymesh/orchestration/graph.py` — replace `_noop_node` with real node

### Changes

#### `state.py`

```python
# BEFORE
book_ranker_output: object | None

# AFTER
from storymesh.schemas.book_ranker import BookRankerAgentOutput
book_ranker_output: BookRankerAgentOutput | None
```

#### `graph.py`

In `build_graph()`, add BookRankerAgent construction and wire the real node:

```python
# ── Stage 2: BookRankerAgent ──────────────────────────────────────────
from storymesh.agents.book_ranker.agent import BookRankerAgent
from storymesh.orchestration.nodes.book_ranker import make_book_ranker_node

book_ranker_cfg = get_agent_config("book_ranker")
book_ranker_llm = _build_llm_client(book_ranker_cfg) if book_ranker_cfg.get("llm_rerank") else None

book_ranker_agent = BookRankerAgent(
    top_n=book_ranker_cfg.get("top_n", 10),
    weights=book_ranker_cfg.get("weights"),
    rating_confidence_threshold=book_ranker_cfg.get("rating_confidence_threshold", 50),
    llm_rerank=book_ranker_cfg.get("llm_rerank", False),
    llm_client=book_ranker_llm,
    temperature=book_ranker_cfg.get("temperature", 0.0),
    max_tokens=book_ranker_cfg.get("max_tokens", 1024),
)
book_ranker_node = make_book_ranker_node(book_ranker_agent, artifact_store=artifact_store)
```

Replace the placeholder in the graph:

```python
# BEFORE
graph.add_node("book_ranker", _noop_node)

# AFTER
graph.add_node("book_ranker", book_ranker_node)
```

### Testing

- Test that `build_graph()` compiles without errors
- Test that the book_ranker node receives book_fetcher_output correctly
- Update any existing graph tests that assert on noop behavior for book_ranker

---

## 9. WI-8: Configuration

### File

- `storymesh.config.yaml` — add `book_ranker` section under `agents:`
- `storymesh.config.yaml.example` — same

### Config Section

```yaml
agents:
  book_ranker:
    top_n: 10
    llm_rerank: false
    provider: anthropic
    model: claude-haiku-4-5-20251001
    temperature: 0.0
    max_tokens: 1024
    rating_confidence_threshold: 50
    weights:
      genre_overlap: 0.40
      reader_engagement: 0.25
      rating_quality: 0.20
      rating_volume: 0.15
```

All values have hardcoded defaults in the agent constructor. This config section is entirely optional — the agent works without it.

---

## 10. WI-9: README Update

### Changes

1. **Current Status:** Add BookRankerAgent to the "Implemented" list with a description of deterministic scoring + optional LLM re-rank
2. **Stage 2 section:** Add a description matching the format of Stage 0 and Stage 1
3. **Known Gaps:** Update to reflect that stages 3–6 remain as placeholders
4. **Roadmap:** Mark "Implement deterministic book ranking" as done

---

## 11. Open Library API Compliance

This section documents how StoryMesh complies with Open Library's API usage terms.

### Rate Limits

- **Identified requests (3 req/sec):** The `OpenLibraryClient` sends a User-Agent header when configured. The `rate_limit_delay` of 0.4s enforces compliance.
- **Non-identified fallback (1 req/sec):** When no User-Agent is set, `rate_limit_delay` is 1.0s.
- Rate limit delays are enforced by the `BookFetcherAgent` between API calls. Cache hits skip both the API call and the delay.

### Data Usage

- StoryMesh is an **open-source educational project** (graduate course).
- API usage is **low-volume**: a single pipeline run makes 1–8 genre queries, cached for 24 hours.
- We use the `fields=` parameter to request only the specific fields needed, per Open Library's recommendation.
- We do NOT use the API for bulk download. Our total data volume is small.
- The new fields added in WI-1 do not increase API call count — they are additional Solr fields in the same response.

### Caching

- All API responses are cached to disk via `diskcache` with a 24-hour TTL.
- Cache hits bypass the API entirely, reducing load on Open Library's infrastructure.
- This is respectful caching that serves both our performance needs and Open Library's capacity constraints.

### User-Agent

The config provides a `user_agent` field under `api_clients.open_library`. Users should set this to `"StoryMesh (their-email@example.com)"` to identify the application and receive the 3x rate limit.

---

## 12. Version Bump Strategy

Bump the package version from `0.5.0` to `0.6.0` in `pyproject.toml`. This is a minor version bump because:

- The `BookRecord` schema has new fields (backward-compatible but semantically significant)
- A new agent (`BookRankerAgent`) is added to the pipeline
- The graph topology changes (real node replaces noop)
- New state type annotation for `book_ranker_output`

---

## 13. Validation Checklist

After all work items are complete:

```bash
# 1. All tests pass
pytest

# 2. Type checking passes
mypy src/storymesh/

# 3. Linting passes
ruff check src/ tests/

# 4. CLI still works
storymesh show-version
storymesh show-config
storymesh show-agent-config book_ranker

# 5. Generate command runs end-to-end
storymesh generate "dark post-apocalyptic detective mystery"

# 6. Verify artifacts now include book_ranker_output.json
ls ~/.storymesh/runs/<latest_run_id>/

# 7. Verify enriched BookRecord fields
# Inspect book_fetcher_output.json — books should now have
# readinglog_count, want_to_read_count, etc.

# 8. Verify BookRanker output
# Inspect book_ranker_output.json — should contain ranked_books
# with score_breakdown and ranked_summaries

# 9. If Open Library cache exists from prior runs, optionally clear it
# to force fresh fetches with enriched fields:
# rm -rf ~/.cache/storymesh/open_library
```

---

## File Summary

| File | Action | Work Item |
|------|--------|-----------|
| `src/storymesh/agents/book_fetcher/client.py` | Update `_FIELDS` | WI-1 |
| `src/storymesh/schemas/book_fetcher.py` | Add 5 fields to `BookRecord` | WI-1 |
| `src/storymesh/agents/book_fetcher/agent.py` | Update `_parse_book_record()` | WI-1 |
| `src/storymesh/versioning/schemas.py` | Bump BookFetcher to 1.2, add BookRanker 1.0 | WI-1, WI-2 |
| `src/storymesh/versioning/agents.py` | Add BookRanker 1.0 | WI-2 |
| `src/storymesh/schemas/book_ranker.py` | CREATE — all ranking schemas | WI-2 |
| `src/storymesh/agents/book_ranker/__init__.py` | CREATE — package init | WI-4 |
| `src/storymesh/agents/book_ranker/scorer.py` | CREATE — pure scoring functions | WI-3 |
| `src/storymesh/agents/book_ranker/agent.py` | CREATE — BookRankerAgent | WI-4 |
| `src/storymesh/prompts/book_ranker.yaml` | CREATE — LLM re-rank prompt | WI-5 |
| `src/storymesh/orchestration/nodes/book_ranker.py` | CREATE — LangGraph node | WI-6 |
| `src/storymesh/orchestration/state.py` | Tighten `book_ranker_output` type | WI-7 |
| `src/storymesh/orchestration/graph.py` | Wire real BookRanker node | WI-7 |
| `storymesh.config.yaml` | Add `book_ranker` agent config | WI-8 |
| `storymesh.config.yaml.example` | Same | WI-8 |
| `README.md` | Update status, add Stage 2 docs | WI-9 |
| `tests/test_schemas_book_fetcher.py` | Update for new fields | WI-1 |
| `tests/test_book_fetcher_client.py` | Update field assertions | WI-1 |
| `tests/test_book_fetcher_agent.py` | Update stubs and parsing tests | WI-1 |
| `tests/test_schemas_book_ranker.py` | CREATE — schema tests | WI-2 |
| `tests/test_book_ranker_scorer.py` | CREATE — scoring math tests | WI-3 |
| `tests/test_book_ranker_agent.py` | CREATE — agent tests | WI-4 |
| `tests/test_graph.py` | Update for real book_ranker node | WI-7 |