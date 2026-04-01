# Pipeline Hardening: LLM Logging, Genre Validation, and Book Diversity

This plan covers three interrelated issues discovered during testing of the Pass 4 holistic genre inference feature. They are ordered by implementation priority: each earlier item makes the later ones easier to build and debug.

---

## WI-1: LLM Call Logging (Priority 1)

### Problem

`LLMClient.complete_json()` logs warnings on parse failures but does not log the actual content of LLM interactions. When Pass 3 or Pass 4 returns a response that fails Pydantic validation, there is no way to see what the LLM actually returned, what system prompt it received, or what user message was sent. Debugging requires inserting print statements and re-running — unacceptable for a pipeline that makes multiple LLM calls per run.

### Design

Add structured logging to `LLMClient.complete_json()` in `src/storymesh/llm/base.py`. Every LLM call logs three things at `DEBUG` level:

1. The system prompt (or `"(none)"` if absent)
2. The user prompt
3. The raw response text

On parse failure, the raw response is also logged at `WARNING` level so it's visible even without debug logging enabled.

### Changes to `src/storymesh/llm/base.py`

In `complete_json()`, add logging around the `complete()` call:

```python
def complete_json(self, prompt, *, system_prompt=None, temperature, max_tokens, max_retries=1):
    for attempt in range(max_retries + 1):
        logger.debug(
            "LLM call [attempt %d/%d]\n--- SYSTEM ---\n%s\n--- USER ---\n%s",
            attempt + 1,
            max_retries + 1,
            system_prompt or "(none)",
            prompt,
        )

        raw = self.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.debug("LLM response [attempt %d/%d]\n--- RESPONSE ---\n%s", attempt + 1, max_retries + 1, raw)

        cleaned = _strip_markdown_fences(raw.strip())

        try:
            parsed = orjson.loads(cleaned)
        except orjson.JSONDecodeError:
            logger.warning(
                "JSON parse failed (attempt %d/%d). Raw response:\n%s",
                attempt + 1,
                max_retries + 1,
                raw,
            )
            if attempt < max_retries:
                continue
            raise

        if not isinstance(parsed, dict):
            logger.warning(
                "Expected JSON object, got %s (attempt %d/%d). Raw response:\n%s",
                type(parsed).__name__,
                attempt + 1,
                max_retries + 1,
                raw,
            )
            if attempt < max_retries:
                continue
            raise ValueError(...)

        return parsed
```

### Log Level Rationale

- `DEBUG` for all request/response content. This is verbose and should only appear when a developer explicitly sets `logging.DEBUG` for the `storymesh.llm.base` logger.
- `WARNING` for parse failures with the raw response included. This ensures that failures are visible in normal operation without requiring debug mode.

### Caller-Side Logging

Additionally, `resolve_llm()` and `resolve_holistic()` in `resolver.py` should log at `WARNING` when the Pydantic validation of the parsed JSON fails, including the parsed dict. Currently `resolve_llm()` logs `"LLM response failed schema validation"` but does not include the actual data that failed. Update to:

```python
except Exception:
    logger.warning(
        "LLM response failed schema validation. Parsed data: %s",
        raw_response,
        exc_info=True,
    )
```

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/llm/base.py` | Add DEBUG logging of prompts/responses, include raw response in WARNING on parse failure |
| `src/storymesh/agents/genre_normalizer/resolver.py` | Include parsed data in validation failure warnings (both `resolve_llm()` and `resolve_holistic()`) |

### Testing

- No new tests needed — this is observability-only. Verify manually that `pytest -s --log-cli-level=DEBUG` shows LLM call content when running integration tests.
- Existing tests should continue to pass unchanged since `FakeLLMClient` goes through the same `complete_json()` path.

---

## WI-2: Inferred Genre Validation and Graceful Failure (Priority 2)

### Problem

Pass 4 can infer genres like `young_adult` that don't correspond to valid Open Library subject categories. When the BookFetcher normalizes this to `"young adult"` and queries the API, the query may return zero results or fail entirely. This has two consequences: it wastes an API call (plus rate-limit delay), and if the failure is not handled gracefully, it can disrupt the pipeline.

Additionally, the LLM occasionally returns responses for Pass 4 that fail to parse, though the improved logging from WI-1 will help diagnose the specific failure modes.

### Root Cause Analysis

The pipeline currently has no validation layer between "genre name the LLM inferred" and "subject string Open Library understands." The GenreNormalizerAgent uses canonical genre names like `science_fiction`, `thriller`, `fantasy` — these all map cleanly to Open Library subjects. But Pass 4's LLM can return any genre it considers valid, including categories like `young_adult`, `workplace_fiction`, `techno_thriller`, or `cli_fi` that may not exist as Open Library subjects.

### Design: Genre-to-Subject Mapping

Create a static mapping file `src/storymesh/data/genre_subject_map.json` that maps canonical genre names to Open Library subject query strings. This serves two purposes:

1. **Validation:** If a genre isn't in the map, we know it's not queryable on Open Library.
2. **Translation:** Some genres need different query strings (e.g., `science_fiction` → `"science fiction"`, `young_adult` → `"juvenile fiction"` or `"young adult fiction"`).

```json
{
  "fantasy": ["fantasy"],
  "science_fiction": ["science fiction"],
  "horror": ["horror"],
  "romance": ["romance"],
  "mystery": ["mystery"],
  "thriller": ["thriller", "suspense"],
  "historical_fiction": ["historical fiction"],
  "literary_fiction": ["literary fiction"],
  "adventure": ["adventure"],
  "crime": ["crime"],
  "western": ["western"],
  "young_adult": ["young adult fiction", "juvenile fiction"],
  "workplace_fiction": [],
  "techno_thriller": ["technothriller", "thriller"]
}
```

Genres with an empty list (like `workplace_fiction`) have no good Open Library subject equivalent. The BookFetcher should skip these silently rather than sending a doomed query.

Some genres map to multiple subjects (like `thriller` → `["thriller", "suspense"]`). The BookFetcher could query each, though for simplicity in the first pass, using just the first entry is sufficient.

### Integration Point: BookFetcher Node Wrapper

The mapping should be applied in the `book_fetcher_node` function in `nodes/book_fetcher.py`, which is where genres are assembled before being passed to the agent. This keeps the BookFetcherAgent itself generic — it just queries whatever subjects it receives.

```python
from storymesh.agents.book_fetcher.subject_map import resolve_subjects

# In book_fetcher_node:
genres_to_query = genre_output.normalized_genres + [
    ig.canonical_genre for ig in genre_output.inferred_genres
]
subjects_to_query = resolve_subjects(genres_to_query)
input_data = BookFetcherAgentInput(normalized_genres=subjects_to_query)
```

### Subject Resolver Module

Create `src/storymesh/agents/book_fetcher/subject_map.py`:

```python
def resolve_subjects(genres: list[str]) -> list[str]:
    """Map canonical genre names to Open Library subject query strings.

    Genres not in the mapping are passed through as-is (underscore → space).
    Genres that map to an empty list are silently dropped.
    Deduplicates the final list.

    Args:
        genres: Canonical genre names (snake_case).

    Returns:
        Deduplicated list of Open Library subject strings.
    """
```

The function loads the mapping file once (module-level cache) and performs the translation. Unknown genres are passed through with basic normalization (replace underscores with spaces) as a fallback — this preserves backward compatibility with genres that happen to work as-is on Open Library.

### Graceful Failure in BookFetcherAgent

The BookFetcher's `run()` method iterates over genres and calls the client for each. If a particular genre query raises `OpenLibraryAPIError`, the current behavior would propagate the exception. Update the loop to catch per-genre failures:

```python
for index, subject in enumerate(genres):
    try:
        # ... existing fetch logic ...
    except OpenLibraryAPIError:
        logger.warning("Query for subject '%s' failed. Skipping.", subject, exc_info=True)
        per_genre_debug[subject] = {"books_fetched": 0, "cache": "error"}
        continue
```

This ensures that one bad genre query (whether from an invalid subject or a transient API error) doesn't prevent the remaining genres from being fetched.

### Pass 4 Prompt Improvement

Update the `genre_inference.yaml` system prompt to constrain the LLM toward genres that are more likely to be queryable. Add to the RULES section:

```
- Prefer widely recognized top-level literary genres from this list:
  fantasy, science_fiction, horror, romance, mystery, thriller,
  historical_fiction, literary_fiction, adventure, crime, western.
- You may suggest subgenres, but always include a parent canonical genre.
- Avoid inventing audience categories (e.g., "young_adult", "middle_grade")
  as genre inferences — these are marketing labels, not literary genres.
```

This doesn't guarantee compliance but significantly reduces the frequency of unqueryable genre names.

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/data/genre_subject_map.json` | **CREATE** — canonical genre → Open Library subject mapping |
| `src/storymesh/agents/book_fetcher/subject_map.py` | **CREATE** — `resolve_subjects()` function |
| `src/storymesh/orchestration/nodes/book_fetcher.py` | Update to use `resolve_subjects()` before constructing input |
| `src/storymesh/agents/book_fetcher/agent.py` | Add per-genre try/except in the fetch loop |
| `src/storymesh/prompts/genre_inference.yaml` | Add genre constraint guidance to system prompt |
| `tests/test_subject_map.py` | **CREATE** — tests for `resolve_subjects()` |
| `tests/test_book_fetcher_agent.py` | Add test for graceful per-genre failure handling |

### Testing

1. `resolve_subjects(["science_fiction"])` → `["science fiction"]`
2. `resolve_subjects(["workplace_fiction"])` → `[]` (empty mapping, dropped)
3. `resolve_subjects(["unknown_genre"])` → `["unknown genre"]` (fallback passthrough)
4. `resolve_subjects(["thriller", "thriller"])` → `["thriller"]` (dedup)
5. BookFetcher continues when one genre query fails — other genres still fetch successfully
6. Debug dict records the failed genre with `"cache": "error"`

---

## WI-3: Book Diversity — MMR-Style Selection (Priority 3)

### Problem

The BookRanker's top-10 output is dominated by well-known classics (Pride and Prejudice, Dracula, Frankenstein, etc.) regardless of the user's prompt. The ranker uses a pure relevance ranking with no diversity mechanism. For a creative synopsis generator, this produces repetitive, uninteresting reference material.

### Root Cause

Two compounding factors, as discussed in our earlier conversation:

**Factor 1 — Homogeneous candidate pool.** BookFetcher sorts by `editions`, which returns the most-republished books first. For broad genres, these are century-old public domain classics.

**Factor 2 — Popularity-dominated scoring.** Three of four scoring signals (60% weight) are popularity proxies. When only one genre is queried, all books have identical genre_overlap scores, and ranking is entirely determined by popularity.

### Design: Two-Layer Fix

#### Layer A: Diversify the BookFetcher Candidate Pool

Currently, all queries use `sort=editions`. Add a configurable multi-sort strategy that fetches books using different sort criteria and merges the results. This injects recency and rating diversity into the candidate pool before ranking even begins.

**Proposed approach:** For each genre, make two queries instead of one:

1. `sort=editions` (existing behavior) — gets the canonical, widely-published books
2. `sort=rating` — gets highly-rated but potentially less-known books

The `limit_per_genre` is split across the two sorts (e.g., 15 each instead of 30 from one sort). Deduplication already handles overlap between the two result sets.

This doubles the API calls, but since most results will be cached after the first run, the ongoing cost is minimal. The rate limit delay already handles spacing between calls.

**Configuration:**

```yaml
api_clients:
  open_library:
    sort_strategies: ["editions", "rating"]
    limit_per_sort: 15  # per-sort limit (replaces limit_per_genre)
```

**Fallback:** If `sort_strategies` is not in config, default to `["editions"]` with the existing `limit_per_genre` behavior.

**Files affected:**

| File | Change |
|---|---|
| `storymesh.config.yaml` | Add `sort_strategies` and `limit_per_sort` |
| `src/storymesh/agents/book_fetcher/agent.py` | Loop over sort strategies per genre, merge results |
| `tests/test_book_fetcher_agent.py` | Test multi-sort behavior |

#### Layer B: MMR-Style Diversity Selection in BookRanker

After deterministic scoring and truncation to a shortlist (e.g., top 20–30), apply an iterative MMR-inspired selection to choose the final `top_n` (default 10). This replaces the current simple truncation.

**Algorithm:**

```
Given: scored_books (sorted by composite_score descending), target top_n, lambda λ
Result: selected (list of top_n books)

1. Select the highest-scored book into `selected`.
2. For each remaining slot:
   a. For each candidate not yet selected, compute:
      MMR(candidate) = λ · composite_score(candidate)
                     - (1 - λ) · max_similarity(candidate, selected)
   b. Select the candidate with the highest MMR score.
3. Return selected.
```

**Similarity function:** Jaccard similarity over the `subjects` field from Open Library. Each book has a list of subject tags. Two books that share many subject tags are considered similar.

```python
def _jaccard_similarity(subjects_a: list[str], subjects_b: list[str]) -> float:
    set_a = set(s.lower() for s in subjects_a)
    set_b = set(s.lower() for s in subjects_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union
```

**Lambda parameter:** Controls the relevance-diversity tradeoff. `λ=1.0` means pure relevance (current behavior). `λ=0.0` means pure diversity. A reasonable default for creative reference material is **λ=0.6** — biased toward relevance but with meaningful diversity pressure.

**Configuration:**

```yaml
agents:
  book_ranker:
    top_n: 10
    mmr_lambda: 0.6         # 0.0 = pure diversity, 1.0 = pure relevance (no MMR)
    mmr_candidates: 30      # Score this many books, then MMR-select top_n from them
```

`mmr_candidates` controls the size of the pre-MMR shortlist. The MMR algorithm has O(n²) complexity in the number of candidates, but at 30 candidates this is negligible.

When `mmr_lambda` is set to `1.0`, MMR is effectively disabled and the behavior matches the current simple truncation — this is a clean migration path.

**Implementation location:** Add a new function in `scorer.py`:

```python
def select_diverse(
    scored: list[tuple[BookRecord, float, ScoreBreakdown]],
    top_n: int,
    mmr_lambda: float = 0.6,
) -> list[tuple[BookRecord, float, ScoreBreakdown]]:
    """Select top_n books using MMR-style diversity-aware selection.

    Iteratively picks books that balance high composite scores with
    low similarity to already-selected books (measured by Jaccard
    similarity over Open Library subject tags).

    Args:
        scored: Books sorted by composite_score descending.
        top_n: Number of books to select.
        mmr_lambda: Tradeoff parameter. 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        Selected books in MMR-selection order (not score order).
    """
```

**Integration into BookRankerAgent:**

In `agent.py`, after `compute_scores()`, replace the simple truncation:

```python
# BEFORE
top_scored = scored[: self._top_n]

# AFTER
candidates = scored[: self._mmr_candidates]
top_scored = select_diverse(candidates, self._top_n, self._mmr_lambda)
```

**Debug output:** The debug dict should include MMR-specific data:

```python
debug["mmr"] = {
    "lambda": self._mmr_lambda,
    "candidates_considered": len(candidates),
    "selection_order": [book.work_key for book, _, _ in top_scored],
}
```

**Files affected:**

| File | Change |
|---|---|
| `storymesh.config.yaml` | Add `mmr_lambda` and `mmr_candidates` under `agents.book_ranker` |
| `src/storymesh/agents/book_ranker/scorer.py` | Add `_jaccard_similarity()` and `select_diverse()` |
| `src/storymesh/agents/book_ranker/agent.py` | Use `select_diverse()` instead of simple truncation, accept new config params |
| `tests/test_book_ranker_scorer.py` | Tests for `_jaccard_similarity()` and `select_diverse()` |
| `tests/test_book_ranker_agent.py` | Test that MMR selection produces different results than pure truncation |

### Testing for MMR

1. **`_jaccard_similarity` unit tests:**
   - Identical subject lists → 1.0
   - Disjoint subject lists → 0.0
   - Partial overlap → expected fraction
   - Empty lists → 0.0
   - Case insensitivity

2. **`select_diverse` unit tests:**
   - `mmr_lambda=1.0` produces same order as simple truncation (pure relevance)
   - `mmr_lambda=0.0` with very similar books reorders to maximize diversity
   - With diverse books and `mmr_lambda=0.6`, the top-scoring book is always selected first
   - `top_n >= len(scored)` returns all books
   - Single book input returns that book
   - Books with no subjects (empty lists) — similarity defaults to 0.0, no crash

3. **Integration test:**
   - Create a batch of books where 5 have nearly identical subjects and 5 have unique subjects. Verify that with `mmr_lambda=0.6`, the unique books are promoted over some of the similar-but-higher-scored books.

---

## Implementation Order

1. **WI-1: LLM Logging** — smallest change, immediate observability payoff
2. **WI-2: Genre Validation** — fixes actual pipeline errors, stabilizes Pass 4
3. **WI-3A: BookFetcher Multi-Sort** — diversifies the candidate pool
4. **WI-3B: BookRanker MMR Selection** — diversifies the final selection

Each step is independently deployable and testable. WI-1 provides the debugging tools needed for WI-2. WI-2 ensures the pipeline runs cleanly, which is a prerequisite for meaningful WI-3 testing.

---

## Schema Version Impact

- **WI-1:** No schema changes.
- **WI-2:** No schema changes (the mapping is internal to the BookFetcher node wrapper).
- **WI-3A:** No schema changes (BookFetcherAgentOutput already records `queries_executed` and dedup data).
- **WI-3B:** Bump `BOOK_RANKER_SCHEMA_VERSION` from `"1.0"` to `"1.1"` — the debug dict gains new `mmr` fields. The `BookRankerAgentOutput` structure itself is unchanged, but the semantic meaning of `ranked_books` changes from "top-N by score" to "top-N by MMR selection." Downstream consumers that depend on strict score ordering should be aware.

---

## Open Questions

- **Should the genre_subject_map.json include subgenre mappings?** Currently, Pass 1–3 subgenres like `solarpunk`, `techno_thriller`, `dark_fantasy` are not passed to BookFetcher (only canonical genres are). If inferred genres start including subgenres, the map would need entries for those too. For now, the fallback (underscore → space passthrough) handles subgenres that happen to work as Open Library subjects.

- **Should MMR selection happen before or after the optional LLM re-rank?** Currently the plan runs MMR after scoring but before LLM re-rank. This makes sense because the LLM re-rank is meant to assess "narrative potential" — it should see the diverse shortlist, not the redundant one. If the LLM re-rank is enabled, the pipeline would be: score → MMR select → LLM reorder. If MMR is `λ=1.0` (disabled), the pipeline is: score → truncate → LLM reorder (current behavior).

- **Multi-sort API cost.** Doubling the API calls per genre from 1 to 2 doubles the initial fetch time (before caching). For a prompt that resolves to 3 genres, that's 6 API calls instead of 3, at ~0.4s spacing each = ~2.4s total. After the first run, caching eliminates this cost. This seems acceptable for a graduate project but is worth noting.