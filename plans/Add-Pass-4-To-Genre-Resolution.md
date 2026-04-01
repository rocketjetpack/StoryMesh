# Pass 4: Holistic Genre Inference

## Problem Statement

The current genre resolution pipeline (Passes 1–3) operates on a **token-classification** paradigm. Passes 1 and 2 perform greedy n-gram matching against static genre and tone maps. Pass 3 sends leftover tokens to an LLM for individual classification. This works well for explicit genre signals ("thriller," "sci-fi," "dark fantasy") but systematically misses **implicit genre signals** — contextual cues, thematic patterns, and compound phrases that imply genres without naming them.

**Example prompt:**
> "techno optimistic thriller about a c++ programmer who is doing everything he can to optimize his code to save one more millisecond"

**Current resolution result:**
- Genres: `[thriller]` (matched in Pass 1)
- Tones: `[optimistic]` (matched in Pass 2)
- Narrative context: `[techno, c, programmer, optimize, code, save, one, more, millisecond, ...]`

**What a human reader would identify:**
- Genres: `[thriller, science_fiction]` — "techno optimistic" strongly implies a sci-fi sensibility
- Subgenres: `[techno_thriller, workplace_fiction]` — a programmer optimizing code in a high-stakes scenario
- Tones: `[optimistic, tense, cerebral]`

The gap between these two results means downstream agents (BookFetcher, ThemeExtractor, ProposalDraft) operate with a narrower creative palette than the prompt warrants. Books like *The Martian* or *Microserfs* would never surface because `science_fiction` was never queried.

## Design Decision

**Add Pass 4** as a post-resolution holistic inference step. Passes 1–3 remain unchanged. Pass 4 receives the full original prompt plus everything already resolved, and asks an LLM: "Given this complete picture, what genres does this prompt *imply* that haven't been captured yet?"

### Why Pass 4 (not modifying Pass 3)

- Passes 1–3 are tested and stable. No risk of regressions.
- Token classification and holistic inference are different cognitive tasks. Keeping them in separate LLM calls (separate prompts, separate response schemas) produces better results from both.
- The new pass is purely additive — if the LLM returns nothing, the pipeline output is identical to the current behavior.

### Key Behavioral Decisions

1. **Always runs.** Pass 4 executes unconditionally after Passes 1–3, regardless of how many genres were already resolved. Even prompts that resolve to 3+ explicit genres may have meaningful implicit ones.

2. **Inferred genres are flagged, not equalized.** A new `ResolutionMethod.LLM_INFERRED` distinguishes "the LLM found this implicitly from context" from `LLM_LIVE` ("the LLM classified this leftover token as a genre"). Downstream agents receive this flag and can decide how to weight inferred genres. BookFetcher will query books for inferred genres; BookRanker's genre_overlap scoring naturally handles weighting.

3. **Only novel genres.** The prompt explicitly tells the LLM to exclude any genre already resolved by Passes 1–3. Each inferred genre must include a brief rationale explaining *why* the prompt implies it. Duplicates of already-resolved genres are discarded during post-processing as a safety net.

4. **Rationale is preserved.** Each inferred genre carries a `rationale` string explaining the inference. This serves two purposes: it forces the LLM to reason before concluding (improving quality), and it provides debug/audit information downstream.

---

## Files Affected

### New Files

| File | Purpose |
|---|---|
| `src/storymesh/prompts/genre_inference.yaml` | System + user prompt template for Pass 4 |

### Modified Files

| File | Change |
|---|---|
| `src/storymesh/schemas/genre_normalizer.py` | Add `LLM_INFERRED` to `ResolutionMethod`, add `InferredGenre` model, add `inferred_genres` field to `GenreNormalizerAgentOutput` |
| `src/storymesh/agents/genre_normalizer/resolver.py` | Add `resolve_holistic()` function, update `resolve_all()` to call it, update `ResolverResult` |
| `src/storymesh/agents/genre_normalizer/agent.py` | Wire inferred genres into the output contract |
| `src/storymesh/versioning/schemas.py` | Bump `GENRE_CONSTRAINT_SCHEMA_VERSION` to `"3.0"` |
| `tests/test_genre_normalizer_resolver.py` | Tests for `resolve_holistic()` and updated `resolve_all()` |
| `tests/test_schemas_genre_normalizer.py` | Tests for new schema fields |
| `tests/test_genre_normalizer_agent.py` | Integration tests for Pass 4 in the agent |

---

## Schema Changes

### `ResolutionMethod` — Add New Enum Value

```python
class ResolutionMethod(StrEnum):
    STATIC_EXACT = "static_exact"
    STATIC_FUZZY = "static_fuzzy"
    LLM_LIVE = "llm_live"
    LLM_CACHED = "llm_cached"
    LLM_INFERRED = "llm_inferred"  # NEW: Holistic inference from full prompt context
```

### New Model: `InferredGenre`

```python
class InferredGenre(BaseModel):
    """A genre inferred from holistic analysis of the full user prompt.

    Unlike GenreResolution (which tracks a specific input token), InferredGenre
    represents a genre that no single token maps to — it emerges from the
    overall context, theme, or setting described in the prompt.
    """

    model_config = {"frozen": True}

    canonical_genre: str = Field(
        min_length=1,
        description="The inferred canonical genre name (snake_case)."
    )

    subgenres: list[str] = Field(
        default_factory=list,
        description="Any subgenres implied by the inference."
    )

    default_tones: list[str] = Field(
        default_factory=list,
        description="Tones commonly associated with this inferred genre."
    )

    rationale: str = Field(
        min_length=1,
        description="Brief explanation of why the prompt implies this genre."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.7,
        description="Confidence score for inferred genres. Default 0.7, lower than LLM_LIVE (0.8)."
    )

    method: ResolutionMethod = Field(
        default=ResolutionMethod.LLM_INFERRED
    )
```

**Design note — why `InferredGenre` instead of reusing `GenreResolution`:**

`GenreResolution` has a required `input_token` field (the specific token that was resolved). Inferred genres don't map to a single input token — they emerge from the holistic prompt. Using `GenreResolution` would require stuffing a synthetic value into `input_token` (like the full prompt or a placeholder), which is misleading. A separate model makes the semantic difference explicit: "this genre came from a token" vs. "this genre came from overall context." Additionally, `GenreResolution.canonical_genres` is a list (a single token can map to multiple genres, e.g. "romantasy" → `[romance, fantasy]`), while each inferred genre entry should represent exactly one inference with its own rationale.

### `GenreNormalizerAgentOutput` — Add Field

```python
class GenreNormalizerAgentOutput(BaseModel):
    # ... existing fields ...

    inferred_genres: list[InferredGenre] = Field(
        default_factory=list,
        description="Genres inferred from holistic prompt analysis (Pass 4). "
                    "These are not mapped from specific tokens but from overall context."
    )
```

The existing `normalized_genres` field continues to hold only explicitly resolved genres from Passes 1–3. The `inferred_genres` list is a separate, additive field. Downstream agents that want the full genre picture combine both; agents that want only high-confidence explicit genres use `normalized_genres` alone.

**Downstream consumer pattern:**

```python
# Full genre list for book fetching (explicit + inferred)
all_genres = output.normalized_genres + [ig.canonical_genre for ig in output.inferred_genres]

# Explicit only (for high-confidence operations)
explicit_genres = output.normalized_genres
```

### `ResolverResult` — Add Field

```python
@dataclass(frozen=True)
class ResolverResult:
    genre_resolutions: list[GenreResolution] = field(default_factory=list)
    tone_resolutions: list[ToneResolution] = field(default_factory=list)
    narrative_context: list[str] = field(default_factory=list)
    unresolved_tokens: list[str] = field(default_factory=list)
    inferred_genres: list[InferredGenre] = field(default_factory=list)  # NEW
```

### Schema Version Bump

Bump `GENRE_CONSTRAINT_SCHEMA_VERSION` from `"2.0"` to `"3.0"` in `src/storymesh/versioning/schemas.py`. This is a major version bump because `GenreNormalizerAgentOutput` has a new field that downstream consumers need to be aware of.

---

## Prompt Design: `genre_inference.yaml`

### System Prompt

The system prompt should establish the LLM's role, provide the list of canonical genres to choose from, define the response format, and include worked examples. Key instructions:

- The LLM receives the **full original prompt**, the **list of already-resolved genres** (from Passes 1–3), the **resolved tones**, and the **narrative context** tokens.
- It must identify genres that the prompt **implies** but that are **not already in the resolved list**.
- Each inferred genre must include a `rationale` explaining the textual evidence.
- Only recognized literary genres/subgenres — no invented categories.
- Use the same canonical genre names as the genre_map (snake_case): `fantasy`, `science_fiction`, `horror`, `romance`, `mystery`, `thriller`, `historical_fiction`, `literary_fiction`, `adventure`, `crime`, `western`.
- If no additional genres are implied, return an empty list. This is a valid and common outcome.
- Do NOT re-infer genres that are already resolved. The resolved list is provided specifically so the LLM can avoid duplication.

### User Prompt Template

```yaml
user: |
  Original user prompt: "{raw_input}"

  Already resolved genres: {resolved_genres}
  Already resolved subgenres: {resolved_subgenres}
  Already resolved tones: {resolved_tones}
  Narrative context tokens: {narrative_context}

  Based on the FULL original prompt — including its themes, setting, character details,
  and overall narrative direction — identify any literary genres or subgenres that are
  strongly implied but NOT already captured in the resolved lists above.

  Only include genres with clear textual evidence. Do not speculate.
```

### Expected Response Format

```json
{
  "inferred_genres": [
    {
      "canonical_genre": "science_fiction",
      "subgenres": ["techno_thriller"],
      "default_tones": ["cerebral", "tense"],
      "rationale": "The prompt describes a programmer optimizing code with 'techno optimistic' framing, implying a technology-forward science fiction sensibility."
    },
    {
      "canonical_genre": "literary_fiction",
      "subgenres": ["workplace_fiction"],
      "default_tones": ["introspective", "driven"],
      "rationale": "The focus on a single professional's obsessive craft — optimizing code to save one millisecond — centers workplace dynamics and professional identity as narrative themes."
    }
  ]
}
```

### Prompt Examples

Include 2–3 worked examples in the system prompt that demonstrate:

1. **A prompt where inference adds value** (like the C++ programmer example).
2. **A prompt where inference returns an empty list** (e.g., "dark fantasy adventure in a medieval kingdom" — all genres are already explicit, nothing additional is implied).
3. **A prompt where inference catches a non-obvious genre** (e.g., "a detective in 1920s Harlem navigating jazz clubs and bootleggers" — already resolved as `mystery`, inference adds `historical_fiction` because of the 1920s Harlem setting).

---

## Resolver Changes: `resolve_holistic()`

### New Function Signature

```python
def resolve_holistic(
    *,
    raw_input: str,
    resolved_genres: list[str],
    resolved_subgenres: list[str],
    resolved_tones: list[str],
    narrative_context: list[str],
    llm_client: LLMClient | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> list[InferredGenre]:
```

### Behavior

1. If `llm_client` is `None`, return an empty list immediately (no-op when LLM is not configured).
2. Load the `genre_inference` prompt template via `load_prompt("genre_inference")`.
3. Format the user prompt with all resolved data.
4. Call `llm_client.complete_json()`.
5. Validate the response against a Pydantic model (`_HolisticInferenceResponse`).
6. **Deduplication safety net:** Filter out any inferred genre where `canonical_genre` is already in `resolved_genres`. The prompt instructs the LLM not to do this, but we don't trust it completely.
7. Return the validated list of `InferredGenre` objects.

### Error Handling

Follow the same pattern as `resolve_llm()`:
- If the LLM call fails (network error, timeout), log a warning and return an empty list. Pass 4 is supplementary — failure should never crash the pipeline.
- If the LLM response fails Pydantic validation, log a warning and return an empty list.
- No retries. Unlike Pass 3 (which handles core unresolved tokens), Pass 4 is a best-effort enhancement. A single failed call is acceptable.

### Confidence Score

Use a fixed default of **0.7** for `LLM_INFERRED` results, lower than the 0.8 used for `LLM_LIVE`. This reflects the inherently lower certainty of inferring a genre from context vs. classifying a specific token that likely names a genre.

Define this as a module-level constant:
```python
_LLM_INFERRED_CONFIDENCE = 0.7
```

---

## `resolve_all()` Changes

Add the Pass 4 call after the existing Pass 3 logic. The function already accepts `llm_client`, `temperature`, and `max_tokens`, so no signature change is needed.

```python
# --- existing Pass 3 code ---

# Pass 4: Holistic genre inference
if allow_llm_fallback and llm_client is not None:
    resolved_genre_names = [
        genre for g in genre_resolutions
              for genre in g.canonical_genres
    ]
    resolved_subgenre_names = [
        sg for g in genre_resolutions
           for sg in g.subgenres
    ]
    resolved_tone_names = [
        tone for t in tone_resolutions
             for tone in t.normalized_tones
    ]

    inferred = resolve_holistic(
        raw_input=raw_input,
        resolved_genres=resolved_genre_names,
        resolved_subgenres=resolved_subgenre_names,
        resolved_tones=resolved_tone_names,
        narrative_context=narrative_context,
        llm_client=llm_client,
        temperature=temperature,
        max_tokens=max_tokens,
    )
else:
    inferred = []

return ResolverResult(
    genre_resolutions=genre_resolutions,
    tone_resolutions=tone_resolutions,
    narrative_context=narrative_context,
    unresolved_tokens=unresolved,
    inferred_genres=inferred,
)
```

**Note on `allow_llm_fallback`:** Pass 4 respects the same toggle as Pass 3. If the user/config has disabled LLM fallback, Pass 4 is also skipped. This is consistent — the flag means "no LLM calls in genre normalization," not "no Pass 3 specifically."

---

## Agent Changes: `agent.py`

In the `run()` method, after the existing output assembly:

```python
# Extract inferred genre data for the output contract
inferred_genres = resolver_result.inferred_genres

# Add Pass 4 data to debug dict
debug["inferred_genres"] = [ig.model_dump() for ig in inferred_genres]
```

Update the output contract construction:

```python
return GenreNormalizerAgentOutput(
    raw_input=input_data.raw_genre,
    normalized_genres=normalized_genres,
    subgenres=subgenres,
    user_tones=tone_result.user_tones,
    tone_override=tone_result.tone_override,
    override_note=tone_result.override_note,
    inferred_genres=inferred_genres,  # NEW
    debug=debug,
)
```

**Important:** Inferred genres are NOT added to `normalized_genres` or `subgenres`. Those fields remain the explicit-resolution outputs from Passes 1–3. The separation is intentional — downstream agents opt in to using inferred genres by reading the `inferred_genres` field.

---

## Config Changes

Add Pass 4 settings under the existing `genre_normalizer` agent config in `storymesh.config.yaml`:

```yaml
agents:
  genre_normalizer:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    temperature: 0.0
    # Pass 4 uses the same provider/model/temperature as Pass 3.
    # No separate config section needed — both are lightweight classification tasks.
```

Pass 4 reuses the same LLM client, provider, model, and temperature as Pass 3. There's no need for a separate config section because both passes are classification tasks appropriate for the same model tier. If this assumption proves wrong in practice (e.g., holistic inference benefits from a smarter model), a separate config section can be added later.

---

## Downstream Impact

### BookFetcherAgent (Stage 1)

The BookFetcher currently reads `normalized_genres` from `GenreNormalizerAgentOutput` to construct its Open Library queries. After this change, the BookFetcher node wrapper (`nodes/book_fetcher.py`) should be updated to also query for inferred genres:

```python
genres_to_query = genre_output.normalized_genres + [
    ig.canonical_genre for ig in genre_output.inferred_genres
]
```

This is a separate, small change to `nodes/book_fetcher.py` — not part of the core Pass 4 implementation, but should be done immediately after to realize the benefit.

### BookRankerAgent (Stage 2)

No changes needed. The ranker's genre_overlap score is computed based on how many of the queried genres returned a given book. If BookFetcher queries more genres, books appearing in multiple queries (including inferred-genre queries) naturally score higher. The ranking system self-adjusts.

### Future Agents (Stages 3–6)

ThemeExtractor and ProposalDraft are not yet implemented. When they are, they should consume the `inferred_genres` field and use the rationale strings to inform creative direction. The rationale is particularly valuable for ProposalDraft — it explains *why* a genre is relevant, which helps the LLM generate more contextually appropriate story proposals.

---

## Testing Strategy

### Unit Tests for `resolve_holistic()`

1. **No LLM client returns empty list.** Verify `resolve_holistic(..., llm_client=None)` returns `[]`.
2. **Valid LLM response is parsed correctly.** Use `FakeLLMClient` with a canned JSON response containing 1–2 inferred genres. Verify `InferredGenre` objects are constructed with correct fields.
3. **Deduplication removes already-resolved genres.** Provide `resolved_genres=["thriller"]` and an LLM response that (incorrectly) includes `thriller`. Verify it's filtered out.
4. **LLM failure returns empty list.** Use `FakeLLMClient` that raises an exception. Verify empty list, no crash.
5. **Malformed LLM response returns empty list.** Use `FakeLLMClient` returning invalid JSON. Verify graceful degradation.
6. **Empty inference is valid.** LLM returns `{"inferred_genres": []}`. Verify empty list, no error.

### Integration Tests for `resolve_all()`

7. **Pass 4 runs after Passes 1–3.** Provide a prompt with one explicit genre and context implying another. Verify `ResolverResult.inferred_genres` is populated.
8. **Pass 4 skipped when `allow_llm_fallback=False`.** Verify `inferred_genres` is empty.
9. **Pass 4 skipped when `llm_client=None`.** Same verification.

### Schema Tests

10. **`InferredGenre` validates correctly.** Test required fields, defaults, and constraints.
11. **`GenreNormalizerAgentOutput` accepts `inferred_genres`.** Test with empty list and populated list.
12. **`ResolutionMethod.LLM_INFERRED` exists and is usable.** Verify enum value.

### Agent-Level Tests

13. **Agent output includes `inferred_genres`.** Run the agent with `FakeLLMClient` and verify the output contract.
14. **Agent debug dict includes Pass 4 data.** Verify `debug["inferred_genres"]` is populated.

---

## Implementation Order

1. **Schema changes** (`schemas/genre_normalizer.py`, `versioning/schemas.py`) — add `LLM_INFERRED`, `InferredGenre`, update `GenreNormalizerAgentOutput`, bump version. Write schema tests.
2. **Prompt file** (`prompts/genre_inference.yaml`) — write the system and user prompts with examples.
3. **Resolver function** (`resolver.py`) — implement `resolve_holistic()`, write unit tests with `FakeLLMClient`.
4. **Wire into `resolve_all()`** (`resolver.py`) — add Pass 4 call, update `ResolverResult`, write integration tests.
5. **Agent integration** (`agent.py`) — wire `inferred_genres` into the output contract, update debug dict, write agent tests.
6. **BookFetcher update** (`nodes/book_fetcher.py`) — include inferred genres in query list.

Steps 1–5 are the core work. Step 6 is a small follow-up that activates the downstream benefit.

---

## Open Questions / Future Considerations

- **Should inferred genre tones participate in tone merging?** Currently `merge_tones()` operates on `GenreResolution` default tones vs. user tones. `InferredGenre` also carries `default_tones`. For now, these are informational only (stored in the `InferredGenre` object) and do NOT feed into the tone merge. This avoids unexpected tone changes from low-confidence inferences. Revisit if downstream agents need richer tone data.

- **Caching for Pass 4.** Pass 3 has a caching design (deferred, using `diskcache`). Pass 4 could use the same pattern, keyed on the combination of resolved genres + narrative context. Not in scope for this implementation but architecturally compatible.

- **Pass 4 prompt iteration.** The prompt examples and instructions will likely need tuning based on real-world results. The YAML prompt file makes this easy to iterate without code changes.