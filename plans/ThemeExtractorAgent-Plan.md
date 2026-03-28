# StoryMesh Implementation Plan — ThemeExtractorAgent & Supporting Changes

**Date:** 2026-03-28
**Scope:** narrative_context promotion, ThemePack schemas, ThemeExtractorAgent, BookRanker MMR diversity
**Target Version:** 0.7.0

This document is the authoritative implementation plan for building the ThemeExtractorAgent (Stage 3) and its supporting infrastructure. It is intended to be consumed by Claude Code or any developer working on the repository. Each work item includes the rationale, the files affected, the exact changes required, testing expectations, and ordering constraints.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Work Item Ordering and Dependencies](#2-work-item-ordering-and-dependencies)
3. [WI-1: Promote `narrative_context` to Top-Level Field](#3-wi-1-promote-narrative_context-to-top-level-field)
4. [WI-2: ThemePack Schemas](#4-wi-2-themepack-schemas)
5. [WI-3: ThemeExtractorAgent Core](#5-wi-3-themeextractoragent-core)
6. [WI-4: ThemeExtractorAgent Prompt](#6-wi-4-themeextractoragent-prompt)
7. [WI-5: ThemeExtractorAgent Node Wrapper & Graph Wiring](#7-wi-5-themeextractoragent-node-wrapper--graph-wiring)
8. [WI-6: Configuration & Versioning](#8-wi-6-configuration--versioning)
9. [WI-7: BookRanker MMR Diversity Selection](#9-wi-7-bookranker-mmr-diversity-selection)
10. [WI-8: README Update](#10-wi-8-readme-update)
11. [Validation Checklist](#11-validation-checklist)

---

## 1. Design Philosophy

### The Creative Thesis

StoryMesh's value proposition is not that it makes more LLM calls — it is that the pipeline produces creative output that a single LLM call cannot, because structured intermediate representations enable *dialectical synthesis*.

The ThemeExtractorAgent is the creative engine of the pipeline. Rather than asking "what themes do these books share?", it identifies the **thematic assumptions** each genre tradition takes for granted, finds where those assumptions **contradict** each other, and frames each contradiction as a **creative question** that a new story could explore.

For example, given "dark post-apocalyptic detective mystery":

- The mystery tradition assumes: truth is discoverable, there is a resolution, a detective figure restores order, justice is achievable.
- The post-apocalyptic tradition assumes: systems have collapsed, truth is subjective, survival trumps justice, the world is fundamentally disordered.
- The **tension**: What does "solving a case" mean when there's no institution to deliver justice to? What does "evidence" mean when the physical world is unreliable?

This tension — not the tropes from either genre individually — is what makes the eventual synopsis interesting.

### Context Engineering Principle

Each agent in the pipeline should see the **minimum context required** for its task. The ThemeExtractorAgent is the first agent that consumes data from multiple upstream stages. Its node wrapper is where context engineering happens — it assembles a clean, self-contained `ThemeExtractorAgentInput` from pipeline state. The agent itself has no knowledge of the pipeline.

The dual-representation pattern established by BookRankerAgent (full `RankedBook` in artifacts, slim `RankedBookSummary` in state) continues here: the ThemePack will contain full analysis detail for artifacts but downstream agents receive only what they need.

### BookRanker Diversity Motivation

OpenLibrary's search API is keyword-based and popularity-biased. Any prompt that includes "fantasy" returns the same dominant fantasy titles regardless of what is interesting about the specific combination of genres the user requested. The MMR (Maximal Marginal Relevance) diversity pass ensures the shortlist covers the thematic space well, including unexpected entries that might yield richer tensions downstream.

---

## 2. Work Item Ordering and Dependencies

```text
WI-1 (narrative_context promotion)  — no dependencies, do first
  │
WI-2 (ThemePack schemas)            — depends on WI-1 (input schema references narrative_context)
  │
WI-3 (ThemeExtractorAgent core)     — depends on WI-2 (schemas must exist)
  │
WI-4 (prompt YAML)                  — depends on WI-2 (prompt references schema structure)
  │
WI-5 (node wrapper & graph wiring)  — depends on WI-3, WI-4 (agent and prompt must exist)
  │
WI-6 (config & versioning)          — depends on WI-5 (config key must match graph node name)
  │
WI-7 (BookRanker MMR diversity)     — independent of WI-1–6, but do after to enable before/after comparison
  │
WI-8 (README update)                — LAST — captures final state of all other changes
```

**Recommended execution order:** WI-1 → WI-2 → WI-3 → WI-4 → WI-5 → WI-6 → WI-7 → WI-8

---

## 3. WI-1: Promote `narrative_context` to Top-Level Field

### Rationale

The ThemeExtractorAgent needs narrative context tokens (settings, time periods, character archetypes like "chicago", "2085", "fallen angel") as a first-class input. Currently these tokens are stored inside `GenreNormalizerAgentOutput.debug["narrative_context"]` — an untyped dict described as being for "observability." Reaching into a debug dict from a downstream agent is fragile, has no schema enforcement, and violates the principle that debug data is for humans inspecting artifacts, not for agents consuming state.

This is a case where the existing design should bend to better practice: `narrative_context` is contract data, not observability data.

### Schema Version

`GENRE_CONSTRAINT_SCHEMA_VERSION` bumps from `"2.0"` to `"3.0"`. This is a structural change to the output contract.

### Files Affected

#### `src/storymesh/schemas/genre_normalizer.py`

Add a new field to `GenreNormalizerAgentOutput`, between `override_note` and `debug`:

```python
narrative_context: list[str] = Field(
    default_factory=list,
    description=(
        "Tokens or phrases from the user input that represent narrative "
        "elements (settings, time periods, character archetypes) rather "
        "than genres or tones. Consumed by downstream agents to anchor "
        "creative output in the user's specific vision."
    ),
)
```

#### `src/storymesh/agents/genre_normalizer/agent.py`

Where the `GenreNormalizerAgentOutput` is constructed (the final return statement of `run()`), add `narrative_context=resolver_result.narrative_context` as a keyword argument.

**Keep** `narrative_context` in the debug dict as well — debug serves as an audit trail showing everything the resolver produced, the top-level field serves as the downstream contract. Both should contain the same data.

#### `src/storymesh/versioning/schemas.py`

Change:
```python
GENRE_CONSTRAINT_SCHEMA_VERSION = "2.0"
```
to:
```python
GENRE_CONSTRAINT_SCHEMA_VERSION = "3.0"
```

Add a version history comment:
```python
# 2026-03-28: Increment Genre Constraint schema to 3.0. Promoted
#             narrative_context from debug dict to a top-level field
#             so downstream agents (ThemeExtractorAgent) can consume it
#             as part of the typed contract.
```

### Testing

#### `tests/test_genre_normalizer_agent.py`

The existing test `test_narrative_context_preserved` currently asserts:
```python
assert result.debug["narrative_context"] == []
```

It should now **also** assert:
```python
assert result.narrative_context == []
```

Any test that checks `narrative_context` in the debug dict should add a parallel assertion on the top-level field. For LLM-based tests that produce non-empty narrative context, assert that `result.narrative_context` matches `result.debug["narrative_context"]`.

#### New test cases to add in the same test file

```
TestNarrativeContextPromotion:
  - test_narrative_context_defaults_to_empty_list: result.narrative_context == [] for "fantasy"
  - test_narrative_context_matches_debug: result.narrative_context == result.debug["narrative_context"]
```

### Backward Compatibility

The field has `default_factory=list`, so any existing serialized artifacts (JSON files from prior runs) will still deserialize correctly — the field defaults to empty. No migration needed.

---

## 4. WI-2: ThemePack Schemas

### Rationale

The ThemeExtractorAgent needs Pydantic input/output schemas following the established patterns. The output schema (`ThemePack`) is the novel contribution — it captures the dialectical structure of genre collisions rather than a flat list of themes.

### Files Affected

- `src/storymesh/schemas/theme_extractor.py` — CREATE
- `src/storymesh/versioning/schemas.py` — verify `THEMEPACK_SCHEMA_VERSION` exists (it does, at `"1.0"`)
- `src/storymesh/versioning/agents.py` — add `THEME_EXTRACTOR_AGENT_VERSION = "1.0"`
- `tests/test_schemas_theme_extractor.py` — CREATE

### Schemas

#### `ThemeExtractorAgentInput`

This is the first agent input that draws from **multiple** upstream stages. The node wrapper (WI-5) assembles this from pipeline state.

```python
class ThemeExtractorAgentInput(BaseModel):
    """Input contract for the ThemeExtractorAgent.

    Assembled by the node wrapper from GenreNormalizerAgentOutput,
    BookRankerAgentOutput, and pipeline state. The agent itself has
    no knowledge of the pipeline.
    """

    ranked_summaries: list[RankedBookSummary] = Field(
        min_length=1,
        description="Slim ranked book summaries from BookRankerAgent.",
    )
    normalized_genres: list[str] = Field(
        min_length=1,
        description="Canonical genre names from GenreNormalizerAgent.",
    )
    subgenres: list[str] = Field(
        default_factory=list,
        description="Subgenre names from GenreNormalizerAgent.",
    )
    user_tones: list[str] = Field(
        default_factory=list,
        description="User-specified tone words from GenreNormalizerAgent.",
    )
    tone_override: bool = Field(
        default=False,
        description="Whether user tones diverge from genre default tones.",
    )
    narrative_context: list[str] = Field(
        default_factory=list,
        description=(
            "Narrative tokens (settings, time periods, character archetypes) "
            "from GenreNormalizerAgent. These anchor the creative output "
            "in the user's specific vision."
        ),
    )
    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string.",
    )
```

Import `RankedBookSummary` from `storymesh.schemas.book_ranker`.

#### `GenreCluster`

```python
class GenreCluster(BaseModel):
    """A group of books belonging to a single genre tradition.

    Captures the thematic assumptions and dominant tropes that the
    LLM identifies as characteristic of this genre tradition, based
    on the books in the cluster.
    """

    model_config = {"frozen": True}

    genre: str = Field(
        min_length=1,
        description="Canonical genre name (e.g., 'mystery', 'post_apocalyptic').",
    )
    books: list[str] = Field(
        min_length=1,
        description="Titles of books grouped into this genre cluster.",
    )
    thematic_assumptions: list[str] = Field(
        min_length=1,
        description=(
            "Core assumptions this genre tradition takes for granted "
            "(e.g., 'truth is discoverable', 'justice is achievable')."
        ),
    )
    dominant_tropes: list[str] = Field(
        default_factory=list,
        description="Common narrative devices in this genre tradition.",
    )
```

#### `ThematicTension`

```python
class ThematicTension(BaseModel):
    """A creative tension between two genre traditions.

    Defined by two opposing assumptions drawn from different genre
    clusters, plus a creative question that frames the tension as
    something a story could explore.
    """

    model_config = {"frozen": True}

    tension_id: str = Field(
        min_length=1,
        description="Short identifier (e.g., 'T1', 'T2').",
    )
    cluster_a: str = Field(
        description="Genre label of the first cluster.",
    )
    assumption_a: str = Field(
        description="The assumption from cluster A that creates the tension.",
    )
    cluster_b: str = Field(
        description="Genre label of the second cluster.",
    )
    assumption_b: str = Field(
        description="The opposing assumption from cluster B.",
    )
    creative_question: str = Field(
        min_length=1,
        description=(
            "The generative question this tension poses for a story "
            "(e.g., 'What does justice look like when there is no one left to enforce it?')."
        ),
    )
    intensity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How fundamentally the assumptions conflict. High intensity "
            "(near 1.0) means a deep thematic contradiction; low intensity "
            "means a stylistic or surface-level difference."
        ),
    )
```

#### `NarrativeSeed`

```python
class NarrativeSeed(BaseModel):
    """A concrete story kernel that emerges from one or more thematic tensions.

    Bridges theme extraction and proposal drafting: ProposalDraftAgent
    receives these seeds as starting points rather than re-interpreting
    raw tensions. This keeps single-responsibility intact — ThemeExtractor
    identifies creative potential, ProposalDraft develops it.
    """

    model_config = {"frozen": True}

    seed_id: str = Field(
        min_length=1,
        description="Short identifier (e.g., 'S1', 'S2').",
    )
    concept: str = Field(
        min_length=10,
        description="A 2–3 sentence story kernel describing the core premise.",
    )
    tensions_used: list[str] = Field(
        min_length=1,
        description="Which tension_ids feed this seed.",
    )
    tonal_direction: list[str] = Field(
        default_factory=list,
        description="Tones this seed leans into (from user tones or genre defaults).",
    )
    narrative_context_used: list[str] = Field(
        default_factory=list,
        description=(
            "Which user narrative context tokens (settings, time periods, etc.) "
            "this seed incorporates."
        ),
    )
```

#### `ThemeExtractorAgentOutput` (the ThemePack)

```python
class ThemeExtractorAgentOutput(BaseModel):
    """Output contract for the ThemeExtractorAgent (Stage 3).

    The ThemePack captures the dialectical structure of genre collisions:
    genre clusters with their thematic assumptions, tensions between those
    assumptions, and narrative seeds that emerge from the tensions. This
    structured intermediate representation is what enables the pipeline to
    produce creative output that a single LLM call cannot.
    """

    model_config = {"frozen": True}

    genre_clusters: list[GenreCluster] = Field(
        min_length=1,
        description="Books grouped by genre tradition with identified thematic assumptions.",
    )
    tensions: list[ThematicTension] = Field(
        min_length=1,
        description="Creative tensions between genre traditions.",
    )
    narrative_seeds: list[NarrativeSeed] = Field(
        min_length=1,
        description=(
            "Concrete story kernels (3–5) that emerge from the tensions. "
            "ProposalDraftAgent selects and develops the best seed."
        ),
    )
    user_tones_carried: list[str] = Field(
        default_factory=list,
        description="User tones passed through for downstream agents.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extraction metadata: number of books processed, genre distribution, "
            "LLM call details, prompt token counts."
        ),
    )
    schema_version: str = THEMEPACK_SCHEMA_VERSION
```

Import `THEMEPACK_SCHEMA_VERSION` from `storymesh.versioning.schemas`.

### Testing

```
TestGenreCluster:
  - test_valid_construction: all required fields provided
  - test_frozen: cannot mutate after construction
  - test_empty_books_rejected: min_length=1 enforced
  - test_empty_assumptions_rejected: min_length=1 enforced

TestThematicTension:
  - test_valid_construction: all fields provided
  - test_intensity_bounds: rejects < 0.0 and > 1.0
  - test_frozen: cannot mutate after construction

TestNarrativeSeed:
  - test_valid_construction: all fields provided
  - test_concept_min_length: rejects concepts shorter than 10 chars
  - test_empty_tensions_used_rejected: min_length=1 enforced
  - test_frozen: cannot mutate after construction

TestThemeExtractorAgentInput:
  - test_valid_construction: all required fields provided
  - test_empty_ranked_summaries_rejected: min_length=1
  - test_empty_genres_rejected: min_length=1
  - test_defaults_for_optional_fields: subgenres, user_tones, narrative_context default to []

TestThemeExtractorAgentOutput:
  - test_valid_construction: all required fields provided
  - test_frozen: cannot mutate after construction
  - test_schema_version_matches: schema_version == THEMEPACK_SCHEMA_VERSION
  - test_empty_clusters_rejected: min_length=1
  - test_empty_tensions_rejected: min_length=1
  - test_empty_seeds_rejected: min_length=1
```

---

## 5. WI-3: ThemeExtractorAgent Core

### Rationale

The agent orchestrates: receive input, load prompt, call LLM, parse JSON response through Pydantic schemas, assemble output. Follows the one-agent-one-tool pattern established by existing agents.

### Files Affected

- `src/storymesh/agents/theme_extractor/__init__.py` — CREATE
- `src/storymesh/agents/theme_extractor/agent.py` — CREATE
- `tests/test_theme_extractor_agent.py` — CREATE

### Package Init

```python
"""ThemeExtractorAgent package — Stage 3 of the StoryMesh pipeline."""
```

### Constructor

```python
class ThemeExtractorAgent:
    """Extracts thematic tensions from ranked books across genre traditions (Stage 3).

    This is the creative engine of the pipeline. Rather than listing themes
    shared across books, it identifies the thematic assumptions each genre
    tradition takes for granted, finds contradictions between traditions,
    and frames those contradictions as creative questions for story generation.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_seeds: int = 5,
    ) -> None:
```

- `llm_client` is **required** — this agent cannot operate without LLM access. Unlike GenreNormalizerAgent which has deterministic fallbacks, theme extraction is fundamentally a creative synthesis task.
- `temperature` defaults to `0.6` — higher than classification agents (0.0) but not maximally creative. This is a deliberate design decision: the agent needs creative interpretation but must still produce structured, parseable JSON.
- `max_seeds` configurable (default 5) — controls how many NarrativeSeeds the LLM generates. Capped to keep ProposalDraft's selection work manageable.

### `run()` Method

```python
def run(self, input_data: ThemeExtractorAgentInput) -> ThemeExtractorAgentOutput:
```

Algorithm:

1. Load the prompt via `load_prompt("theme_extractor")`.
2. Serialize `input_data.ranked_summaries` to a compact JSON string for the prompt. Include only: title, authors, source_genres, rank. Strip work_key, composite_score — the LLM does not need them.
3. Format the user prompt with: serialized book list, normalized_genres, subgenres, user_tones, narrative_context, user_prompt, max_seeds.
4. Call `self._llm_client.complete_json()` with the system prompt, formatted user prompt, temperature, and max_tokens.
5. Parse the JSON response through Pydantic models:
   - Build `GenreCluster` list from `response["genre_clusters"]`
   - Build `ThematicTension` list from `response["tensions"]`
   - Build `NarrativeSeed` list from `response["narrative_seeds"]`
6. Carry forward `input_data.user_tones` as `user_tones_carried`.
7. Populate debug dict with: number of books processed, number of clusters found, number of tensions found, number of seeds generated.
8. Construct and return `ThemeExtractorAgentOutput`.

### Error Handling

- If `complete_json()` raises (LLM error, JSON parse failure after retry), let the exception propagate. The node wrapper will catch it.
- If the parsed JSON is missing required fields or fails Pydantic validation, raise a clear `ValueError` with context about what field was missing.

### Testing

Use a `FakeLLMClient` pattern (same as used in GenreNormalizerAgent tests) that returns pre-canned JSON responses.

```
TestBasicExtraction:
  - test_returns_theme_extractor_output_type: output is ThemeExtractorAgentOutput
  - test_genre_clusters_populated: at least one GenreCluster returned
  - test_tensions_populated: at least one ThematicTension returned
  - test_narrative_seeds_populated: at least one NarrativeSeed returned
  - test_user_tones_carried_through: output.user_tones_carried matches input.user_tones
  - test_schema_version_set: output.schema_version == THEMEPACK_SCHEMA_VERSION

TestLLMInteraction:
  - test_llm_called_with_system_prompt: FakeLLMClient records that system_prompt was passed
  - test_llm_called_with_correct_temperature: temperature matches configured value
  - test_llm_failure_propagates: RuntimeError from LLM propagates up

TestDebugMetadata:
  - test_debug_contains_book_count: debug["books_processed"] == len(ranked_summaries)
  - test_debug_contains_cluster_count: debug["clusters_found"] matches output
  - test_debug_contains_tension_count: debug["tensions_found"] matches output
  - test_debug_contains_seed_count: debug["seeds_generated"] matches output

TestEdgeCases:
  - test_single_genre_still_produces_output: even with one genre, agent should find internal tensions or trope subversions
  - test_empty_narrative_context_ok: agent works fine without narrative context
  - test_max_seeds_respected: output has <= max_seeds narrative seeds
```

---

## 6. WI-4: ThemeExtractorAgent Prompt

### Rationale

Prompts reside in `src/storymesh/prompts/` as YAML files, loaded by the existing `load_prompt()` utility. This agent's prompt is the most creatively demanding in the pipeline.

### File

- `src/storymesh/prompts/theme_extractor.yaml` — CREATE

### Prompt Design

The system prompt establishes the agent's role as a literary analyst who identifies thematic tensions between genre traditions. The user prompt provides the data and instructions.

Key design decisions:

- **Temperature guidance is not in the prompt** — temperature is set at the API call level in the agent config.
- **The prompt explicitly asks for contradictions**, not similarities. This is the core creative insight.
- **JSON-only output** is enforced in the system prompt, consistent with the genre_normalizer pattern.
- **Examples are included** in the system prompt to anchor the LLM's understanding of what "thematic assumptions" and "creative questions" mean.

#### System Prompt Structure

The system prompt should instruct the LLM to:

1. Group the provided books by their primary genre tradition (using `source_genres`).
2. For each genre cluster, identify 2–4 core thematic assumptions that the genre tradition takes for granted.
3. For each genre cluster, identify 2–4 dominant narrative tropes.
4. Compare assumptions across clusters and identify contradictions.
5. For each contradiction, frame it as a creative question a story could explore and assign an intensity score (0.0–1.0).
6. Generate narrative seeds (story kernels) that resolve or explore the tensions, incorporating any provided narrative context tokens (settings, time periods, character archetypes).
7. Return ONLY a JSON object matching the specified schema. No preamble, no markdown fences.

#### User Prompt Template Placeholders

```
{user_prompt}          — original raw user input
{normalized_genres}    — list of canonical genres
{subgenres}            — list of subgenres
{user_tones}           — list of user-specified tones
{narrative_context}    — list of narrative tokens
{book_list}            — serialized book data (title, authors, source_genres, rank)
{max_seeds}            — maximum number of narrative seeds to generate
```

#### Expected LLM Response Structure

```json
{
  "genre_clusters": [
    {
      "genre": "mystery",
      "books": ["The Big Sleep", "Gone Girl"],
      "thematic_assumptions": [
        "Truth is discoverable through investigation",
        "There is a resolution to be found",
        "A detective figure can restore order"
      ],
      "dominant_tropes": ["unreliable narrator", "red herring", "locked room"]
    },
    {
      "genre": "post_apocalyptic",
      "books": ["The Road", "Station Eleven"],
      "thematic_assumptions": [
        "Civilization has collapsed or is collapsing",
        "Survival is the primary moral imperative",
        "Institutional justice no longer exists"
      ],
      "dominant_tropes": ["journey narrative", "found family", "resource scarcity"]
    }
  ],
  "tensions": [
    {
      "tension_id": "T1",
      "cluster_a": "mystery",
      "assumption_a": "Truth is discoverable through investigation",
      "cluster_b": "post_apocalyptic",
      "assumption_b": "Institutional knowledge systems have collapsed",
      "creative_question": "What does investigation look like when there are no records, no witnesses, and no institutions to consult?",
      "intensity": 0.85
    }
  ],
  "narrative_seeds": [
    {
      "seed_id": "S1",
      "concept": "A former forensic accountant in a post-collapse settlement is the only person who can trace the missing grain stores. But her methods — spreadsheets, databases, cross-referencing — are artifacts of a dead world. She must reinvent investigation from first principles.",
      "tensions_used": ["T1"],
      "tonal_direction": ["dark", "gritty"],
      "narrative_context_used": []
    }
  ]
}
```

### Testing

```
TestPromptLoading:
  - test_load_prompt_succeeds: load_prompt("theme_extractor") returns a PromptTemplate
  - test_system_prompt_non_empty: template.system is a non-empty string
  - test_user_template_has_required_placeholders: all placeholders are present
  - test_format_user_with_valid_data: format_user() succeeds with all required kwargs
  - test_format_user_missing_placeholder_raises: PromptFormattingError on missing key
```

---

## 7. WI-5: ThemeExtractorAgent Node Wrapper & Graph Wiring

### Rationale

The node wrapper is where context engineering happens for this agent. It is the first wrapper that assembles input from **multiple upstream stages** — both `genre_normalizer_output` and `book_ranker_output`. The agent itself remains pipeline-unaware.

### Files Affected

- `src/storymesh/orchestration/nodes/theme_extractor.py` — CREATE
- `src/storymesh/orchestration/graph.py` — UPDATE (replace noop with real node)
- `src/storymesh/orchestration/state.py` — UPDATE (tighten type annotation)
- `tests/test_graph.py` — UPDATE (add node tests)

### Node Wrapper

```python
def make_theme_extractor_node(
    agent: ThemeExtractorAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for ThemeExtractorAgent (Stage 3).

    This is the first node that draws from multiple upstream stages. It reads
    genre_normalizer_output and book_ranker_output from the pipeline state,
    assembles a self-contained ThemeExtractorAgentInput, runs the agent, and
    returns a partial state dict.

    Args:
        agent: A fully constructed ThemeExtractorAgent instance.
        artifact_store: Optional store for per-node artifact persistence.

    Returns:
        A node callable with signature StoryMeshState -> dict[str, Any].
    """
```

The node function inside:

1. Read `genre_normalizer_output` from state. Raise `RuntimeError` if None.
2. Read `book_ranker_output` from state. Raise `RuntimeError` if None.
3. Assemble `ThemeExtractorAgentInput`:
   - `ranked_summaries` from `book_ranker_output.ranked_summaries`
   - `normalized_genres` from `genre_normalizer_output.normalized_genres`
   - `subgenres` from `genre_normalizer_output.subgenres`
   - `user_tones` from `genre_normalizer_output.user_tones`
   - `tone_override` from `genre_normalizer_output.tone_override`
   - `narrative_context` from `genre_normalizer_output.narrative_context` (the NEW top-level field from WI-1)
   - `user_prompt` from `state["user_prompt"]`
4. Call `agent.run(input_data)`.
5. Persist artifact if `artifact_store` is provided.
6. Return `{"theme_extractor_output": output}`.

### State Type Update

In `src/storymesh/orchestration/state.py`, change:
```python
# ── Stage 3: ThemeExtractorAgent (LLM) ────────────────────────────────────
# TODO: Replace object with ThemePack once implemented.
theme_extractor_output: object | None
```
to:
```python
# ── Stage 3: ThemeExtractorAgent (LLM) ────────────────────────────────────
theme_extractor_output: ThemeExtractorAgentOutput | None
```

Add the import:
```python
from storymesh.schemas.theme_extractor import ThemeExtractorAgentOutput
```

### Graph Wiring

In `src/storymesh/orchestration/graph.py`, replace the `_noop_node` for `theme_extractor` with a real node. Follow the pattern established by Stages 0–2:

1. Import `ThemeExtractorAgent` and `make_theme_extractor_node`.
2. Under a `# ── Stage 3: ThemeExtractorAgent ──` comment:
   - Read config via `get_agent_config("theme_extractor")`.
   - Build an LLM client via `_build_llm_client(theme_cfg)`.
   - If `theme_llm` is `None` (no API key configured), keep the noop node and log a warning. The pipeline should degrade gracefully, not crash.
   - Otherwise, construct `ThemeExtractorAgent(llm_client=theme_llm, temperature=..., max_tokens=..., max_seeds=...)`.
   - Create the node via `make_theme_extractor_node(agent, artifact_store=artifact_store)`.
3. Register the node in the graph (replacing the noop).

### Testing

```
TestThemeExtractorNode:
  - test_returns_theme_extractor_output_type: output dict has "theme_extractor_output" key
  - test_none_genre_output_raises_runtime_error: RuntimeError if genre_normalizer_output is None
  - test_none_book_ranker_output_raises_runtime_error: RuntimeError if book_ranker_output is None
  - test_assembles_input_from_multiple_stages: verify the input constructed has data from both upstream outputs
  - test_only_returns_own_key: partial state dict has exactly one key
```

---

## 8. WI-6: Configuration & Versioning

### Files Affected

- `storymesh.config.yaml` — add `theme_extractor` agent config
- `storymesh.config.yaml.example` — same
- `src/storymesh/versioning/agents.py` — add `THEME_EXTRACTOR_AGENT_VERSION`

### Config Entry

Add under `agents:` in both config files:

```yaml
  theme_extractor:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.6
    max_tokens: 4096
    max_seeds: 5
```

Design note: This agent uses `claude-sonnet-4-6` rather than Haiku. Theme extraction is a creative synthesis task that benefits from a more capable model. The higher temperature (0.6 vs 0.0 for classification agents) is also deliberate — the agent needs creative interpretation but must still produce structured JSON.

### Versioning

In `src/storymesh/versioning/agents.py`, add:

```python
THEME_EXTRACTOR_AGENT_VERSION = "1.0"
```

And add the entry to the `AGENT_VERSIONS` dict.

---

## 9. WI-7: BookRanker MMR Diversity Selection

### Rationale

OpenLibrary's search is popularity-biased. The same dominant titles appear for any prompt that touches a given genre. This means ThemeExtractorAgent draws from a predictable, stable pool — which limits the creativity of the output. Adding a Maximal Marginal Relevance (MMR) diversity pass ensures the shortlist covers the thematic space well, including unexpected entries.

**Important:** Build this AFTER the ThemeExtractorAgent is working. This enables before/after comparison: run the same prompt with and without diversity, compare the ThemePack output, and discuss the difference in the paper.

### Files Affected

- `src/storymesh/agents/book_ranker/scorer.py` — UPDATE (add diversity selection)
- `src/storymesh/agents/book_ranker/agent.py` — UPDATE (call diversity selection)
- `storymesh.config.yaml` — add `diversity_weight` under `agents.book_ranker`
- `storymesh.config.yaml.example` — same
- `tests/test_book_ranker_scorer.py` — UPDATE (add diversity tests)
- `tests/test_book_ranker_agent.py` — UPDATE (add diversity integration tests)

### Algorithm

After computing composite scores and before truncating to `top_n`, apply iterative MMR selection:

```python
def select_with_diversity(
    scored_books: list[ScoredBook],
    top_n: int,
    diversity_weight: float = 0.3,
) -> list[ScoredBook]:
    """Select top_n books balancing relevance against redundancy.

    Uses Maximal Marginal Relevance (MMR): each selection maximizes
    (1 - diversity_weight) * relevance - diversity_weight * max_similarity_to_selected.

    Similarity is Jaccard similarity over source_genres sets.

    Args:
        scored_books: All books with computed composite scores, sorted by score desc.
        top_n: Number of books to select.
        diversity_weight: 0.0 = pure relevance (current behavior), 1.0 = maximum diversity.

    Returns:
        Selected books in MMR order.
    """
```

The similarity function:

```python
def _jaccard_similarity(genres_a: list[str], genres_b: list[str]) -> float:
    """Jaccard similarity between two genre lists."""
    set_a, set_b = set(genres_a), set(genres_b)
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
```

### Config

Add to `agents.book_ranker` in both config files:

```yaml
    diversity_weight: 0.3   # 0.0 = pure relevance, 1.0 = max diversity
```

### Constructor Change

Add `diversity_weight: float = 0.0` to `BookRankerAgent.__init__()`. Default `0.0` preserves backward compatibility — existing behavior is unchanged unless the config explicitly enables diversity.

### Debug Output

Add to the debug dict:
- `diversity_weight`: the weight used
- `diversity_applied`: bool (whether weight > 0.0)
- `selection_order`: list of work_keys in the order MMR selected them (useful for understanding why a surprising book was included)

### Testing

```
TestJaccardSimilarity:
  - test_identical_genres: ["fantasy", "mystery"] vs ["fantasy", "mystery"] → 1.0
  - test_disjoint_genres: ["fantasy"] vs ["mystery"] → 0.0
  - test_partial_overlap: ["fantasy", "mystery"] vs ["mystery", "thriller"] → 0.333...
  - test_empty_both: [] vs [] → 1.0
  - test_empty_one: ["fantasy"] vs [] → 0.0

TestSelectWithDiversity:
  - test_zero_weight_matches_pure_relevance: diversity_weight=0.0 produces same order as sorted by score
  - test_diversity_changes_selection: diversity_weight=0.5 produces different set than weight=0.0 given redundant books
  - test_top_n_respected: output length == top_n
  - test_single_book: works with one book
  - test_all_identical_genres: all books have same genres, diversity has no effect on set (only order)

TestBookRankerAgentDiversity:
  - test_diversity_weight_from_config: agent reads diversity_weight from constructor
  - test_debug_records_diversity_metadata: debug dict contains diversity_weight, diversity_applied
  - test_default_zero_weight: backward compatible, no diversity by default
```

### Schema Version

`BOOK_RANKER_SCHEMA_VERSION` stays at `"1.0"` — the output schema is unchanged. The diversity pass changes which books are selected, not the shape of the output.

---

## 10. WI-8: README Update

### Rationale

Capture the final state of all changes in the README.

### Changes

- Update "Current Status" section: move `ThemeExtractorAgent` from "Not implemented" to "Implemented" with a description of the dialectical synthesis approach.
- Update Stage 3 description under "Architecture" to describe the genre-cluster / tension / narrative-seed pipeline.
- Add `theme_extractor` to the config section listing.
- Update "Known Gaps" to remove ThemeExtractor and note remaining gaps.
- Update "Roadmap" to reflect progress.
- Mention the BookRanker MMR diversity feature under Stage 2.

---

## 11. Validation Checklist

After all work items are complete, run the following:

```bash
# 1. Full test suite
pytest

# 2. Type checking
mypy src/

# 3. Lint
ruff check src/ tests/

# 4. Verify CLI works
storymesh show-version
storymesh show-config
storymesh show-agent-config theme_extractor

# 5. End-to-end run (requires API keys)
storymesh generate "dark post-apocalyptic detective mystery"
# Verify: theme_extractor shows "✓ done" in the stage table
# Verify: ~/.storymesh/runs/<run_id>/theme_extractor_output.json exists and contains valid ThemePack

# 6. Verify diversity effect (requires API keys)
# Run the same prompt twice: once with diversity_weight: 0.0, once with 0.3
# Compare book_ranker_output.json and theme_extractor_output.json between runs
```