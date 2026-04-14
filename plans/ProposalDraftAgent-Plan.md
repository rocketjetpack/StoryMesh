# ProposalDraftAgent Implementation Plan

## Overview

ProposalDraftAgent is Stage 4 of the StoryMesh pipeline. It receives a ThemePack (genre clusters, thematic tensions with clichéd resolutions, and narrative seeds) from ThemeExtractorAgent and produces a fully developed story proposal — characters, setting, plot arc, thematic structure — that RubricJudgeAgent can evaluate and SynopsisWriterAgent can expand.

This agent uses a **multi-sample with self-selection** architecture:

1. **Generate N candidate proposals** (default 3), each steered toward a different narrative seed, using independent stateless LLM calls at elevated temperature (1.2).
2. **Evaluate and select the best candidate** via a separate low-temperature (0.2) critic call that checks candidates against clichéd resolutions, thematic coherence, and tonal alignment.

This implements a propose-evaluate-select decision cycle grounded in the CoALA framework (Sumers et al., 2024), producing structurally diverse creative output that a single LLM call cannot achieve.

### Why Multi-Sample?

A single LLM call at high temperature produces one roll of the dice. The rubric retry loop helps, but retrying the same prompt often yields minor variations rather than genuinely different creative directions. Multi-sample with seed-steering guarantees structural divergence at the input level:

- Each candidate is assigned a different narrative seed as its primary starting point
- Each call is fully stateless (no conversational history, no shared context between candidates)
- The selection step uses the ThemePack's `cliched_resolutions` as a concrete, verifiable evaluation criterion
- On rubric retry, the agent generates N fresh samples — a structurally different search, not "try again and hope"

### Context Contamination Prevention

Each candidate call **must** be a fully independent `complete_json()` invocation. The `LLMClient.complete()` method is stateless — each call sends a fresh `messages` array with only system + user content. No conversational history is carried between calls.

To prevent the model from converging on similar creative patterns despite identical input context, each candidate receives a **differentiated user prompt**:

- **Seed-steering**: Candidate 1 is assigned seed S1, Candidate 2 gets S2, etc.
- **Candidate index**: Each prompt identifies itself as "Candidate {n} of {N}" with an explicit instruction to prioritize originality
- **All other seeds are visible** as context but not as the primary assignment

This ensures divergence at the prompt level rather than relying on temperature alone.

---

## Work Item Ordering and Dependencies

```
WI-1: Pydantic Schemas (proposal_draft.py)
  │
  └─ WI-2: Prompts (proposal_draft_generate.yaml + proposal_draft_select.yaml)
       │
       └─ WI-3: Agent Core (agents/proposal_draft/agent.py)
            │
            └─ WI-4: Node Wrapper (orchestration/nodes/proposal_draft.py)
                 │
                 └─ WI-5: Graph Wiring + Config (graph.py, state.py, config)
                      │
                      └─ WI-6: README + Version Updates
```

Recommended execution order: WI-1 → WI-2 → WI-3 → WI-4 → WI-5 → WI-6. Each step is independently testable before the next begins.

---

## 1. WI-1: Pydantic Schemas

### Rationale

Define the input/output contracts before anything else. These schemas are the typed interface between agents and are consumed by the node wrapper, the agent, and downstream stages.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/schemas/proposal_draft.py` | CREATE |
| `src/storymesh/versioning/schemas.py` | Bump `PROPOSAL_SCHEMA_VERSION` from `"1.0"` to `"1.1"` |
| `tests/test_schemas_proposal_draft.py` | CREATE |

### `ProposalDraftAgentInput`

```python
class ProposalDraftAgentInput(BaseModel):
    """Input contract for ProposalDraftAgent (Stage 4).

    Assembled by the node wrapper from ThemeExtractorAgentOutput,
    GenreNormalizerAgentOutput, and pipeline state. The agent itself
    has no knowledge of the pipeline.
    """

    narrative_seeds: list[NarrativeSeed] = Field(
        min_length=1,
        description="Narrative seeds from ThemeExtractorAgent. Each candidate is steered toward a different seed.",
    )
    tensions: list[ThematicTension] = Field(
        min_length=1,
        description="Thematic tensions with clichéd resolutions. Used for candidate evaluation.",
    )
    genre_clusters: list[GenreCluster] = Field(
        min_length=1,
        description="Genre clusters with thematic assumptions. Provides genre context for proposals.",
    )
    normalized_genres: list[str] = Field(
        min_length=1,
        description="Canonical genre names from GenreNormalizerAgent.",
    )
    user_tones: list[str] = Field(
        default_factory=list,
        description="User-specified tone words carried through from earlier stages.",
    )
    narrative_context: list[str] = Field(
        default_factory=list,
        description="Narrative tokens (settings, time periods, character archetypes) from GenreNormalizerAgent.",
    )
    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string.",
    )
```

Note: `NarrativeSeed`, `ThematicTension`, and `GenreCluster` are imported from `storymesh.schemas.theme_extractor`. These are already frozen Pydantic models.

### `StoryProposal`

This is the core creative output — a single developed story proposal.

```python
class StoryProposal(BaseModel):
    """A fully developed story proposal generated from a narrative seed.

    Contains enough structural detail for RubricJudgeAgent to evaluate
    and SynopsisWriterAgent to expand into a full synopsis.
    """

    model_config = {"frozen": True}

    seed_id: str = Field(
        min_length=1,
        description="Which narrative seed this proposal was primarily developed from.",
    )
    title: str = Field(
        min_length=1,
        description="Working title for the story.",
    )
    protagonist: str = Field(
        min_length=10,
        description=(
            "The main character: name, defining trait, internal conflict, "
            "and what they want vs. what they need."
        ),
    )
    setting: str = Field(
        min_length=10,
        description=(
            "Where and when the story takes place. Must reflect the "
            "narrative context tokens and genre traditions."
        ),
    )
    plot_arc: str = Field(
        min_length=50,
        description=(
            "A 3-act plot summary (setup, confrontation, resolution) "
            "with specific story beats. 150-300 words."
        ),
    )
    thematic_thesis: str = Field(
        min_length=10,
        description=(
            "The central thematic argument the story makes — what it "
            "says about the tensions it explores. Not a moral or lesson, "
            "but the story's philosophical stance."
        ),
    )
    key_scenes: list[str] = Field(
        min_length=2,
        description=(
            "3-5 pivotal scenes described in 1-2 sentences each. "
            "These are the moments where the thematic tensions become visible."
        ),
    )
    tensions_addressed: list[str] = Field(
        min_length=1,
        description="Which tension_ids from the ThemePack this proposal explores.",
    )
    tone: list[str] = Field(
        min_length=1,
        description="The tonal qualities of this proposal (e.g., 'dark', 'cerebral', 'hopeful').",
    )
    genre_blend: list[str] = Field(
        min_length=1,
        description="Which genres from the input this proposal blends.",
    )
```

### `SelectionRationale`

```python
class SelectionRationale(BaseModel):
    """The critic's reasoning for selecting the winning proposal.

    Persisted in the debug artifacts so the selection decision is
    auditable and inspectable.
    """

    model_config = {"frozen": True}

    selected_index: int = Field(
        ge=0,
        description="0-based index of the winning candidate in the candidates list.",
    )
    rationale: str = Field(
        min_length=10,
        description="Why the critic selected this candidate over the others.",
    )
    cliche_violations: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of candidate index (as string) to list of clichéd resolutions "
            "the critic identified in that candidate. Empty list means no violations."
        ),
    )
    runner_up_index: int | None = Field(
        default=None,
        description="Index of the second-best candidate, if applicable.",
    )
```

### `ProposalDraftAgentOutput`

```python
class ProposalDraftAgentOutput(BaseModel):
    """Output contract for ProposalDraftAgent (Stage 4).

    Contains the selected proposal as the primary output, plus all
    candidates and selection rationale in the debug dict for artifact
    inspection.
    """

    model_config = {"frozen": True}

    proposal: StoryProposal = Field(
        description="The selected (winning) story proposal.",
    )
    all_candidates: list[StoryProposal] = Field(
        min_length=1,
        description=(
            "All valid candidate proposals, including the winner. "
            "Persisted for artifact inspection and debugging."
        ),
    )
    selection_rationale: SelectionRationale = Field(
        description="The critic's reasoning for the selection.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Generation metadata: num_candidates_requested, num_valid_candidates, "
            "num_parse_failures, draft_temperature, selection_temperature, "
            "seed_assignments, total_llm_calls."
        ),
    )
    schema_version: str = PROPOSAL_SCHEMA_VERSION
```

### Schema Version

In `src/storymesh/versioning/schemas.py`, bump:

```python
PROPOSAL_SCHEMA_VERSION = "1.1"
```

Add version history comment:

```python
# 2026-04-XX: Increment Proposal schema to 1.1. Introduced full schema:
#             ProposalDraftAgentInput, StoryProposal, SelectionRationale,
#             ProposalDraftAgentOutput. Multi-sample architecture with
#             seed-steering and self-selection.
```

### Testing (`tests/test_schemas_proposal_draft.py`)

```
TestStoryProposal:
  - test_valid_construction: all required fields provided
  - test_frozen: cannot mutate after construction
  - test_title_min_length: rejects empty string
  - test_protagonist_min_length: rejects strings < 10 chars
  - test_setting_min_length: rejects strings < 10 chars
  - test_plot_arc_min_length: rejects strings < 50 chars
  - test_thematic_thesis_min_length: rejects strings < 10 chars
  - test_key_scenes_min_length: rejects lists with < 2 items
  - test_tensions_addressed_min_length: rejects empty list
  - test_tone_min_length: rejects empty list
  - test_genre_blend_min_length: rejects empty list

TestSelectionRationale:
  - test_valid_construction: all required fields provided
  - test_frozen: cannot mutate
  - test_selected_index_ge_zero: rejects negative index
  - test_rationale_min_length: rejects strings < 10 chars
  - test_runner_up_defaults_to_none: optional field

TestProposalDraftAgentInput:
  - test_valid_construction: all required fields provided
  - test_empty_seeds_rejected: min_length=1
  - test_empty_tensions_rejected: min_length=1
  - test_empty_clusters_rejected: min_length=1
  - test_empty_genres_rejected: min_length=1
  - test_defaults_for_optional_fields: user_tones, narrative_context default to []

TestProposalDraftAgentOutput:
  - test_valid_construction: all required fields provided
  - test_frozen: cannot mutate
  - test_schema_version_matches: schema_version == PROPOSAL_SCHEMA_VERSION
  - test_all_candidates_min_length: rejects empty list
  - test_debug_defaults_to_empty_dict: default_factory
```

---

## 2. WI-2: Prompt Design

### Rationale

ProposalDraftAgent requires **two** prompt files — one for the drafting phase and one for the selection phase. This is different from all previous agents which use a single prompt file. The `load_prompt()` utility supports this naturally — each prompt file is loaded independently by name.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/proposal_draft_generate.yaml` | CREATE |
| `src/storymesh/prompts/proposal_draft_select.yaml` | CREATE |
| `tests/test_prompt_loader.py` | ADD integration tests for new prompts |

### Drafting Prompt (`proposal_draft_generate.yaml`)

**System prompt** establishes the LLM as a fiction development editor who creates detailed story proposals from narrative seeds. Key instructions:

1. You are developing **one specific narrative seed** into a full story proposal
2. Your proposal must address the thematic tensions identified in the seed
3. **Avoid** the clichéd resolutions listed for each tension — these are explicitly flagged as tropey and predictable
4. The protagonist must have a clear internal conflict that mirrors the thematic tension
5. The plot arc must be specific — name scenes, describe turning points, show how the tension manifests in concrete story moments
6. Respect the user's tonal preferences and narrative context tokens (settings, time periods)
7. Return **only** valid JSON matching the StoryProposal schema. No markdown fences, no commentary
8. Include 2-3 sentence examples of what a good `plot_arc` and `thematic_thesis` look like

**User prompt template** placeholders:

- `{candidate_index}` — 1-based candidate number (e.g., "1 of 3")
- `{total_candidates}` — total number of candidates being generated
- `{assigned_seed}` — JSON serialization of the NarrativeSeed this candidate must develop
- `{all_seeds}` — JSON serialization of all seeds (for context, not primary development)
- `{tensions}` — JSON serialization of all ThematicTensions (with cliched_resolutions)
- `{genre_clusters}` — JSON serialization of GenreClusters
- `{normalized_genres}` — list of canonical genre names
- `{user_tones}` — list of tone words
- `{narrative_context}` — list of narrative context tokens
- `{user_prompt}` — original user input

The user prompt must include the anti-overlap instruction:

> You are generating candidate {candidate_index} of {total_candidates}. Your proposal must be substantially different from any other proposal that could be generated from these same inputs. Prioritize bold, surprising creative choices over safe or expected ones. Develop the ASSIGNED SEED below as your primary starting point.

### Selection Prompt (`proposal_draft_select.yaml`)

**System prompt** establishes the LLM as an editorial critic evaluating story proposals. Key instructions:

1. You are selecting the **strongest** proposal from a set of candidates
2. Evaluation criteria (in priority order):
   a. **Cliché avoidance**: Does the proposal fall into any of the flagged clichéd resolutions? Proposals with cliché violations are penalized
   b. **Thematic depth**: Does the proposal make a genuine thematic argument, or does it merely gesture at themes?
   c. **Specificity**: Are the characters, scenes, and plot beats concrete and vivid, or generic and abstract?
   d. **Tonal coherence**: Does the proposal honor the user's tonal preferences?
   e. **Internal conflict**: Does the protagonist's internal arc mirror the thematic tension?
3. Return JSON matching the SelectionRationale schema
4. You **must** check every candidate against every clichéd resolution and report violations

**User prompt template** placeholders:

- `{candidates}` — JSON array of all valid StoryProposal objects
- `{tensions}` — JSON array of ThematicTensions (with cliched_resolutions)
- `{user_tones}` — list of user tone preferences
- `{user_prompt}` — original user input

### Testing

```
TestProposalDraftGeneratePrompt:
  - test_load_prompt_succeeds: load_prompt("proposal_draft_generate") returns PromptTemplate
  - test_system_prompt_non_empty: template.system is non-empty string
  - test_user_template_has_required_placeholders: all placeholders present
  - test_format_user_with_valid_data: format_user() succeeds

TestProposalDraftSelectPrompt:
  - test_load_prompt_succeeds: load_prompt("proposal_draft_select") returns PromptTemplate
  - test_system_prompt_non_empty: template.system is non-empty string
  - test_user_template_has_required_placeholders: all placeholders present
  - test_format_user_with_valid_data: format_user() succeeds
```

---

## 3. WI-3: Agent Core

### Rationale

The agent orchestrates the multi-sample generation and selection flow. Follows the one-agent-one-tool pattern: its "tool" is "produce a story proposal," and the multi-sample architecture is the implementation strategy.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/agents/proposal_draft/__init__.py` | CREATE |
| `src/storymesh/agents/proposal_draft/agent.py` | CREATE |
| `tests/test_proposal_draft_agent.py` | CREATE |

### Package Init

```python
"""ProposalDraftAgent package — Stage 4 of the StoryMesh pipeline."""
```

### Constructor

```python
class ProposalDraftAgent:
    """Develops narrative seeds into full story proposals (Stage 4).

    Uses a multi-sample with self-selection architecture: generates N
    candidate proposals from different narrative seeds, then uses a
    critic call to select the strongest one. This produces structurally
    diverse creative output that a single LLM call cannot achieve.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 1.2,
        max_tokens: int = 4096,
        num_candidates: int = 3,
        selection_temperature: float = 0.2,
        selection_max_tokens: int = 2048,
    ) -> None:
```

Parameters:

- `llm_client` — **required**. No deterministic fallback; this is a creative generation task.
- `temperature` — default `1.2`. Elevated above 1.0 to maximize creative variance. The selection step acts as a quality filter, so we can afford to push variance higher. If parse failures become excessive, this is the first knob to turn down.
- `max_tokens` — default `4096`. Story proposals are detailed; this needs to be generous.
- `num_candidates` — default `3`. Minimum 2, maximum limited by the number of available seeds. Configurable in `storymesh.config.yaml`.
- `selection_temperature` — default `0.2`. The critic call should be analytical and consistent, not creative.
- `selection_max_tokens` — default `2048`. Selection rationale is shorter than proposals.

The constructor eagerly loads **both** prompts:

```python
self._generate_prompt = load_prompt("proposal_draft_generate")
self._select_prompt = load_prompt("proposal_draft_select")
```

This ensures misconfiguration is caught at construction time, not mid-pipeline.

### `run()` Method

```python
def run(self, input_data: ProposalDraftAgentInput) -> ProposalDraftAgentOutput:
```

**Algorithm:**

1. **Assign seeds to candidates.** Build a mapping of candidate index → assigned seed. If `num_candidates <= len(seeds)`, assign one seed per candidate in order. If `num_candidates > len(seeds)`, wrap around (candidate 4 gets seed S1 again but with an additional prompt instruction to take a different angle). In practice, the default config has `max_seeds=5` and `num_candidates=3`, so wrapping is unlikely.

2. **Generate N candidates.** For each candidate `i` in `range(num_candidates)`:
   a. Serialize the assigned seed, all seeds, tensions (including `cliched_resolutions`), genre clusters, genres, tones, and narrative context into the user prompt
   b. Call `self._llm_client.complete_json()` with `temperature=self._temperature`, `max_tokens=self._max_tokens`
   c. Parse the response through `StoryProposal(**response)`
   d. If parsing fails (Pydantic validation or JSON decode), log a warning and record as a parse failure. Do **not** retry with the same prompt — this avoids burning budget on a prompt that may be structurally problematic.
   e. Append valid candidates to a list

3. **Handle insufficient candidates.** If fewer than 2 valid candidates survived parsing:
   - If 1 valid candidate: use it directly without a selection step. Log a warning. Set `selection_rationale` to a synthetic rationale noting only one candidate was available.
   - If 0 valid candidates: raise `RuntimeError("ProposalDraftAgent: all candidate proposals failed parsing.")`. The node wrapper will catch this and add it to the errors list.

4. **Selection step.** If 2+ valid candidates:
   a. Serialize all valid candidates and the tensions (with `cliched_resolutions`) into the selection prompt
   b. Call `self._llm_client.complete_json()` with `temperature=self._selection_temperature`, `max_tokens=self._selection_max_tokens`
   c. Parse the response through `SelectionRationale(**response)`
   d. Validate that `selected_index` is within range of the candidates list
   e. If the selection call fails, fall back to selecting candidate 0 (the first valid one) with a synthetic rationale

5. **Assemble output.**
   - `proposal` = the selected candidate
   - `all_candidates` = all valid candidates (preserves order for index reference)
   - `selection_rationale` = the critic's rationale
   - `debug` dict containing:
     - `num_candidates_requested`: how many were attempted
     - `num_valid_candidates`: how many passed parsing
     - `num_parse_failures`: how many failed
     - `draft_temperature`: the temperature used for drafting
     - `selection_temperature`: the temperature used for selection
     - `seed_assignments`: mapping of candidate index → seed_id
     - `total_llm_calls`: total API calls made (candidates + selection)

### Important Implementation Detail: Stateless Calls

Each `complete_json()` call is fully independent. The `LLMClient.complete()` method sends a fresh `messages` array with only system + user content. There is no conversational history, no shared context, no message accumulation between calls. This is already how `LLMClient` works — this note is here to ensure the implementer does not accidentally introduce any form of state sharing (e.g., by building a conversation history list and appending to it).

### Error Handling

- If `complete_json()` raises during a **candidate call**: catch the exception, log it as a parse failure, continue to the next candidate. Do not re-raise — individual candidate failures are expected and handled by the minimum-candidates check.
- If `complete_json()` raises during the **selection call**: catch the exception, fall back to selecting candidate 0 with a synthetic rationale. Log a warning.
- If **all** candidate calls fail: raise `RuntimeError` so the node wrapper can record the error.
- If the LLM returns a `selected_index` outside the valid range: clamp to 0, log a warning.

### Testing (`tests/test_proposal_draft_agent.py`)

Use the `FakeLLMClient` pattern. The client accepts a `responses` list and returns them in order. For N candidates + 1 selection, provide N+1 responses.

```
TestBasicGeneration:
  - test_returns_proposal_draft_output_type: output is ProposalDraftAgentOutput
  - test_proposal_is_story_proposal: output.proposal is a StoryProposal
  - test_all_candidates_populated: len(output.all_candidates) >= 1
  - test_selection_rationale_populated: output.selection_rationale is SelectionRationale
  - test_schema_version_set: output.schema_version == PROPOSAL_SCHEMA_VERSION
  - test_selected_proposal_in_candidates: output.proposal in output.all_candidates

TestSeedSteering:
  - test_each_candidate_gets_different_seed: inspect captured prompts, verify each contains a different seed_id assignment
  - test_more_candidates_than_seeds_wraps: with 2 seeds and 3 candidates, candidate 3 gets seed S1 with alternate-angle instruction
  - test_single_seed_all_candidates_get_same_seed: with 1 seed, all candidates develop the same seed (divergence from temperature + anti-overlap instruction only)

TestLLMInteraction:
  - test_num_candidates_llm_calls: FakeLLMClient.call_count == num_candidates + 1 (drafts + selection)
  - test_draft_temperature_used: captured temperature matches configured draft temperature
  - test_selection_temperature_used: captured temperature for last call matches selection_temperature
  - test_system_prompt_passed_to_drafts: FakeLLMClient records system_prompt was non-None for draft calls
  - test_system_prompt_passed_to_selection: FakeLLMClient records system_prompt was non-None for selection call

TestParseFailures:
  - test_one_candidate_fails_others_succeed: 3 candidates, 1 returns bad JSON → output has 2 candidates, selection still runs
  - test_all_candidates_fail_raises_runtime_error: all return bad JSON → RuntimeError
  - test_single_valid_candidate_skips_selection: only 1 survives → no selection call, synthetic rationale
  - test_selection_call_fails_falls_back_to_first: selection returns bad JSON → candidate 0 selected with synthetic rationale
  - test_selected_index_out_of_range_clamped: selection returns index 99 → clamped to 0

TestDebugMetadata:
  - test_debug_contains_num_candidates_requested: present and correct
  - test_debug_contains_num_valid_candidates: present and correct
  - test_debug_contains_num_parse_failures: present and correct
  - test_debug_contains_seed_assignments: present, maps candidate index to seed_id
  - test_debug_contains_total_llm_calls: present and correct
  - test_debug_contains_temperatures: both draft and selection temperatures recorded

TestEdgeCases:
  - test_single_seed_input: agent works with only 1 narrative seed
  - test_empty_user_tones_ok: agent works without user tones
  - test_empty_narrative_context_ok: agent works without narrative context
  - test_two_candidates_minimum_selection_runs: 2 candidates → selection step runs normally
```

---

## 4. WI-4: Node Wrapper

### Rationale

The node wrapper bridges the pipeline state to the agent's input schema. ProposalDraftAgent draws from ThemeExtractorAgentOutput (tensions, seeds, clusters) and GenreNormalizerAgentOutput (genres, tones, narrative context).

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/nodes/proposal_draft.py` | CREATE |
| `tests/test_graph.py` | ADD node wrapper tests |

### Node Wrapper

```python
def make_proposal_draft_node(
    agent: ProposalDraftAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
```

The node function inside:

1. Read `theme_extractor_output` from state. Raise `RuntimeError` if None.
2. Read `genre_normalizer_output` from state. Raise `RuntimeError` if None.
3. Set `current_run_id` from state `run_id` (same ContextVar pattern as other wrappers).
4. Assemble `ProposalDraftAgentInput`:
   - `narrative_seeds` = `theme_extractor_output.narrative_seeds`
   - `tensions` = `theme_extractor_output.tensions`
   - `genre_clusters` = `theme_extractor_output.genre_clusters`
   - `normalized_genres` = `genre_normalizer_output.normalized_genres`
   - `user_tones` = `theme_extractor_output.user_tones_carried`
   - `narrative_context` = `genre_normalizer_output.narrative_context`
   - `user_prompt` = `state["user_prompt"]`
5. Call `agent.run(input_data)`.
6. Persist output artifact if `artifact_store` is provided.
7. Return `{"proposal_draft_output": output}`.

### Testing

```
TestProposalDraftNodeWrapper:
  - test_missing_theme_extractor_output_raises: RuntimeError if theme_extractor_output is None
  - test_missing_genre_normalizer_output_raises: RuntimeError if genre_normalizer_output is None
  - test_output_key_is_proposal_draft_output: returned dict has correct key
  - test_assembles_input_from_multiple_stages: input carries data from both upstream outputs
  - test_artifact_persisted_when_store_provided: artifact_store.save_output called
  - test_current_run_id_set_during_execution: ContextVar is set correctly
```

---

## 5. WI-5: Graph Wiring, State Update, and Config

### Rationale

Replace the `_noop_node` placeholder for `proposal_draft` with the real node. Update state types. Add config parameters.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/graph.py` | Replace noop with real node |
| `src/storymesh/orchestration/state.py` | Tighten `proposal_draft_output` type |
| `storymesh.config.yaml` | Add new config keys |
| `storymesh.config.yaml.example` | Same |

### `state.py` Changes

Replace:
```python
proposal_draft_output: object | None
```
With:
```python
from storymesh.schemas.proposal_draft import ProposalDraftAgentOutput
# ...
proposal_draft_output: ProposalDraftAgentOutput | None
```

### `graph.py` Changes

In `build_graph()`, under the Stage 4 comment:

1. Get the `proposal_draft` config via `get_agent_config("proposal_draft")`
2. Build the `LLMClient` using the provider registry (same pattern as theme_extractor)
3. Construct `ProposalDraftAgent` with config values
4. Create the node via `make_proposal_draft_node(agent, artifact_store=artifact_store)`
5. Replace `graph.add_node("proposal_draft", _noop_node)` with the real node

The LLM client should follow the same graceful-degradation pattern as theme_extractor: if no API key is available, the node falls back to `_noop_node` and logs a warning.

### Config Changes

Update `storymesh.config.yaml` and `storymesh.config.yaml.example`:

```yaml
  proposal_draft:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 1.2
    max_tokens: 4096
    num_candidates: 3
    selection_temperature: 0.2
    selection_max_tokens: 2048
```

### Testing

- Existing graph compilation tests must still pass
- The `test_full_initial_state_is_valid` test in `test_graph.py` should still pass with the tightened type

---

## 6. WI-6: README and Version Updates

### Files Affected

| File | Action |
|------|--------|
| `README.md` | Update status, architecture notes |
| `src/storymesh/versioning/schemas.py` | Version history comment (done in WI-1) |
| `src/storymesh/versioning/agents.py` | Add ProposalDraftAgent version if applicable |

### README Changes

1. Move `ProposalDraftAgent` from "Not implemented yet" to "Implemented"
2. Add description: "ProposalDraftAgent with multi-sample seed-steering and self-selection for creative proposal generation"
3. Update "Current runtime behavior" step 5 to describe real proposal generation
4. Update the Known Gaps section if applicable
5. Update the Roadmap — mark "Implement ProposalDraftAgent" as done

---

## Design Decision Record

### Why temperature 1.2?

At temperature 1.0, the model produces substantial creative variance but gravitates toward "high-quality" safe continuations. At 1.2, the model makes more genuinely unexpected choices — unusual character professions, atypical setting combinations, structural surprises. Since the selection step acts as a quality filter, we can afford to push variance higher. If parse failures become excessive in practice, this is a single config value to dial back to 1.0.

Both Anthropic and OpenAI APIs support temperatures up to 2.0, so this is portable across registered providers. If a future provider has a tighter temperature range, the agent should clamp rather than error — add a note in the LLMClient constructor or the provider implementation.

### Why two prompt files instead of one?

The drafting and selection phases have fundamentally different system prompts. The drafter is told to be a creative fiction developer; the critic is told to be an analytical editorial evaluator. Combining these into a single prompt file with conditional sections would be fragile and hard to maintain. Two files is cleaner and follows the principle of single responsibility.

The `load_prompt()` utility supports this naturally — each file is loaded independently by name. The naming convention `proposal_draft_generate` and `proposal_draft_select` is clear and discoverable.

### Why not separate agents for draft and selection?

The one-agent-one-tool philosophy could be interpreted as requiring separate `ProposalDrafter` and `ProposalSelector` agents. However, the "tool" here is "produce a story proposal," and the multi-sample architecture is the implementation strategy for that tool. Splitting into two agents would require a new graph node, additional state fields, and wiring — all for an internal implementation detail that the rest of the pipeline doesn't need to know about. The selection step is not independently useful; it only makes sense in the context of the draft step.

### Why persist all candidates?

Debugging creative agents requires understanding *what the model considered*, not just what it chose. Persisting all candidates in the output lets you:

- Compare rejected proposals to understand the selection criteria in practice
- Identify if the seed-steering is producing enough divergence
- Debug cases where the "wrong" candidate was selected
- Provide material for future evaluation/scoring work

The `all_candidates` field in the output schema and the `debug` dict serve this purpose.

### Why raise RuntimeError on total failure instead of returning a partial output?

If all candidate proposals fail parsing, the agent has nothing meaningful to return. A partial output with an empty `proposal` field would violate the schema contract and cause downstream failures anyway. Raising `RuntimeError` lets the node wrapper handle the error consistently (add to state errors list, log, potentially trigger a graceful pipeline degradation). This matches the pattern used by ThemeExtractorAgent.

---

## Validation Checklist

After all work items are complete:

```bash
# 1. All tests pass
pytest

# 2. Type checking passes
mypy src/storymesh/

# 3. Linting passes
ruff check src/ tests/

# 4. CLI commands work
storymesh show-version
storymesh show-config
storymesh show-agent-config proposal_draft

# 5. Generate command runs end-to-end (requires API key)
storymesh generate "dark post-apocalyptic detective mystery"

# 6. Artifacts now include proposal_draft_output.json
ls ~/.storymesh/runs/<latest_run_id>/

# 7. Inspect proposal_draft_output.json
# - proposal field contains a StoryProposal
# - all_candidates has >= 1 entries
# - selection_rationale has rationale text
# - debug dict has seed_assignments, temperatures, etc.

# 8. LLM call log shows N+1 calls for this stage
# (N draft calls + 1 selection call)
cat ~/.storymesh/runs/<latest_run_id>/llm_calls.jsonl | grep proposal_draft
```

---

## File Summary

| File | Action | Work Item |
|------|--------|-----------|
| `src/storymesh/schemas/proposal_draft.py` | CREATE | WI-1 |
| `src/storymesh/versioning/schemas.py` | UPDATE — bump PROPOSAL_SCHEMA_VERSION | WI-1 |
| `src/storymesh/prompts/proposal_draft_generate.yaml` | CREATE | WI-2 |
| `src/storymesh/prompts/proposal_draft_select.yaml` | CREATE | WI-2 |
| `src/storymesh/agents/proposal_draft/__init__.py` | CREATE | WI-3 |
| `src/storymesh/agents/proposal_draft/agent.py` | CREATE | WI-3 |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | CREATE | WI-4 |
| `src/storymesh/orchestration/state.py` | UPDATE — tighten type | WI-5 |
| `src/storymesh/orchestration/graph.py` | UPDATE — wire real node | WI-5 |
| `storymesh.config.yaml` | UPDATE — add new config keys | WI-5 |
| `storymesh.config.yaml.example` | UPDATE — same | WI-5 |
| `README.md` | UPDATE — status, architecture | WI-6 |
| `src/storymesh/versioning/agents.py` | UPDATE — add agent version | WI-6 |
| `tests/test_schemas_proposal_draft.py` | CREATE | WI-1 |
| `tests/test_prompt_loader.py` | UPDATE — add integration tests | WI-2 |
| `tests/test_proposal_draft_agent.py` | CREATE | WI-3 |
| `tests/test_graph.py` | UPDATE — add node wrapper tests | WI-4, WI-5 |