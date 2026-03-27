# StoryMesh Implementation Plan — Architectural Improvements

**Date:** 2026-03-27
**Version:** 0.4.0 → 0.5.0
**Scope:** Infrastructure, config, CLI, graph topology, artifact persistence, naming consistency

This document is the authoritative implementation plan for a set of architectural improvements to the StoryMesh pipeline. It is intended to be consumed by Claude Code or any developer working on the repository. Each work item includes the rationale, the files affected, the exact changes required, testing expectations, and ordering constraints.

---

## Table of Contents

1. [Work Item Ordering and Dependencies](#1-work-item-ordering-and-dependencies)
2. [WI-1: Lazy Config Validation](#2-wi-1-lazy-config-validation)
3. [WI-2: Rename `genre` to `user_prompt` Across the Public API](#3-wi-2-rename-genre-to-user_prompt-across-the-public-api)
4. [WI-3: Config Naming Alignment](#4-wi-3-config-naming-alignment)
5. [WI-4: LLM Provider Registry](#5-wi-4-llm-provider-registry)
6. [WI-5: Per-Node Artifact Persistence](#6-wi-5-per-node-artifact-persistence)
7. [WI-6: BookFetcher `max_books` Config](#7-wi-6-bookfetcher-max_books-config)
8. [WI-7: Graph Topology — Rubric Retry Loop](#8-wi-7-graph-topology--rubric-retry-loop)
9. [WI-8: Rich CLI Output](#9-wi-8-rich-cli-output)
10. [WI-9: Docstring Typo Fixes](#10-wi-9-docstring-typo-fixes)
11. [WI-10: README.md Update](#11-wi-10-readmemd-update)
12. [Version Bump Strategy](#12-version-bump-strategy)
13. [Validation Checklist](#13-validation-checklist)

---

## 1. Work Item Ordering and Dependencies

The work items must be executed in a specific order because later items depend on naming conventions, state fields, and infrastructure established by earlier ones. The dependency graph is:

```text
WI-9 (typos)           — no dependencies, do first
  │
WI-1 (lazy config)     — no dependencies on other WIs
  │
WI-2 (rename genre)    — no dependencies on other WIs, but touches many files
  │
WI-3 (config naming)   — depends on WI-2 (state field names must be settled)
  │
WI-4 (provider registry) — depends on WI-1 (config loading semantics must be settled)
  │
WI-5 (per-node artifacts) — depends on WI-2 (state field names), WI-3 (config names)
  │
WI-6 (max_books)       — depends on WI-3 (config structure settled)
  │
WI-7 (graph topology)  — depends on WI-5 (artifact persistence pattern established)
  │
WI-8 (rich CLI)        — depends on WI-2 (parameter names), WI-5 (artifact paths to display)
  │
WI-10 (README update)  — LAST — captures final state of all other changes
```

**Recommended execution order:** WI-9 → WI-1 → WI-2 → WI-3 → WI-4 → WI-5 → WI-6 → WI-7 → WI-8 → WI-10

---

## 2. WI-1: Lazy Config Validation

### Rationale

Currently, `get_config()` in `src/storymesh/config.py` validates that API keys exist for *every* provider referenced anywhere in the config at load time. This means CLI commands like `show-config`, `show-agent-config`, and even `show-version` (which may trigger config loading transitively) fail if API keys are absent. Since only one stage currently uses an LLM (and it gracefully degrades when the key is missing via `_build_llm_client` returning `None`), the eager validation is counterproductive. Config loading should **never** require API keys.

### Files Affected

- `src/storymesh/config.py`
- `tests/test_config.py` (if it exists; create if not)
- `README.md` (Known Gaps section — remove the relevant bullet)

### Changes

#### `src/storymesh/config.py`

1. **Remove** the call to `_get_required_env_keys()` from `get_config()`.
2. **Remove** the call to `_load_env(required_keys)` from `get_config()`.
3. **Remove** the call to `_validate_env(required_keys)` from `get_config()`.
4. **Keep** the functions `_get_required_env_keys()`, `_load_env()`, and `_validate_env()` but make them available as a public utility: rename to `validate_provider_keys()` that can be called explicitly by callers who need it.
5. **Move** `.env` loading into `get_config()` unconditionally (without requiring specific keys). The `.env` file should always be loaded if present so that API keys, LangSmith settings, etc. are available in the environment. Replace the current conditional `.env` loading with:
   ```python
   # Load .env unconditionally (best-effort, no error if missing)
   _load_env_best_effort()
   ```
6. **Create** a new function `_load_env_best_effort()` that loads `.env` from CWD or `~/.storymesh/.env` without checking for specific keys. It should not raise on missing files.
7. **Add** a new public function `validate_provider_keys(config: dict) -> None` that performs the current validation logic. This function should be called explicitly in `build_graph()` (in `graph.py`) or in `StoryMeshPipeline.generate()` just before graph invocation — i.e., only when the pipeline is actually about to run.

#### Decision: Where to call `validate_provider_keys()`

**Option chosen:** Call it in `StoryMeshPipeline.generate()` right before `build_graph()`. This means:
- `show-config`, `show-agent-config`, `show-version` all work without API keys.
- The validation still happens before any LLM-dependent agent is constructed.
- The error message is clear and early (before graph compilation, not during node execution).

However, the validation should only **warn** for missing keys rather than raising, since `_build_llm_client()` already handles missing keys gracefully by returning `None`. Change `_validate_env` to log warnings instead of raising `OSError`. This way a user who only has Anthropic keys configured can still run the pipeline even if the config references OpenAI in a section for a stage that isn't implemented yet.

#### `src/storymesh/orchestration/pipeline.py`

Add the `.env` loading call before graph construction (it is idempotent since `get_config()` also does it):

```python
if self._graph is None:
    from storymesh.config import get_config  # noqa: PLC0415
    config = get_config()  # loads config + .env, no key validation
    # Warn about missing provider keys (non-fatal)
    from storymesh.config import warn_missing_provider_keys  # noqa: PLC0415
    warn_missing_provider_keys(config)
    from storymesh.orchestration.graph import build_graph  # noqa: PLC0415
    self._graph = build_graph()
```

### Testing

- Test that `get_config()` succeeds when no API keys are set in the environment and no `.env` file exists.
- Test that `show-config` CLI command works without API keys.
- Test that `show-agent-config genre_normalizer` works without API keys.
- Test that `warn_missing_provider_keys()` logs appropriate warnings for missing keys.
- Ensure existing `test_generate.py` tests still pass (they should, since the test environment presumably has keys or mocks).

---

## 3. WI-2: Rename `genre` to `user_prompt` Across the Public API

### Rationale

The pipeline accepts a free-text string that can include genres, tones, settings, time periods, and narrative concepts. Calling this parameter `genre` is misleading. The rename should be consistent across every layer: CLI argument, Python API, state field, metadata keys, and result schema.

### Naming Decision

The new name is `user_prompt`. This was chosen over `raw_input` (shadows Python builtin) and `user_input` (ambiguous in UI contexts).

### Files Affected (exhaustive list)

- `src/storymesh/__init__.py` — `generate_synopsis()` parameter
- `src/storymesh/cli.py` — CLI argument name and help text
- `src/storymesh/orchestration/pipeline.py` — `generate()` parameter, initial state dict, metadata dict
- `src/storymesh/orchestration/state.py` — `input_genre` field rename to `user_prompt`
- `src/storymesh/orchestration/nodes/genre_normalizer.py` — reads `state["input_genre"]`
- `src/storymesh/orchestration/nodes/book_fetcher.py` — may reference state field indirectly
- `src/storymesh/schemas/result.py` — `metadata` dict keys
- `tests/test_cli.py` — mock metadata keys
- `tests/test_graph.py` — state dict construction
- `tests/test_generate.py` — state dict / metadata assertions
- Any other test files that construct `StoryMeshState` dicts with `input_genre`

### Changes

#### `src/storymesh/orchestration/state.py`

Rename the field:

```python
# BEFORE
input_genre: str
"""Raw genre string supplied by the caller."""

# AFTER
user_prompt: str
"""Raw user input string describing the desired fiction synopsis."""
```

#### `src/storymesh/__init__.py`

```python
# BEFORE
def generate_synopsis(genre: str) -> GenerationResult:

# AFTER
def generate_synopsis(user_prompt: str) -> GenerationResult:
```

Update the docstring parameter description accordingly.

#### `src/storymesh/cli.py`

```python
# BEFORE
def generate(
    genre: str = typer.Argument(...,
           help="Fiction genre or genre list to generate a synopsis for."),
) -> None:

# AFTER
def generate(
    user_prompt: str = typer.Argument(...,
           help="Describe the fiction you want a synopsis for (genres, tones, setting, etc.)."),
) -> None:
```

Update the body to pass `user_prompt` instead of `genre`.

#### `src/storymesh/orchestration/pipeline.py`

Rename the `generate()` parameter from `genre` to `user_prompt`. Update all references:
- `initial_state` dict: `"user_prompt": user_prompt` (was `"input_genre": genre`)
- `artifact_store.save_run()` metadata: key becomes `"user_prompt"`
- The placeholder synopsis string: reference `user_prompt` instead of `genre`

#### `src/storymesh/orchestration/nodes/genre_normalizer.py`

```python
# BEFORE
raw_input = state["input_genre"]

# AFTER
raw_input = state["user_prompt"]
```

#### All test files

Search for `"input_genre"` and replace with `"user_prompt"` in state dict construction. Search for parameter `genre=` in API calls and replace with `user_prompt=`.

### Migration Note

This is a breaking change to the Python API (`generate_synopsis()`). Since this is a pre-1.0 project with no external consumers, this is acceptable. The CLI usage changes from `storymesh generate "dark fantasy"` to the same command (positional argument, so the shell invocation is identical — only the help text changes).

### Testing

- All existing tests must be updated to use the new names and must pass.
- `test_cli.py`: verify `generate` command still works with a positional argument.
- `test_graph.py`: verify state dicts use `user_prompt`.
- Run `ruff check` and `mypy` to catch any missed references.

---

## 4. WI-3: Config Naming Alignment

### Rationale

The config file has a `proposal_generation` section (action-noun style) while the graph looks up agent configs by names matching agent class conventions (e.g., `genre_normalizer`). The README calls this out as a known gap. All config section names under `agents:` should match the graph node names, which in turn match the agent class name pattern.

### Canonical Name Mapping

| Graph Node Name     | Agent Class Name        | Config Key (NEW)     | Config Key (OLD, if different) |
|---------------------|-------------------------|----------------------|-------------------------------|
| `genre_normalizer`  | `GenreNormalizerAgent`  | `genre_normalizer`   | *(already correct)*           |
| `book_fetcher`      | `BookFetcherAgent`      | `book_fetcher`       | *(already correct)*           |
| `book_ranker`       | `BookRankerAgent`       | `book_ranker`        | *(new, not yet in config)*    |
| `theme_extractor`   | `ThemeExtractorAgent`   | `theme_extractor`    | *(new, not yet in config)*    |
| `proposal_draft`    | `ProposalDraftAgent`    | `proposal_draft`     | `proposal_generation`         |
| `rubric_judge`      | `RubricJudgeAgent`      | `rubric_judge`       | *(new, not yet in config)*    |
| `synopsis_writer`   | `SynopsisWriterAgent`   | `synopsis_writer`    | *(new, not yet in config)*    |

### Files Affected

- `storymesh.config.yaml` — rename `proposal_generation` to `proposal_draft`
- `storymesh.config.yaml.example` — same rename
- `README.md` — Known Gaps section (remove the naming inconsistency bullet)

### Changes

#### `storymesh.config.yaml`

Rename the section key:
```yaml
# BEFORE
agents:
  genre_normalizer:
    ...
  proposal_generation:
    ...

# AFTER
agents:
  genre_normalizer:
    ...
  proposal_draft:
    ...
```

#### `storymesh.config.yaml.example`

Apply the same rename.

### Testing

- `storymesh show-agent-config proposal_draft` should return the config (not a "using defaults" warning).
- `storymesh show-agent-config proposal_generation` should return a "using defaults" warning (the old name is gone).

---

## 5. WI-4: LLM Provider Registry

### Rationale

`_build_llm_client()` in `graph.py` uses a hardcoded `if/elif` chain to select the correct `LLMClient` subclass. Adding a new provider requires modifying this function. A registry pattern is cleaner and aligns with the roadmap item "Expand provider support."

### Files Affected

- `src/storymesh/llm/__init__.py` — export the registry
- `src/storymesh/llm/base.py` — add registry dict and registration function
- `src/storymesh/llm/anthropic.py` — register itself on import
- `src/storymesh/orchestration/graph.py` — replace `_build_llm_client()` with registry lookup
- `tests/test_llm_registry.py` — new test file

### Changes

#### `src/storymesh/llm/base.py`

Add a module-level registry and a registration function:

```python
# Provider registry: maps provider name strings to LLMClient subclass constructors.
_PROVIDER_REGISTRY: dict[str, type[LLMClient]] = {}


def register_provider(name: str, cls: type[LLMClient]) -> None:
    """Register an LLMClient subclass for a given provider name.

    Args:
        name: Provider name as it appears in storymesh.config.yaml (e.g., 'anthropic').
        cls: The concrete LLMClient subclass to instantiate for this provider.

    Raises:
        ValueError: If the provider name is already registered to a different class.
    """
    if name in _PROVIDER_REGISTRY and _PROVIDER_REGISTRY[name] is not cls:
        raise ValueError(
            f"Provider '{name}' is already registered to {_PROVIDER_REGISTRY[name].__name__}, "
            f"cannot re-register to {cls.__name__}."
        )
    _PROVIDER_REGISTRY[name] = cls


def get_provider_class(name: str) -> type[LLMClient]:
    """Look up the LLMClient subclass for a given provider name.

    Args:
        name: Provider name string.

    Returns:
        The registered LLMClient subclass.

    Raises:
        ValueError: If no provider is registered under that name.
    """
    if name not in _PROVIDER_REGISTRY:
        registered = ", ".join(sorted(_PROVIDER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown LLM provider: '{name}'. Registered providers: {registered}"
        )
    return _PROVIDER_REGISTRY[name]
```

#### `src/storymesh/llm/anthropic.py`

Add self-registration at module level (after the class definition):

```python
from storymesh.llm.base import register_provider

register_provider("anthropic", AnthropicClient)
```

#### `src/storymesh/orchestration/graph.py`

Replace `_build_llm_client()` with:

```python
from storymesh.llm.base import get_provider_class

_PROVIDER_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _build_llm_client(agent_cfg: dict[str, Any]) -> LLMClient | None:
    """Instantiate the correct LLMClient subclass from an agent config dict.

    Uses the provider registry in storymesh.llm.base. Returns None with a
    warning if the required API key is not set.
    """
    provider: str | None = agent_cfg.get("provider")
    model: str | None = agent_cfg.get("model")

    if provider is None:
        return None

    env_key = _PROVIDER_KEY_MAP.get(provider)
    if env_key and not os.environ.get(env_key):
        logger.warning(
            "%s is not set — the agent will run in static-only mode (no LLM fallback).",
            env_key,
        )
        return None

    # Import the provider module to trigger registration.
    # This is necessary because Python only executes module-level code on import.
    _ensure_provider_imported(provider)

    cls = get_provider_class(provider)
    return cls(model=model)


def _ensure_provider_imported(provider: str) -> None:
    """Import the provider module so its register_provider() call executes."""
    import importlib
    module_map = {
        "anthropic": "storymesh.llm.anthropic",
        "openai": "storymesh.llm.openai",
    }
    module_name = module_map.get(provider)
    if module_name:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.warning(
                "Provider module '%s' could not be imported. "
                "Install the corresponding extra: pip install storymesh[%s]",
                module_name, provider,
            )
```

Note: The `_PROVIDER_KEY_MAP` in `graph.py` duplicates the one in `config.py`. Consider consolidating into a single location (e.g., `config.py` exports it) during this work item.

### Testing

- Test that `register_provider` correctly registers a class.
- Test that `register_provider` raises on conflicting re-registration.
- Test that `get_provider_class` returns the correct class.
- Test that `get_provider_class` raises for unknown providers.
- Test that `_build_llm_client` returns `None` when the API key is missing.
- Test that `_build_llm_client` returns the correct client type when the key is present.

---

## 6. WI-5: Per-Node Artifact Persistence

### Rationale

Currently, `pipeline.py` iterates over all stage outputs *after* the graph completes and serializes them. If the pipeline crashes at stage 4, artifacts from stages 0–3 are lost. Moving artifact persistence into each node means artifacts are written as they complete, enabling partial-run inspection and crash recovery.

### Design

1. Generate `run_id` *before* graph invocation and add it to `StoryMeshState`.
2. Add `run_id: str` as a new field on `StoryMeshState`.
3. Create a utility function `persist_stage_output(state, stage_name, output)` that nodes call after computing their output.
4. `pipeline.py` writes only `run_metadata.json` before invocation and assembles `GenerationResult` after.
5. The `ArtifactStore` instance should be created once and passed via state or closure. Passing via state is problematic (not serializable). **Use closure injection:** the `make_*_node()` factory functions already accept the agent; also accept an `ArtifactStore` instance.

### Files Affected

- `src/storymesh/orchestration/state.py` — add `run_id` field
- `src/storymesh/orchestration/pipeline.py` — generate `run_id` before invocation, write `run_metadata.json` before graph runs, remove post-invocation artifact loop
- `src/storymesh/orchestration/nodes/genre_normalizer.py` — accept `ArtifactStore`, persist output
- `src/storymesh/orchestration/nodes/book_fetcher.py` — accept `ArtifactStore`, persist output
- `src/storymesh/orchestration/graph.py` — pass `ArtifactStore` to node factories
- `src/storymesh/core/artifacts.py` — no changes to the class itself, but add a helper function
- `tests/test_graph.py` — update node factory calls to include `ArtifactStore`
- `tests/test_artifacts.py` — test per-node persistence

### Changes

#### `src/storymesh/orchestration/state.py`

Add:
```python
run_id: str
"""Unique run identifier, generated before graph invocation."""
```

#### `src/storymesh/core/artifacts.py`

Add a helper function for use by nodes:

```python
def persist_node_output(
    artifact_store: ArtifactStore,
    run_id: str,
    stage_name: str,
    output: Any,
) -> None:
    """Persist a node's output as a JSON artifact within the run directory.

    Handles both Pydantic models (via model_dump) and plain dicts.
    No-ops if output is None.

    Args:
        artifact_store: The ArtifactStore instance for this run.
        run_id: The unique run identifier.
        stage_name: Name of the pipeline stage (e.g., 'genre_normalizer').
        output: The stage output (Pydantic model, dict, or None).
    """
    if output is None:
        return
    if hasattr(output, "model_dump"):
        data = output.model_dump()
    elif isinstance(output, dict):
        data = output
    else:
        return
    artifact_store.save_run_file(run_id, f"{stage_name}_output.json", data)
```

#### `src/storymesh/orchestration/nodes/genre_normalizer.py`

Update the factory signature to accept an `ArtifactStore`:

```python
def make_genre_normalizer_node(
    agent: GenreNormalizerAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:

    def genre_normalizer_node(state: StoryMeshState) -> dict[str, Any]:
        raw_input = state["user_prompt"]
        output = agent.run(raw_input)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output
            persist_node_output(artifact_store, state["run_id"], "genre_normalizer", output)

        return {"genre_normalizer_output": output}

    return genre_normalizer_node
```

Apply the same pattern to `make_book_fetcher_node`.

For placeholder noop nodes, no artifact persistence is needed (they produce no output).

#### `src/storymesh/orchestration/graph.py`

In `build_graph()`, create an `ArtifactStore` and pass it to node factories:

```python
artifact_store = ArtifactStore()

genre_node = make_genre_normalizer_node(genre_agent, artifact_store=artifact_store)
book_fetcher_node = make_book_fetcher_node(book_fetcher_agent, artifact_store=artifact_store)
```

#### `src/storymesh/orchestration/pipeline.py`

Update `generate()`:

```python
def generate(self, user_prompt: str) -> GenerationResult:
    if self._graph is None:
        # ... build graph ...

    run_id = uuid.uuid4().hex

    # Write run_metadata.json BEFORE graph invocation
    artifact_store = ArtifactStore()
    artifact_store.save_run(run_id, {
        "user_prompt": user_prompt,
        "pipeline_version": storymesh_version,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
    })

    initial_state: StoryMeshState = {
        "user_prompt": user_prompt,
        "pipeline_version": storymesh_version,
        "run_id": run_id,
        # ... other fields ...
    }

    final_state = self._graph.invoke(initial_state)

    # NOTE: Individual stage artifacts are now written by nodes.
    # No post-invocation artifact loop needed.

    # Assemble GenerationResult from final_state
    # ...
```

**Remove** the entire `stage_outputs` loop and per-stage `save_run_file` calls from `pipeline.py`.

### Testing

- Test that running the pipeline creates individual `<stage>_output.json` files in the run directory.
- Test that `run_metadata.json` is written before graph invocation (simulate a crash mid-pipeline and verify metadata exists).
- Test that node factories work with `artifact_store=None` (no persistence, for unit tests that don't need it).
- Update `test_graph.py` to pass `artifact_store=None` to node factories.

---

## 7. WI-6: BookFetcher `max_books` Config

### Rationale

If a user inputs a rich prompt that resolves to many genres, the BookFetcher could return hundreds of deduplicated books. Downstream LLM stages (ThemeExtractor, ProposalDraft) would either be overwhelmed or expensive. A configurable total cap keeps the pipeline bounded.

### Files Affected

- `storymesh.config.yaml` — add `max_books` under `api_clients.open_library`
- `storymesh.config.yaml.example` — same
- `src/storymesh/agents/book_fetcher/agent.py` — read `max_books` from config, truncate output
- `src/storymesh/schemas/book_fetcher.py` — add `max_books` to `BookFetcherAgentOutput.debug`
- `tests/test_book_fetcher_agent.py` — test truncation behavior

### Changes

#### `storymesh.config.yaml`

```yaml
api_clients:
  open_library:
    base_url: "https://openlibrary.org"
    max_books: 50    # Total cap after deduplication across all genre queries
    # ... existing fields ...
```

#### `src/storymesh/agents/book_fetcher/agent.py`

After deduplication, if `len(books) > max_books`, sort by a priority heuristic (e.g., number of `source_genres` descending, then `edition_count` descending) and truncate:

```python
max_books = client_config.get("max_books", 50)

if len(deduplicated_books) > max_books:
    # Prioritize books that appeared in more genre queries, then by popularity
    deduplicated_books.sort(
        key=lambda b: (len(b.source_genres), b.edition_count),
        reverse=True,
    )
    deduplicated_books = deduplicated_books[:max_books]
```

Add `"max_books_applied": len(books) > max_books` and `"max_books_limit": max_books` to the debug dict.

### Testing

- Test that when more than `max_books` books are fetched, the output is truncated to `max_books`.
- Test that truncation preserves books with the most `source_genres` (cross-genre relevance).
- Test that the debug dict records whether truncation was applied.
- Test default behavior when `max_books` is not in config (should default to 50).

---

## 8. WI-7: Graph Topology — Rubric Retry Loop

### Rationale

The current graph is fully linear. The rubric judge → proposal draft retry loop is the most architecturally interesting non-linear topology in the pipeline and should be designed now, even though the agents themselves are placeholders.

### Target Topology

```text
START
  → genre_normalizer
  → book_fetcher
  → book_ranker
  → theme_extractor
  → proposal_draft
  → rubric_judge
  → [conditional edge]
      ├── PASS → synopsis_writer → END
      └── FAIL → proposal_draft  (retry, max 2 attempts)
```

The retry loop requires tracking the attempt count. After `max_retries` failures, the pipeline proceeds to `synopsis_writer` with the best available proposal (or a failure marker).

### Files Affected

- `src/storymesh/orchestration/state.py` — add `rubric_retry_count: int` field
- `src/storymesh/orchestration/graph.py` — replace linear edge with conditional edge
- `tests/test_graph.py` — test conditional routing logic

### Changes

#### `src/storymesh/orchestration/state.py`

Add:
```python
rubric_retry_count: int
"""Number of times the rubric_judge has sent the proposal back for revision. Starts at 0."""
```

#### `src/storymesh/orchestration/graph.py`

Replace the direct `rubric_judge → synopsis_writer` edge with a conditional edge:

```python
MAX_RUBRIC_RETRIES = 2

def _rubric_route(state: StoryMeshState) -> str:
    """Route from rubric_judge: retry proposal_draft or proceed to synopsis_writer.

    Once the real RubricJudgeAgent is implemented, this function should inspect
    the rubric_judge_output for a pass/fail signal. For now, it always passes
    (since the node is a noop).
    """
    retry_count = state.get("rubric_retry_count", 0)
    rubric_output = state.get("rubric_judge_output")

    # Placeholder logic: always pass. Replace with real pass/fail check.
    passed = True

    if passed or retry_count >= MAX_RUBRIC_RETRIES:
        return "synopsis_writer"
    return "proposal_draft"

# In build_graph():
# BEFORE:
#   graph.add_edge("rubric_judge", "synopsis_writer")
# AFTER:
graph.add_conditional_edges(
    "rubric_judge",
    _rubric_route,
    {
        "synopsis_writer": "synopsis_writer",
        "proposal_draft": "proposal_draft",
    },
)
```

The noop `proposal_draft` node must also increment the retry counter when it's a retry (i.e., when `rubric_retry_count > 0` already). Update the noop or the eventual real node to return `{"rubric_retry_count": state.get("rubric_retry_count", 0) + 1}` alongside its output when it detects a retry scenario. For the noop, this is a simple increment:

```python
def _proposal_draft_noop(state: StoryMeshState) -> dict[str, Any]:
    """Placeholder for proposal_draft that increments retry count."""
    return {"rubric_retry_count": state.get("rubric_retry_count", 0) + 1}
```

**Important:** The retry count starts at 0 in the initial state. The first normal pass through `proposal_draft` sets it to 1. If `rubric_judge` fails it and routes back, `proposal_draft` runs again and increments to 2. After `MAX_RUBRIC_RETRIES` (2), the routing function forces progression to `synopsis_writer`.

### Testing

- Test that `_rubric_route` returns `"synopsis_writer"` when `passed=True`.
- Test that `_rubric_route` returns `"proposal_draft"` when `passed=False` and retry count < max.
- Test that `_rubric_route` returns `"synopsis_writer"` when `passed=False` but retry count >= max.
- Test the full graph compiles without errors with the conditional edge.

---

## 9. WI-8: Rich CLI Output

### Rationale

The CLI currently outputs a bare synopsis string and a metadata dict. For a graduate project demonstration, the output should show per-stage progress with timing and artifact paths. The `rich` library provides styled terminal output with panels, tables, and status indicators.

### Dependency Addition

Add `rich` to the core `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "rich",
]
```

### Files Affected

- `pyproject.toml` — add `rich` dependency
- `src/storymesh/cli.py` — rewrite `generate` command output
- `src/storymesh/orchestration/pipeline.py` — emit timing events or return timing data
- `tests/test_cli.py` — update output assertions

### Design: How Timing Data Flows

**Option chosen:** `StoryMeshPipeline.generate()` collects timing data and returns it as part of `GenerationResult.metadata`. The CLI reads this metadata and formats it with Rich. This keeps the pipeline class UI-agnostic and the CLI responsible for presentation only.

To collect timing data, each node wrapper should record `time.perf_counter()` before and after agent execution and store the duration in the state. Add a new state field:

```python
stage_timings: dict[str, float]
"""Mapping of stage name to execution duration in seconds."""
```

Each node appends its timing:
```python
start = time.perf_counter()
output = agent.run(input_data)
elapsed = time.perf_counter() - start

current_timings = dict(state.get("stage_timings", {}))
current_timings[stage_name] = elapsed
return {"<stage>_output": output, "stage_timings": current_timings}
```

### CLI Output Design

The `generate` command should produce output resembling:

```
╭──────────────────────────────────────────────────────╮
│ StoryMesh v0.5.0 — Run abc123def456                  │
│ Input: "dark post-apocalyptic detective mystery"     │
╰──────────────────────────────────────────────────────╯

 Stage                    Status    Time     Artifacts
 ─────────────────────────────────────────────────────
 genre_normalizer         ✓ done    0.03s    ~/.storymesh/runs/abc.../genre_normalizer_output.json
 book_fetcher             ✓ done    1.24s    ~/.storymesh/runs/abc.../book_fetcher_output.json
 book_ranker              ○ noop    0.00s    —
 theme_extractor          ○ noop    0.00s    —
 proposal_draft           ○ noop    0.00s    —
 rubric_judge             ○ noop    0.00s    —
 synopsis_writer          ○ noop    0.00s    —

 Total: 1.27s

╭─ Synopsis ───────────────────────────────────────────╮
│ Placeholder synopsis for ...                         │
╰──────────────────────────────────────────────────────╯

Artifacts saved to: ~/.storymesh/runs/abc123def456/
```

### Implementation Notes for `cli.py`

Use Rich's `Console`, `Table`, `Panel`, and `Text` objects:

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
```

The `show-version` and `show-config` commands should also be updated to use Rich for consistent styling, but this is lower priority. At minimum, `generate` should use Rich output.

### Testing

- Test that `generate` CLI output contains key elements (run ID, stage names, timing).
- Since Rich output includes ANSI escape codes, use `CliRunner(mix_stderr=False)` and test for semantic content rather than exact formatting.
- Consider testing with `Console(force_terminal=False)` to get plain text output in tests.

---

## 10. WI-9: Docstring Typo Fixes

### Rationale

Minor typos in docstrings violate the project's CLAUDE.md standard of "accurate and descriptive docstrings."

### Files Affected

- `src/storymesh/llm/base.py`

### Changes

```python
# Line ~1 of the module docstring:
# BEFORE: """Abstract class defining the requiements for LLM provider classes."""
# AFTER:  """Abstract class defining the requirements for LLM provider classes."""

# In complete_json() docstring:
# BEFORE: "Call the complete() implementation and parse the resposne as a JSON object."
# AFTER:  "Call the complete() implementation and parse the response as a JSON object."
```

### Testing

- No functional tests needed. Run `ruff check` to ensure no formatting issues introduced.

---

## 11. WI-10: README.md Update

### Rationale

After all other work items are complete, the README must reflect the new state of the project. This is the **last** work item.

### Changes

1. **Current Status section:** Update to reflect the new naming (`user_prompt`), per-node artifact persistence, Rich CLI output, and the rubric retry loop topology.

2. **Architecture section:** Replace the linear graph diagram with:

    ```text
    START
      → genre_normalizer
      → book_fetcher
      → book_ranker
      → theme_extractor
      → proposal_draft
      → rubric_judge
      → [conditional]
          ├── PASS → synopsis_writer → END
          └── FAIL → proposal_draft (max 2 retries)
    ```

3. **Configuration section:**
    - Remove the note about config validation requiring API keys.
    - Document the `max_books` config option under `api_clients.open_library`.
    - Note that config section names under `agents:` match graph node names.

4. **Usage section:**
    - Update CLI examples to show Rich-formatted output (or describe it).
    - Update Python API example to use `generate_synopsis(user_prompt=...)`.

5. **Known Gaps section:**
    - Remove the bullet about config validation blocking runs without API keys.
    - Remove the bullet about `proposal_generation` naming mismatch.
    - Add any new known gaps discovered during implementation.

6. **Roadmap section:**
    - Mark "Tighten config semantics" as done.
    - Add any new roadmap items that emerged.

---

## 12. Version Bump Strategy

Bump the package version from `0.4.0` to `0.5.0` in `pyproject.toml`. This is a minor version bump because:

- The public Python API has a breaking parameter rename (`genre` → `user_prompt`).
- The state schema has new fields (`run_id`, `rubric_retry_count`, `stage_timings`).
- The graph topology changes (conditional edge).
- A new dependency is added (`rich`).

Update `pyproject.toml`:
```toml
version = "0.5.0"
```

The version is sourced from `storymesh.versioning.package.__version__`, which reads from `pyproject.toml`. Verify this chain works after the bump.

---

## 13. Validation Checklist

After all work items are complete, run the following validation steps:

```bash
# 1. All tests pass
pytest

# 2. Type checking passes
mypy src/storymesh/

# 3. Linting passes
ruff check src/ tests/

# 4. CLI commands work without API keys
storymesh show-version
storymesh show-config
storymesh show-agent-config genre_normalizer
storymesh show-agent-config proposal_draft

# 5. Generate command runs (with API keys or in static-only mode)
storymesh generate "dark post-apocalyptic detective mystery"

# 6. Artifacts are written per-node
ls ~/.storymesh/runs/<latest_run_id>/

# 7. Rich output displays correctly in terminal
# (visual inspection)

# 8. Config naming is consistent
storymesh show-agent-config book_ranker       # should warn "using defaults" (not yet configured)
storymesh show-agent-config proposal_draft    # should return config (not "proposal_generation")
```