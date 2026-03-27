# StoryMesh

StoryMesh is a Python package for building an agentic fiction-synopsis pipeline. The repository includes a working LangGraph pipeline, a fully implemented genre normalization stage, an implemented Open Library book-fetching stage, per-node artifact persistence, a Rich-formatted CLI, and a Python API for running the current pipeline.

The later creative stages are scaffolded but not implemented yet. Today, a full `generate` run executes the implemented stages, persists artifacts, and returns a placeholder synopsis while stages 2 to 6 remain stubs.

## Current Status

Implemented:

- Package, CLI, config loader, and version reporting
- LangGraph orchestration with typed shared state and per-node artifact persistence
- `GenreNormalizerAgent` with deterministic mapping, fuzzy matching, and optional LLM fallback
- `BookFetcherAgent` backed by the Open Library Search API with disk cache, deduplication, and configurable `max_books` cap
- LLM provider registry for extensible provider support
- Rubric retry loop topology (conditional edge wired; noop placeholder always passes)
- Rich-formatted CLI output with per-stage timing and artifact paths
- Test coverage for import, CLI, schemas, graph node wrappers, prompt loading, book fetching, genre normalization, LLM registry, and artifacts

Not implemented yet:

- `BookRankerAgent`
- `ThemeExtractorAgent`
- `ProposalDraftAgent`
- `RubricJudgeAgent`
- `SynopsisWriterAgent`

Current runtime behavior:

1. Normalize the user's prompt into genres and tones.
2. Fetch matching books from Open Library.
3. Pass through placeholder nodes for stages 2 to 6.
4. Return a `GenerationResult` with a placeholder `final_synopsis`.

## Architecture

The pipeline is built as a LangGraph `StateGraph` with this topology:

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

Only `genre_normalizer` and `book_fetcher` currently perform real work. The remaining nodes are registered as no-op placeholders. The rubric retry loop is wired via a conditional edge; the noop rubric judge always passes so the pipeline runs linearly until the real agent is implemented.

### Stage 0: GenreNormalizerAgent

Status: implemented

- Reads taxonomy data from `src/storymesh/data/genre_map.json` and `src/storymesh/data/tone_map.json`
- Resolves genre and tone tokens deterministically where possible
- Uses fuzzy matching for near-matches
- Can fall back to an LLM client when unresolved tokens remain and API keys are configured
- Produces a strict `GenreNormalizerAgentOutput` including debug traces and preserved narrative context

### Stage 1: BookFetcherAgent

Status: implemented

- Queries Open Library's search API by subject
- Uses disk cache via `diskcache`
- Deduplicates books by Open Library work key
- Preserves all matched source genres per book
- Caps results at `max_books` (default 50) after deduplication, prioritising cross-genre books
- Emits debug metadata for cache hits, misses, truncation, and per-genre counts

### Stages 2 to 6

Status: scaffolded only

The state fields, graph nodes, and versioning hooks exist, but the runtime logic for ranking, thematic extraction, proposal drafting, rubric evaluation, and synopsis synthesis has not been implemented yet.

## Requirements

- Python 3.12+
- `pip`

Optional extras:

- `storymesh[anthropic]`
- `storymesh[openai]`
- `storymesh[gemini]`
- `storymesh[langsmith]`
- `storymesh[dev]`
- `storymesh[all-providers]`
- `storymesh[langgraph-studio]`

Core dependencies are declared in [pyproject.toml](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/pyproject.toml).

## Setup

```bash
git clone https://github.com/<your-username>/storymesh.git
cd storymesh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[anthropic]"
```

API keys are loaded from `.env` (CWD or `~/.storymesh/.env`) at pipeline run time. Absent keys produce a warning; agents fall back to static-only mode. CLI commands like `show-config` and `show-version` never require API keys.

Create a `.env` from `.env.example`:

```bash
cp .env.example .env
```

Then set any required keys:

```dotenv
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=storymesh-dev
```

## Configuration

The main config file is `storymesh.config.yaml`.

Important behavior:

- `get_config()` loads the YAML and `.env` files without requiring any API keys. Key presence is validated (as a warning) only when the pipeline is about to run.
- `.env` is loaded from the current working directory first, then `~/.storymesh/.env`
- Cache directories are derived from `cache.dir`
- Logging is configured from `logging.level`
- All config section names under `agents:` match graph node names (e.g. `genre_normalizer`, `book_fetcher`, `proposal_draft`)

Current committed config includes:

- LLM defaults
- `genre_normalizer` and `proposal_draft` agent overrides
- Open Library client settings including `max_books`
- Cache and logging settings
- Optional LangSmith project settings

An example config is also provided at `storymesh.config.yaml.example`.

## Usage

### CLI

Generate with the current pipeline:

```bash
storymesh generate "dark post-apocalyptic detective mystery"
```

Example output:

```
╭──────────────────────────────────────────────────────────────╮
│ StoryMesh v0.5.0  Run abc123def456                           │
│ Input: "dark post-apocalyptic detective mystery"             │
╰──────────────────────────────────────────────────────────────╯

 Stage                   Status    Time     Artifact
 ────────────────────────────────────────────────────────────────────────
 genre_normalizer        ✓ done    0.03s    ~/.storymesh/runs/abc.../genre_normalizer_output.json
 book_fetcher            ✓ done    1.24s    ~/.storymesh/runs/abc.../book_fetcher_output.json
 book_ranker             ○ noop    0.00s    —
 theme_extractor         ○ noop    0.00s    —
 proposal_draft          ○ noop    0.00s    —
 rubric_judge            ○ noop    0.00s    —
 synopsis_writer         ○ noop    0.00s    —
 ────────────────────────────────────────────────────────────────────────
                         Total: 1.27s

╭─ Synopsis ───────────────────────────────────────────────────╮
│ Placeholder synopsis for 'dark post-apocalyptic detective    │
│ mystery'. SynopsisWriterAgent is not yet implemented.        │
╰──────────────────────────────────────────────────────────────╯

Artifacts saved to: ~/.storymesh/runs/abc123def456/
```

Inspect version information:

```bash
storymesh show-version
```

Inspect resolved config:

```bash
storymesh show-config
storymesh show-agent-config genre_normalizer
storymesh show-agent-config proposal_draft
```

### Python API

```python
from storymesh import generate_synopsis

result = generate_synopsis("dark post-apocalyptic detective mystery")
print(result.final_synopsis)
print(result.metadata)
```

The public return type is `GenerationResult`, defined in `src/storymesh/schemas/result.py`. The `metadata` dict includes `user_prompt`, `pipeline_version`, `run_id`, `stage_timings`, and `run_dir`.

## Artifacts and Caching

StoryMesh persists run artifacts under `~/.storymesh`. Artifacts are written as each node completes (not after the full graph finishes), so a crash mid-pipeline leaves partial artifacts on disk for inspection.

```
~/.storymesh/
├── runs/
│   └── <run_id>/
│       ├── run_metadata.json         ← written before graph invocation
│       ├── genre_normalizer_output.json
│       └── book_fetcher_output.json
└── stages/                           ← stage-level cache (content-addressed)
```

Open Library responses are cached under the configured cache root, typically:

```
~/.cache/storymesh/open_library
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Real API coverage exists behind the `real_api` marker:

```bash
pytest -m real_api
```

Useful files:

- `src/storymesh/orchestration/graph.py`
- `src/storymesh/orchestration/pipeline.py`
- `src/storymesh/cli.py`
- `src/storymesh/config.py`

## Known Gaps

- The graph includes placeholder nodes for stages 2 to 6; the final synopsis is a placeholder until `SynopsisWriterAgent` is implemented.
- The rubric retry loop topology is wired but the routing function always passes (noop placeholder).

## Roadmap

- Implement deterministic book ranking
- Add theme extraction and proposal-generation stages
- Activate rubric-based retry logic with real pass/fail signal
- Implement final synopsis synthesis
- Expand provider support beyond the current Anthropic implementation
