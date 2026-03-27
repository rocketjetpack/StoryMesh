# StoryMesh

StoryMesh is a Python package for building an agentic fiction-synopsis pipeline. The repository includes a working LangGraph pipeline, a fully implemented genre normalization stage, an implemented Open Library book-fetching stage, artifact persistence, and a CLI/API surface for running the current pipeline.

The later creative stages are scaffolded but not implemented yet. Today, a full `generate` run executes the implemented stages, persists artifacts, and returns a placeholder synopsis while stages 2 to 6 remain stubs.

## Current Status

Implemented:

- Package, CLI, config loader, and version reporting
- LangGraph orchestration and typed shared state
- `GenreNormalizerAgent` with deterministic mapping, fuzzy matching, and optional LLM fallback
- `BookFetcherAgent` backed by the Open Library Search API with disk cache and deduplication
- Artifact persistence for pipeline runs under `~/.storymesh/runs`
- Test coverage for import, CLI, schemas, graph node wrappers, prompt loading, book fetching, and genre normalization

Not implemented yet:

- `BookRankerAgent`
- `ThemeExtractorAgent`
- `ProposalDraftAgent`
- `RubricJudgeAgent`
- `SynopsisWriterAgent`

Current runtime behavior:

1. Normalize the user’s genre input.
2. Fetch matching books from Open Library.
3. Pass through placeholder nodes for stages 2 to 6.
4. Return a `GenerationResult` with a placeholder `final_synopsis`.

## Architecture

The pipeline is built as a LangGraph `StateGraph` with this topology:

```text
START
  -> genre_normalizer
  -> book_fetcher
  -> book_ranker
  -> theme_extractor
  -> proposal_draft
  -> rubric_judge
  -> synopsis_writer
  -> END
```

Only `genre_normalizer` and `book_fetcher` currently perform real work. The remaining nodes are registered as no-op placeholders so the graph shape is in place before the later stages are implemented.

### Stage 0: GenreNormalizerAgent

Status: implemented

- Reads taxonomy data from [src/storymesh/data/genre_map.json](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/data/genre_map.json) and [src/storymesh/data/tone_map.json](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/data/tone_map.json)
- Resolves genre and tone tokens deterministically where possible
- Uses fuzzy matching for near-matches
- Can fall back to an LLM client when unresolved tokens remain and API keys are configured
- Produces a strict `GenreNormalizerAgentOutput` including debug traces and preserved narrative context

### Stage 1: BookFetcherAgent

Status: implemented

- Queries Open Library’s search API by subject
- Uses disk cache via `diskcache`
- Deduplicates books by Open Library work key
- Preserves all matched source genres per book
- Emits debug metadata for cache hits, misses, and per-genre counts

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

If you only want the currently implemented deterministic and Open Library stages, the package still expects a configured LLM provider in `storymesh.config.yaml` during config validation. In practice, that means you should either:

- install a matching provider extra and set the corresponding API key, or
- change the committed config locally to point at a provider you have configured

Create a `.env` from [.env.example](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/.env.example):

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

The main config file is [storymesh.config.yaml](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/storymesh.config.yaml).

Important behavior:

- `get_config()` validates that API keys exist for every provider referenced in the config
- `.env` is loaded from the current working directory first, then `~/.storymesh/.env`
- cache directories are derived from `cache.dir`
- logging is configured from `logging.level`

Current committed config includes:

- LLM defaults
- `genre_normalizer` overrides
- a `proposal_generation` section that is not currently consumed by the graph
- Open Library client settings
- cache and logging settings
- optional LangSmith project settings

An example config is also provided at [storymesh.config.yaml.example](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/storymesh.config.yaml.example).

## Usage

### CLI

Generate with the current pipeline:

```bash
storymesh generate "dark post-apocalyptic detective mystery"
```

Current output shape in the CLI:

```text
Generated Synopsis:
Placeholder synopsis for genre 'dark post-apocalyptic detective mystery'. SynthesisWriterAgent is not yet implemented.
Metadata: {'input_genre': 'dark post-apocalyptic detective mystery', 'pipeline_version': '0.4.0', 'run_id': '...'}
```

Inspect version information:

```bash
storymesh show-version
```

Inspect resolved config:

```bash
storymesh show-config
storymesh show-agent-config genre_normalizer
```

### Python API

```python
from storymesh import generate_synopsis

result = generate_synopsis("dark fantasy")
print(result.final_synopsis)
print(result.metadata)
```

The public return type is `GenerationResult`, defined in [src/storymesh/schemas/result.py](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/schemas/result.py).

## Artifacts and Caching

StoryMesh persists run artifacts under `~/.storymesh`:

- `~/.storymesh/runs/<run_id>/run_metadata.json`
- `~/.storymesh/runs/<run_id>/<stage>_output.json`
- `~/.storymesh/stages/` for stage-level persisted artifacts when used

Open Library responses are cached under the configured cache root, typically:

- `~/.cache/storymesh/open_library`

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Real API coverage exists behind the `real_api` marker.

Useful files:

- [src/storymesh/orchestration/graph.py](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/orchestration/graph.py)
- [src/storymesh/orchestration/pipeline.py](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/orchestration/pipeline.py)
- [src/storymesh/cli.py](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/cli.py)
- [src/storymesh/config.py](/mnt/c/Users/Kali/Documents/Devel/StoryMesh/src/storymesh/config.py)

## Known Gaps

- The README previously described the full planned pipeline as if it were implemented; that was inaccurate.
- A successful `generate()` call still requires config/API-key validation even though only one stage currently needs optional LLM access.
- The graph includes placeholder nodes for stages 2 to 6, so the final synopsis is intentionally not a real generated synopsis yet.
- The current config file includes a `proposal_generation` section, while the graph looks up agent configs by names such as `genre_normalizer`; later stage config names will need alignment when those stages are implemented.

## Roadmap

- Implement deterministic book ranking
- Add theme extraction and proposal-generation stages
- Add rubric-based retry logic with conditional graph edges
- Implement final synopsis synthesis
- Expand provider support beyond the current Anthropic implementation
- Tighten config semantics so unimplemented providers and stages do not block local development
