# StoryMesh

StoryMesh is a Python package for building an agentic fiction-synopsis pipeline. The repository includes a working LangGraph pipeline, a fully implemented genre normalization stage, an implemented Open Library book-fetching stage, a deterministic book ranking stage with optional LLM re-ranking and MMR diversity selection, a creative theme extraction stage, per-node artifact persistence, a Rich-formatted CLI, and a Python API for running the current pipeline.

The later creative stages (proposal drafting, rubric judging, synopsis writing) are scaffolded but not implemented yet. Today, a full `generate` run executes stages 0–3, persists artifacts, and returns a placeholder synopsis while stages 4–6 remain stubs.

## Current Status

Implemented:

- Package, CLI, config loader, and version reporting
- LangGraph orchestration with typed shared state and per-node artifact persistence
- `GenreNormalizerAgent` with deterministic mapping, fuzzy matching, and optional LLM fallback
- `BookFetcherAgent` backed by the Open Library Search API with disk cache, deduplication, and configurable `max_books` cap
- `BookRankerAgent` with deterministic weighted composite scoring (genre overlap, reader engagement, rating quality, rating volume), optional LLM re-ranking by narrative potential, and MMR diversity selection to ensure the shortlist covers the thematic space
- `ThemeExtractorAgent` — the creative engine of the pipeline. Identifies the thematic assumptions each genre tradition takes for granted, finds contradictions between traditions, frames them as creative questions, and generates concrete narrative seeds for the downstream proposal stage
- LLM provider registry for extensible provider support
- Rubric retry loop topology (conditional edge wired; noop placeholder always passes)
- Rich-formatted CLI output with per-stage timing and artifact paths
- Test coverage for import, CLI, schemas, graph node wrappers, prompt loading, book fetching, book ranking, genre normalization, theme extraction, LLM registry, and artifacts

Not implemented yet:

- `ProposalDraftAgent`
- `RubricJudgeAgent`
- `SynopsisWriterAgent`

Current runtime behavior:

1. Normalize the user's prompt into genres, tones, and narrative context tokens.
2. Fetch matching books from Open Library.
3. Rank books by composite score with MMR diversity selection (deterministic; optionally refined by LLM).
4. Extract thematic tensions, genre clusters, and narrative seeds (LLM — requires API key).
5. Pass through placeholder nodes for stages 4 to 6.
6. Return a `GenerationResult` with a placeholder `final_synopsis`.

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

Stages 0, 1, and 2 currently perform real work. Stages 3 to 6 are registered as no-op placeholders. The rubric retry loop is wired via a conditional edge; the noop rubric judge always passes so the pipeline runs linearly until the real agent is implemented.

### Stage 0: GenreNormalizerAgent

Status: implemented

Four-pass resolution pipeline:

- **Pass 1** — Greedy longest-match against `genre_map.json` (exact then fuzzy)
- **Pass 2** — Greedy longest-match against `tone_map.json` (exact then fuzzy)
- **Pass 3** — LLM fallback for unresolved tokens: classifies each remaining token as genre, tone, narrative context, or unknown
- **Pass 4** — Holistic genre inference: sends the full original prompt plus everything already resolved to an LLM and asks what genres are *implied* by the overall context but not yet captured. Results are stored as `InferredGenre` objects in a separate `inferred_genres` field, distinct from `normalized_genres`. Each inference carries a `rationale` string and a `confidence` score (default 0.7, lower than Pass 3's 0.8 to reflect the higher uncertainty of holistic inference). If Passes 1–3 find no explicit genres but Pass 4 infers some, the inferred genres are promoted into `normalized_genres` as a last-resort fallback so the pipeline can continue.

Other behaviors:
- Reads taxonomy data from `src/storymesh/data/genre_map.json` and `src/storymesh/data/tone_map.json`
- Passes 3 and 4 share the same LLM client, model, and temperature (both are lightweight classification tasks)
- Produces a strict `GenreNormalizerAgentOutput` including debug traces and preserved narrative context
- `InferredGenre.default_tones` is informational only and does **not** feed into the tone-merge pipeline. Revisit when downstream agents (ProposalDraft, SynopsisWriter) need richer tone data from implied genres.

### Stage 1: BookFetcherAgent

Status: implemented

- Queries Open Library's search API by subject
- Queries both `normalized_genres` (explicit, Passes 1–3) and `inferred_genres` (holistic, Pass 4) so books implied by contextual signals surface alongside explicitly named genres
- Uses disk cache via `diskcache`
- Deduplicates books by Open Library work key
- Preserves all matched source genres per book
- Caps results at `max_books` (default 50) after deduplication, prioritising cross-genre books
- Emits debug metadata for cache hits, misses, truncation, and per-genre counts

### Stage 2: BookRankerAgent

Status: implemented

- Scores books using a weighted composite of four signals:
  - **Genre overlap** (weight 0.40): fraction of queried genres that returned this book
  - **Reader engagement** (weight 0.25): min-max normalized `readinglog_count` from Open Library
  - **Rating quality** (weight 0.20): confidence-adjusted average rating (discounts low sample sizes)
  - **Rating volume** (weight 0.15): min-max normalized `ratings_count`
- Applies **MMR diversity selection** after scoring: uses Maximal Marginal Relevance with Jaccard genre similarity to ensure the shortlist covers distinct genre traditions rather than returning the same dominant titles repeatedly; configurable via `diversity_weight` (default 0.3; set to 0.0 for pure relevance)
- Output is returned in MMR selection order so downstream LLM agents see genre-diverse books early in the list, which directly influences which genre clusters ThemeExtractorAgent identifies as primary
- Truncates to a configurable `top_n` (default 10)
- Emits dual-representation output: full `RankedBook` records in artifacts, slim `RankedBookSummary` objects for downstream LLM token efficiency
- Optional LLM re-rank pass: when `llm_rerank: true`, passes the shortlist to an LLM which re-orders by narrative potential for the user's creative brief; falls back gracefully on LLM failure
- All scoring weights, `top_n`, and `diversity_weight` are configurable in `storymesh.config.yaml`

### Stage 3: ThemeExtractorAgent

Status: implemented

The creative engine of the pipeline. Rather than asking "what themes do these books share?", it identifies the **thematic assumptions** each genre tradition takes for granted, finds where those assumptions **contradict** each other, and frames each contradiction as a **creative question** that a story could explore.

For example, given "dark post-apocalyptic detective mystery":
- The mystery tradition assumes: truth is discoverable, there is a resolution, a detective figure restores order
- The post-apocalyptic tradition assumes: systems have collapsed, survival trumps justice, the world is fundamentally disordered
- The tension: What does "solving a case" mean when there's no institution to deliver justice to?

Output (the ThemePack) contains:
- **Genre clusters**: books grouped by tradition with their thematic assumptions and dominant tropes
- **Thematic tensions**: pairs of opposing assumptions framed as creative questions, each scored by intensity and accompanied by 2–4 **clichéd resolutions** — the predictable narrative moves that lazy writing defaults to. Downstream agents use these as explicit exclusions (ProposalDraft) and evaluation criteria (RubricJudge)
- **Narrative seeds**: 3–5 concrete 2–3 sentence story kernels that emerge from the tensions, incorporating the user's narrative context tokens (settings, time periods, character archetypes)

Uses `claude-sonnet-4-6` at temperature 0.6 — more capable than classification agents, with enough creative latitude to produce novel tensions while still generating valid structured JSON.

### Stages 4 to 6

Status: scaffolded only

The state fields, graph nodes, and versioning hooks exist, but the runtime logic for proposal drafting, rubric evaluation, and synopsis synthesis has not been implemented yet.

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
- `genre_normalizer`, `book_ranker`, `theme_extractor`, and `proposal_draft` agent overrides
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
│ StoryMesh v0.6.0  Run abc123def456                           │
│ Input: "dark post-apocalyptic detective mystery"             │
╰──────────────────────────────────────────────────────────────╯

 Stage                   Status    Time     Artifact
 ────────────────────────────────────────────────────────────────────────
 genre_normalizer        ✓ done    0.03s    ~/.storymesh/runs/abc.../genre_normalizer_output.json
 book_fetcher            ✓ done    1.24s    ~/.storymesh/runs/abc.../book_fetcher_output.json
 book_ranker             ✓ done    0.01s    ~/.storymesh/runs/abc.../book_ranker_output.json
 theme_extractor         ○ noop    0.00s    —
 proposal_draft          ○ noop    0.00s    —
 rubric_judge            ○ noop    0.00s    —
 synopsis_writer         ○ noop    0.00s    —
 ────────────────────────────────────────────────────────────────────────
                         Total: 1.28s

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
│       ├── book_fetcher_output.json
│       └── book_ranker_output.json
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

- The graph includes placeholder nodes for stages 4 to 6; the final synopsis is a placeholder until `SynopsisWriterAgent` is implemented.
- The rubric retry loop topology is wired but the routing function always passes (noop placeholder).
- `ThemeExtractorAgent` requires a configured API key; without one, stage 3 runs as a noop and the ThemePack is not produced.

## Roadmap

- Implement `ProposalDraftAgent` to select and develop the best narrative seed into a full proposal
- Activate rubric-based retry logic with real pass/fail signal
- Implement final synopsis synthesis
- Expand provider support beyond the current Anthropic implementation
