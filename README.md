# StoryMesh

StoryMesh is an agentic AI workflow that generates **original fiction book synopses** from structured genre input.

The system is built on a strict **One-Agent-One-Tool** philosophy, combining deterministic constraint mapping with controlled LLM creativity. It emphasizes:

- Novelty enforcement as a first-class system concern
- Strict JSON contracts between agents
- Observability and reproducibility
- Vendor-agnostic LLM integration
- Legal defensibility (no scraping, no copyrighted text storage)

StoryMesh is designed to be modular, inspectable, and extensible.

---

# Overview

## High-Level Workflow

StoryMesh takes a user prompt containing genre preferences and narrative ideas as input and outputs a polished, original book synopsis. The pipeline discovers popular books in the requested genres, extracts thematic patterns from them, and uses those patterns alongside the user's creative intent to draft, evaluate, iteratively refine, and synthesize a final synopsis.

Pipeline stages:

1. Genre normalization
2. Book discovery
3. Book ranking
4. Theme extraction
5. Proposal drafting and rubric evaluation (iterative refinement loop)
6. Synopsis writing

Each stage is implemented as a narrow agent with a strict JSON schema. Orchestration is handled by a LangGraph-based DAG controller with conditional edges for the iterative refinement loop.

---

## Architectural Philosophy

### One Agent = One Tool

Each agent:

- Performs exactly one operation
- Wraps exactly one tool (API call, transformation, or LLM call)
- Has strict Pydantic input/output schemas
- Is independently testable and cacheable

This design keeps each node auditable and replaceable without breaking the broader pipeline.

### Iterative Refinement

The proposal-drafting stage is not a single pass. The ProposalDraftAgent generates a structured book proposal, the RubricJudgeAgent scores it against a multi-dimensional rubric, and if the proposal does not meet a configurable quality threshold, feedback is routed back to the ProposalDraftAgent for revision. This loop runs for a configurable maximum number of attempts. All attempts and their scores are tracked, and the best-scoring proposal across all attempts is forwarded to the final synthesis stage.

### Multi-Provider Readiness

The LLM abstraction layer (`LLMClient`) supports swapping providers (Anthropic, OpenAI, Gemini) per agent via configuration. The architecture is designed so that in a future iteration, the ProposalDraftAgent could fan out to multiple LLM providers simultaneously, generating competing proposals that are each evaluated and refined independently before the best elements are synthesized.

---

## Pipeline Data Flow

```
User: "dark post-apocalyptic detective mystery about a crime in 2075 Chicago"
  │
  ▼
[0] GenreNormalizerAgent
  │  genres: [mystery, post_apocalyptic]
  │  subgenres: [detective]
  │  tones: [dark]
  │  narrative_context: ["crime", "2075", "chicago"]
  │
  ▼
[1] BookFetcherAgent
  │  queries Open Library: /subjects/mystery.json, /subjects/post_apocalyptic.json
  │  returns: ~50 raw book metadata records
  │
  ▼
[2] BookRankerAgent
  │  deduplicates, scores cross-genre alignment, truncates
  │  returns: top 25 books with composite scores
  │
  ▼
[3] ThemeExtractorAgent  (LLM)
  │  "Given these 25 popular mystery/post-apocalyptic books,
  │   what are the genre obligations, conventions, clichés,
  │   and innovation opportunities?"
  │  returns: ThemePack
  │
  ▼
[4] ProposalDraftAgent  (LLM)  ◄──────────────────┐
  │  inputs: ThemePack + genres + tones              │
  │          + "crime in 2075 Chicago"               │
  │  returns: structured proposal JSON               │
  │                                                  │
  ▼                                                  │
[5] RubricJudgeAgent  (LLM)                          │
  │  scores the proposal against rubric dimensions   │
  │  if fail and attempts < max → retry with feedback┘
  │  if pass or max attempts reached → select best proposal
  │
  ▼
[6] SynopsisWriterAgent  (LLM)
  │  receives ALL proposals + ALL rubric scores
  │  synthesizes the best elements from each attempt
  │  into a polished narrative synopsis
  │
  ▼
Final Output: polished synopsis + scores + metadata
```

---

## Core Agents

### 0. GenreNormalizerAgent

**Status:** Implemented

**Tool:** Deterministic mapping with LLM fallback

**Purpose:** Transforms raw user input into a structured constraint object that governs all downstream generation. Separates genre tokens, tone modifiers, and narrative context (freeform user directives like settings, time periods, and plot elements) from the input string.

**Design:**
- Three-pass resolution: greedy longest-match against genre index, then tone index, then LLM fallback for remaining tokens
- Fuzzy matching via `rapidfuzz` for misspellings and close variants
- LLM classification at temperature 0 for deterministic categorization of unrecognized tokens
- Narrative context tokens (non-genre, non-tone) are preserved for downstream agents
- Resolution audit trails available in the `debug` dict for observability

**Output contract:** `GenreNormalizerAgentOutput` — includes `normalized_genres`, `subgenres`, `user_tones`, `tone_override`, `override_note`, and `debug` containing full resolution traces and `narrative_context`.

---

### 1. BookFetcherAgent

**Status:** Not yet implemented

**Tool:** Open Library Search API

**Purpose:** Discover genre-relevant, culturally significant books using Open Library's subject and search endpoints. Results provide the raw seed data that downstream agents use for theme extraction and market awareness.

**Design:**
- Queries the Open Library subjects endpoint (e.g., `/subjects/mystery.json`) for each normalized genre from Stage 0
- Uses edition count, number of ratings, and rating average as popularity proxies
- Respects Open Library rate limits (1 req/sec default, 3 req/sec with User-Agent identification)
- Caching via `diskcache` is permitted; bulk downloads are not
- No copyrighted content is stored — only catalog metadata (title, author, subjects, edition count, first publish year)

**Output:** A list of book metadata objects with fields sourced from Open Library's API response.

---

### 2. BookRankerAgent

**Status:** Not yet implemented

**Tool:** Deterministic scoring function

**Purpose:** Merge, deduplicate, score, and filter the raw book list from Stage 1 into a manageable working set.

When multiple genres are queried, the BookFetcherAgent returns overlapping result sets. This agent merges them by work ID, scores each book for composite genre alignment (a book matching 3 of 3 requested genres scores higher than one matching 1 of 3), applies recency and popularity weighting, and truncates to the top N books.

**Design:**
- Fully deterministic — no LLM call
- Scoring factors: genre alignment breadth, edition count (cultural reach proxy), rating signals, and publication recency
- Configurable output size (default: 25 books)

**Output:** A ranked list of book metadata objects with composite scores.

---

### 3. ThemeExtractorAgent

**Status:** Not yet implemented

**Tool:** LLM call

**Purpose:** Extract genre-level thematic patterns from the ranked book list. This is the pipeline's primary knowledge-compression step — it distills what the genre demands, what it tends to overuse, and where space exists for innovation.

The LLM receives the full ranked book list (titles, authors, subjects) and uses its world knowledge of these books to produce a structured ThemePack. This approach leverages the fact that widely-published books are well-represented in LLM training data, making individual per-book profiling unnecessary.

**Design:**
- Single LLM call with the full book list as context
- Temperature set for analytical reasoning (low, e.g. 0.2)
- The prompt explicitly asks for obligations (what the genre requires), conventions (what readers expect), clichés (what is overused), and innovation axes (where fresh approaches are possible)

**Output (ThemePack):**
```json
{
  "genre_obligations": [],
  "genre_conventions": [],
  "genre_cliches": [],
  "innovation_axes": [],
  "market_patterns": []
}
```

---

### 4. ProposalDraftAgent

**Status:** Not yet implemented

**Tool:** LLM call

**Purpose:** Generate a structured fiction proposal that satisfies genre constraints while targeting identified innovation axes and honoring the user's narrative context.

This agent participates in an iterative refinement loop with the RubricJudgeAgent. On the initial call, it receives the ThemePack, genre constraints, tone information, and narrative context. On retry calls, it additionally receives its previous proposal and the rubric feedback identifying specific weaknesses to address.

**Design:**
- Temperature set for creative generation (e.g. 0.7)
- Initial and retry prompts are distinct — the retry prompt instructs the LLM to revise based on specific feedback while preserving strengths
- Retry prompts may encourage exploration of different structural approaches to maintain diversity across attempts
- The provider is configurable via `storymesh.config.yaml`, supporting future multi-provider fan-out

**Output:**
```json
{
  "logline": "",
  "setting": "",
  "protagonist": {},
  "antagonist": {},
  "act_structure": {},
  "themes": [],
  "twist": "",
  "comparable_titles": [],
  "novelty_claim": ""
}
```

---

### 5. RubricJudgeAgent

**Status:** Not yet implemented

**Tool:** LLM call

**Purpose:** Evaluate a proposal against a structured rubric and return scored feedback. Routes the pipeline either back to the ProposalDraftAgent for revision or forward to the SynopsisWriterAgent.

**Rubric dimensions:**
- Genre fit — does the proposal satisfy the genre obligations from the ThemePack?
- Internal coherence — do the plot, characters, and setting work together?
- Emotional arc — is there a compelling character journey?
- Market hook — would this stand out on a shelf?
- Novelty — does it avoid the clichés identified in the ThemePack?
- User intent fidelity — does it honor the user's narrative context and tone?

**Design:**
- Temperature 0 for consistent evaluation
- Each dimension receives a numeric score (0.0–1.0) and a text explanation
- Pass/fail is determined by whether the average score meets or exceeds a configurable threshold (default: 0.7)
- The pipeline tracks all proposals and their rubric results across attempts
- When max attempts are exhausted without passing, the pipeline accepts the highest-scoring proposal rather than the most recent one
- Maximum attempts configurable (default: 3, meaning initial + 2 retries)

**Output:**
```json
{
  "scores": {},
  "feedback": {},
  "pass": true,
  "attempt_number": 1
}
```

**Graph topology:** The RubricJudgeAgent node uses a LangGraph conditional edge. On failure with remaining retries, it routes back to ProposalDraftAgent. On pass or max retries exhausted, it routes forward to SynopsisWriterAgent.

---

### 6. SynopsisWriterAgent

**Status:** Not yet implemented

**Tool:** LLM call

**Purpose:** Synthesize the best elements from all proposal attempts into a polished, cohesive narrative synopsis.

This agent receives every proposal draft and every rubric evaluation from the refinement loop. Its job is editorial: review what each attempt did well, identify the strongest elements across all drafts (the best setting from attempt 1, the most compelling character arc from attempt 2, the freshest twist from attempt 3), and weave them into a unified synopsis that is greater than the sum of its parts.

**Design:**
- Receives all proposals and all rubric results, not just the highest-scoring one
- The rubric feedback guides synthesis — dimensions with high scores indicate strengths to preserve; low scores indicate elements to rework or draw from a different attempt
- Output is narrative prose suitable for a publisher's pitch or book jacket
- Temperature set for polished creative writing (e.g. 0.5)

**Output:**
```json
{
  "final_synopsis": ""
}
```

---

## Pydantic Schemas

StoryMesh uses strict Pydantic v2 models for:

- Inter-agent contracts with frozen, versioned output models
- Validation of LLM outputs with automatic JSON parsing
- Retry logic on schema violations
- Separation of downstream-consumable fields from debug/audit data
- Downstream stability — malformed output from one agent never silently propagates to the next

Schemas are versioned. Every agent boundary is a validated checkpoint.

---

## Legal & Ethical Design

StoryMesh intentionally avoids:

- Scraping Goodreads or Amazon
- Storing copyrighted review text or long-form summaries
- Reproducing publisher copy verbatim
- Using APIs whose terms prohibit AI-related use of their data

Instead it uses:

- Open Library catalog metadata (titles, authors, subjects, edition counts) under CC0-compatible terms
- LLM world knowledge for thematic analysis of well-known books
- Derived structured summaries (plot skeletons, market signals)
- Aggregated thematic extraction

All final outputs are AI-generated and explicitly acknowledged as such.

---

## Configuration

### API Keys (.env — not committed to repo)

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

### Model and Agent Settings (storymesh.config.yaml — committed to repo)

```yaml
llm:
  default_provider: anthropic
  default_model: claude-haiku-4-5-20251001

agents:
  genre_normalizer:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    temperature: 0.0
  theme_extractor:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.2
  proposal_draft:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.7
    max_attempts: 3
    pass_threshold: 0.7
  rubric_judge:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.0
  synopsis_writer:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
```

### Prompts (src/storymesh/prompts/)

Prompts are stored as standalone text files, one per agent. They are never embedded in agent code or configuration. This separation enables independent iteration on prompts without modifying application logic.

### External Data Sources

The Open Library API does not require an API key for read access. Identified requests (with a User-Agent header containing the app name and contact email) receive a 3x rate limit increase. See the [Open Library API documentation](https://openlibrary.org/developers/api) for details.

---

## Production-Grade Features

- Strict Pydantic v2 schema validation at every agent boundary
- Rate limiting and retry logic for external API calls
- Disk-based API response caching (`diskcache`)
- Deterministic hashing for reproducibility
- Bounded iterative refinement with configurable attempt limits and quality thresholds
- Best-of-N proposal selection across refinement attempts
- Per-run artifact persistence for full pipeline auditing
- LangGraph-based orchestration with conditional edges
- Vendor-agnostic LLM abstraction with per-agent provider and model configuration

---

# Requirements

## Core Dependencies

- Python 3.12+
- `pydantic >= 2.0`
- `rapidfuzz`
- `diskcache`
- `orjson`
- `httpx`
- `typer`
- `python-dotenv`
- `pyyaml`
- `langgraph >= 0.2`
- Anthropic, OpenAI, or Gemini SDK (at least one required)

---

# Setup

```bash
git clone https://github.com/<your-username>/storymesh.git
cd storymesh
python -m venv venv
source venv/bin/activate
pip install -e ".[anthropic]"  # or openai, or gemini
```

Create a `.env` file using `.env.example` as a base and add appropriate API keys.

Run tests:
```bash
pytest
```

---

# Usage

## CLI

```bash
storymesh generate "post-apocalyptic eco-thriller romance"
```

Output:
```json
{
  "final_synopsis": "...",
  "scores": {},
  "metadata": {}
}
```

## Python

```python
from storymesh import generate_synopsis

result = generate_synopsis("dark fantasy enemies to lovers")
print(result.final_synopsis)
```

---

# Development Roadmap

- [x] Project scaffolding, CI, versioning infrastructure
- [x] GenreNormalizerAgent — deterministic mapping with LLM fallback
- [x] LLM abstraction layer (LLMClient, AnthropicClient)
- [x] Configuration system (YAML config + .env secrets)
- [x] Prompt externalization (storymesh/prompts/)
- [x] LangGraph orchestration scaffolding with artifact persistence
- [ ] Implement BookFetcherAgent with Open Library integration and caching
- [ ] Implement BookRankerAgent (deterministic scoring)
- [ ] Implement ThemeExtractorAgent
- [ ] Implement ProposalDraftAgent with initial and retry prompt variants
- [ ] Implement RubricJudgeAgent with conditional retry edge
- [ ] Implement SynopsisWriterAgent with multi-proposal synthesis
- [ ] Add observability and structured logging

---

# History

## v0.1 — Concept & Architecture
- Defined One-Agent-One-Tool philosophy
- Designed full agent graph
- Established legal and novelty constraints
- Set up packaging, CI, linting, type checking, and test scaffolding

## v0.2 — GenreNormalizerAgent
- Hybrid classification design with three-pass resolution pipeline
- Genre and tone mapping with fuzzy matching
- Pydantic schema design with versioning

## v0.3 — LLM Integration & Data Source Migration
- Created YAML configuration file with per-agent LLM settings
- Built vendor-agnostic LLM abstraction (LLMClient base class)
- Implemented AnthropicClient as first provider
- Wired LLM fallback into GenreNormalizerAgent Pass 3
- Replaced NYT Books API with Open Library API for legal compliance
- Renamed NYTBestsellerFetcherAgent to BookFetcherAgent
- Removed Hardcover API references

## v0.4 — Orchestration & Pipeline Architecture
- Integrated LangGraph for graph-based pipeline orchestration
- Built artifact persistence system for per-run auditing
- Scaffolded pipeline state and node wrappers for all stages
- Revised pipeline architecture: consolidated BookProfileSynthesizer and ThemeAggregator into ThemeExtractorAgent
- Designed iterative refinement loop for proposal drafting
- Renamed agents for clarity and consistency

---

# Vision

StoryMesh demonstrates that AI-generated fiction can be:

- Structurally rigorous
- Market-aware
- Legally defensible
- Explicitly novel
- Architecturally transparent

It treats originality as a systems problem, not a stylistic accident.

---

# License

MIT

---

# Status

Active development. GenreNormalizerAgent and LLM abstraction layer are implemented. Remaining pipeline agents are in progress — see roadmap above.