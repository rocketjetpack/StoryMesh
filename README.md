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

StoryMesh takes a genre (or combination of genres) as input and outputs a polished, original book synopsis.

Pipeline stages:

1. Genre normalization
2. Genre seed discovery
3. Seed ranking
4. Book profile synthesis
5. Theme aggregation
6. Proposal generation
7. Rubric evaluation
8. Final synthesis

Each stage is implemented as a narrow agent with a strict JSON schema. Orchestration is handled by a graph-based DAG controller.

---

## Architectural Philosophy

### One Agent = One Tool

Each agent:

- Performs exactly one operation
- Wraps exactly one tool (API call, transformation, or LLM call)
- Has strict Pydantic input/output schemas
- Is independently testable and cacheable

This design keeps each node auditable and replaceable without breaking the broader pipeline.

---

## Core Agents

### 0. GenreNormalizerAgent

**Purpose:**
Transforms raw genre input into a structured constraint object that governs all downstream generation.

**Design:**
- LLM classification at temperature = 0 for determinism
- Deterministic constraint expansion from a fixed taxonomy

**Output Schema:**
```json
{
  "normalized_genres": [],
  "subgenres": [],
  "tone": "",
  "audience": "",
  "genre_obligations": [],
  "genre_conventions": [],
  "genre_cliches": [],
  "innovation_axes": [],
  "constraints": {}
}
```

---

### 1. GenreSeedFetcherAgent

**Tool:** Open Library Search API
**Purpose:** Discover genre-relevant, culturally significant books using Open Library's subject and search APIs. Results are used as seed data for downstream theme extraction and market awareness.

**Design:**
- Queries Open Library subjects endpoint (e.g., `/subjects/mystery.json`) filtered by the normalized genres from the GenreNormalizerAgent.
- Uses edition count, number of ratings, and rating average as popularity proxies in place of bestseller rankings.
- Respects Open Library rate limits (1 req/sec default, 3 req/sec with User-Agent identification).
- Caching via `diskcache` is permitted; bulk downloads are not.
- No copyrighted content is stored — only catalog metadata (title, author, subjects, edition count, first publish year).

**Output:**
```json
{
  "seed_books": []
}
```

---

### 2. SeedRankerAgent

**Tool:** Deterministic scoring function
**Purpose:** Rank and filter the raw seed list to a manageable working set (20–30 books).

Scoring factors include genre alignment, recency, edition count (as a cultural reach proxy), and rating signals. This is a fully deterministic node — no LLM call is made.

**Output:**
```json
{
  "ranked_books": []
}
```

---

### 3. BookProfileSynthesizerAgent

**Tool:** LLM call
**Purpose:** Produce a compressed structured summary for each ranked seed book based on available metadata.

No copyrighted text is stored. Summaries are derived, structured, and compressed.

**Output (per book):**
```json
{
  "plot_skeleton": {},
  "appeal_factors": [],
  "market_signals": {}
}
```

---

### 4. ThemeAggregatorAgent

**Tool:** LLM call
**Purpose:** Aggregate all book profiles into a single `ThemePack` representing the genre's structural landscape.

This is the pipeline's primary knowledge-compression step. It distills what the genre demands, what it tends to overuse, and where space exists for innovation.

**Output:**
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

### 5. ProposalAgent

**Tool:** LLM call (single provider)
**Purpose:** Generate a structured fiction proposal that satisfies the genre constraints while targeting identified innovation axes.

The proposal conforms to a strict schema, ensuring downstream agents can operate on predictable fields.

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

**Design note:** The provider is configurable via environment variable. The schema is intentionally provider-agnostic, making it straightforward to extend to parallel multi-provider generation in a future iteration.

---

### 6. RubricJudgeAgent

**Tool:** LLM call
**Purpose:** Evaluate the proposal against a structured rubric and return scored feedback.

**Rubric dimensions:**

- Genre fit
- Internal coherence
- Emotional arc
- Market hook
- Novelty
- Sequel potential

**Output:**
```json
{
  "scores": {},
  "feedback": {},
  "pass": true
}
```

If the proposal does not pass, the pipeline triggers a single regeneration attempt from the ProposalAgent before halting. There is no unbounded retry loop.

---

### 7. SynthesisWriterAgent

**Tool:** LLM call
**Purpose:** Produce the final polished synopsis from the approved proposal and its rubric feedback.

**Output:**
```json
{
  "final_synopsis": "",
}
```

---

## Pydantic Schemas

StoryMesh uses strict Pydantic v2 models for:

- Inter-agent contracts
- Validation of LLM outputs
- JSON-only enforcement at LLM boundaries
- Retry logic on schema violations
- Downstream stability

Schemas are versioned. Every agent boundary is a validated checkpoint — malformed output from one agent never silently propagates to the next.

---

## Legal & Ethical Design

StoryMesh intentionally avoids:

- Scraping Goodreads or Amazon
- Storing copyrighted review text or long-form summaries
- Reproducing publisher copy verbatim

Instead it uses:

- Open Library catalog metadata (titles, authors, subjects, edition counts) under CC0-compatible terms
- Derived structured summaries (plot skeletons, market signals)
- Aggregated thematic extraction

All final outputs are AI-generated and explicitly acknowledged as such.

---

## Production-Grade Features

- Strict Pydantic v2 schema validation at every agent boundary
- Rate limiting and retry logic for external API calls
- Disk-based API response caching (`diskcache`)
- Deterministic hashing for reproducibility
- Single-revision regeneration rule (no unbounded loops)
- Novelty guardrails via ThemePack + RubricJudge

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
- Anthropic, OpenAI, or Gemini SDK (one required)

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
- [ ] Finalize GenreNormalizerAgent taxonomy and schema
- [ ] Implement GenreSeedFetcherAgent with Open Library integration and caching
- [ ] Implement SeedRankerAgent (deterministic scoring)
- [ ] Implement BookProfileSynthesizerAgent
- [ ] Implement ThemeAggregatorAgent
- [ ] Implement ProposalAgent
- [ ] Implement RubricJudgeAgent with regeneration rule
- [ ] Implement SynthesisWriterAgent
- [ ] Wire DAG orchestrator (pipeline.py)
- [ ] Add observability and structured logging

---

# History

## v0.1 — Concept & Architecture
- Defined One-Agent-One-Tool philosophy
- Designed full agent graph
- Established legal and novelty constraints
- Set up packaging, CI, linting, type checking, and test scaffolding

## v0.2 — GenreNormalizerAgent (In Progress)
- Hybrid classification design
- Taxonomy planning
- Schema planning

## v0.3 - Implementation of LLM (In Progress)
- Creation of YAML configuration file
- Parsing of .env and configuration file for LLM settings
- Add a generic LLM interface that has common function definitions for providers
- Add Anthropic as the first provider
- Replaced NYT Books API with Open Library API for legal compliance
- Renamed NYTBestsellerFetcherAgent to GenreSeedFetcherAgent
- Updated scoring model to use edition count and ratings as popularity proxies
- Removed Hardcover API references (insufficient documentation)

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

Active development. Core pipeline not yet implemented — see roadmap above.