# StoryMesh

StoryMesh is a production-grade, agentic AI workflow that generates **original fiction book synopses** from structured genre input.

The system is built on a strict **One-Agent-One-Tool** philosophy, combining deterministic constraint mapping with controlled LLM creativity. It emphasizes:

- Legal defensibility (no scraping, no copyrighted text storage)
- Novelty enforcement as a first-class system concern
- Strict JSON contracts between agents
- Observability and reproducibility
- Vendor-agnostic LLM integration

StoryMesh is designed to be modular, inspectable, and extensible — suitable for research, productization, or publishing-facing applications.

---

# Overview

## High-Level Workflow

StoryMesh takes a genre (or combination of genres) as input and outputs a polished, original book synopsis.

Pipeline stages:

1. Genre normalization
2. Bestseller seeding
3. Metadata enrichment
4. Theme aggregation
5. Multi-model proposal generation
6. Evaluation + similarity analysis
7. Final synthesis

Each stage is implemented as a narrow agent with a strict JSON schema.

---

## Architectural Philosophy

### One Agent = One Tool

Each agent:

- Performs exactly one operation
- Wraps exactly one tool (API call, transformation, or LLM call)
- Has strict Pydantic input/output schemas
- Is independently testable and cacheable

Orchestration is handled via a graph-based controller (e.g., LangGraph or custom DAG).

---

## Core Agents

### 0. GenreNormalizerAgent

**Purpose:**  
Transforms raw genre input into a structured constraint object.

**Hybrid design:**
- LLM classification (temperature = 0)
- Deterministic constraint expansion

**Output Schema (simplified):**

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

This agent defines the structural boundaries of generation.

---

### 1. NYTBestsellerFetcherAgent

**Tool:** NYT Books API  
**Purpose:** Retrieve historical bestseller data aligned with normalized genres.

**Output:**

```json
{
  "seed_books": []
}
```

Aggressive caching is required due to API rate limits.

---

### 2. SeedRankerAgent

**Tool:** Deterministic scoring function  
Ranks and filters seed books to a manageable set (e.g., 20–30).

---

### 3. Metadata Enrichment Agents

Optional enrichment:

- GoogleBooksMetadataAgent
- HardcoverMetadataAgent

These agents retrieve:
- Publisher descriptions
- Categories
- Ratings
- Identifiers

Important:
- No long copyrighted text is stored.
- Only derived structured summaries are retained.

---

### 4. BookProfileSynthesizerAgent

Produces structured summaries:

```json
{
  "plot_skeleton": {},
  "appeal_factors": [],
  "market_signals": {}
}
```

All summaries are compressed and structured.

---

### 5. ThemeAggregatorAgent

Aggregates seed book profiles into a `ThemePack`:

```json
{
  "genre_obligations": [],
  "genre_conventions": [],
  "genre_cliches": [],
  "innovation_axes": [],
  "market_patterns": []
}
```

This replaces scraped review analysis.

---

### 6. Proposal Generation Agents (Parallel)

- OpenAIProposalAgent
- GeminiProposalAgent
- AnthropicProposalAgent

Each produces:

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

All outputs conform to identical schemas.

---

### 7. RubricJudgeAgent

Evaluates proposals across:

- Genre fit
- Coherence
- Emotional arc
- Market hook
- Novelty
- Sequel potential

---

### 8. SimilarityRiskAgent

Compares proposals against:

- Seed plot structures
- Aggregated ThemePack

Flags:
- Structural overlap
- Derivative arcs
- Excessive similarity

Novelty enforcement is a first-class concern.

---

### 9. SynthesisWriterAgent

Produces:

- Final polished synopsis
- Optional back-cover copy
- Optional pitch package

---

## Pydantic Schemas

StoryMesh uses strict Pydantic models for:

- Inter-agent contracts
- Validation of LLM outputs
- JSON-only enforcement
- Retry logic
- Downstream stability

Schemas are versioned and validated at every boundary.

Future:
- Schema registry
- JSON schema export
- Snapshot-based drift testing

---

## Legal & Ethical Design

StoryMesh intentionally avoids:

- Scraping Goodreads
- Scraping Amazon
- Storing copyrighted review text
- Long-form copyrighted summaries

Instead it uses:

- Bestseller signals
- Ratings metadata
- Structured summaries
- Aggregated thematic extraction

All outputs are AI-generated and explicitly acknowledged as such.

---

## Production-Grade Features

- Strict JSON schema validation
- Rate limiting and retry logic
- API response caching
- Budget-aware LLM routing
- Single-revision regeneration rule
- Deterministic hashing for reproducibility
- Novelty guardrails

---

# Requirements

## Core Dependencies

- Python 3.10+
- pydantic
- rapidfuzz
- diskcache or redis
- orjson
- OpenAI / Anthropic SDK (for classification + generation)

## Optional

- LangGraph (for orchestration)
- FastAPI (for API layer)
- Docker (deployment)
- CI/CD pipeline (TBD)

---

# Setup

```bash
git clone https://github.com/<your-username>/storymesh.git
cd storymesh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
NYT_API_KEY=
GOOGLE_BOOKS_API_KEY=
HARDCOVER_API_KEY=
```

Run initial tests:

```bash
pytest
```

---

# Usage

## CLI (Planned)

```bash
storymesh generate --genre "post-apocalyptic eco-thriller romance"
```

Output:

```json
{
  "final_synopsis": "...",
  "scores": {},
  "similarity_risk": {}
}
```

## Python (Planned)

```python
from storymesh import generate_synopsis

result = generate_synopsis("dark fantasy enemies to lovers")
print(result.final_synopsis)
```

---

# Development Roadmap

- [ ] Finalize taxonomy for GenreNormalizerAgent
- [ ] Implement strict schema validation
- [ ] Integrate NYT Books API client
- [ ] Implement ThemeAggregator clustering
- [ ] Add SimilarityRisk structural comparison
- [ ] Add orchestration layer
- [ ] Add observability + logging
- [ ] Add drift monitoring

---

# History

## v0.1 — Concept & Architecture
- Defined One-Agent-One-Tool philosophy
- Designed full agent graph
- Established legal and novelty constraints

## v0.2 — GenreNormalizerAgent (In Progress)
- Hybrid classification design
- Taxonomy planning
- Schema planning

---

# Vision

StoryMesh aims to demonstrate that AI-generated fiction can be:

- Structurally rigorous
- Market-aware
- Legally defensible
- Explicitly novel
- Architecturally transparent

It treats originality as a systems problem, not a stylistic accident.

---

# License

TBD

---

# Status

Active development.