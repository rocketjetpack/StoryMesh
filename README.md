# StoryMesh

StoryMesh is a Python package for a fully implemented agentic fiction pipeline. A single `generate` call normalizes a free-text creative prompt, fetches thematically relevant books from Open Library, extracts the tensions between genre traditions, drafts and edits a story proposal (with rubric-driven retry), selects a voice profile, writes a complete short story in three prose passes, reviews it for near-miss moments, generates a cover image, and assembles a PDF and EPUB.

## Current Status

All pipeline stages are implemented:

- Package, CLI, config loader, and version reporting
- LangGraph orchestration with typed shared state and per-node artifact persistence
- `GenreNormalizerAgent` with deterministic mapping, fuzzy matching, and optional LLM fallback
- `BookFetcherAgent` backed by the Open Library Search API with disk cache, deduplication, and configurable `max_books` cap
- `BookRankerAgent` with deterministic weighted composite scoring (genre overlap, reader engagement, rating quality, rating volume), optional LLM re-ranking by narrative potential, and MMR diversity selection
- `ThemeExtractorAgent` — identifies thematic assumptions each genre tradition takes for granted, finds contradictions between traditions, frames them as creative questions, and generates concrete narrative seeds
- `ProposalDraftAgent` with multi-sample seed-steering and self-selection. Generates N candidate proposals from different seeds at elevated temperature, then uses a critic call to select the strongest
- `RubricJudgeAgent` with 6-dimension craft quality rubric, cross-provider evaluation (OpenAI evaluating Anthropic output), cliché detection, composite scoring with configurable pass threshold, and structured feedback for the retry loop
- `VoiceProfileSelectorAgent` — classifies the user's prompt into one of three voice profiles (`literary_restraint`, `cozy_warmth`, `genre_active`) to produce genre-appropriate prose without homogenizing output across the portfolio
- `StoryWriterAgent` — three-pass prose generation: scene outline → full draft → back-cover summary. Voice profile overlays are applied to all three passes for register-appropriate output
- `ResonanceReviewerAgent` — identifies 0–3 near-miss moments where the draft implies depth but retreats before engaging. Expands avoidance moments with targeted prose additions; re-generates the back-cover summary to reflect the revised draft. Cross-provider review (GPT-4o reads what Claude wrote)
- `CoverArtAgent` — generates a cover image via gpt-image-1 using the `image_prompt` from the selected proposal. Title and byline are composited onto the image using Pillow
- `BookAssemblerAgent` — assembles a PDF (WeasyPrint) and EPUB (ebooklib) from the final prose draft and cover art
- LLM and ImageClient provider registries for extensible backend support
- Rubric retry loop with configurable pass threshold, max retries, and min retries
- Quality presets: `draft`, `standard`, `high`, `very_high`
- Offline stylometric counter (`storymesh stylometrics`) for per-run tic frequency diagnostics
- Rich-formatted CLI and Python API

## Architecture

The pipeline is built as a LangGraph `StateGraph` with this topology:

```text
START
  → genre_normalizer
  → voice_profile_selector         ← selects literary_restraint / cozy_warmth / genre_active
  → book_fetcher
  → book_ranker
  → theme_extractor
  → proposal_draft
  → rubric_judge
  → [conditional]
      ├── PASS → story_writer → resonance_reviewer → cover_art → book_assembler → END
      └── FAIL → proposal_draft (max retries configurable per quality preset)
```

All stages perform real work. The rubric retry loop is wired via a conditional edge; a failing rubric score triggers re-entry into `proposal_draft` with targeted feedback injected into the prompt. `resonance_reviewer` only runs at `high` and `very_high` quality presets.

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
- `InferredGenre.default_tones` is informational only and does **not** feed into the tone-merge pipeline

### Stage 0.5: VoiceProfileSelectorAgent

Status: implemented

Runs immediately after `genre_normalizer` and before `book_fetcher`. Classifies the user's prompt, normalized genres, and tone words into one of three voice profiles using a Haiku classifier at T=0:

| Profile | Intended for | Key behaviors |
|---|---|---|
| `literary_restraint` | dark, mystery, literary, dread, psychological | Current default prose style; subtext over statement |
| `cozy_warmth` | bedtime, cute, comfy, gentle, quietly wondrous | Direct emotion-naming; ritualistic repetition; soft anthropomorphism |
| `genre_active` | action, adventure, pulp, high-energy fanfic | Kinetic over interior; dialogue-forward; sentence economy |

The selected profile flows through pipeline state and is consumed by `StoryWriterAgent` (all three passes) and `ResonanceReviewerAgent` (revision and summary passes). If no API key is configured, the node defaults to `literary_restraint` without making any LLM calls. The profile can also be pinned via `voice_profile_override` in `storymesh.config.yaml` to bypass the classifier.

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

### Stage 4: ProposalDraftAgent

Status: implemented

Develops narrative seeds into fully realised story proposals using a **multi-sample with self-selection** architecture:

1. **Generate N candidates** (default 3) — each candidate is assigned a different narrative seed and calls the LLM independently at elevated temperature (1.2). Each call is fully stateless; divergence is enforced at the prompt level via seed-steering, candidate indexing, and an anti-overlap instruction
2. **Select the strongest** — a separate low-temperature (0.2) critic call evaluates all valid candidates against the thematic tensions, checking for cliché violations, thematic depth, specificity, tonal coherence, and internal conflict mirroring

Output contains the selected `StoryProposal`, all candidate proposals (for artifact inspection), a `SelectionRationale` with the critic's reasoning, and a debug dict with per-run metadata (temperatures, seed assignments, call counts).

If all candidate calls fail to parse, a `RuntimeError` is raised so the pipeline handles it gracefully. If only one candidate survives, it is selected without a critic call. If the critic call fails, candidate 0 is selected as a fallback.

Uses `claude-sonnet-4-6` for both drafting (temperature 1.2) and selection (temperature 0.2).

### Stage 5: RubricJudgeAgent

Status: implemented

Evaluates the selected `StoryProposal` from Stage 4 against six craft quality dimensions:

- **D-1 Tension Inhabitation** (weight 0.25): Does the proposal live inside the identified thematic tension rather than resolving or avoiding it?
- **D-2 Specificity Density** (weight 0.20): Are setting, character, and conflict expressed through concrete particulars rather than genre-level abstractions?
- **D-3 Craft Discipline** (weight 0.20): Does the proposal honour the Craft Directives from Stage 4, especially CD-1 (no resolution) and CD-2 (specificity)?
- **D-4 Protagonist Interiority** (weight 0.15): Does the protagonist have a clearly differentiated want/need split with an internal stake distinct from the surface plot?
- **D-5 Structural Surprise** (weight 0.10): Is there at least one structural or tonal element that subverts the expected arc for the genre combination?
- **D-6 User Intent Fidelity** (weight 0.10): Does the proposal faithfully honour the user's original creative brief?

Scoring is an integer composite out of 10, with each dimension scored 0 (fail) / 1 (acceptable) / 2 (strong). If the composite falls below the configured `pass_threshold` (default 6), the agent emits a structured `RubricJudgeAgentOutput` with `passed=False` and dimension-level feedback; the conditional edge re-routes to `proposal_draft` for a targeted retry. A `min_retries` setting forces at least one editorial revision even when the proposal passes on the first attempt.

Deliberately uses a **different LLM provider** than ProposalDraftAgent (`provider: openai`, `model: gpt-4o`) to prevent the evaluator from inheriting the generator's blind spots. The prompt is calibrated as antagonistic: it is instructed to find weaknesses, with anchors like "score of 2 is rare" and "most proposals score 5–7."

### Stage 6: StoryWriterAgent

Status: implemented

Produces a complete short story in three sequential LLM passes:

1. **Outline pass** (T=0.5): Expands the `StoryProposal`'s `key_scenes` into 6–10 `SceneOutline` objects. Each outline carries an `opens_with` sentence used verbatim by the draft pass — preventing generic AI opening lines. Voice profile exemplars replace the default few-shot examples at this stage.

2. **Draft pass** (T=0.8): Writes the full prose using scene outlines as structure. All craft principles (sentence rhythm, subtext, concrete detail, temporal irregularity) are enforced via the prompt. Voice profile `craft_overlay` and `avoid_overlay` are injected into the system prompt, producing register-appropriate prose without duplicating the full prompt for each profile.

3. **Summary pass** (T=0.4): Writes ~300-word back-cover marketing copy from the completed draft. Writing after the draft ensures it accurately reflects what was written rather than what was planned. Voice profile `summary_overlay` adjusts register here too.

### Stage 6b: ResonanceReviewerAgent

Status: implemented. Only runs at `high` and `very_high` quality presets.

Reviews the completed prose draft for near-miss moments — places where the story implies depth but retreats before engaging — and produces targeted expansions. Uses three internal LLM passes:

1. **Review pass** (GPT-4o, T=0.4): Identifies 0–3 near-miss moments, classifying each as *restraint* (earned silence) or *avoidance* (missed opportunity). Using a different provider than the writer avoids shared blind spots.

2. **Revision pass** (Claude, T=0.7): Expands avoidance moments in-place within the existing draft, matching the original voice. Adds roughly 50–150 words per moment. The voice profile `craft_overlay` is forwarded as a register reminder.

3. **Summary pass** (Claude, T=0.4): Re-generates the back-cover summary from the revised draft so it reflects the final text.

The node replaces `story_writer_output` in the pipeline state with the revised draft and summary, so downstream nodes (`cover_art`, `book_assembler`) consume the revised text without any changes to their code.

### Stage 7: CoverArtAgent

Status: implemented

Generates a cover image for the selected `StoryProposal` using the gpt-image-1 model via the OpenAI Images API.

Before the prompt reaches the API, `CoverArtAgent` appends a flat-canvas enforcement suffix that instructs the model to produce art filling the entire canvas with no book object, no perspective frame, and no text or lettering. Title and byline ("A StoryMesh Production") are composited onto the raw PNG programmatically via Pillow, giving reliable typography independent of the model's text rendering.

Configuration options (all in the `cover_art:` agent block of `storymesh.config.yaml`):

- `image_provider`: image generation backend (default `openai`)
- `image_model`: model identifier (default `gpt-image-1`)
- `image_size`: portrait `1024x1792` is the default for book covers
- `image_quality`: `auto`, `low`, `medium`, or `high` (default `auto`)

If `OPENAI_API_KEY` is absent, the node runs as a noop.

### Stage 8: BookAssemblerAgent

Status: implemented

Assembles the final short story into a PDF (via WeasyPrint) and EPUB (via ebooklib) using the prose draft, scene list, back-cover summary, and cover art PNG from earlier stages. No LLM calls. Runs as a noop if neither WeasyPrint nor ebooklib is installed.

## Requirements

- Python 3.12+
- `pip`

Optional extras:

- `storymesh[anthropic]`
- `storymesh[openai]`
- `storymesh[gemini]` *(dependency declared; LLM client not yet implemented)*
- `storymesh[langsmith]`
- `storymesh[dev]`
- `storymesh[all-providers]`
- `storymesh[langgraph-studio]`

Core dependencies are declared in [pyproject.toml](pyproject.toml).

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
- `genre_normalizer`, `book_ranker`, `theme_extractor`, `proposal_draft`, and `cover_art` agent overrides
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
│ StoryMesh v0.9.0  Run abc123def456                           │
│ Input: "dark post-apocalyptic detective mystery"             │
╰──────────────────────────────────────────────────────────────╯

 Stage                    Status    Time      Artifact
 ───────────────────────────────────────────────────────────────────────────
 genre_normalizer         ✓ done    0.03s     ~/.storymesh/runs/abc.../genre_normalizer_output.json
 book_fetcher             ✓ done    1.24s     ~/.storymesh/runs/abc.../book_fetcher_output.json
 book_ranker              ✓ done    0.01s     ~/.storymesh/runs/abc.../book_ranker_output.json
 theme_extractor          ✓ done    3.21s     ~/.storymesh/runs/abc.../theme_extractor_output.json
 proposal_draft           ✓ done    8.74s     ~/.storymesh/runs/abc.../proposal_draft_output.json
 rubric_judge             ✓ done    4.12s     ~/.storymesh/runs/abc.../rubric_judge_output.json
 story_writer             ✓ done    22.40s    ~/.storymesh/runs/abc.../story_writer_output.json
 resonance_reviewer       ✓ done    18.33s    ~/.storymesh/runs/abc.../resonance_reviewer_output.json
 cover_art                ✓ done    3.54s     ~/.storymesh/runs/abc.../cover_art_output.json
 book_assembler           ✓ done    0.88s     ~/.storymesh/runs/abc.../book_assembler_output.json
 ───────────────────────────────────────────────────────────────────────────
                          Total: 62.50s

╭─ Synopsis ───────────────────────────────────────────────────────────╮
│ In the weeks after the flood receded, Mara Voss returned to what     │
│ she knew. The records were gone. The precinct was a waterline stain  │
│ on brick. She kept working anyway.                                   │
╰──────────────────────────────────────────────────────────────────────╯

PDF:  ~/.storymesh/runs/abc123def456/story.pdf
EPUB: ~/.storymesh/runs/abc123def456/story.epub
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

Inspect a past run stage-by-stage:

```bash
storymesh inspect-run                          # most recent run
storymesh inspect-run <run_id>                 # specific run
storymesh inspect-run <run_id> --llm all       # include full LLM prompts and responses
storymesh inspect-run <run_id> --html out.html # export a self-contained HTML report
```

Count prose tics in a story draft (offline, no LLM calls):

```bash
storymesh stylometrics                  # most recent run, JSON output
storymesh stylometrics <run_id>         # specific run
storymesh stylometrics --pretty         # human-readable table
storymesh stylometrics --all            # all runs in the store
```

Regenerate the cover art for a previous run without re-running the full pipeline:

```bash
storymesh rerun cover_art               # regenerate for the most recent run
storymesh rerun cover_art <run_id>      # regenerate for a specific run
```

Purge caches and run data:

```bash
storymesh purge-cache                # purge both stage and API response caches
storymesh purge-cache --stages-only  # stage artifact cache only
storymesh purge-cache --api-only     # Open Library API response cache only
storymesh purge-runs                 # delete all per-run artifact directories
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
│       ├── run_metadata.json                   ← written before graph invocation
│       ├── llm_calls.jsonl                     ← per-call log (agent, prompt, response)
│       ├── genre_normalizer_output.json
│       ├── voice_profile_selector_output.json
│       ├── book_fetcher_output.json
│       ├── book_ranker_output.json
│       ├── theme_extractor_output.json
│       ├── proposal_draft_output.json
│       ├── rubric_judge_output.json
│       ├── story_writer_output.json
│       ├── resonance_reviewer_output.json
│       ├── cover_art_output.json
│       ├── cover_art.png
│       ├── book_assembler_output.json
│       ├── story.pdf
│       └── story.epub
└── stages/                                     ← stage-level cache (content-addressed)
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

## Known Gaps and Limitations

- LLM-required stages (`ThemeExtractorAgent`, `ProposalDraftAgent`, `RubricJudgeAgent`, `VoiceProfileSelectorAgent`, `StoryWriterAgent`, `ResonanceReviewerAgent`) degrade to noops when the relevant API key is absent.
- `CoverArtAgent` requires `OPENAI_API_KEY`; without one, Stage 7 runs as a noop.
- `BookAssemblerAgent` requires WeasyPrint (PDF) and ebooklib (EPUB); without them it runs as a noop.
- **Rubric-judge aesthetic bias**: the current rubric rewards literary-fiction qualities regardless of voice profile. Proposals for `cozy_warmth` or `genre_active` stories will be scored against literary criteria, which may cause the retry loop to pull them back toward literary aesthetic. This is a known v1 limitation; per-profile rubric conditioning is deferred.
- **ThemeExtractorAgent's dialectical framework** assumes every story argues a thesis via genre tension. This works well for literary and genre fiction but produces over-philosophical proposals for cozy or action genres.

## Roadmap

- ~~Implement `ProposalDraftAgent` with multi-sample drafting and critic selection~~
- ~~Activate rubric-based retry logic with real pass/fail signal~~
- ~~Implement `StoryWriterAgent` for three-pass prose generation~~
- ~~Implement `ResonanceReviewerAgent` for near-miss detection and targeted expansion~~
- ~~Implement `CoverArtAgent` for gpt-image-1 cover image generation~~
- ~~Implement `BookAssemblerAgent` for PDF and EPUB output~~
- ~~Add `VoiceProfileSelectorAgent` to produce genre-appropriate prose~~
- ~~Add offline stylometric counter (`storymesh stylometrics`)~~
- Per-profile rubric conditioning (v2: `cozy_warmth` and `genre_active` proposals scored against appropriate criteria)
- Per-profile theme extraction (v2: lighter extraction path for non-literary genres)
