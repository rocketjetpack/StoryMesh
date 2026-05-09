# StoryMesh

StoryMesh is an agentic fiction pipeline for turning a free-text creative brief into a complete short story package: proposal, prose draft, cover image, and assembled PDF/EPUB artifacts.

The current package version is `0.7.0`.

## What It Does

Given a prompt like:

```text
A quiet literary mystery set in a flood-damaged coastal city...
```

StoryMesh will:

1. normalize genres, tones, and narrative context from the prompt
2. optionally infer implied genres from the prompt holistically
3. fetch comparable books from Open Library
4. rank them for narrative usefulness and diversity
5. extract thematic tensions and generate narrative seeds
6. draft one or more story proposals and select the strongest
7. judge the proposal against a rubric and optionally retry with feedback
8. select a voice profile
9. write a full short story in outline -> draft -> summary passes
10. optionally run a resonance review pass on the prose
11. optionally generate cover art
12. assemble the result into PDF and EPUB

StoryMesh also includes:

- a `compare` mode for StoryMesh vs. single-call baseline runs
- prompt-style switching at runtime
- offline stylometric diagnostics
- a kiosk web app with a React frontend and FastAPI backend

## Current Status

The core pipeline is implemented end-to-end.

Implemented subsystems include:

- Python package + Typer CLI
- LangGraph orchestration with typed shared state
- per-stage artifact persistence under `~/.storymesh/runs/<run_id>/`
- `GenreNormalizerAgent`
- `VoiceProfileSelectorAgent`
- `BookFetcherAgent`
- `BookRankerAgent`
- `ThemeExtractorAgent`
- `ProposalDraftAgent`
- `RubricJudgeAgent`
- `ProposalReaderAgent` on the retry path
- `StoryWriterAgent`
- `ResonanceReviewerAgent`
- `CoverArtAgent`
- `BookAssemblerAgent`
- compare-mode baseline generation and blinded eval packet output
- prompt-style loading with `default` fallback
- kiosk backend + frontend

## Architecture

The pipeline is built as a LangGraph `StateGraph`.

High-level flow:

```text
START
  -> genre_normalizer
  -> voice_profile_selector
  -> book_fetcher
  -> book_ranker
  -> theme_extractor
  -> proposal_draft
  -> rubric_judge
       -> PASS -> story_writer -> resonance_reviewer? -> cover_art? -> book_assembler -> END
       -> FAIL -> proposal_reader -> proposal_draft (revision) -> rubric_judge
```

Notes:

- `proposal_reader` only runs on the retry path.
- `resonance_reviewer` only runs for `high` and `very_high` quality presets.
- `cover_art` is a noop when no image provider is available.
- `book_assembler` may emit PDF, EPUB, both, or neither depending on installed extras and config.

## Pipeline Stages

### 1. Genre Normalization

`GenreNormalizerAgent` resolves:

- explicit genres
- tone words
- narrative context
- implied genres from holistic prompt inference

It uses a multi-pass approach:

- static map matching for genres
- static map matching for tones
- LLM fallback for unresolved tokens
- holistic genre inference over the full prompt

If no explicit genres are found but inference produces viable ones, inferred genres are promoted so the pipeline can continue.

### 2. Voice Profile Selection

`VoiceProfileSelectorAgent` picks a voice profile for the run.

Profiles live in:

- `src/storymesh/prompts/voice_profiles/`

The classifier can be bypassed with:

- CLI: `--voice <profile_id>`
- config: `agents.voice_profile_selector.voice_profile_override`

### 3. Book Fetching

`BookFetcherAgent` queries Open Library by subject, caches responses on disk, and deduplicates books by work key.

### 4. Book Ranking

`BookRankerAgent` scores fetched books using weighted signals like:

- genre overlap
- reader engagement
- rating quality
- rating volume

It also applies MMR-based diversity selection. An optional LLM rerank pass can be enabled in config.

### 5. Theme Extraction

`ThemeExtractorAgent` turns genre traditions and comparable books into:

- genre clusters
- thematic tensions
- narrative seeds

This is the stage that frames contradiction between genre assumptions as creative pressure for the story.

### 6. Proposal Drafting

`ProposalDraftAgent` drafts candidate proposals from the seeds, then selects the strongest one.

It supports:

- initial proposal generation
- retry generation after rubric feedback
- directed revision after both rubric and reader feedback

### 7. Rubric Evaluation

`RubricJudgeAgent` scores the proposal and decides whether it passes.

If it fails and retry budget remains, the pipeline loops through:

- `proposal_reader`
- `proposal_draft` revision/retry
- `rubric_judge`

### 8. Story Writing

`StoryWriterAgent` writes the story in three passes:

1. scene outline
2. full draft
3. back-cover summary

Voice-profile overlays are injected into this stage.

### 9. Resonance Review

`ResonanceReviewerAgent` runs only for higher quality settings.

It:

- identifies near-miss moments in the draft
- revises the prose in-place
- regenerates the summary from the revised draft

### 10. Cover Art

`CoverArtAgent` uses the selected proposal's `image_prompt` and appends a flat-canvas enforcement suffix before calling the image backend.

The generated image is then composited with:

- title
- byline

using Pillow.

### 11. Book Assembly

`BookAssemblerAgent` creates:

- `output.pdf`
- `output.epub`

in the run directory.

## Prompt Styles

Prompts are organized under:

- `src/storymesh/prompts/styles/`

Prompt resolution order for style `X` is:

1. `styles/X/<prompt>.yaml`
2. `styles/default/<prompt>.yaml`

That means a style only needs to override the prompt files it wants to change.

Current style directories in the repo include:

- `default` — canonical prompt set
- `slim` — selective shorter overrides
- `bare_minimum` — starter/template style
- `context_priming` — experiments with prepend-based priming
- `verbalized_sampling` — experiments in explicit candidate comparison
- `test` — internal/testing style

Runtime selection:

```bash
storymesh generate "your prompt" --prompt-style default
storymesh generate "your prompt" --prompt-style bare_minimum
storymesh generate "your prompt" --prompt-style slim
```

The default configured style is read from:

- `storymesh.config.yaml -> prompts.style`

### Context Priming

StoryMesh also supports an optional `prompts.prepend` pool in config. When a prompt template contains the literal `{prepend}` token, a random configured prepend line is substituted into the system prompt.

This is process-wide prompt decoration, not a separate style system.

## Voice Profiles

Voice profiles live in:

- `src/storymesh/prompts/voice_profiles/`

They are prompt-adjacent data files, not code. They can alter:

- craft overlay
- avoid overlay
- summary overlay
- exemplars

Current repo profiles include:

- `literary_restraint`
- `cozy_warmth`
- `genre_active`
- plus several additional experimental profiles

## CLI

### Generate

```bash
storymesh generate "dark post-apocalyptic detective mystery"
```

Useful options:

```bash
storymesh generate "prompt" --quality standard
storymesh generate "prompt" --prompt-style bare_minimum
storymesh generate "prompt" --voice cozy_warmth
storymesh generate "prompt" --email you@example.com
storymesh generate "prompt" --run-id my_custom_run_id
```

Quality presets:

- `draft`
- `standard`
- `high`
- `very_high`

During generation, the CLI shows:

- a live per-stage progress table
- elapsed wall-clock time
- approximate token totals aggregated from `llm_calls.jsonl`

### Compare

```bash
storymesh compare "A literary mystery in a flood-damaged city"
```

`compare`:

1. runs the full StoryMesh pipeline
2. reads `story_writer_output.json`
3. uses StoryMesh's final word count as the target length for a one-shot baseline
4. runs a single-call baseline using the `story_writer` provider/model by default
5. writes comparison artifacts into the same run directory

Baseline options:

```bash
storymesh compare "prompt" \
  --quality standard \
  --prompt-style default \
  --baseline-provider openai \
  --baseline-model gpt-4o \
  --baseline-temperature 0.8 \
  --baseline-max-tokens 8000
```

### Rerun

Re-run selected downstream stages without rerunning the whole graph:

```bash
storymesh rerun cover_art
storymesh rerun cover_art <run_id>
storymesh rerun book_assembler
storymesh rerun book_assembler <run_id> --email you@example.com
```

### Inspection / Utilities

```bash
storymesh show-version
storymesh show-config
storymesh show-agent-config genre_normalizer
storymesh inspect-run
storymesh inspect-run <run_id>
storymesh inspect-run <run_id> --llm all
storymesh inspect-run <run_id> --html out.html
storymesh stylometrics
storymesh stylometrics <run_id> --pretty
storymesh stylometrics --all
storymesh purge-cache
storymesh purge-cache --stages-only
storymesh purge-cache --api-only
storymesh purge-runs
```

### Kiosk

The kiosk commands are mounted under the main CLI when the `kiosk` extra is installed:

```bash
storymesh kiosk start
storymesh kiosk start --foreground
storymesh kiosk start --host 0.0.0.0 --port 8000
storymesh kiosk stop
storymesh kiosk status
```

## Python API

```python
from storymesh import generate_synopsis

result = generate_synopsis(
    "A literary mystery in a drought-stricken mountain town",
    prompt_style="default",
    voice_profile_override="literary_restraint",
)

print(result.final_synopsis)
print(result.metadata["run_id"])
```

Public entrypoint:

- `storymesh.generate_synopsis()`

Important keyword arguments:

- `pass_threshold`
- `max_retries`
- `min_retries`
- `skip_resonance_review`
- `prompt_style`
- `email_recipient`
- `run_id`
- `voice_profile_override`

## Installation

### Core

```bash
git clone https://github.com/<your-username>/storymesh.git
cd storymesh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[anthropic]"
```

### Optional Extras

Declared extras:

- `anthropic`
- `openai`
- `gemini`
- `pdf`
- `kiosk`
- `langsmith`
- `dev`
- `all-providers`
- `langgraph-studio`

Examples:

```bash
pip install -e ".[anthropic,openai,pdf]"
pip install -e ".[dev]"
pip install -e ".[kiosk]"
pip install -e ".[all-providers,pdf,kiosk]"
```

Notes:

- `gemini` is declared as an optional dependency, but there is not currently a registered Gemini client implementation under `src/storymesh/llm/`.
- PDF/EPUB generation depends on the `pdf` extra.
- kiosk mode depends on the `kiosk` extra.

## Frontend / Kiosk Development

The kiosk frontend lives in:

- `frontend/`

It is a React + Vite + TypeScript app.

Frontend commands:

```bash
cd frontend
npm install
npm run dev
npm run build
npm run preview
```

Backend:

- FastAPI app: `src/storymesh/kiosk/app.py`
- SSE-backed job updates
- built frontend is served from `frontend/dist` when present

Main kiosk API routes:

- `GET /healthz`
- `GET /api/prompt-styles`
- `POST /api/submit`
- `GET /api/jobs`
- `GET /api/gallery`
- `GET /api/cover/{run_id}`
- `GET /api/run/{run_id}/synopsis`
- `GET /api/events`

Important privacy invariant:

- kiosk request models do accept an email address on submission
- kiosk response models never expose that email
- the subprocess receives the email through `STORYMESH_EMAIL_RECIPIENT`, not argv

## Configuration

Primary project config:

- `storymesh.config.yaml`

Config layering:

1. project config discovered by walking up from `src/storymesh/`
2. optional user override at `~/.storymesh/storymesh.config.yaml`

The user config is deep-merged on top of the project config.

`.env` loading order:

1. `.env` in the current working directory
2. `~/.storymesh/.env`

Important config sections:

- `llm`
- `prompts`
- `agents`
- `api_clients`
- `cache`
- `logging`
- `email`
- `kiosk`
- `langsmith`

The repo also includes:

- `storymesh.config.yaml.example`

## Environment Variables

Typical variables:

```dotenv
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=storymesh-dev
STORYMESH_SMTP_USER=
STORYMESH_SMTP_PASSWORD=
```

When provider keys are missing, StoryMesh warns and affected LLM-backed stages may degrade to static-only/noop behavior depending on the agent.

## Artifacts

Run artifacts are written under:

```text
~/.storymesh/runs/<run_id>/
```

Typical files include:

```text
run_metadata.json
llm_calls.jsonl
genre_normalizer_output.json
voice_profile_selector_output.json
book_fetcher_output.json
book_ranker_output.json
theme_extractor_output.json
proposal_draft_output.json
rubric_judge_output.json
proposal_reader_feedback_output.json
story_writer_output.json
resonance_reviewer_output.json
cover_art_output.json
cover_art.png
book_assembler_output.json
output.pdf
output.epub
```

Compare-mode runs add:

```text
baseline_output.json
comparison.json
blinded_eval_packet.json
blinded_eval_key.json
```

Artifact notes:

- `llm_calls.jsonl` includes approximate token counts per call
- the one-shot baseline call is logged under agent name `compare_baseline`
- stage output files are written incrementally as the graph progresses

Stage cache location:

```text
~/.storymesh/stages/
```

API cache root defaults to:

```text
~/.cache/storymesh/
```

## LLM-as-Judge Workflow

If you are using `storymesh compare` to evaluate StoryMesh against a one-shot baseline:

- use `blinded_eval_packet.json` as the judge input
- keep `blinded_eval_key.json` out of the judging prompt

The packet contains:

- original user prompt
- two anonymized candidates labeled `A` and `B`
- each story's full draft and word count

This is designed for either:

- human blind review
- LLM-as-judge workflows

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run real API tests:

```bash
pytest -m real_api
```

Useful source files:

- `src/storymesh/cli.py`
- `src/storymesh/config.py`
- `src/storymesh/orchestration/graph.py`
- `src/storymesh/orchestration/pipeline.py`
- `src/storymesh/core/stage_progress.py`
- `src/storymesh/kiosk/app.py`
- `src/storymesh/kiosk/jobs.py`

## Known Limitations

- Many stages still depend on external provider availability; absent keys can turn parts of the pipeline into static-only/noop behavior.
- `CoverArtAgent` currently depends on the OpenAI image backend.
- the rubric is still biased toward the house literary sensibility more than all voice profiles equally
- the theme-extraction framework is strongest on literary / mystery / speculative tension-heavy prompts and can feel overdetermined for lighter commercial modes
- Gemini is listed as an optional dependency but is not yet wired as a provider implementation

## Repo Layout

High-signal directories:

- `src/storymesh/agents/` — agent implementations
- `src/storymesh/orchestration/` — LangGraph graph, nodes, pipeline wrapper
- `src/storymesh/prompts/styles/` — prompt styles
- `src/storymesh/prompts/voice_profiles/` — voice profiles
- `src/storymesh/kiosk/` — FastAPI kiosk backend
- `frontend/` — kiosk frontend
- `tests/` — test suite
- `plans/` — design / implementation notes
