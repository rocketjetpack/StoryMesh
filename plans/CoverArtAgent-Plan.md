# CoverArtAgent Implementation Plan

## Overview

CoverArtAgent is Stage 7 of the StoryMesh pipeline. It receives the selected `StoryProposal`
from `ProposalDraftAgentOutput` and generates a book cover image using DALL-E 3 via the OpenAI
Images API.

Rather than crafting the image generation prompt in a separate LLM call, the image prompt is
produced by `ProposalDraftAgent` as a new field on `StoryProposal`. This is architecturally
sound for two reasons: the proposal generator has the richest creative context (all seeds,
tensions, genre clusters, and tonal direction) at the moment of story synthesis, and the LLM is
already constructing the story's visual world — producing a DALL-E prompt is a natural extension
of that work rather than a second pass over the same material.

`CoverArtAgent` itself is therefore a pure image-generation wrapper: it takes the
pre-crafted image prompt from the proposal, calls the DALL-E 3 API, and returns the raw image
result. The node wrapper handles saving the PNG to disk and constructing the final output schema.

### Pipeline Position

```
rubric_judge
  → [conditional]
      ├── PASS → synopsis_writer → cover_art → END
      └── FAIL → proposal_draft (retry)
```

Stage 7 runs after `synopsis_writer`. It depends only on `proposal_draft_output`, which is
available from Stage 4 onward — but placing it after `synopsis_writer` preserves the correct
ordering for when that stage is implemented.

### Noop Behaviour

If `OPENAI_API_KEY` is not set, `cover_art` runs as a noop (consistent with all other LLM
stages). A warning is logged and `cover_art_output` remains `None` in state.

---

## Work Item Ordering and Dependencies

```
WI-1: Schema changes
  │   - StoryProposal: add image_prompt field (proposal_draft.py, v1.1 → v1.2)
  │   - New CoverArtAgentOutput schema (cover_art.py, v1.0)
  │   - versioning/schemas.py: bump + add new constant
  │
  └─ WI-2: Prompt changes
       │   - proposal_draft_generate.yaml: add image_prompt to JSON schema
       │   - proposal_draft_retry.yaml: same
       │
       └─ WI-3: ImageClient abstraction
            │   - llm/image_base.py: ImageClient ABC + registry
            │   - llm/openai_image.py: DallEImageClient
            │
            └─ WI-4: ArtifactStore binary support
                 │   - core/artifacts.py: save_run_binary()
                 │
                 └─ WI-5: Agent core
                      │   - agents/cover_art/agent.py
                      │
                      └─ WI-6: Node wrapper
                           │   - orchestration/nodes/cover_art.py
                           │
                           └─ WI-7: Graph wiring, state, config
                                │   - orchestration/state.py
                                │   - orchestration/graph.py
                                │   - storymesh.config.yaml
                                │   - storymesh.config.yaml.example
                                │
                                └─ WI-8: CLI and versioning
                                        - cli.py: _STAGE_NAMES
                                        - versioning/schemas.py
                                        - versioning/agents.py
                                        - README.md
```

Recommended execution order: WI-1 through WI-8 in sequence. Each step is independently
testable before the next begins.

---

## 1. WI-1: Schema Changes

### 1a. `StoryProposal` — add `image_prompt` field

**File:** `src/storymesh/schemas/proposal_draft.py`

Add the following field to `StoryProposal`, after `genre_blend`:

```python
image_prompt: str = Field(
    min_length=30,
    description=(
        "A DALL-E 3-ready image generation prompt for the book cover. "
        "Describes the dominant visual (a scene, object, or atmosphere from "
        "the story world — not a named character portrait), art style, mood, "
        "color palette, and period or setting details. Contains no character "
        "names, text, or readable symbols."
    ),
)
```

This is a **breaking schema change**: any existing `proposal_draft_output.json` artifacts on
disk will fail to deserialize because they lack `image_prompt`. This is acceptable — the project
is in active development and pre-existing run artifacts are exploratory. The schema version bump
makes the break explicit and auditable.

**Schema version:** `PROPOSAL_SCHEMA_VERSION` 1.1 → 1.2 (in `versioning/schemas.py`).

### 1b. New `CoverArtAgentOutput` schema

**File:** `src/storymesh/schemas/cover_art.py` (CREATE)

```python
"""Pydantic schemas for CoverArtAgent (Stage 7)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.versioning.schemas import COVER_ART_SCHEMA_VERSION


class CoverArtAgentInput(BaseModel):
    """Input contract for CoverArtAgent (Stage 7).

    Assembled by the node wrapper from ProposalDraftAgentOutput.
    The agent has no knowledge of the pipeline.
    """

    image_prompt: str = Field(
        min_length=30,
        description="DALL-E 3 image generation prompt from StoryProposal.",
    )
    title: str = Field(
        min_length=1,
        description="Story title — used only in debug metadata.",
    )


class CoverArtAgentOutput(BaseModel):
    """Output contract for CoverArtAgent (Stage 7).

    image_path points to the PNG saved in the run artifact directory.
    The JSON artifact (cover_art_output.json) holds everything except
    the raw image bytes, which live only in the PNG file.
    """

    model_config = {"frozen": True}

    image_path: str = Field(
        description=(
            "Absolute path to the generated cover PNG in the run directory. "
            "Empty string when running without an artifact store (e.g. tests)."
        ),
    )
    image_prompt: str = Field(
        description="The prompt submitted to the image generation API.",
    )
    revised_prompt: str | None = Field(
        default=None,
        description=(
            "DALL-E 3 may rewrite the submitted prompt for safety or quality. "
            "When present this is what the model actually used. None otherwise."
        ),
    )
    model: str = Field(description="Image model identifier (e.g. 'dall-e-3').")
    image_size: str = Field(description="Image dimensions (e.g. '1024x1024').")
    image_quality: str = Field(description="Quality setting ('standard' or 'hd').")
    image_style: str = Field(description="Style setting ('vivid' or 'natural').")
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Generation metadata: latency_ms, title, provider.",
    )
    schema_version: str = COVER_ART_SCHEMA_VERSION
```

### 1c. Versioning

**File:** `src/storymesh/versioning/schemas.py`

Add:
```python
COVER_ART_SCHEMA_VERSION = "1.0"
```

Update `SCHEMA_VERSIONS`:
```python
SCHEMA_VERSIONS: dict[str, str] = {
    ...
    "Cover Art": COVER_ART_SCHEMA_VERSION,
}
```

Add version history comment:
```python
# 2026-04-29: Increment Proposal schema to 1.2. Added image_prompt (str, min_length=30)
#             to StoryProposal. ProposalDraftAgent generates this field alongside the
#             narrative fields; CoverArtAgent consumes it directly. Breaking change —
#             existing artifacts lack the field and will not deserialize.
# 2026-04-29: Add Cover Art schema 1.0. Introduces CoverArtAgentInput and
#             CoverArtAgentOutput. CoverArtAgent wraps DALL-E 3 image generation;
#             image_path points to the PNG saved in the run artifact directory.
```

### Tests

**File:** `tests/test_schemas_cover_art.py` (CREATE)

```
TestCoverArtAgentInput:
  - test_valid_construction
  - test_image_prompt_min_length: rejects strings < 30 chars
  - test_title_min_length: rejects empty string

TestCoverArtAgentOutput:
  - test_valid_construction
  - test_frozen: cannot mutate after construction
  - test_revised_prompt_defaults_to_none
  - test_debug_defaults_to_empty_dict
  - test_schema_version_matches: schema_version == COVER_ART_SCHEMA_VERSION
  - test_image_path_can_be_empty_string: allows empty path (no-artifact-store case)
```

**File:** `tests/test_schemas_proposal_draft.py` (UPDATE)

- Add `test_image_prompt_required`: construction fails without `image_prompt`
- Add `test_image_prompt_min_length`: rejects strings < 30 chars
- Update all `StoryProposal` fixture factories to include `image_prompt`

---

## 2. WI-2: Prompt Changes

Both prompt files need the `image_prompt` field added to their JSON output schema section. The
instructions for generating it belong in the system prompt.

### 2a. `proposal_draft_generate.yaml` — system prompt additions

Add a new step after STEP 3 (CRAFT A THEMATIC THESIS):

```
STEP 4 — WRITE A COVER IMAGE PROMPT:
The image_prompt field is a DALL-E 3-ready description of the book cover image.

Rules:
- 1-3 sentences only
- Describe a dominant visual: a scene, landscape, object, or atmosphere derived from
  the story world. NOT a named character's face or portrait.
- Specify: art style (e.g., painterly oil, gritty noir ink, ethereal watercolor,
  stark photorealism), mood, dominant color palette, and any period or setting
  details that establish the world at a glance.
- Do NOT include: character names, titles, text, readable symbols, or anything
  that references the story by name.
- Think: what image would stop a reader browsing a bookshelf? Aim for atmospheric
  and genre-communicating, not illustrative.

Example (dark post-apocalyptic detective):
"A rain-slicked street in a flooded cityscape at dusk, viewed from below, a single
figure in a long coat silhouetted against the pale light of a collapsed skyscraper.
Gritty noir ink wash style with muted greys and a single amber light source.
Atmosphere of isolation and institutional decay."
```

Add `image_prompt` to the JSON schema in RESPONSE FORMAT:

```json
"image_prompt": "<DALL-E 3 cover image prompt: dominant visual, art style, mood, color palette, setting — no character names or text>"
```

### 2b. `proposal_draft_retry.yaml` — same additions

Apply the identical STEP 4 instruction block and `image_prompt` field to the retry prompt.
The retry prompt has the same system section as the generate prompt — add it in the same
position so the field is always required regardless of which prompt fires.

### Tests

**File:** `tests/test_prompt_loader.py` (UPDATE)

Add to the existing proposal draft prompt tests:

```
TestProposalDraftGeneratePrompt (update):
  - test_schema_contains_image_prompt: "image_prompt" appears in the user template string
  - test_system_contains_step_4: system prompt contains STEP 4 instruction block

TestProposalDraftRetryPrompt (update):
  - test_schema_contains_image_prompt: "image_prompt" appears in the user template string
  - test_system_contains_step_4: system prompt contains STEP 4 instruction block
```

---

## 3. WI-3: ImageClient Abstraction

The existing `LLMClient` base class is designed for text completions. Image generation is a
fundamentally different API surface (different parameters, binary output), so a separate
abstract base class and registry is appropriate. This mirrors the `LLMClient` pattern exactly.

### 3a. `src/storymesh/llm/image_base.py` (CREATE)

```python
"""Abstract base class and registry for image generation clients."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass(frozen=True)
class GeneratedImage:
    """Raw result from an image generation API call.

    image_bytes is raw PNG data ready to write to disk.
    revised_prompt is the provider-rewritten prompt, if any.
    """

    image_bytes: bytes
    revised_prompt: str | None


class ImageClient(ABC):
    """Vendor-agnostic interface for image generation.

    Subclasses implement generate() for a specific provider.
    """

    def __init__(self, *, model: str, agent_name: str = "unknown") -> None:
        self.model = model
        self.agent_name = agent_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
        style: str,
    ) -> GeneratedImage:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions (e.g. '1024x1024').
            quality: Quality tier ('standard' or 'hd').
            style: Generation style ('vivid' or 'natural').

        Returns:
            GeneratedImage with raw PNG bytes and optional revised prompt.
        """
        ...


# ---------------------------------------------------------------------------
# Provider registry — mirrors storymesh.llm.base
# ---------------------------------------------------------------------------

_IMAGE_REGISTRY: dict[str, type[ImageClient]] = {}


def register_image_provider(name: str, cls: type[ImageClient]) -> None:
    """Register an ImageClient subclass for a provider name.

    Idempotent for the same class; raises if a different class is registered
    under an already-taken name.
    """
    if name in _IMAGE_REGISTRY and _IMAGE_REGISTRY[name] is not cls:
        raise ValueError(
            f"Image provider '{name}' is already registered to "
            f"{_IMAGE_REGISTRY[name].__name__}, cannot re-register to {cls.__name__}."
        )
    _IMAGE_REGISTRY[name] = cls


def get_image_provider_class(name: str) -> type[ImageClient]:
    """Return the ImageClient subclass registered for the given provider name.

    Raises:
        ValueError: If no provider is registered under name.
    """
    if name not in _IMAGE_REGISTRY:
        registered = ", ".join(sorted(_IMAGE_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown image provider: '{name}'. Registered providers: {registered}"
        )
    return _IMAGE_REGISTRY[name]
```

### 3b. `src/storymesh/llm/openai_image.py` (CREATE)

```python
"""DALL-E image generation client backed by the OpenAI Images API."""

from __future__ import annotations

import base64
import os

import openai

from storymesh.llm.image_base import GeneratedImage, ImageClient, register_image_provider

_DEFAULT_MODEL = "dall-e-3"


class DallEImageClient(ImageClient):
    """ImageClient implementation backed by the OpenAI Images API (DALL-E 3).

    API key is resolved from the api_key argument or OPENAI_API_KEY env var.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: Model identifier. Defaults to 'dall-e-3'.
        agent_name: Identifies the calling agent in log output.

    Raises:
        ValueError: If no API key is found in args or environment.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        agent_name: str = "unknown",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key has not been provided. "
                "Pass api_key or set OPENAI_API_KEY in the environment."
            )
        resolved_model = model or _DEFAULT_MODEL
        super().__init__(model=resolved_model, agent_name=agent_name)
        self._api_key = resolved_key
        self.client = openai.OpenAI(api_key=resolved_key)

    def generate(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
        style: str,
    ) -> GeneratedImage:
        """Call the OpenAI Images API and return PNG bytes.

        Requests b64_json response format so the image can be written to
        disk without a second HTTP round-trip to fetch from a URL.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions. DALL-E 3 supports '1024x1024',
                '1024x1792', and '1792x1024'.
            quality: 'standard' or 'hd'.
            style: 'vivid' (painterly/dramatic) or 'natural' (photorealistic).

        Returns:
            GeneratedImage with decoded PNG bytes and optional revised_prompt.

        Raises:
            openai.OpenAIError: On API-level failures.
            ValueError: If the API response is missing expected fields.
        """
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size=size,  # type: ignore[arg-type]
            quality=quality,  # type: ignore[arg-type]
            style=style,  # type: ignore[arg-type]
            response_format="b64_json",
        )

        if not response.data:
            raise ValueError("OpenAI Images API returned empty data list.")

        image_data = response.data[0]
        if image_data.b64_json is None:
            raise ValueError("OpenAI Images API response missing b64_json field.")

        image_bytes = base64.b64decode(image_data.b64_json)
        revised_prompt = image_data.revised_prompt  # str | None

        return GeneratedImage(image_bytes=image_bytes, revised_prompt=revised_prompt)


register_image_provider("openai", DallEImageClient)
```

### Tests

**File:** `tests/test_llm_registry.py` (UPDATE)

Add image registry tests alongside the existing LLM registry tests:

```
TestImageRegistry:
  - test_register_and_retrieve: register a FakeImageClient, get_image_provider_class returns it
  - test_duplicate_registration_same_class_ok: idempotent
  - test_duplicate_registration_different_class_raises: ValueError
  - test_unknown_provider_raises: ValueError with helpful message
```

**Note on DallEImageClient tests:** The OpenAI Images API call itself is not unit-tested
(same policy as `OpenAIClient` and `AnthropicClient`). Integration tests behind the
`real_api` marker can be added later if desired.

---

## 4. WI-4: ArtifactStore Binary Support

**File:** `src/storymesh/core/artifacts.py` (UPDATE)

Add one method to `ArtifactStore`:

```python
def save_run_binary(self, run_id: str, filename: str, data: bytes) -> None:
    """Write raw bytes to a file within a run directory.

    Used for non-JSON artifacts such as the cover art PNG. The run
    directory is created if it does not already exist.

    Args:
        run_id: Unique run identifier (matches the run directory name).
        filename: Name of the file to write (e.g. 'cover_art.png').
        data: Raw bytes to write.
    """
    run_dir = self.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_bytes(data)
```

### Tests

**File:** `tests/test_artifacts.py` (UPDATE)

```
TestArtifactStoreBinary:
  - test_save_run_binary_writes_file: file exists after call with correct content
  - test_save_run_binary_creates_run_dir: run directory created if absent
  - test_save_run_binary_overwrites: second call with same filename replaces content
```

---

## 5. WI-5: Agent Core

### Files

| File | Action |
|------|--------|
| `src/storymesh/agents/cover_art/__init__.py` | CREATE |
| `src/storymesh/agents/cover_art/agent.py` | CREATE |

### `__init__.py`

```python
"""CoverArtAgent package — Stage 7 of the StoryMesh pipeline."""
```

### `agent.py`

The agent is intentionally thin. All creative work has already been done by
`ProposalDraftAgent`; this agent's responsibility is API orchestration only.

It returns a `GeneratedCoverImage` dataclass (not the full `CoverArtAgentOutput`). The
`image_path` field in the output schema requires knowing the run directory, which is a
pipeline concern — the node wrapper handles that after receiving the raw result.

```python
"""CoverArtAgent — Stage 7 of the StoryMesh pipeline.

Generates a book cover image from the image_prompt field on the selected
StoryProposal. The prompt was written by ProposalDraftAgent at the moment
of story synthesis, when the model had the fullest creative context.

This agent makes a single API call to the image generation provider and
returns raw PNG bytes. Filesystem persistence is handled by the node wrapper.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

from storymesh.llm.image_base import ImageClient
from storymesh.schemas.cover_art import CoverArtAgentInput

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GeneratedCoverImage:
    """Raw result from image generation, before filesystem persistence.

    Returned by CoverArtAgent.run(). The node wrapper saves image_bytes
    to disk and constructs the final CoverArtAgentOutput with image_path.
    """

    image_bytes: bytes
    image_prompt: str
    revised_prompt: str | None
    model: str
    image_size: str
    image_quality: str
    image_style: str
    latency_ms: int


class CoverArtAgent:
    """Generates a book cover image from the story proposal (Stage 7).

    Makes a single call to the configured image generation provider using
    the image_prompt field from the selected StoryProposal.
    """

    def __init__(
        self,
        *,
        image_client: ImageClient,
        image_size: str = "1024x1024",
        image_quality: str = "standard",
        image_style: str = "vivid",
    ) -> None:
        """Construct the agent.

        Args:
            image_client: Image generation client instance. Required.
            image_size: Image dimensions. Default '1024x1024'.
            image_quality: Quality tier. 'standard' (default) or 'hd'.
            image_style: Generation style. 'vivid' (default) or 'natural'.
        """
        self._image_client = image_client
        self._image_size = image_size
        self._image_quality = image_quality
        self._image_style = image_style

    def run(self, input_data: CoverArtAgentInput) -> GeneratedCoverImage:
        """Generate a cover image from the proposal's image prompt.

        Args:
            input_data: Input assembled by the node wrapper from the selected proposal.

        Returns:
            GeneratedCoverImage with raw PNG bytes and generation metadata.

        Raises:
            openai.OpenAIError: On API-level failures from the image provider.
            ValueError: If the image provider returns an unexpected response.
        """
        logger.info(
            "CoverArtAgent starting | title=%r model=%s size=%s quality=%s style=%s",
            input_data.title,
            self._image_client.model,
            self._image_size,
            self._image_quality,
            self._image_style,
        )

        t0 = time.perf_counter()
        result = self._image_client.generate(
            input_data.image_prompt,
            size=self._image_size,
            quality=self._image_quality,
            style=self._image_style,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        logger.info(
            "CoverArtAgent complete | latency_ms=%d revised=%s",
            latency_ms,
            result.revised_prompt is not None,
        )

        return GeneratedCoverImage(
            image_bytes=result.image_bytes,
            image_prompt=input_data.image_prompt,
            revised_prompt=result.revised_prompt,
            model=self._image_client.model,
            image_size=self._image_size,
            image_quality=self._image_quality,
            image_style=self._image_style,
            latency_ms=latency_ms,
        )
```

### Tests

**File:** `tests/test_cover_art_agent.py` (CREATE)

Use a `FakeImageClient` stub (defined in the test file, not in the codebase — no production
code should need a fake implementation):

```python
class FakeImageClient(ImageClient):
    def __init__(self, image_bytes: bytes = b"PNG", revised_prompt: str | None = None):
        super().__init__(model="fake-image-model", agent_name="fake")
        self._image_bytes = image_bytes
        self._revised_prompt = revised_prompt
        self.calls: list[dict] = []

    def generate(self, prompt, *, size, quality, style):
        self.calls.append({"prompt": prompt, "size": size, "quality": quality, "style": style})
        return GeneratedImage(image_bytes=self._image_bytes, revised_prompt=self._revised_prompt)
```

Tests:

```
TestCoverArtAgentRun:
  - test_returns_generated_cover_image_type
  - test_image_bytes_from_client: output.image_bytes matches FakeImageClient bytes
  - test_image_prompt_preserved: output.image_prompt == input.image_prompt
  - test_revised_prompt_propagated: None when client returns None
  - test_revised_prompt_propagated_when_present: value carried through when set
  - test_model_from_client: output.model == image_client.model
  - test_size_quality_style_passed_to_client: FakeImageClient.calls[0] has correct params
  - test_latency_ms_is_non_negative: output.latency_ms >= 0

TestCoverArtAgentConfig:
  - test_default_size: default image_size == "1024x1024"
  - test_default_quality: default image_quality == "standard"
  - test_default_style: default image_style == "vivid"
  - test_custom_params_respected: non-default values passed through to client

TestCoverArtAgentErrors:
  - test_client_error_propagates: ValueError from image_client propagates unchanged
```

---

## 6. WI-6: Node Wrapper

**File:** `src/storymesh/orchestration/nodes/cover_art.py` (CREATE)

The node wrapper:

1. Reads `proposal_draft_output` from state. If `None` (e.g. all upstream stages nooped), returns `{}` silently.
2. Assembles `CoverArtAgentInput` from `proposal_draft_output.proposal`.
3. Calls `agent.run(input_data)` → `GeneratedCoverImage`.
4. Saves PNG via `artifact_store.save_run_binary()` if store is provided.
5. Constructs `CoverArtAgentOutput` with `image_path` set to the saved file path (empty string if no store).
6. Persists JSON artifact via `persist_node_output()`.
7. Returns `{"cover_art_output": output}`.

No `current_run_id` ContextVar is needed — `CoverArtAgent` makes no LLM calls.

```python
"""LangGraph node wrapper for CoverArtAgent (Stage 7)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.cover_art.agent import CoverArtAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.cover_art import CoverArtAgentInput, CoverArtAgentOutput
from storymesh.versioning.schemas import COVER_ART_SCHEMA_VERSION

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def make_cover_art_node(
    agent: CoverArtAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for CoverArtAgent (Stage 7).

    Args:
        agent: A fully constructed CoverArtAgent instance.
        artifact_store: Optional store for artifact persistence.
            Pass None (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature StoryMeshState -> dict[str, Any].
    """

    def cover_art_node(state: StoryMeshState) -> dict[str, Any]:
        """Generate a cover image from the selected proposal.

        If proposal_draft_output is absent (upstream nooped), returns an
        empty dict so the pipeline progresses without error.
        """
        proposal_draft_output = state.get("proposal_draft_output")
        if proposal_draft_output is None:
            logger.warning(
                "cover_art_node: proposal_draft_output is None — skipping cover art generation."
            )
            return {}

        proposal = proposal_draft_output.proposal
        input_data = CoverArtAgentInput(
            image_prompt=proposal.image_prompt,
            title=proposal.title,
        )

        raw = agent.run(input_data)

        # Persist PNG and set image_path; skip gracefully without a store.
        image_path = ""
        if artifact_store is not None:
            run_id: str = state.get("run_id", "")
            artifact_store.save_run_binary(run_id, "cover_art.png", raw.image_bytes)
            image_path = str(artifact_store.runs_dir / run_id / "cover_art.png")

        output = CoverArtAgentOutput(
            image_path=image_path,
            image_prompt=raw.image_prompt,
            revised_prompt=raw.revised_prompt,
            model=raw.model,
            image_size=raw.image_size,
            image_quality=raw.image_quality,
            image_style=raw.image_style,
            debug={
                "title": proposal.title,
                "latency_ms": raw.latency_ms,
            },
            schema_version=COVER_ART_SCHEMA_VERSION,
        )

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(artifact_store, state.get("run_id", ""), "cover_art", output)

        return {"cover_art_output": output}

    return cover_art_node
```

### Tests

**File:** `tests/test_graph.py` (UPDATE)

```
TestCoverArtNodeWrapper:
  - test_noop_when_proposal_draft_output_absent: returns {} when state has no proposal_draft_output
  - test_output_key_is_cover_art_output: returned dict has "cover_art_output" key
  - test_output_is_cover_art_agent_output_type
  - test_image_path_empty_without_artifact_store: image_path == "" when store is None
  - test_image_path_set_with_artifact_store: image_path non-empty when store provided
  - test_png_written_to_artifact_store: artifact_store.save_run_binary called with "cover_art.png"
  - test_json_artifact_persisted: persist_node_output called
  - test_debug_contains_title_and_latency
```

---

## 7. WI-7: Graph Wiring, State, and Config

### 7a. `state.py`

Add import and field:

```python
from storymesh.schemas.cover_art import CoverArtAgentOutput

# ── Stage 7: CoverArtAgent ─────────────────────────────────────────────────
cover_art_output: CoverArtAgentOutput | None
```

Also update `pipeline.py` initial state dict to include `"cover_art_output": None`.

### 7b. `graph.py`

Add two new helpers alongside the existing `_PROVIDER_KEY_MAP` and `_build_llm_client`:

```python
_IMAGE_PROVIDER_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
}

_IMAGE_PROVIDER_MODULE_MAP: dict[str, str] = {
    "openai": "storymesh.llm.openai_image",
}


def _ensure_image_provider_imported(provider: str) -> None:
    """Import the image provider module so its register_image_provider() call executes."""
    import importlib  # noqa: PLC0415
    module_name = _IMAGE_PROVIDER_MODULE_MAP.get(provider)
    if module_name:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.warning(
                "Image provider module '%s' could not be imported. "
                "Install the corresponding extra: pip install storymesh[%s]",
                module_name,
                provider,
            )


def _build_image_client(
    agent_cfg: dict[str, Any],
    agent_name: str = "unknown",
) -> "ImageClient | None":
    """Instantiate the correct ImageClient from an agent config dict.

    Returns None with a warning if the required API key is not set.
    """
    from storymesh.llm.image_base import get_image_provider_class  # noqa: PLC0415

    provider: str | None = agent_cfg.get("image_provider")
    model: str | None = agent_cfg.get("image_model")

    if provider is None:
        return None

    env_key = _IMAGE_PROVIDER_KEY_MAP.get(provider)
    if env_key and not os.environ.get(env_key):
        logger.warning(
            "%s is not set — the cover_art stage will run as noop.",
            env_key,
        )
        return None

    _ensure_image_provider_imported(provider)
    cls = get_image_provider_class(provider)
    return cls(model=model, agent_name=agent_name)
```

In `build_graph()`, add Stage 7 block before graph assembly:

```python
# ── Stage 7: CoverArtAgent ────────────────────────────────────────────────
from storymesh.agents.cover_art.agent import CoverArtAgent  # noqa: PLC0415
from storymesh.orchestration.nodes.cover_art import make_cover_art_node  # noqa: PLC0415

cover_cfg = get_agent_config("cover_art")
cover_image_client = _build_image_client(cover_cfg, agent_name="cover_art")

if cover_image_client is None:
    logger.warning("CoverArtAgent: no image client available — stage 7 will run as noop.")
    cover_art_node: Any = _noop_node
else:
    cover_agent = CoverArtAgent(
        image_client=cover_image_client,
        image_size=cover_cfg.get("image_size", "1024x1024"),
        image_quality=cover_cfg.get("image_quality", "standard"),
        image_style=cover_cfg.get("image_style", "vivid"),
    )
    cover_art_node = make_cover_art_node(cover_agent, artifact_store=artifact_store)
```

Update graph topology (replace current `synopsis_writer → END` with):

```python
graph.add_node("cover_art", cover_art_node)   # Stage 7

graph.add_edge("synopsis_writer", "cover_art")
graph.add_edge("cover_art", END)
```

Remove `graph.add_edge("synopsis_writer", END)`.

Update the docstring topology comment in `build_graph()` to reflect Stage 7.

### 7c. Config files

**`storymesh.config.yaml`** — add under `agents:`:

```yaml
  cover_art:
    image_provider: openai
    image_model: dall-e-3
    image_size: "1024x1024"
    image_quality: standard
    image_style: vivid
```

**`storymesh.config.yaml.example`** — same addition.

---

## 8. WI-8: CLI and Versioning

### 8a. `cli.py`

Add `"cover_art"` to `_STAGE_NAMES`:

```python
_STAGE_NAMES = [
    "genre_normalizer",
    "book_fetcher",
    "book_ranker",
    "theme_extractor",
    "proposal_draft",
    "rubric_judge",
    "synopsis_writer",
    "cover_art",           # Stage 7
]
```

The existing stage table logic already handles this: it checks for `{stage}_output.json` and
shows `✓ done` / `○ noop` accordingly. The PNG file appears in the artifact column as
`cover_art_output.json` (the JSON sidecar); the PNG path is visible inside that JSON if the
user inspects it.

### 8b. `versioning/schemas.py`

Already covered in WI-1: add `COVER_ART_SCHEMA_VERSION = "1.0"` and update `SCHEMA_VERSIONS`.

### 8c. `versioning/agents.py`

```python
COVER_ART_AGENT_VERSION = "1.0"

AGENT_VERSIONS: dict[str, str] = {
    ...
    "Cover Art": COVER_ART_AGENT_VERSION,
}
```

### 8d. `README.md`

- Add `CoverArtAgent` to the Implemented list with a one-line description
- Add Stage 7 to the Architecture topology diagram
- Add Stage 7 to the "Current runtime behavior" numbered list
- Remove cover art from the Not Implemented / Known Gaps section (it won't be there yet — but add it there while WI-1 through WI-7 are in progress, then move it to Implemented on completion)
- Update the pipeline version comment to reflect v0.8.0 (if applicable)

---

## Design Decision Record

### Why `image_prompt` on `StoryProposal` rather than a separate LLM call in CoverArtAgent?

The proposal generator has the fullest creative context at the moment of synthesis: it is
reasoning simultaneously about narrative seeds, thematic tensions, genre traditions, tonal
direction, and the protagonist's world. Generating a visual description at this moment means
the image prompt is coherent with the story's texture rather than derived from it after the
fact. A second LLM call in CoverArtAgent would read the finished proposal and work backward
to reconstruct the same creative context, paying LLM cost for inferior results.

The tradeoff: the proposal schema now carries a field that is only consumed by Stage 7. This
creates mild coupling — proposals are slightly larger, and old artifacts break. Both are
acceptable given the project is in active development.

### Why a separate `ImageClient` ABC rather than adding image generation to `LLMClient`?

Image generation APIs have a fundamentally different parameter surface (size, quality, style,
n) and return binary data rather than text. Extending `LLMClient` with image generation would
force every text-only client to implement `NotImplementedError` stubs and make the interface
harder to reason about. A separate ABC with its own registry is five lines of boilerplate and
results in a clean separation of concerns.

### Why return `GeneratedCoverImage` from the agent rather than `CoverArtAgentOutput`?

`CoverArtAgentOutput` contains `image_path`, which requires knowing the run artifact directory.
That is pipeline infrastructure knowledge, not agent knowledge. The agent has no access to the
`ArtifactStore` and should not. Returning an intermediate dataclass keeps the agent pure (no
filesystem side effects) and consistent with the project principle that agents are stateless
input→output functions. The node wrapper composes the final schema from the raw result after
handling persistence.

### Why `b64_json` response format instead of fetching the URL?

DALL-E 3 returns either a hosted URL or base64-encoded data. The URL is valid for ~60 minutes
then expires. Using `b64_json` delivers the image data in the same API response, eliminating
the second HTTP round-trip and the expiry window. The PNG is written to the run artifact
directory where it lives indefinitely alongside the other stage artifacts.

### Why `vivid` as the default image style?

`vivid` produces painterly, dramatic results well-suited to book cover aesthetics. `natural`
produces photorealistic output that reads as a photograph rather than an illustration, which
is less appropriate for most genre fiction covers. Both are configurable in
`storymesh.config.yaml`.

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

# 4. CLI shows cover_art stage
storymesh show-version      # Cover Art appears in schema/agent version tables
storymesh show-agent-config cover_art   # Outputs resolved config

# 5. Generate command produces a PNG (requires OPENAI_API_KEY)
storymesh generate "dark post-apocalyptic detective mystery"

# 6. Run artifacts include cover_art.png and cover_art_output.json
ls ~/.storymesh/runs/<latest_run_id>/
# Expected: cover_art.png  cover_art_output.json  proposal_draft_output.json  ...

# 7. Inspect cover_art_output.json
# - image_path points to the PNG file above
# - image_prompt is non-empty
# - revised_prompt is either null or a non-empty string
# - model == "dall-e-3"

# 8. Verify proposal_draft_output.json includes image_prompt on the proposal
cat ~/.storymesh/runs/<latest_run_id>/proposal_draft_output.json | python -m json.tool | grep image_prompt

# 9. Verify the cover_art.png is a valid PNG
file ~/.storymesh/runs/<latest_run_id>/cover_art.png
# Expected: PNG image data, 1024 x 1024, ...
```

---

## File Summary

| File | Action | Work Item |
|------|--------|-----------|
| `src/storymesh/schemas/proposal_draft.py` | UPDATE — add `image_prompt` to `StoryProposal` | WI-1 |
| `src/storymesh/schemas/cover_art.py` | CREATE | WI-1 |
| `src/storymesh/versioning/schemas.py` | UPDATE — bump `PROPOSAL_SCHEMA_VERSION`, add `COVER_ART_SCHEMA_VERSION` | WI-1 |
| `src/storymesh/prompts/proposal_draft_generate.yaml` | UPDATE — STEP 4 + `image_prompt` in JSON schema | WI-2 |
| `src/storymesh/prompts/proposal_draft_retry.yaml` | UPDATE — same | WI-2 |
| `src/storymesh/llm/image_base.py` | CREATE | WI-3 |
| `src/storymesh/llm/openai_image.py` | CREATE | WI-3 |
| `src/storymesh/core/artifacts.py` | UPDATE — add `save_run_binary()` | WI-4 |
| `src/storymesh/agents/cover_art/__init__.py` | CREATE | WI-5 |
| `src/storymesh/agents/cover_art/agent.py` | CREATE | WI-5 |
| `src/storymesh/orchestration/nodes/cover_art.py` | CREATE | WI-6 |
| `src/storymesh/orchestration/state.py` | UPDATE — add `cover_art_output` field | WI-7 |
| `src/storymesh/orchestration/pipeline.py` | UPDATE — add `cover_art_output: None` to initial state | WI-7 |
| `src/storymesh/orchestration/graph.py` | UPDATE — `_build_image_client`, Stage 7 node, edge wiring | WI-7 |
| `storymesh.config.yaml` | UPDATE — add `cover_art` agent config | WI-7 |
| `storymesh.config.yaml.example` | UPDATE — same | WI-7 |
| `src/storymesh/cli.py` | UPDATE — add `"cover_art"` to `_STAGE_NAMES` | WI-8 |
| `src/storymesh/versioning/agents.py` | UPDATE — add `COVER_ART_AGENT_VERSION` | WI-8 |
| `README.md` | UPDATE — status, architecture, runtime behavior | WI-8 |
| `tests/test_schemas_cover_art.py` | CREATE | WI-1 |
| `tests/test_schemas_proposal_draft.py` | UPDATE — `image_prompt` field tests + fixture updates | WI-1 |
| `tests/test_prompt_loader.py` | UPDATE — verify `image_prompt` in both prompt files | WI-2 |
| `tests/test_llm_registry.py` | UPDATE — image registry tests | WI-3 |
| `tests/test_artifacts.py` | UPDATE — `save_run_binary` tests | WI-4 |
| `tests/test_cover_art_agent.py` | CREATE | WI-5 |
| `tests/test_graph.py` | UPDATE — node wrapper + graph wiring tests | WI-6, WI-7 |
