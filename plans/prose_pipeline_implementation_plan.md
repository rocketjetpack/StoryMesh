# Prose Pipeline Redesign — Implementation Plan

**Created:** 2026-05-01
**Status:** In progress
**Based on:** `plans/prose_pipeline_redesign_proposal.md` + discussion refinements

---

## Design Decisions (Agreed)

| Decision | Resolution |
|----------|-----------|
| Scoring | 3-tier (0=fail, 1=acceptable, 2=strong), sum composite (max 10), configurable threshold via `--quality` presets |
| Unknowns | Optional `list[str]` field on StoryProposal, guidance not requirement |
| Principle 2 | Reframe from "convention then departure" to "story-serving choices" with concrete examples |
| Cover art | "Art direction for a human artist" framing; drop prescriptive genre table; demand medium/technique specificity |
| 3-act arc | Replaced with "narrative progression with named scenes and concrete turning points" |
| thematic_thesis | Keep field name, change prompt guidance to pressure/tension/unresolved direction |
| thematic_function | Rename to `narrative_pressure` in SceneOutline schema |
| Requirements vs. guidance | Cross-cutting audit: every "must" and "always" evaluated for structural vs creative |
| Field renaming | Keep `thematic_thesis` name in schema to limit cascades; rename `thematic_function` → `narrative_pressure` (smaller blast radius) |

---

## Phase 1: Proposal Draft (highest leverage)

### 1a. Schema: `src/storymesh/schemas/proposal_draft.py`
- Add `unknowns: list[str] = Field(default_factory=list, ...)` to `StoryProposal`
- Bump `PROPOSAL_SCHEMA_VERSION` in `src/storymesh/versioning/schemas.py`

### 1b. Prompt: `src/storymesh/prompts/proposal_draft_generate.yaml`
Key changes:
- **thematic_thesis guidance**: Reframe from "the story's philosophical answer" to thematic pressure — what the story circles around, what contradiction it can't settle
- **unknowns**: Add as optional creative tool, not requirement — "if the story benefits from unresolved questions, include 1-3"
- **Principle 2 reframe**: From "convention then departure" to "story-serving choices" — everything serves the story, departures aren't engineered
- **3-act arc**: → "narrative progression with named scenes and concrete turning points"
- **Cover art overhaul**: "Art direction for a human illustrator" framing; drop mandatory "Flat 2D" opener; drop prescriptive genre-mood table; demand specific medium/technique; anti-3D instruction is "do not depict a book cover mockup or product rendering"
- **Requirements → guidance audit**: Soften creative "musts" to guidance throughout
- Add `unknowns` to JSON response format

### 1c. Prompt: `src/storymesh/prompts/proposal_draft_retry.yaml`
- Mirror all system prompt changes from 1b (shared system prompt)
- User prompt section unchanged (revision context framing is fine)

---

## Phase 2: Rubric Judge

### 2a. Prompt: `src/storymesh/prompts/rubric_judge.yaml`
Key changes:
- **3-tier scoring**: 0 (fail) / 1 (acceptable) / 2 (strong) per dimension
- **D-2 reframe**: "Convention and Departure" → "Story-Serving Choices"
- **D-1 expansion**: Integrate texture/residue — reward productive unevenness, penalize overdetermination
- **D-3 expansion**: Include non-functional detail and observational residue as positives
- **Composite**: Sum of all dimensions (max 10), no pass/fail computed by LLM
- **Tier anchors**: Sharp, example-anchored descriptions for each tier of each dimension
- Unknowns awareness: judge should note if unknowns are present and well-used

### 2b. Schema: `src/storymesh/schemas/rubric_judge.py`
- Change dimension `score` type from `float` (0.0-1.0) to `int` (0, 1, 2)
- Update `composite_score` computation to sum of ints
- Bump `RUBRIC_SCHEMA_VERSION`

### 2c. Agent code: `src/storymesh/agents/rubric_judge/agent.py`
- Update composite calculation logic
- Update pass/fail threshold logic for int-based scoring
- Config: update default threshold in `storymesh.config.yaml`

### 2d. CLI: `src/storymesh/cli.py`
- Add `--quality` flag with presets: draft (5), standard (6), high (8)
- Expand max retries from 2 to 3 (configurable)

---

## Phase 3: Story Writer Outline

### 3a. Schema: `src/storymesh/schemas/story_writer.py`
- Rename `thematic_function` → `narrative_pressure` in `SceneOutline`
- Add `observational_anchor: str` field to `SceneOutline`
- Bump `STORY_WRITER_SCHEMA_VERSION`

### 3b. Prompt: `src/storymesh/prompts/story_writer_outline.yaml`
- Replace `thematic_function` with `narrative_pressure` in prompt + JSON format
- Add `observational_anchor` guidance: concrete physical/sensory detail per scene
- Add prompt-level flexibility instruction (scaffold not prison)
- Soften scene summaries: keep concrete and behavioral, avoid embedded interpretation
- Thread `{unknowns}` if present

---

## Phase 4: Story Writer Draft

### 4a. Prompt: `src/storymesh/prompts/story_writer_draft.yaml`
- Compress repeated restatements (says same thing several ways)
- Thread `{unknowns}` into user prompt
- Soften anti-AI rhetoric → focus on craft targets not failure avoidance
- Align THEMATICS section with new thematic_thesis-as-pressure framing

---

## Phase 5: Supporting Files

### 5a. Prompt: `src/storymesh/prompts/proposal_draft_select.yaml`
- Update evaluation criteria to match new Principle 2 (story-serving choices)
- Align with guidance-over-requirements tone

### 5b. Tests
- Update all tests that reference changed schema fields
- Update rubric score assertions from float to int
- Add tests for new `unknowns` field (optional, defaults to empty list)
- Update SceneOutline field references

---

## Key Principle: Story-Serving Choices

The central creative reframe across all prompts:

> Every choice — whether conventional or unexpected — should exist because the story
> needs it, not because the writer is performing restraint, originality, or craft.
> A conventional scene is not a failure. An unusual scene is not an achievement.
> The only question is: does this serve what the story is trying to do?

This replaces the "convention then departure" framework which inadvertently
engineered surprises rather than letting them emerge from the story's needs.

---

## Implementation Notes

- `proposal_draft_retry.yaml` shares its system prompt with `_generate.yaml` — changes must be mirrored
- The `proposal_draft_select.yaml` critic also needs alignment with the new principles
- Schema version bumps trigger history comments in `src/storymesh/versioning/schemas.py`
- All prompt changes should be audited for requirements-vs-guidance: creative suggestions should not be framed as laws
