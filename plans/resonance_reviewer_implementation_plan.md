# Resonance Reviewer — Implementation Plan

**Created:** 2026-05-01
**Status:** Proposed
**Based on:** `plans/resonance_reviewer_concept.md` + discussion of "The Tuesday Variable"

---

## Problem Statement

The story writer agent produces structurally sound, tonally controlled prose that
frequently introduces moments of genuine emotional or philosophical significance —
then moves past them without engaging. These "near-miss" moments arise because
the writer optimizes for restraint, coherence, and anti-melodrama simultaneously,
creating a bias toward *composure over pressure*. The result is prose that
*implies meaning* but does not *spend meaning*.

A separate review stage is needed because the writing agent cannot detect this
pattern from within its own generation process.

---

## Design Principles

### 1. Reader, not critic.

The reviewer's prompt must frame it as a *reader who cares*, not a literary
analyst. The question is not "what themes are underdeveloped" but **"where did
I want to stay longer, and why?"**

This reframes the diagnostic from academic ("queer subtext is unexplored") to
felt ("Data engineered a situation where O'Brien would discover how closely he's
been watching him, then waited. That's someone showing you their heart and
pretending it was an accident. The story never lets anyone feel the weight of
that.").

### 2. Name the felt implication, not the literary one.

Each near-miss diagnostic must articulate:
- What the moment *implies* (not in literary terms, but in human terms)
- What the reader *wanted* to happen
- What the story did instead (retreated, deflected, moved to action/procedure)

### 3. Stay, don't add.

Expansion directives should not request new scenes, characters, or plot events.
They should ask the writer to *stay in the moment longer*: let a silence extend,
let a character's hands stop moving, let the implication land before the story
moves on.

### 4. Distinguish restraint from avoidance.

- **Restraint**: the silence *says* the thing (e.g., walking away together with
  uneven and even footsteps — earned ending)
- **Avoidance**: the silence *replaces* the thing (e.g., Data says "I do not have
  adequate language to categorize the gap" and the story cuts to a timing update)

The reviewer must be calibrated to know the difference. Not every understated
moment is a miss — some are the point.

### 5. Selective: 2–3 moments maximum.

The reviewer identifies at most 3 near-miss moments. This preserves structural
integrity and prevents bloat. The revision pass is surgical, not global.

---

## Architecture

### Pipeline Insertion

The resonance reviewer sits between `story_writer` and `cover_art` as a
single-pass review/revise cycle (no retry loop):

```
story_writer → resonance_reviewer → story_reviser → cover_art → book_assembler
```

- **resonance_reviewer** (new Stage 6b): reads the draft, outputs structured
  diagnostics identifying 2–3 near-miss moments
- **story_reviser** (new Stage 6c): receives the original draft + diagnostics,
  produces a revised draft with targeted expansions

Alternative: combine review + revision into a single two-call agent (like the
story writer's outline→draft passes). This keeps the pipeline simpler.

**Recommended approach: single agent, two LLM passes.** The `ResonanceReviewerAgent`
runs two passes internally:

1. **Review pass** (cross-provider recommended): identify near-miss moments
2. **Revision pass** (same provider as writer): expand the identified moments
   within the existing draft

This mirrors how the story writer already uses multiple internal passes. The
pipeline sees it as one node.

```
story_writer → resonance_reviewer → cover_art → book_assembler
```

### Cross-Provider Review

The review pass MUST use a different provider than the writer. The writer's own
model shares the same biases that created the near-misses — asking it to find
its own blind spots is asking it to see what it cannot see. If the writer is
Claude, the reviewer should be GPT-4o or o3.

The review task — reading fiction with emotional sensitivity, distinguishing
restraint from avoidance, holding multiple unstated implications simultaneously
— is fundamentally a reasoning task, not a generation task. A high-capability
reasoning model (o3 or GPT-4o) is recommended for the review pass.

The revision pass uses the *same* provider as the writer (Claude Sonnet) to
maintain voice consistency. The revision is a creative generation task that must
match the existing prose style.

### State Changes

Add to `StoryMeshState`:
```python
# ── Stage 6b: ResonanceReviewerAgent ────────────────────────────────
resonance_reviewer_output: ResonanceReviewerAgentOutput | None
```

The reviewer *replaces* the `story_writer_output.full_draft` with the revised
version. More precisely, the node returns a new `story_writer_output` with the
updated `full_draft` and `word_count`, preserving the original `scene_list`,
`back_cover_summary`, and `debug` (augmented with review metadata).

Alternative: store the revised draft in a separate state field. But downstream
consumers (cover_art, book_assembler) already read `story_writer_output` — a
separate field would require updating every downstream node. Replacing in-place
is cleaner.

---

## Schema Design

### `src/storymesh/schemas/resonance_reviewer.py`

```python
class NearMissMoment(BaseModel):
    """A moment where the story implies depth but does not engage with it."""

    model_config = {"frozen": True}

    passage_ref: str = Field(
        min_length=10,
        description=(
            "Direct quote or close paraphrase (2-3 sentences) from the draft "
            "that constitutes the near-miss moment. Must be specific enough "
            "to locate in the text."
        ),
    )
    what_it_implies: str = Field(
        min_length=20,
        description=(
            "What the moment implies in human terms — not literary analysis, "
            "but what a reader would feel pulling at them. Written as felt "
            "experience, not academic observation."
        ),
    )
    what_the_reader_wanted: str = Field(
        min_length=20,
        description=(
            "What the reader wanted to happen next — not plot, but emotional "
            "or relational follow-through. 'I wanted to know what O'Brien's "
            "face did. I wanted to know if he recognized that Data was "
            "describing love.'"
        ),
    )
    what_the_story_did: str = Field(
        min_length=10,
        description=(
            "How the story retreated: deflected to action, cut to a new scene, "
            "moved to procedure, ended the paragraph."
        ),
    )
    expansion_directive: str = Field(
        min_length=20,
        description=(
            "Specific instruction for the revision pass. Framed as 'stay, "
            "don't add': extend the moment, let a silence land, let the "
            "character's reaction show before the story moves on. Must not "
            "request new scenes, characters, or plot events."
        ),
    )
    is_restraint_or_avoidance: str = Field(
        description=(
            "Either 'avoidance' or 'restraint'. Avoidance: the silence replaces "
            "the thing. Restraint: the silence says the thing. Only 'avoidance' "
            "moments should be expanded. This field forces the reviewer to "
            "explicitly classify before recommending expansion."
        ),
    )


class ResonanceReviewerAgentInput(BaseModel):
    """Input contract for ResonanceReviewerAgent."""

    model_config = {"frozen": True}

    full_draft: str = Field(
        min_length=500,
        description="Complete prose draft from StoryWriterAgent.",
    )
    proposal_title: str = Field(
        min_length=1,
        description="Title of the story, for context.",
    )
    thematic_thesis: str = Field(
        min_length=1,
        description=(
            "The story's central pressure/tension — what it circles around. "
            "Helps the reviewer distinguish thematically relevant near-misses "
            "from incidental ones."
        ),
    )
    scene_list_summary: str = Field(
        description=(
            "Brief scene-by-scene summary for structural context. The reviewer "
            "needs to understand the story's shape to judge whether restraint "
            "is earned at a given point."
        ),
    )


class ResonanceReviewerAgentOutput(BaseModel):
    """Output contract for ResonanceReviewerAgent."""

    model_config = {"frozen": True}

    near_miss_moments: list[NearMissMoment] = Field(
        max_length=3,
        description=(
            "0–3 identified near-miss moments, ordered by significance. "
            "Only includes moments classified as 'avoidance', not 'restraint'."
        ),
    )
    revised_draft: str = Field(
        min_length=500,
        description=(
            "The full prose draft with targeted expansions applied to the "
            "identified near-miss moments. Untouched passages must remain "
            "exactly as they were."
        ),
    )
    revised_summary: str | None = Field(
        default=None,
        description=(
            "Re-generated back-cover summary reflecting the revised draft. "
            "None when no moments were expanded (draft unchanged)."
        ),
    )
    revision_word_delta: int = Field(
        description="Word count change: revised minus original.",
    )
    moments_found: int = Field(
        ge=0,
        le=3,
        description="Total near-miss moments identified (before filtering to avoidance-only).",
    )
    moments_expanded: int = Field(
        ge=0,
        le=3,
        description="Number of moments that were actually expanded in the revised draft.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Review metadata: temperatures, token counts, provider info.",
    )
    schema_version: str = RESONANCE_REVIEWER_SCHEMA_VERSION
```

---

## Prompt Design

### Review Pass: `src/storymesh/prompts/resonance_reviewer_review.yaml`

Core framing (not final prompt text — will be refined):

```
You are a reader, not a critic. You have just finished reading a short story and
something is bothering you. Not that the story is bad — it isn't. It's good. But
there are moments where it made you feel something pulling at you, where the story
seemed to be about to say something that mattered, and then it moved on. You
wanted it to stay.

Your job is to identify those moments. Not themes. Not literary devices. Moments.
Places where, as a human reader, you would have leaned forward — and the story
leaned back.

Read the following story. Then identify up to 3 moments where:
- The story implies something emotionally or relationally significant
- A reader who cared about the characters would want the story to stay longer
- The story instead retreats: moves to action, cuts to a new scene, deflects
  into procedure, or simply ends the paragraph

For each moment, you must also decide: is this RESTRAINT or AVOIDANCE?

- RESTRAINT is when the silence IS the point. The story communicates through what
  it doesn't say. The ending of a story where two people walk away together
  without resolving anything — if the walking together IS the resolution, that's
  restraint. Don't touch it.

- AVOIDANCE is when the silence REPLACES the point. The story had something to
  say and chose not to. A character begins to confront something difficult and
  the scene cuts away. A moment of recognition is followed immediately by a
  return to procedure. The story sets up meaning and then treats the setup as
  sufficient. That's avoidance. That's what we want to fix.

Only recommend expansion for moments you classify as AVOIDANCE.

When describing what a moment implies, do not use literary terminology. Describe
it the way a person would describe it to a friend: "There's this part where Data
basically admits he's been in love with O'Brien for four years by showing him this
document, and O'Brien just... changes the subject to a seal coupling. And the
story lets him."

When describing what the reader wanted, describe it as desire, not critique:
"I wanted to know what O'Brien's face did. I wanted the story to let him sit
with the fact that someone understood him better than he understood himself."

When writing the expansion directive, frame it as STAYING, not ADDING:
"Don't add new information. Let O'Brien's hands stop moving. Let the silence be
three sentences longer. Let Data's pause before 'an unusual situation' contain
the thing he can't say. The reader needs to feel the weight of this moment
before the story is allowed to move on."
```

### Revision Pass: `src/storymesh/prompts/resonance_reviewer_revise.yaml`

Core framing:

```
You are the author of the following story. A trusted reader has identified
moments where you retreated from your own best ideas — places where you set up
something meaningful and then moved on before letting the reader feel it.

You are going to revise the story. The revisions are surgical:

- You will expand ONLY the identified moments. Nothing else changes.
- You are STAYING in the moment, not ADDING to it. Extend the beat. Let a
  silence land. Let a character's body react before their words do. Let the
  implication sit in the room for a sentence or two longer.
- Match the existing voice exactly. The expanded passage must be indistinguishable
  from the surrounding prose in style, rhythm, and register.
- Each expansion should add roughly 50–150 words. Not more. The goal is pressure,
  not bulk.
- Do not resolve the tension. Do not explain the subtext. Do not have characters
  articulate what they feel. Just let the moment breathe before the story moves on.
- Preserve all SCENE_BREAK delimiters exactly as they appear.

Output the complete revised story. Every passage that was NOT identified as a
near-miss must appear EXACTLY as it was — word for word, punctuation for
punctuation.
```

---

## Agent Design

### `src/storymesh/agents/resonance_reviewer/agent.py`

```python
class ResonanceReviewerAgent:
    """Reviews a story draft for near-miss moments and produces targeted expansions.

    Two internal LLM passes:
    1. Review pass: identify 0–3 near-miss moments (cross-provider recommended)
    2. Revision pass: expand identified avoidance moments within the existing draft
    """

    def __init__(
        self,
        *,
        review_llm_client: LLMClient,      # cross-provider (e.g., GPT-4o)
        revision_llm_client: LLMClient,     # same provider as writer (e.g., Claude)
        review_temperature: float = 0.4,    # analytical, but with sensitivity
        revision_temperature: float = 0.7,  # creative but controlled
        review_max_tokens: int = 4096,
        revision_max_tokens: int = 8000,    # must hold the full revised draft
    ): ...

    def run(self, input_data: ResonanceReviewerAgentInput) -> ResonanceReviewerAgentOutput:
        """Run review + revision + optional summary re-run."""
        # Pass 1: identify near-miss moments (cross-provider)
        moments = self._run_review_pass(input_data)

        # Filter to avoidance-only
        avoidance_moments = [m for m in moments if m.is_restraint_or_avoidance == "avoidance"]

        if not avoidance_moments:
            # No expansion needed — return original draft unchanged
            return ResonanceReviewerAgentOutput(
                near_miss_moments=[],
                revised_draft=input_data.full_draft,
                revised_summary=None,
                revision_word_delta=0,
                moments_found=len(moments),
                moments_expanded=0,
                ...
            )

        # Pass 2: revise with expansion directives (same provider as writer)
        revised_draft = self._run_revision_pass(input_data.full_draft, avoidance_moments)

        # Pass 3: re-run back-cover summary on revised draft (same provider as writer)
        revised_summary = self._run_summary_pass(revised_draft, input_data)

        return ResonanceReviewerAgentOutput(
            near_miss_moments=avoidance_moments,
            revised_draft=revised_draft,
            revised_summary=revised_summary,
            ...
        )
```

### Key Design Choices

**Two separate LLM clients.** Unlike every other agent which takes a single
`llm_client`, this agent takes two: one for review (cross-provider) and one for
revision (same provider as writer). The revision client must match the writer's
voice.

**No-op when zero avoidance moments found.** If the reviewer classifies all
moments as restraint (or finds none), the draft passes through unchanged. This
prevents unnecessary revision.

**Expansion budget.** The revision prompt constrains each expansion to ~50–150
words. With at most 3 moments, the maximum total expansion is ~450 words on a
~3000-word story — roughly 15%. This prevents bloat while allowing meaningful
deepening.

---

## Node Wrapper

### `src/storymesh/orchestration/nodes/resonance_reviewer.py`

Standard pattern: assemble input from state, run agent, persist artifact,
update state.

The node *replaces* `story_writer_output` with a modified version containing
the revised draft and updated word count. This ensures downstream nodes
(cover_art, book_assembler) consume the revised text without any changes to
their code.

```python
def make_resonance_reviewer_node(
    agent: ResonanceReviewerAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:

    def resonance_reviewer_node(state: StoryMeshState) -> dict[str, Any]:
        story_output = state.get("story_writer_output")
        if story_output is None:
            raise RuntimeError(...)

        proposal_output = state.get("proposal_draft_output")
        proposal = proposal_output.proposal

        # Build scene summary for structural context
        scene_summary = "\n".join(
            f"- {s.title}: {s.summary}" for s in story_output.scene_list
        )

        input_data = ResonanceReviewerAgentInput(
            full_draft=story_output.full_draft,
            proposal_title=proposal.title,
            thematic_thesis=proposal.thematic_thesis,
            scene_list_summary=scene_summary,
        )

        output = agent.run(input_data)

        # Persist the review output as its own artifact
        if artifact_store is not None:
            persist_node_output(artifact_store, state["run_id"],
                                "resonance_reviewer", output)

        # Replace story_writer_output with revised draft + summary
        revised_story_output = StoryWriterAgentOutput(
            back_cover_summary=(
                output.revised_summary
                if output.revised_summary is not None
                else story_output.back_cover_summary
            ),
            scene_list=story_output.scene_list,
            full_draft=output.revised_draft,
            word_count=len(output.revised_draft.split()),
            debug={
                **story_output.debug,
                "resonance_review_applied": True,
                "moments_expanded": output.moments_expanded,
                "revision_word_delta": output.revision_word_delta,
            },
            schema_version=story_output.schema_version,
        )

        return {
            "resonance_reviewer_output": output,
            "story_writer_output": revised_story_output,
        }

    return resonance_reviewer_node
```

---

## Graph Changes

In `build_graph()`:

```python
# ── Stage 6b: ResonanceReviewerAgent ──────────────────────────────
# (insert after Stage 6, before Stage 7)

graph.add_node("resonance_reviewer", resonance_reviewer_node)

# Change edge: story_writer → resonance_reviewer (was → cover_art)
graph.add_edge("story_writer", "resonance_reviewer")
graph.add_edge("resonance_reviewer", "cover_art")
# Remove: graph.add_edge("story_writer", "cover_art")
```

---

## Config

### `storymesh.config.yaml`

```yaml
resonance_reviewer:
  review_provider: openai          # cross-provider: different model finds different blind spots
  review_model: gpt-4o             # reasoning model for emotional/contextual sensitivity
  revision_provider: anthropic     # same as writer for voice consistency
  revision_model: claude-sonnet-4-6
  review_temperature: 0.4          # analytical but sensitive
  revision_temperature: 0.7        # creative but controlled
  review_max_tokens: 4096
  revision_max_tokens: 8000        # must hold full revised draft
  summary_max_tokens: 1024         # for back-cover summary re-run
  summary_temperature: 0.4
```

---

## CLI Changes

### Stage table update

Add `resonance_reviewer` to `_STAGE_NAMES` in `cli.py` between `story_writer`
and `cover_art`.

### Quality-gated resonance review

The resonance reviewer only runs at `high` and `very_high` quality presets.
At `draft` and `standard`, the node returns the draft unchanged (no LLM calls).

Quality presets become 4-tuples: `(pass_threshold, max_retries, min_retries, enable_resonance_review)`:

```python
_QUALITY_PRESETS: dict[str, tuple[int, int, int, bool]] = {
    "draft":     (5, 1, 0, False),
    "standard":  (6, 2, 1, False),
    "high":      (8, 3, 1, True),
    "very_high": (9, 3, 2, True),
}
```

A `skip_resonance_review` bool is threaded through:
`CLI → generate_synopsis() → StoryMeshPipeline → build_graph() → node`

The node checks this flag and returns the original draft unchanged when `True`.

---

## Versioning

### `src/storymesh/versioning/schemas.py`
- Add `RESONANCE_REVIEWER_SCHEMA_VERSION = "1.0"`

### `src/storymesh/versioning/agents.py`
- Add `RESONANCE_REVIEWER_AGENT_VERSION = "1.0"`

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/storymesh/schemas/resonance_reviewer.py` | Pydantic contracts |
| `src/storymesh/agents/resonance_reviewer/__init__.py` | Package init |
| `src/storymesh/agents/resonance_reviewer/agent.py` | Two-pass agent |
| `src/storymesh/orchestration/nodes/resonance_reviewer.py` | Node wrapper |
| `src/storymesh/prompts/resonance_reviewer_review.yaml` | Review pass prompt |
| `src/storymesh/prompts/resonance_reviewer_revise.yaml` | Revision pass prompt |
| `tests/test_resonance_reviewer_agent.py` | Agent unit tests |
| `tests/test_schemas_resonance_reviewer.py` | Schema validation tests |

## Files to Modify

| File | Change |
|------|--------|
| `src/storymesh/orchestration/state.py` | Add `resonance_reviewer_output` field |
| `src/storymesh/orchestration/graph.py` | Wire new node, adjust edges |
| `src/storymesh/cli.py` | Add stage name, optional `--skip-review` |
| `src/storymesh/versioning/schemas.py` | Add version constant |
| `src/storymesh/versioning/agents.py` | Add version constant |
| `storymesh.config.yaml` | Add agent config block |
| `README.md` | Update stage table |

---

## Testing Strategy

1. **Schema tests**: validate `NearMissMoment` and output contracts
2. **Agent unit tests**: mock both LLM clients, verify two-pass flow,
   verify no-op when zero avoidance moments
3. **Node wrapper tests**: verify state assembly, draft replacement logic,
   artifact persistence
4. **Graph integration tests**: verify edge wiring, stage ordering
5. **Real API test** (`real_api` marker): end-to-end with actual LLM calls

---

## Resolved Design Decisions

1. **Contextually independent reviewer.** The reviewer does NOT receive rubric
   feedback, creative direction, or any upstream evaluation context. It gets the
   draft, the title, the thematic thesis, and the scene summary — nothing else.
   This prevents context pollution and ensures the reviewer reads the draft as a
   fresh human reader encountering the story for the first time. The near-miss
   moments the rubric cares about (proposal-level structural issues) are
   categorically different from what the reviewer should find (prose-level
   emotional follow-through).

2. **Quality-gated: high and very_high only.** The resonance reviewer runs only
   at `high` and `very_high` quality presets. At `draft` and `standard`, the
   node is skipped entirely (returns draft unchanged). A `skip_resonance_review`
   bool is threaded through the pipeline and set by the quality preset.

3. **Back-cover summary re-run.** When the reviewer expands moments, the
   summary pass is re-run inside the resonance reviewer node to ensure the
   back-cover copy reflects the final text. Future work: migrate the summary
   pass entirely to the book assembler so it always reflects the final output
   regardless of pipeline changes.

4. **Dual LLM client wiring.** Call `_build_llm_client` twice with separate
   config keys (`review_provider`/`review_model` and `revision_provider`/
   `revision_model`). Simpler than extending the helper.

5. **Review model selection.** The review pass should use a high-capability
   reasoning model (GPT-4o or o3) since the task — reading fiction with
   emotional sensitivity, distinguishing restraint from avoidance, holding
   multiple unstated implications simultaneously — is fundamentally a reasoning
   task, not a generation task. The revision pass uses the same provider as the
   story writer (Claude Sonnet) to maintain voice consistency.

## Remaining Open Questions

1. **o3 vs GPT-4o for review pass.** o3 has stronger reasoning but higher
   cost/latency. Worth testing both. The review pass is a single call on a
   ~3000-word input, so latency is bounded. Cost may be justified given this
   is only triggered at high/very_high quality.

2. **Summary pass migration.** Moving the back-cover summary from story_writer
   to book_assembler is a cleaner architecture (summary always reflects final
   text). Should this be done as part of the resonance reviewer implementation
   or as a separate follow-up task?
