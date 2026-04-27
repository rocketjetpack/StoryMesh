# Craft Quality System & RubricJudgeAgent Implementation Plan

## Overview

This plan covers two tightly coupled changes that together form the **creative quality system** of the StoryMesh pipeline:

1. **Rewrite the ProposalDraftAgent prompts** to include explicit Craft Directives (CDs) — specific, verifiable instructions that push LLM output away from generic AI writing patterns and toward specific, surprising, human-interesting fiction.
2. **Implement RubricJudgeAgent** (Stage 5) — a new agent on a **different LLM provider** than ProposalDraftAgent that evaluates proposals against a rubric whose dimensions directly reference the Craft Directives.

These two pieces share a vocabulary (CD codes map to evaluation dimensions) and must be designed together, even though they are separate agents executed by separate LLM providers.

### Why a Different Provider?

ProposalDraftAgent generates the creative proposal. RubricJudgeAgent evaluates it. If both use the same LLM (e.g., both use Claude), the evaluator inherits the generator's blind spots — it tends to rate its own model's output favorably and is less likely to catch stylistic tics or structural laziness that are characteristic of that model. Using a different provider (e.g., ProposalDraft on Anthropic, RubricJudge on OpenAI) provides a genuinely independent editorial perspective.

The existing provider registry, per-agent config, and `_build_llm_client()` routing already support this — it's a config-level change, not a code change.

### How the Retry Loop Works (Mechanical Summary)

The LangGraph topology for the retry loop is already wired:

```
proposal_draft → rubric_judge → [conditional edge]
                                  ├── PASS → synopsis_writer → END
                                  └── FAIL + retries remaining → proposal_draft
```

**State is the communication channel.** The `rubric_judge_output` field in `StoryMeshState` carries the scores and feedback. When ProposalDraftAgent runs on retry, the node wrapper detects existing rubric feedback in state and switches to a **retry prompt template** that includes the evaluator's specific critique. The agent generates N fresh candidates (new LLM calls, new seeds or different angles), self-selects, and writes the new proposal to state. RubricJudge evaluates again.

The `rubric_retry_count` field tracks attempts. After `MAX_RUBRIC_RETRIES` (default 2, meaning initial + 2 retries = 3 total attempts), the routing function forces progression to SynopsisWriter with the best-scoring attempt across all rounds.

**Key detail for the retry prompt:** on retry, ProposalDraftAgent receives the rubric feedback verbatim in its user prompt. The feedback is dimension-specific and actionable (e.g., "The protagonist's flaw is generic and interchangeable — consider what 'trust' means uniquely in this world"). This gives the generator *targeted creative direction* rather than "try again."

---

## Dependency Chain

```
WI-1: Craft Directives — rewrite proposal_draft_generate.yaml
WI-2: Retry Prompt — create proposal_draft_retry.yaml  
WI-3: ProposalDraft Node Wrapper — detect retry state, switch prompt
      ↓ (can test WI-1–3 independently with FakeLLMClient)
WI-4: RubricJudge Schema — Pydantic models for rubric output
WI-5: RubricJudge Prompt — rubric_judge.yaml with evaluation dimensions
WI-6: RubricJudge Agent — agent core
WI-7: RubricJudge Node Wrapper + Graph Wiring — activate the real loop
WI-8: State + History Tracking — track all attempts and best score
WI-9: Config, README, Version Updates
```

Each WI leaves the codebase in a testable state. WI-1–3 can be implemented and tested before WI-4–7 exist — the retry prompt template is ready, just never triggered until the rubric agent produces real failures.

---

## 1. WI-1: Craft Directives — Rewrite `proposal_draft_generate.yaml`

### Rationale

The current system prompt tells the LLM to be a "fiction development editor" and to avoid clichéd resolutions. This is necessary but insufficient — it produces competent-but-predictable output because it doesn't actively suppress the specific patterns that make LLM writing feel like LLM writing. The Craft Directives are explicit, verifiable instructions that push the generator toward surprising, specific, human-interesting fiction.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/proposal_draft_generate.yaml` | REWRITE system prompt |

### Design Philosophy

The Craft Directives serve two audiences simultaneously:
1. **The generating LLM** — they are instructions it must follow during generation
2. **The evaluating LLM** (RubricJudge) — they are criteria it can check against

Each directive has a **code** (CD-1, CD-2, ...) and a **priority level** (`CRITICAL`, `HIGH`, `MEDIUM`). The priority levels serve as tuning knobs during eval rounds — promote a directive to CRITICAL if the model keeps ignoring it, demote to LOW or remove if it's causing stiffness.

### Craft Directives (to be added to system prompt)

The following section replaces or augments the creative instructions in the current system prompt. The existing instructions about developing the assigned seed, avoiding clichéd resolutions, and returning valid JSON are preserved.

```
CRAFT DIRECTIVES
================
Follow ALL directives below. An independent evaluator on a different AI model will 
check your output against each one. Violations will cause rejection and revision.

CD-1 [CRITICAL] — NO TENSION RESOLUTION
The central thematic tension must remain productively unresolved at the end of the 
story. Both sides of the tension must remain true simultaneously. Do NOT pick a winner.
Do NOT find a compromise that dissolves the tension. The story's power comes from the 
reader holding two contradictory truths at once. If your plot arc ends with one side 
of the tension triumphing, you have failed this directive.

EXAMPLE OF VIOLATION: A romance/horror story where "love defeats the monster" — this 
resolves the tension by picking romance's worldview over horror's.
EXAMPLE OF COMPLIANCE: A romance/horror story where the couple's love is genuine AND 
the cosmos remains indifferent — the love doesn't save them, but it existed, and the 
story asks whether that's enough.

CD-2 [CRITICAL] — SPECIFICITY OVER ABSTRACTION
Every character trait, setting detail, and plot mechanism must be concrete and 
particular to THIS story. Replace every vague phrase with a specific one.
- NOT "a grizzled detective" → "a census-taker who memorized the population of every 
  district before the collapse"
- NOT "a dark secret" → "a room in the basement where she keeps detailed records of 
  every lie she's told since age twelve, organized by recipient"
- NOT "a world ravaged by war" → "a city where the ceasefire is maintained by a 
  shared delusion that the war never happened"

If a detail could appear unchanged in a different story with different themes, it is 
too generic. Every detail should be load-bearing — it should connect to the thematic 
tensions or the protagonist's internal conflict.

CD-3 [HIGH] — PACING ASYMMETRY
Your three-act arc must NOT distribute weight equally across acts. Real stories have 
a center of gravity — a section where most of the story's emotional and intellectual 
weight lives. Your arc description should make clear where that weight falls.
- If the story's power lives in a slow unraveling of a single relationship, Act 2 
  could be 80% of the story.
- If the story is about a sudden violent disruption of normalcy, Act 1 could be the 
  vast majority.
- If the story builds to a moment of terrible clarity, Acts 1 and 2 could be 
  misdirection and Act 3 could be where everything actually happens.
State explicitly which act carries the most weight and why.

CD-4 [HIGH] — ONE GENUINELY UNUSUAL ELEMENT
Include at least one creative choice — a character detail, a setting mechanism, a plot 
device, a structural decision — that is genuinely surprising. The test: would this 
element appear in the first ten versions of this story that a writing workshop would 
produce? If yes, it is not unusual enough. The element should not be random or 
arbitrary — it should feel like it belongs once you see it, but you would not have 
predicted it.

CD-5 [HIGH] — PROTAGONIST WANT/NEED SPECIFICITY
The protagonist's internal conflict must be SPECIFIC TO THIS CHARACTER IN THIS WORLD.
- NOT "must learn to trust again" (generic, interchangeable with any story)
- NOT "struggles with their dark past" (every protagonist has this)
- The want/need split should mirror the thematic tension: if the tension is between 
  "truth is knowable" and "all records are destroyed," the protagonist might WANT to 
  reconstruct a specific historical truth but NEED to accept that some truths can only 
  exist as collective memory, never as verified fact.

CD-6 [MEDIUM] — ANTI-PATTERN AVOIDANCE
Do NOT use any of the following LLM writing patterns:
- Characters whose eyes "reflect," "mirror," or "hold" anything
- The word "juxtaposition" or "dichotomy"
- The phrase "little did they know" or "unbeknownst to"
- Weather used to signal mood (storms during conflict, sunshine during resolution)
- The protagonist "realizing" or having an "epiphany" in the climax — have the 
  understanding change nothing about their circumstances, only about how they 
  carry the weight
- Ending any scene with a character reflecting on what just happened
- A villain/antagonist whose motivation is explained by a tragic backstory that 
  makes them sympathetic — let opposing forces be partially opaque
- The construction "In a world where..." anywhere in the proposal
- Describing anything as "a dance" or "a tapestry" metaphorically
- Characters who are "haunted by" anything

CD-7 [MEDIUM] — STRUCTURAL SURPRISE IN THE ARC
The three-act arc should contain at least one beat where the reader's understanding 
of the premise shifts. Not a "twist" in the cheap sense (hidden identity reveals, 
it-was-all-a-dream), but a moment where Act 3 reveals that the story was actually 
about something other than what Acts 1 and 2 suggested. The thematic thesis should 
only become fully visible in retrospect.

CD-8 [LOW] — SCENE SPECIFICITY
Each key scene description should include at least one concrete sensory detail 
(sound, texture, smell, temperature, taste) that is specific to the setting and 
could not appear in a different story. Avoid visual-only descriptions.
```

### What Changes in the YAML

The system prompt section of `proposal_draft_generate.yaml` gets the Craft Directives block appended after the existing instructions about developing seeds and avoiding clichéd resolutions. The existing JSON schema instructions and user prompt template are **unchanged** — the user template already has all the right placeholders (`{assigned_seed}`, `{tensions}`, etc.).

### What Does NOT Change

- The user prompt template — no new placeholders needed
- The `ProposalDraftAgent` Python code — it reads the prompt file as-is
- The `StoryProposal` Pydantic schema — the same JSON fields are populated, just with higher-quality content
- The selection prompt (`proposal_draft_select.yaml`) — this is ProposalDraft's internal critic for choosing among N candidates, separate from the rubric judge

### Testing

- All existing `TestProposalDraftGeneratePrompt` tests must still pass (the template placeholders are unchanged)
- Manual inspection: load the prompt and verify the Craft Directives section is present and well-formed
- No new automated tests needed for WI-1 alone — the validation happens when the rubric agent (WI-6) evaluates against these directives

---

## 2. WI-2: Retry Prompt — `proposal_draft_retry.yaml`

### Rationale

When RubricJudgeAgent fails a proposal and routes back to ProposalDraftAgent, the generator needs to know *what went wrong*. A retry with the same prompt produces similar output. A retry with targeted editorial feedback produces meaningfully different output. This requires a separate prompt template that includes the rubric feedback.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/proposal_draft_retry.yaml` | CREATE |
| `tests/test_prompt_loader.py` | ADD integration tests |

### Prompt Design

The **system prompt** is identical to `proposal_draft_generate.yaml` — same role, same Craft Directives. The Craft Directives are the standards; they don't change on retry.

The **user prompt template** is different. It includes everything from the original user template PLUS:

- `{previous_proposal}` — JSON serialization of the failed proposal
- `{rubric_feedback}` — the dimension-by-dimension feedback from RubricJudge
- `{rubric_scores}` — the numeric scores per dimension
- `{attempt_number}` — which attempt this is (2 or 3)

The user prompt should frame the retry explicitly:

```
REVISION CONTEXT
================
This is attempt {attempt_number}. A previous proposal was evaluated by an independent 
editorial judge and did not pass. The specific feedback is below.

Your job is to generate a NEW proposal that addresses the identified weaknesses while 
preserving any strengths noted. Do NOT simply patch the previous proposal — generate 
fresh creative work that takes the feedback as creative direction.

PREVIOUS PROPOSAL (for reference — do NOT copy structure or details):
{previous_proposal}

EDITORIAL FEEDBACK:
{rubric_feedback}

SCORES BY DIMENSION:
{rubric_scores}

--- 
(remainder of the standard input: user prompt, genres, tones, seeds, tensions, etc.)
```

### Testing

```
TestProposalDraftRetryPrompt:
  - test_load_prompt_succeeds: load_prompt("proposal_draft_retry") returns PromptTemplate
  - test_system_prompt_matches_generate: retry system prompt is identical to generate system prompt
  - test_user_template_has_standard_placeholders: all original placeholders present
  - test_user_template_has_retry_placeholders: {previous_proposal}, {rubric_feedback}, 
    {rubric_scores}, {attempt_number} are present
  - test_format_user_with_valid_retry_data: format_user() succeeds with all fields
```

### Design Decision: Why a Separate File Instead of Conditional Logic

The alternative is one prompt file with an `{optional_retry_section}` placeholder that's empty on first attempt and filled on retry. This is simpler but has a problem: the retry framing ("A previous proposal was evaluated...") is substantial enough that including it as an empty-or-filled section makes the template harder to read and maintain. Two files is cleaner, follows the precedent of `proposal_draft_generate.yaml` vs. `proposal_draft_select.yaml`, and makes it obvious which prompt is used when.

---

## 3. WI-3: ProposalDraft Node Wrapper — Retry Detection

### Rationale

The node wrapper must detect whether this is a first attempt or a retry, and load the appropriate prompt template. The agent itself doesn't know about retries — the node wrapper is the orchestration layer that bridges state to agent input.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE |
| `tests/test_graph.py` | ADD retry-specific node wrapper tests |

### Changes to `proposal_draft_node` Inner Function

The current node wrapper assembles `ProposalDraftAgentInput` and calls `agent.run()`. The change:

1. **Check state for rubric feedback.** If `state.get("rubric_judge_output")` is not None and has `passed == False`, this is a retry.

2. **On first attempt:** behave exactly as today — the agent uses `proposal_draft_generate.yaml` (its default prompt, loaded in the constructor).

3. **On retry:** the node wrapper must pass the rubric feedback into the agent so it can use the retry prompt. This requires a small change to `ProposalDraftAgent.run()` — add an optional `rubric_feedback` parameter:

```python
def run(
    self, 
    input_data: ProposalDraftAgentInput,
    *,
    rubric_feedback: RubricFeedback | None = None,
) -> ProposalDraftAgentOutput:
```

When `rubric_feedback` is provided, the agent uses `self._retry_prompt` instead of `self._generate_prompt` and formats the user template with the additional retry-specific fields.

4. **Increment retry count.** On retry, the node wrapper returns `rubric_retry_count` incremented by 1 alongside `proposal_draft_output`.

### Agent Constructor Change

Add a third prompt load to the constructor:

```python
self._retry_prompt = load_prompt("proposal_draft_retry")
```

This is eagerly loaded so misconfiguration is caught at construction time.

### RubricFeedback Type

This is a simple NamedTuple or dataclass (not a Pydantic schema — it's internal plumbing, not a persisted artifact):

```python
@dataclasses.dataclass(frozen=True)
class RubricFeedback:
    """Internal carrier for rubric feedback passed to retry attempts."""
    previous_proposal_json: str
    feedback_text: str
    scores_text: str
    attempt_number: int
```

Define this in `src/storymesh/agents/proposal_draft/agent.py` — it's internal to this agent, not shared.

### Node Wrapper Logic (Pseudocode)

```python
rubric_output = state.get("rubric_judge_output")
is_retry = (
    rubric_output is not None 
    and hasattr(rubric_output, "passed") 
    and not rubric_output.passed
)

rubric_feedback = None
if is_retry:
    rubric_feedback = RubricFeedback(
        previous_proposal_json=json.dumps(
            state["proposal_draft_output"].proposal.model_dump(), indent=2
        ),
        feedback_text=_format_feedback(rubric_output),
        scores_text=_format_scores(rubric_output),
        attempt_number=state.get("rubric_retry_count", 0) + 1,
    )

output = agent.run(input_data, rubric_feedback=rubric_feedback)

result = {"proposal_draft_output": output}
if is_retry:
    result["rubric_retry_count"] = state.get("rubric_retry_count", 0) + 1
return result
```

### Helper Functions

```python
def _format_feedback(rubric_output: RubricJudgeAgentOutput) -> str:
    """Format dimension-by-dimension feedback as human-readable text for the retry prompt."""
    lines = []
    for dim, text in rubric_output.feedback.items():
        score = rubric_output.scores.get(dim, "N/A")
        lines.append(f"[{dim}] (score: {score}): {text}")
    return "\n".join(lines)

def _format_scores(rubric_output: RubricJudgeAgentOutput) -> str:
    """Format scores as a compact summary."""
    lines = [f"  {dim}: {score}" for dim, score in rubric_output.scores.items()]
    lines.append(f"  COMPOSITE: {rubric_output.composite_score}")
    lines.append(f"  THRESHOLD: {rubric_output.pass_threshold}")
    return "\n".join(lines)
```

### Testing

```
TestProposalDraftNodeRetry:
  - test_first_attempt_uses_generate_prompt: when rubric_judge_output is None, 
    agent.run() is called without rubric_feedback
  - test_retry_uses_retry_prompt: when rubric_judge_output has passed=False,
    agent.run() is called with rubric_feedback populated
  - test_retry_increments_count: returned dict includes incremented rubric_retry_count
  - test_noop_rubric_treated_as_first_attempt: when rubric_judge_output is None 
    (noop placeholder), no retry logic triggers
  - test_passed_rubric_treated_as_first_attempt: when rubric_judge_output has 
    passed=True, no retry logic triggers (this shouldn't happen in practice, 
    but defensive)
```

---

## 4. WI-4: RubricJudge Schema

### Rationale

The RubricJudgeAgent needs Pydantic schemas for its input and output. The output schema is the most important — it defines the scores, feedback, and pass/fail signal that drive the retry loop.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/schemas/rubric_judge.py` | CREATE |
| `src/storymesh/versioning/schemas.py` | ADD `RUBRIC_SCHEMA_VERSION` |
| `tests/test_schemas_rubric_judge.py` | CREATE |

### Schema Design

```python
RUBRIC_SCHEMA_VERSION = "1.0"

class RubricJudgeAgentInput(BaseModel):
    """Input to the RubricJudgeAgent."""
    model_config = {"frozen": True}
    
    proposal: StoryProposal
    tensions: list[ThematicTension]
    cliched_resolutions: dict[str, list[str]]  # tension_id → list of clichés
    user_tones: list[str]
    user_prompt: str
    normalized_genres: list[str]
    attempt_number: int = Field(ge=1, default=1)

class DimensionResult(BaseModel):
    """Score and feedback for a single rubric dimension."""
    model_config = {"frozen": True}
    
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(min_length=10)
    directive_ref: str = Field(
        min_length=1,
        description="The CD code this dimension evaluates (e.g., 'CD-1', 'CD-2')."
    )

class RubricJudgeAgentOutput(BaseModel):
    """Complete rubric evaluation of a story proposal."""
    model_config = {"frozen": True}
    
    passed: bool
    composite_score: float = Field(ge=0.0, le=1.0)
    pass_threshold: float = Field(ge=0.0, le=1.0)
    dimensions: dict[str, DimensionResult] = Field(
        min_length=1,
        description=(
            "Mapping of dimension name to score and feedback. "
            "Dimension names: tension_inhabitation, specificity_density, "
            "craft_discipline, protagonist_interiority, structural_surprise, "
            "user_intent_fidelity."
        ),
    )
    cliche_violations: list[str] = Field(
        default_factory=list,
        description="Specific clichéd resolutions detected in the proposal."
    )
    overall_feedback: str = Field(
        min_length=10,
        description="Holistic editorial assessment summarizing key strengths and weaknesses."
    )
    debug: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = RUBRIC_SCHEMA_VERSION
```

### Computed Properties

The `passed` field is computed by the agent, not the LLM. The LLM returns scores and feedback; the agent computes `composite_score` from the weighted dimension scores and compares against `pass_threshold`. This keeps the pass/fail decision deterministic and auditable — we don't ask the LLM "did it pass?"

### Design Decision: `dimensions` as Dict vs. Fixed Fields

Using a dict (`dimensions: dict[str, DimensionResult]`) rather than fixed fields (`tension_inhabitation: DimensionResult`, `specificity_density: DimensionResult`, etc.) has a tradeoff:

**Advantage:** Adding, removing, or renaming dimensions during eval rounds doesn't require schema changes or version bumps. You edit the prompt, the LLM returns the new dimension names, and the dict carries them.

**Disadvantage:** No compile-time guarantee that all expected dimensions are present. The agent must validate this at runtime.

**Decision: Dict.** The eval-round flexibility is more valuable for this project than compile-time dimension validation. The agent checks that all expected dimensions are present after parsing and raises a clear error if any are missing.

### Testing

```
TestDimensionResult:
  - test_valid_construction
  - test_frozen
  - test_score_bounds: rejects < 0.0 and > 1.0
  - test_feedback_min_length: rejects short strings

TestRubricJudgeAgentOutput:
  - test_valid_construction
  - test_frozen
  - test_passed_reflects_threshold: passed=True when composite >= threshold
  - test_dimensions_min_length: rejects empty dict
  - test_cliche_violations_defaults_empty
  - test_schema_version_matches
  - test_overall_feedback_min_length

TestRubricJudgeAgentInput:
  - test_valid_construction
  - test_frozen
  - test_attempt_number_ge_one
```

---

## 5. WI-5: RubricJudge Prompt — `rubric_judge.yaml`

### Rationale

The prompt is the heart of the quality system. It defines the evaluation dimensions, their weights, and the criteria for each. The dimensions directly reference the CD codes from the ProposalDraft prompt, creating a shared vocabulary between generator and evaluator.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/rubric_judge.yaml` | CREATE |
| `tests/test_prompt_loader.py` | ADD integration tests |

### System Prompt Content

```
You are a senior fiction editor evaluating a story proposal for an AI-assisted creative
writing pipeline. Your evaluation must be rigorous, specific, and honest. You are NOT
the same AI that generated this proposal — you are an independent evaluator providing
a second opinion.

Your goal is to identify whether this proposal demonstrates genuine creative thinking
or whether it falls into the patterns of generic, predictable AI-generated fiction. 
Good fiction surprises. Good fiction makes specific choices. Good fiction sits inside 
tension rather than resolving it.

EVALUATION DIMENSIONS
=====================
Score each dimension from 0.0 to 1.0 and provide specific, actionable feedback.
Reference the Craft Directive (CD) codes — the generator was given these as 
instructions.

D-1: TENSION INHABITATION [weight: 0.25]
Evaluates: CD-1 (No Tension Resolution)
Does the proposal LIVE INSIDE the thematic tension, or does it RESOLVE it?
- Score 0.0–0.3: The tension is cleanly resolved. One side wins. The story 
  picks a worldview.
- Score 0.4–0.6: The tension is partially preserved but the ending leans 
  toward resolution. The reader is let off the hook.
- Score 0.7–0.8: Both sides of the tension remain true. The ending is 
  ambiguous in a productive way.
- Score 0.9–1.0: The story's power comes from the reader holding two 
  contradictory truths simultaneously. The tension generates meaning 
  precisely because it is not resolved.
Check: Does the plot_arc ending allow both sides of each thematic tension 
to remain true?

D-2: SPECIFICITY DENSITY [weight: 0.20]
Evaluates: CD-2 (Specificity Over Abstraction)
Does the proposal make concrete, particular creative choices?
- Count SPECIFIC details: named mechanisms, particular professions, 
  unusual combinations, load-bearing details that connect to theme.
- Count GENERIC phrases: "a dark secret," "a dangerous journey," 
  "a world on the brink," "struggles with their past."
- The ratio of specific to generic is the score signal.
- Score 0.0–0.3: Mostly generic. Details could be swapped into any story 
  in this genre.
- Score 0.7–1.0: Nearly every detail is specific to THIS story and 
  connects to the thematic tensions.

D-3: CRAFT DISCIPLINE [weight: 0.20]
Evaluates: CD-3 (Pacing Asymmetry), CD-6 (Anti-Pattern Avoidance), 
CD-8 (Scene Specificity)
- Does the three-act arc have asymmetric weight distribution? Is the 
  center of gravity stated explicitly?
- Does the proposal avoid the anti-patterns listed in CD-6? Check each one.
- Do key scenes include non-visual sensory details?
Score is the proportion of craft directives followed. Each CD-6 violation 
is a deduction.

D-4: PROTAGONIST INTERIORITY [weight: 0.15]
Evaluates: CD-5 (Protagonist Want/Need Specificity)
- Is the protagonist's want/need split specific to THIS character in 
  THIS world?
- Does the internal conflict mirror the thematic tension?
- Could this internal conflict belong to a protagonist in a completely 
  different story? If yes, score low.
- Score 0.0–0.3: Generic internal conflict ("learns to trust," "overcomes 
  fear," "finds redemption")
- Score 0.7–1.0: Internal conflict is unique to this character and 
  inseparable from the thematic landscape

D-5: STRUCTURAL SURPRISE [weight: 0.10]
Evaluates: CD-4 (One Genuinely Unusual Element), CD-7 (Structural 
Surprise in Arc)
- Does the proposal contain at least one creative choice that would not 
  appear in the first ten versions of this story?
- Does Act 3 recontextualize Acts 1 and 2? Does the thematic thesis 
  become fully visible only in retrospect?
- Score 0.0–0.3: Every element is predictable from the premise.
- Score 0.7–1.0: At least one element genuinely surprises AND the arc 
  has a recontextualization beat.

D-6: USER INTENT FIDELITY [weight: 0.10]
Evaluates: Alignment with the user's original prompt.
- Does the proposal honor the genres, tones, and narrative context 
  the user specified?
- This is a floor, not a ceiling — penalize only when the proposal 
  actively contradicts the user's request, not when it interprets 
  it creatively.
- Score 0.0–0.3: The proposal ignores the user's stated preferences.
- Score 0.7–1.0: The proposal fulfills the user's intent while 
  making surprising creative choices within that intent.

CLICHÉ CHECK
============
Separately from the dimension scores, check the proposal against EVERY 
clichéd resolution provided in the tensions list. For each cliché that 
the proposal falls into, add it to the cliche_violations list with a 
brief explanation of how the proposal matches it. Cliché violations are 
a hard penalty — each one reduces the composite score by 0.05 (applied 
after weighted average).

SCORING
=======
Compute the composite score as:
  composite = (D1 * 0.25) + (D2 * 0.20) + (D3 * 0.20) + (D4 * 0.15) + 
              (D5 * 0.10) + (D6 * 0.10) - (num_cliche_violations * 0.05)
  composite = max(0.0, composite)  # floor at 0

Do NOT compute pass/fail. Return only the scores, feedback, and 
cliché violations. The pipeline will determine pass/fail from the 
composite score and a configurable threshold.

RESPONSE FORMAT
===============
Return ONLY a JSON object. No markdown fences, no commentary.

{
  "dimensions": {
    "tension_inhabitation": {
      "score": <0.0-1.0>,
      "feedback": "<specific, actionable feedback referencing CD-1>",
      "directive_ref": "CD-1"
    },
    "specificity_density": {
      "score": <0.0-1.0>,
      "feedback": "<specific feedback with examples from the proposal>",
      "directive_ref": "CD-2"
    },
    "craft_discipline": {
      "score": <0.0-1.0>,
      "feedback": "<list any CD-6 violations found; note pacing assessment>",
      "directive_ref": "CD-3,CD-6,CD-8"
    },
    "protagonist_interiority": {
      "score": <0.0-1.0>,
      "feedback": "<assess want/need specificity>",
      "directive_ref": "CD-5"
    },
    "structural_surprise": {
      "score": <0.0-1.0>,
      "feedback": "<identify the unusual element if present; assess arc recontextualization>",
      "directive_ref": "CD-4,CD-7"
    },
    "user_intent_fidelity": {
      "score": <0.0-1.0>,
      "feedback": "<assess alignment with user prompt>",
      "directive_ref": "user_prompt"
    }
  },
  "cliche_violations": [
    "<specific clichéd resolution text that was matched, with explanation>"
  ],
  "overall_feedback": "<2-3 sentence editorial summary: strongest element, 
    weakest element, one specific suggestion for improvement>"
}
```

### User Prompt Template

```
USER PROMPT: "{user_prompt}"
GENRES: {normalized_genres}
USER TONES: {user_tones}

THEMATIC TENSIONS (with clichéd resolutions to check against):
{tensions}

PROPOSAL TO EVALUATE:
{proposal}
```

Placeholders:
- `{user_prompt}` — original user input
- `{normalized_genres}` — list of genre names
- `{user_tones}` — list of tone words
- `{tensions}` — JSON array of ThematicTension objects including cliched_resolutions
- `{proposal}` — JSON serialization of the StoryProposal being evaluated

### Testing

```
TestRubricJudgePrompt:
  - test_load_prompt_succeeds: load_prompt("rubric_judge") returns PromptTemplate
  - test_system_prompt_non_empty
  - test_system_prompt_contains_dimension_names: all 6 dimension names appear
  - test_system_prompt_contains_cd_references: CD-1 through CD-8 are referenced
  - test_user_template_has_required_placeholders: {user_prompt}, {normalized_genres}, 
    {user_tones}, {tensions}, {proposal}
  - test_format_user_with_valid_data: format_user() succeeds
```

---

## 6. WI-6: RubricJudge Agent Core

### Rationale

The agent sends the proposal and evaluation criteria to the LLM, parses the response, computes the composite score and pass/fail, and returns the structured output.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/agents/rubric_judge/__init__.py` | CREATE |
| `src/storymesh/agents/rubric_judge/agent.py` | CREATE |
| `tests/test_rubric_judge_agent.py` | CREATE |

### Constructor

```python
class RubricJudgeAgent:
    """Evaluates story proposals against a craft-quality rubric (Stage 5).

    Uses a DIFFERENT LLM provider than ProposalDraftAgent to provide
    an independent editorial evaluation. Computes pass/fail from the
    weighted composite score against a configurable threshold.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        pass_threshold: float = 0.7,
        dimension_weights: dict[str, float] | None = None,
        cliche_penalty: float = 0.05,
    ) -> None:
```

Parameters:
- `llm_client` — **required**. Should be a different provider than ProposalDraft.
- `temperature` — default `0.0`. Evaluation should be deterministic and reproducible.
- `max_tokens` — default `4096`. Rubric feedback can be detailed.
- `pass_threshold` — default `0.7`. Configurable in `storymesh.config.yaml`. This is the primary tuning knob for eval rounds.
- `dimension_weights` — optional override of the default weights. Defaults to the weights in the prompt (0.25, 0.20, 0.20, 0.15, 0.10, 0.10).
- `cliche_penalty` — per-violation score deduction. Default `0.05`.

The constructor eagerly loads the prompt:
```python
self._prompt = load_prompt("rubric_judge")
```

### Default Dimension Weights

```python
DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "tension_inhabitation": 0.25,
    "specificity_density": 0.20,
    "craft_discipline": 0.20,
    "protagonist_interiority": 0.15,
    "structural_surprise": 0.10,
    "user_intent_fidelity": 0.10,
}
```

### `run()` Method

```python
def run(self, input_data: RubricJudgeAgentInput) -> RubricJudgeAgentOutput:
```

Algorithm:

1. Serialize `input_data.proposal` to JSON for the user prompt.
2. Serialize `input_data.tensions` to JSON (including `cliched_resolutions`).
3. Format the user prompt with all placeholders.
4. Call `self._llm_client.complete_json()` with the system prompt and formatted user prompt at `self._temperature`.
5. Parse the response into individual `DimensionResult` objects.
6. **Validate dimension coverage:** check that all expected dimension names are present in the response. If any are missing, log a warning and assign a score of 0.0 with feedback "Dimension not evaluated by the model."
7. **Compute composite score:** `sum(weight * score for each dimension) - (num_cliches * cliche_penalty)`, floored at 0.0.
8. **Determine pass/fail:** `composite_score >= self._pass_threshold`.
9. Assemble and return `RubricJudgeAgentOutput`.

### Error Handling

- If `complete_json()` raises or returns unparseable JSON, the agent should return a **default fail** with `composite_score=0.0`, `passed=False`, and `overall_feedback="Rubric evaluation failed: [error message]. Treating as fail to trigger retry."` This ensures the retry loop gets a chance to improve the proposal rather than the pipeline crashing.
- If the LLM returns scores outside [0.0, 1.0], clamp them.

### Testing

```
TestRubricJudgeAgent:
  - test_passing_proposal: FakeLLMClient returns high scores → passed=True
  - test_failing_proposal: FakeLLMClient returns low scores → passed=False
  - test_composite_score_computation: verify weighted average math
  - test_cliche_penalty_applied: composite reduced by 0.05 per violation
  - test_cliche_penalty_floor_at_zero: many violations don't go negative
  - test_custom_threshold: pass_threshold=0.5 passes lower-scoring proposals
  - test_custom_weights: overridden weights change composite calculation
  - test_missing_dimension_handled: LLM omits a dimension → score=0.0, warning logged
  - test_llm_failure_returns_default_fail: exception → passed=False, score=0.0
  - test_temperature_zero: verify LLM is called with temperature=0.0
  - test_debug_contains_metadata: debug dict has weights_used, threshold, raw_scores
```

---

## 7. WI-7: RubricJudge Node Wrapper + Graph Wiring

### Rationale

Replace the `_noop_node` placeholder for `rubric_judge` with the real node. Activate the real pass/fail signal in `_rubric_route`.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/nodes/rubric_judge.py` | CREATE |
| `src/storymesh/orchestration/state.py` | UPDATE type annotation |
| `src/storymesh/orchestration/graph.py` | UPDATE — wire real node, update routing |
| `tests/test_graph.py` | ADD rubric node wrapper tests, update routing tests |

### Node Wrapper

```python
def make_rubric_judge_node(
    agent: RubricJudgeAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
```

The node function inside:

1. Read `proposal_draft_output` from state. Raise `RuntimeError` if None.
2. Read `theme_extractor_output` from state (for tensions and clichéd resolutions).
3. Read `genre_normalizer_output` from state (for genres, tones).
4. Assemble `RubricJudgeAgentInput`:
   - `proposal` = `proposal_draft_output.proposal`
   - `tensions` = `theme_extractor_output.tensions`
   - `cliched_resolutions` = extract from tensions (dict comprehension)
   - `user_tones` = `theme_extractor_output.user_tones_carried`
   - `user_prompt` = `state["user_prompt"]`
   - `normalized_genres` = `genre_normalizer_output.normalized_genres`
   - `attempt_number` = `state.get("rubric_retry_count", 0) + 1`
5. Call `agent.run(input_data)`.
6. Persist output artifact.
7. Return `{"rubric_judge_output": output}`.

**Note:** The node wrapper does NOT increment `rubric_retry_count`. That's done by the proposal_draft node wrapper on retry (WI-3). The rubric judge just evaluates and returns — the routing function reads the result and decides what to do.

### State Type Update

Replace:
```python
rubric_judge_output: object | None
```
With:
```python
from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput
# ...
rubric_judge_output: RubricJudgeAgentOutput | None
```

### Graph Wiring Update

In `build_graph()`:

1. Get the `rubric_judge` config via `get_agent_config("rubric_judge")`.
2. Build the `LLMClient` using the provider registry — **this should resolve to a different provider** than proposal_draft based on config.
3. Construct `RubricJudgeAgent` with config values.
4. Create the node via `make_rubric_judge_node(agent, artifact_store=artifact_store)`.
5. Replace `graph.add_node("rubric_judge", _noop_node)` with the real node.

### Routing Function Update

The existing `_rubric_route` already has the right structure. The only change: remove the comment "Placeholder logic: always pass" and verify the `getattr(output, 'passed', True)` pattern works with the real `RubricJudgeAgentOutput` (it will, since `passed` is a field on the model).

### Testing

```
TestRubricJudgeNodeWrapper:
  - test_returns_rubric_judge_output_type
  - test_only_returns_own_key
  - test_missing_proposal_draft_output_raises
  - test_missing_theme_extractor_output_raises
  - test_artifact_persisted_when_store_provided

TestRubricRouteWithRealAgent:
  - test_passed_routes_to_synopsis_writer
  - test_failed_routes_to_proposal_draft
  - test_failed_at_max_retries_routes_to_synopsis_writer
```

---

## 8. WI-8: State History Tracking — All Attempts

### Rationale

The SynopsisWriterAgent (Stage 6, next sprint) is designed to receive ALL proposal attempts and ALL rubric evaluations, not just the final one. It synthesizes across attempts, taking the best elements from each. This requires tracking attempt history in state.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/state.py` | ADD history fields |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE — append to history |
| `src/storymesh/orchestration/nodes/rubric_judge.py` | UPDATE — append to history |

### New State Fields

```python
# ── Attempt history (for SynopsisWriter synthesis) ─────────────────────
proposal_history: list[ProposalDraftAgentOutput]
"""All proposal attempts in order. Appended by proposal_draft node on each run."""

rubric_history: list[RubricJudgeAgentOutput]
"""All rubric evaluations in order. Appended by rubric_judge node on each run."""

best_proposal_index: int
"""Index into proposal_history of the highest-scoring attempt."""
```

### LangGraph State Accumulation

LangGraph's `TypedDict` state uses **last-write-wins** by default. For list fields that need to *accumulate* across iterations, the node wrapper must read the existing list from state and return the appended version:

```python
# In proposal_draft node:
history = list(state.get("proposal_history", []))
history.append(output)
return {
    "proposal_draft_output": output,
    "proposal_history": history,
}

# In rubric_judge node:
history = list(state.get("rubric_history", []))
history.append(output)

# Track best proposal
best_idx = state.get("best_proposal_index", 0)
proposals = state.get("proposal_history", [])
if proposals:
    current_best_score = -1.0
    for i, p_output in enumerate(proposals):
        # Use the rubric score if available, otherwise treat as 0
        if i < len(history):
            score = history[i].composite_score
            if score > current_best_score:
                current_best_score = score
                best_idx = i

return {
    "rubric_judge_output": output,
    "rubric_history": history,
    "best_proposal_index": best_idx,
}
```

### Interaction with `_rubric_route`

When `_rubric_route` forces progression (max retries exhausted), the SynopsisWriter node wrapper will use `best_proposal_index` to identify which proposal to use as the primary, while still having access to all attempts via `proposal_history` and `rubric_history`.

### Testing

```
TestAttemptHistory:
  - test_first_attempt_starts_history: proposal_history has length 1 after first run
  - test_retry_appends_to_history: proposal_history has length 2 after retry
  - test_best_proposal_index_updated: index points to highest-scoring attempt
  - test_rubric_history_accumulated: rubric_history grows with each evaluation
```

---

## 9. WI-9: Config, README, Version Updates

### Files Affected

| File | Action |
|------|--------|
| `storymesh.config.yaml` | ADD rubric_judge section |
| `storymesh.config.yaml.example` | Same |
| `README.md` | UPDATE status, architecture |
| `src/storymesh/versioning/schemas.py` | ADD RUBRIC_SCHEMA_VERSION |
| `src/storymesh/versioning/agents.py` | ADD RubricJudgeAgent version |

### Config Addition

```yaml
agents:
  # ... existing agents ...
  
  rubric_judge:
    provider: openai          # DIFFERENT provider than proposal_draft
    model: gpt-4o
    temperature: 0.0          # Deterministic evaluation
    max_tokens: 4096
    pass_threshold: 0.7       # Primary tuning knob for eval rounds
    cliche_penalty: 0.05      # Per-violation composite score deduction
    # dimension_weights:      # Uncomment to override defaults
    #   tension_inhabitation: 0.25
    #   specificity_density: 0.20
    #   craft_discipline: 0.20
    #   protagonist_interiority: 0.15
    #   structural_surprise: 0.10
    #   user_intent_fidelity: 0.10
```

### README Changes

1. Move `RubricJudgeAgent` from "Not implemented yet" to "Implemented"
2. Add description: "RubricJudgeAgent with 6-dimension craft quality rubric, cross-provider evaluation (OpenAI evaluating Anthropic output), cliché violation detection, and actionable feedback for retry loop"
3. Update "Current runtime behavior" to describe real rubric evaluation
4. Update the rubric retry loop section — it's now fully functional, not a placeholder
5. Add a "Craft Directives" section explaining the CD system
6. Roadmap: mark "Activate rubric-based retry logic" as done

---

## Design Decision Record

### Why Not Embed Weights in the Prompt?

The prompt tells the LLM what the weights are so it understands relative importance. But the actual composite score computation happens in Python (in `agent.py`), not in the LLM. This is deliberate — the LLM returns raw dimension scores and feedback, the agent computes the weighted average. This means:
- The weights in the prompt are for the LLM's interpretive guidance ("tension inhabitation matters more than user intent fidelity")
- The weights in the agent code are for the actual pass/fail computation
- These SHOULD match, but if they diverge slightly during tuning, the Python weights are authoritative

If you change weights during eval rounds, update BOTH the prompt and the config. The agent reads weights from config (or uses defaults that match the prompt).

### Why Temperature 0.0 for the Rubric Judge?

Evaluation should be reproducible. If you run the same proposal through the rubric judge twice, you should get the same scores (within floating-point noise). Temperature 0.0 gives you this. Temperature > 0 would introduce variance in how strictly the judge evaluates, making it harder to tune the system.

### Why a Cliché Penalty Instead of a Cliché Dimension?

Cliché violations are qualitatively different from the other dimensions. They're binary checks against a specific list — either the proposal matches a flagged cliché or it doesn't. Making them a separate per-violation penalty (rather than a 0-1 dimension score) means a single cliché in an otherwise excellent proposal doesn't tank the score, but multiple clichés accumulate into a meaningful penalty. The penalty is additive on top of (well, subtractive from) the dimension-weighted score.

### The Eval Round Workflow

After this plan is implemented, the eval workflow is:

1. Run `storymesh generate "your prompt here"`
2. Inspect `~/.storymesh/runs/<run_id>/proposal_draft_output.json` — read the proposal
3. Inspect `~/.storymesh/runs/<run_id>/rubric_judge_output.json` — read the scores and feedback
4. If the output is still too generic: tighten the rubric (lower `pass_threshold` in config, or promote CDs from MEDIUM to HIGH/CRITICAL in the prompt)
5. If the output is being rejected too harshly: relax (raise `pass_threshold`, or demote CDs)
6. If a specific anti-pattern keeps appearing: add it to CD-6 in the prompt
7. If the retry feedback isn't helping: edit the feedback formatting in the retry prompt
8. Re-run and compare

No code changes needed for steps 4–7. Just prompt and config edits.

### Conflict with README: "Market Hook" Dimension

The README lists "market hook" as a rubric dimension. This plan deliberately drops it. The reasoning: "market hook" incentivizes the LLM to optimize for what it thinks publishers want, which is a statistical average of successful books. That's the opposite of surprising output. If you want to add it back as a low-weight dimension, it can be added to the prompt and the dimension_weights dict without any code changes.

### Conflict with README: "Genre Fit" and "Internal Coherence" Dimensions

Also dropped. Genre fit is handled by user_intent_fidelity (lower-weighted, as a floor not a ceiling). Internal coherence is table stakes — if the proposal doesn't cohere, it fails parsing or the ProposalDraft selection step. Neither needs its own rubric dimension. If you disagree, they can be added as additional dimensions in the prompt with corresponding weights in config.

---

## File Summary

| File | Action | WI |
|------|--------|----|
| `src/storymesh/prompts/proposal_draft_generate.yaml` | REWRITE system prompt | WI-1 |
| `src/storymesh/prompts/proposal_draft_retry.yaml` | CREATE | WI-2 |
| `tests/test_prompt_loader.py` | UPDATE — add retry prompt tests | WI-2 |
| `src/storymesh/agents/proposal_draft/agent.py` | UPDATE — add retry prompt, RubricFeedback | WI-3 |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE — retry detection | WI-3 |
| `tests/test_graph.py` | UPDATE — add retry node wrapper tests | WI-3 |
| `src/storymesh/schemas/rubric_judge.py` | CREATE | WI-4 |
| `src/storymesh/versioning/schemas.py` | UPDATE — add RUBRIC_SCHEMA_VERSION | WI-4 |
| `tests/test_schemas_rubric_judge.py` | CREATE | WI-4 |
| `src/storymesh/prompts/rubric_judge.yaml` | CREATE | WI-5 |
| `tests/test_prompt_loader.py` | UPDATE — add rubric prompt tests | WI-5 |
| `src/storymesh/agents/rubric_judge/__init__.py` | CREATE | WI-6 |
| `src/storymesh/agents/rubric_judge/agent.py` | CREATE | WI-6 |
| `tests/test_rubric_judge_agent.py` | CREATE | WI-6 |
| `src/storymesh/orchestration/nodes/rubric_judge.py` | CREATE | WI-7 |
| `src/storymesh/orchestration/state.py` | UPDATE — type + history fields | WI-7, WI-8 |
| `src/storymesh/orchestration/graph.py` | UPDATE — wire real node | WI-7 |
| `tests/test_graph.py` | UPDATE — rubric node + routing tests | WI-7 |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE — history append | WI-8 |
| `src/storymesh/orchestration/nodes/rubric_judge.py` | UPDATE — history + best tracking | WI-8 |
| `storymesh.config.yaml` | UPDATE — add rubric_judge section | WI-9 |
| `storymesh.config.yaml.example` | UPDATE — same | WI-9 |
| `README.md` | UPDATE — status, craft directives, architecture | WI-9 |
| `src/storymesh/versioning/agents.py` | UPDATE — add agent version | WI-9 |

---

## Validation Checklist

After all work items are complete, verify the following:

- [ ] `pytest` passes with no new failures
- [ ] `ruff check src/ tests/` clean
- [ ] `mypy src/` clean
- [ ] Running `storymesh generate "dark post-apocalyptic detective mystery"` with both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set produces a real rubric evaluation in `~/.storymesh/runs/<run_id>/rubric_judge_output.json`
- [ ] The rubric output contains all 6 dimension scores with non-empty feedback
- [ ] If the proposal fails the rubric, the pipeline retries and the retry prompt includes the rubric feedback
- [ ] After max retries, the pipeline proceeds to synopsis_writer with the best-scoring proposal
- [ ] The `proposal_history` and `rubric_history` lists in the final state contain all attempts
- [ ] With only `ANTHROPIC_API_KEY` set (no OpenAI key), the rubric_judge node falls back to noop and the pipeline still completes
- [ ] Changing `pass_threshold` in config changes the pass/fail behavior without code changes
- [ ] Editing the Craft Directives in `proposal_draft_generate.yaml` changes generator behavior without code changes
- [ ] The anti-pattern list in CD-6 is reflected in the rubric judge's D-3 evaluation (inspect the feedback text)