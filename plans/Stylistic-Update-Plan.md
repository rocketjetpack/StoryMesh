# Craft Quality System & RubricJudgeAgent Implementation Plan (v2)

## Overview

This plan covers the **creative quality system** of the StoryMesh pipeline:

1. **Reframe clichéd resolutions** from a blacklist into a departure map — conventions the story follows *until* it doesn't, at a moment that matters.
2. **Rewrite the ProposalDraftAgent prompts** to replace prescriptive rules with a creative sensibility: tonal register guidance, restraint over spectacle, and permission to be conventional.
3. **Implement RubricJudgeAgent** (Stage 5) — a new agent on a **different LLM provider** that evaluates proposals against a rubric aligned with that sensibility.

### Philosophical Shift from v1

The v1 plan treated quality as a checklist: eight coded Craft Directives (CD-1 through CD-8), each with a priority level, mapped to six rubric dimensions. This produced technically competent proposals that still felt artificial — the model was following rules rather than inhabiting a sensibility. The output had an impression of depth without substance because every instruction was still telling the model to *perform*.

v2 makes three changes:

**Sensibility over rules.** Instead of eight directives, three guiding principles. Instead of "do this, don't do that," an aesthetic north star with tonal reference passages. The model is given a *voice to match* rather than a *checklist to satisfy*.

**Clichés as departure maps.** The upstream ThemeExtractor already identifies clichéd resolutions per tension. v1 said "avoid these." v2 says "your story may follow these patterns — they are the genre contract — but identify ONE moment where you depart from a familiar resolution, and make that departure structurally significant." Clichés are reliable because humans enjoy them. The magic is in the departure, not the avoidance.

**Restraint over spectacle.** The single most important instruction in the revised prompts is "resist the urge to be impressive." LLMs are trained to maximize apparent interestingness. Every instinct they have pushes toward escalation, juxtaposition, thematic spectacle. Real fiction has long stretches of the familiar punctuated by moments of genuine surprise. The ratio is maybe 90/10 familiar-to-surprising. The model defaults to 10/90.

### Why a Different Provider?

ProposalDraftAgent generates the creative proposal. RubricJudgeAgent evaluates it. If both use the same LLM, the evaluator inherits the generator's blind spots — it tends to rate its own model's output favorably and is less likely to catch stylistic tics or structural laziness characteristic of that model. Using a different provider (e.g., ProposalDraft on Anthropic, RubricJudge on OpenAI) provides a genuinely independent editorial perspective.

The existing provider registry, per-agent config, and `_build_llm_client()` routing already support this — it's a config-level change, not a code change.

### How the Retry Loop Works (Mechanical Summary)

The LangGraph topology for the retry loop is already wired:

```
proposal_draft → rubric_judge → [conditional edge]
                                  ├── PASS → synopsis_writer → END
                                  └── FAIL + retries remaining → proposal_draft
```

**State is the communication channel.** The `rubric_judge_output` field in `StoryMeshState` carries the scores and feedback. When ProposalDraftAgent runs on retry, the node wrapper detects existing rubric feedback in state and switches to a **retry prompt template** that includes the evaluator's specific critique. The agent generates N fresh candidates, self-selects, and writes the new proposal to state. RubricJudge evaluates again.

The `rubric_retry_count` field tracks attempts. After `MAX_RUBRIC_RETRIES` (default 2, meaning initial + 2 retries = 3 total attempts), the routing function forces progression to SynopsisWriter with the best-scoring attempt across all rounds.

---

## Dependency Chain

```
WI-0: ThemeExtractor Cliché Reframe — minor prompt language change
WI-1: Creative Sensibility — rewrite proposal_draft_generate.yaml
WI-2: Retry Prompt — create proposal_draft_retry.yaml
WI-3: ProposalDraft Node Wrapper — detect retry state, switch prompt
      ↓ (can test WI-0–3 independently with FakeLLMClient)
WI-4: RubricJudge Schema — Pydantic models for rubric output
WI-5: RubricJudge Prompt — rubric_judge.yaml with evaluation dimensions
WI-6: RubricJudge Agent — agent core
WI-7: RubricJudge Node Wrapper + Graph Wiring — activate the real loop
WI-8: State + History Tracking — track all attempts and best score
WI-9: Config, README, Version Updates
```

Each WI leaves the codebase in a testable state. WI-0–3 can be implemented and tested before WI-4–7 exist.

---

## 0. WI-0: ThemeExtractor Cliché Reframe

### Rationale

The ThemeExtractor currently frames `cliched_resolutions` as "the predictable, tropey approaches a lazy or unimaginative writer would default to." This framing is wrong for the v2 philosophy. Clichés are reliable narrative patterns that form the genre contract. They're cliché because they *work* — readers enjoy them. The creative opportunity is in following the pattern and then departing from it at a moment that matters.

The data structure doesn't change. The field name `cliched_resolutions` is still accurate (they *are* clichéd). What changes is the prompt language that instructs the LLM *how to think about* these patterns.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/theme_extractor.yaml` | UPDATE — reframe cliché language |

### Changes

In the system prompt, find and replace the cliché instruction block. The current language:

```
Identify 2–4 clichéd resolutions: the predictable, tropey approaches a lazy or 
unimaginative writer would default to when resolving this particular tension. These 
must be SPECIFIC and RECOGNIZABLE — not vague platitudes like "the hero saves the day", 
but concrete narrative moves like "a lone detective rebuilds justice single-handedly 
through sheer determination." The more specific the cliché, the more useful it is as 
an exclusion for downstream agents.
```

Replace with:

```
Identify 2–4 familiar resolutions: the reliable, conventional narrative moves that 
readers of this genre intersection expect. These are the genre contract — the patterns 
that are cliché precisely because they work and readers enjoy them. They must be 
SPECIFIC and RECOGNIZABLE — not vague platitudes like "the hero saves the day", but 
concrete narrative moves like "a lone detective rebuilds justice single-handedly 
through sheer determination." The more specific the pattern, the more useful it is 
as a departure point for downstream agents — the story may follow these patterns for 
most of its arc and then break from one at a structurally significant moment.
```

### What Does NOT Change

- The field name `cliched_resolutions` in the `ThematicTension` Pydantic schema — it's still an accurate name and renaming it would require a schema version bump and cascade through ProposalDraft and RubricJudge input schemas. Not worth it.
- The data structure — still a `list[str]` of specific narrative patterns.
- The examples in the prompt — "a lone detective rebuilds justice single-handedly" is still a good example of a specific, recognizable pattern.
- The ThemeExtractorAgent code — no changes.

### Testing

- All existing `TestThemeExtractorPrompt` tests must still pass.
- Manual inspection: verify the reframed language is present and the exclusion-oriented language is removed.

---

## 1. WI-1: Creative Sensibility — Rewrite `proposal_draft_generate.yaml`

### Rationale

The v1 plan had eight coded Craft Directives (CD-1 through CD-8) with priority levels. This produced proposals that followed rules but felt artificial — like furniture assembled from instructions. The model was constructing stories from the outside in (structure → fill) rather than the inside out (core feeling → structure that serves it).

v2 replaces the directive list with three guiding principles and tonal reference passages. The model is given a sensibility to inhabit rather than a checklist to satisfy.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/proposal_draft_generate.yaml` | REWRITE system prompt |

### Design Philosophy

The key insight: you can't instruct an LLM into authenticity. More rules produce more rule-following, which is itself a kind of artificiality. What you *can* do is:

1. Give the model a **tonal register** to match (concrete examples of the prose quality you want)
2. Give it **permission to be conventional** (follow genre patterns; the departure is what matters, not the avoidance)
3. Give it a single meta-instruction that cuts against the model's deepest training incentive: **resist the urge to be impressive**

### Creative Sensibility Section (replaces Craft Directives)

The following section replaces the Craft Directives block in the system prompt. The existing instructions about developing the assigned seed and returning valid JSON are preserved.

```
CREATIVE SENSIBILITY
====================
An independent editor on a different AI model will evaluate your proposal. 
They are looking for authentic, human-interesting fiction — not technically 
impressive AI output. Read the following principles carefully.

PRINCIPLE 1: RESTRAINT OVER SPECTACLE
Resist the urge to be impressive. The most common failure mode of AI-generated 
fiction is that every detail calls attention to itself, every character is 
extraordinary, every plot beat subverts expectations. Real fiction has long 
stretches of the familiar — comfortable, even predictable. The significant 
moments land because the story earned them through patience, not because the 
writer forced intensity into every paragraph.

Do not announce your themes. Do not escalate beyond what the scene warrants. 
Treat significant moments with the same plainness as insignificant ones. If 
you find yourself reaching for a metaphor, a juxtaposition, or a moment of 
dramatic irony — pause and ask whether the scene is better without it. Usually 
it is.

PRINCIPLE 2: CONVENTION THEN DEPARTURE
The clichéd resolutions listed for each thematic tension are the genre contract. 
Your reader expects them. Your story may — and in many places SHOULD — follow 
these familiar patterns. They are reliable and satisfying for a reason.

But your proposal must identify ONE moment where the story departs from a 
familiar resolution. This is the moment that makes the story worth telling. 
The departure should:
- Come at a structurally significant point (not a throwaway scene)
- Feel earned by everything conventional that preceded it
- Not announce itself — the reader should realize what happened a beat late
- Recontextualize the familiar patterns that came before it, so the reader 
  thinks "oh — THAT'S what this story was actually about"

Do not depart from ALL conventions. A story that subverts everything is as 
predictable as a story that subverts nothing. One departure, well-placed.

PRINCIPLE 3: SPECIFICITY WITHOUT PERFORMANCE
Details should be concrete and particular to this story. A census-taker who 
memorized district populations before the collapse is specific. A detective 
who is "grizzled" is generic. But — and this is crucial — specific details 
should not call attention to themselves. State them plainly. If the prose 
lingers on a detail, admiring its own specificity, it becomes performative.

Good: "She kept records of every lie she'd told since age twelve, organized 
by recipient." (Stated as fact. The reader does the work of finding it 
strange.)

Bad: "In a peculiar habit that spoke volumes about her fractured relationship 
with truth, she maintained meticulous records of every lie..." (The prose is 
explaining why the detail is interesting, which means the detail isn't doing 
its job.)

REGISTER GUIDANCE
=================
The following passages demonstrate the tonal register your proposal should 
aim for. Study what they DON'T do: they don't announce their themes, they 
don't escalate, they treat significant moments plainly. Match this register.

[TONAL_REFERENCES]

(The project maintainer will insert 2-4 short reference passages here during 
eval rounds. These are the aesthetic north star for the pipeline's creative 
output and represent a design choice, not something derived from data.

Until reference passages are inserted, follow this guidance: write as though 
you are summarizing a novel you read years ago and remember fondly but 
imperfectly. The details you recall are the ones that mattered. The ones 
you've forgotten were connective tissue. Your proposal should read like 
those remembered details — specific, slightly melancholy, without commentary.)

ANTI-PATTERNS
=============
The following are LLM writing tics. They are not forbidden — but if you notice 
yourself reaching for one, it's a signal that you're constructing rather than 
telling. Find a plainer way to say it.

- Characters whose eyes "reflect," "mirror," or "hold" anything
- The words "juxtaposition," "dichotomy," "tapestry," "dance" used metaphorically
- "In a world where..." constructions
- Weather signaling mood
- Ending scenes with a character reflecting on what just happened
- Antagonists explained by tragic backstories
- The protagonist having an "epiphany" or "realization" in the climax
- Describing anything as "a testament to" anything else
- The phrase "little did they know"
- Characters who are "haunted by" something
```

### Placeholder: `[TONAL_REFERENCES]`

This is a deliberate placeholder. The project maintainer (you) inserts 2-4 short reference passages during eval rounds. These are the *aesthetic north star* — the answer to "what should good output feel like?" That's a design choice only a human can make.

The fallback guidance ("write as though you are summarizing a novel you read years ago...") is a reasonable default that encourages understatement, selectivity, and the sense that the story existed before the proposal described it.

As you do eval rounds, you can swap reference passages in and out to steer the register. This is the primary creative tuning knob — more powerful than rubric weights because it shapes *how the model thinks* rather than how it's judged.

### What Changes in the YAML

The system prompt section of `proposal_draft_generate.yaml` gets the Creative Sensibility block replacing any Craft Directives or cliché-avoidance instructions. The existing JSON schema instructions and user prompt template are **unchanged** — the user template already has all the right placeholders.

### What Does NOT Change

- The user prompt template — no new placeholders needed
- The `ProposalDraftAgent` Python code — it reads the prompt file as-is
- The `StoryProposal` Pydantic schema — same JSON fields, better content
- The selection prompt (`proposal_draft_select.yaml`) — this is ProposalDraft's internal critic for choosing among N candidates, separate from the rubric judge

### Changes to the Selection Prompt

The selection prompt (`proposal_draft_select.yaml`) should also be updated to align with the v2 philosophy. The current selection criteria (in priority order) are: cliché avoidance, thematic depth, specificity, tonal coherence, internal conflict. These should be revised to:

1. **Restraint**: Which candidate is least performative? Which one resists the urge to be impressive?
2. **Convention-departure quality**: Which candidate follows genre conventions and then departs at the right moment?
3. **Specificity without showiness**: Which candidate's details are concrete but stated plainly?
4. **Tonal coherence**: Which candidate matches the user's tonal preferences?
5. **Internal conflict**: Which protagonist's internal arc feels specific to this character, not interchangeable?

The cliché-avoidance criterion is removed. The cliché-as-departure-map framing means clichés in the proposal are expected and welcome — what matters is the quality of the departure.

### Testing

- All existing `TestProposalDraftGeneratePrompt` tests must still pass (template placeholders unchanged)
- Manual inspection: verify the Creative Sensibility section is present and the old Craft Directives are removed
- No new automated tests needed for WI-1 alone

---

## 2. WI-2: Retry Prompt — `proposal_draft_retry.yaml`

### Rationale

When RubricJudgeAgent fails a proposal and routes back to ProposalDraftAgent, the generator needs to know *what went wrong*. A retry with the same prompt produces similar output. A retry with targeted editorial feedback produces meaningfully different output.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/proposal_draft_retry.yaml` | CREATE |
| `tests/test_prompt_loader.py` | ADD integration tests |

### Prompt Design

The **system prompt** is identical to `proposal_draft_generate.yaml` — same role, same Creative Sensibility section. The principles don't change on retry.

The **user prompt template** includes everything from the original template PLUS:

- `{previous_proposal}` — JSON serialization of the failed proposal
- `{rubric_feedback}` — the dimension-by-dimension feedback from RubricJudge
- `{rubric_scores}` — the numeric scores per dimension
- `{attempt_number}` — which attempt this is (2 or 3)

The user prompt frames the retry:

```
REVISION CONTEXT
================
This is attempt {attempt_number}. A previous proposal was evaluated by an 
independent editor and did not pass. The specific feedback is below.

Your job is to generate a NEW proposal that addresses the identified 
weaknesses while preserving any strengths noted. Do NOT simply patch the 
previous proposal — generate fresh creative work that takes the feedback 
as editorial direction.

Read the feedback carefully. If the editor says the proposal was too 
performative, write with more restraint. If they say the departure moment 
was weak, find a different and better departure. If they say the details 
were generic, make them specific — but don't overcompensate by making every 
detail aggressively unusual. Find the plain, specific, right detail.

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
  - test_system_prompt_matches_generate: retry system prompt is identical to generate
  - test_user_template_has_standard_placeholders: all original placeholders present
  - test_user_template_has_retry_placeholders: {previous_proposal}, {rubric_feedback},
    {rubric_scores}, {attempt_number} are present
  - test_format_user_with_valid_retry_data: format_user() succeeds with all fields
```

### Design Decision: Why a Separate File Instead of Conditional Logic

Two files is cleaner than one file with an `{optional_retry_section}` placeholder. The retry framing is substantial, and having it as a separately visible file makes it obvious which prompt is used when. Follows the precedent of `proposal_draft_generate.yaml` vs. `proposal_draft_select.yaml`.

---

## 3. WI-3: ProposalDraft Node Wrapper — Retry Detection

### Rationale

The node wrapper must detect whether this is a first attempt or a retry, and pass the appropriate context to the agent. The agent itself doesn't know about retries — the node wrapper bridges state to agent input.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE |
| `src/storymesh/agents/proposal_draft/agent.py` | UPDATE — add retry prompt, RubricFeedback |
| `tests/test_graph.py` | ADD retry-specific node wrapper tests |

### Changes to `ProposalDraftAgent.run()`

Add an optional `rubric_feedback` parameter:

```python
def run(
    self,
    input_data: ProposalDraftAgentInput,
    *,
    rubric_feedback: RubricFeedback | None = None,
) -> ProposalDraftAgentOutput:
```

When `rubric_feedback` is provided, the agent uses `self._retry_prompt` instead of `self._generate_prompt` and formats the user template with the additional retry-specific fields.

### Agent Constructor Change

Add a third prompt load:

```python
self._retry_prompt = load_prompt("proposal_draft_retry")
```

Eagerly loaded so misconfiguration is caught at construction time.

### RubricFeedback Type

Internal plumbing, not a persisted artifact:

```python
@dataclasses.dataclass(frozen=True)
class RubricFeedback:
    """Internal carrier for rubric feedback passed to retry attempts."""
    previous_proposal_json: str
    feedback_text: str
    scores_text: str
    attempt_number: int
```

Defined in `src/storymesh/agents/proposal_draft/agent.py`.

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
    """Format dimension-by-dimension feedback for the retry prompt."""
    lines = []
    for dim_name, dim_result in rubric_output.dimensions.items():
        lines.append(f"[{dim_name}] (score: {dim_result.score}): {dim_result.feedback}")
    return "\n".join(lines)

def _format_scores(rubric_output: RubricJudgeAgentOutput) -> str:
    """Format scores as a compact summary."""
    lines = []
    for dim_name, dim_result in rubric_output.dimensions.items():
        lines.append(f"  {dim_name}: {dim_result.score}")
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
    passed=True, no retry logic triggers
```

---

## 4. WI-4: RubricJudge Schema

### Rationale

The RubricJudgeAgent needs Pydantic schemas for its input and output. The output schema defines the scores, feedback, and pass/fail signal that drive the retry loop.

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
    cliched_resolutions: dict[str, list[str]]  # tension_id → list of familiar patterns
    user_tones: list[str]
    user_prompt: str
    normalized_genres: list[str]
    attempt_number: int = Field(ge=1, default=1)

class DimensionResult(BaseModel):
    """Score and feedback for a single rubric dimension."""
    model_config = {"frozen": True}

    score: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(min_length=10)
    principle_ref: str = Field(
        min_length=1,
        description=(
            "Which creative principle this dimension evaluates "
            "(e.g., 'restraint', 'convention_departure', 'specificity')."
        ),
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
            "Dimension names: restraint, convention_departure, "
            "specificity, protagonist_interiority, user_intent_fidelity."
        ),
    )
    convention_departures: list[str] = Field(
        default_factory=list,
        description=(
            "Specific genre conventions the proposal follows and the "
            "departure moment(s) identified, if any."
        ),
    )
    overall_feedback: str = Field(
        min_length=10,
        description="Holistic editorial assessment.",
    )
    debug: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = RUBRIC_SCHEMA_VERSION
```

### Key Change from v1: `cliche_violations` → `convention_departures`

v1 had a `cliche_violations: list[str]` field that tracked clichés the proposal *failed to avoid*. v2 replaces this with `convention_departures: list[str]` that tracks genre conventions the proposal *follows* and the moment(s) where it *departs*. The evaluator identifies both the convention and the quality of the departure. An empty list means the proposal had no meaningful departure — which is a scoring issue, not a penalty.

### Design Decision: `dimensions` as Dict

Same rationale as v1. Dict provides eval-round flexibility. Adding/removing/renaming dimensions doesn't require schema changes. The agent validates dimension presence at runtime.

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
  - test_convention_departures_defaults_empty
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

The prompt defines the evaluation dimensions and their criteria. v2 dimensions align with the three creative principles rather than mapping to coded directives.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/prompts/rubric_judge.yaml` | CREATE |
| `tests/test_prompt_loader.py` | ADD integration tests |

### System Prompt Content

```
You are a senior fiction editor evaluating a story proposal. You are NOT the 
same AI that generated this proposal — you are an independent evaluator 
providing a second opinion.

You are not looking for technical competence. The proposal will be structurally 
sound — it will have a protagonist, a plot arc, thematic tensions. That is 
table stakes. What you are looking for is whether this proposal feels like it 
was written by someone with a genuine sensibility, or whether it reads like 
competent AI output — impressive on the surface but hollow underneath.

The hallmarks of hollow AI fiction:
- Every detail is maximally interesting. Nothing is plain or ordinary.
- Themes are announced rather than embodied. The prose explains its own 
  significance.
- Tensions are either resolved cleanly or left dramatically unresolved 
  as a performance of ambiguity.
- The pacing is even — each act gets roughly equal weight, tension 
  escalates smoothly, there are no digressions or lulls.
- Metaphors and juxtapositions are overused. Everything is "a testament to" 
  or "a dance between" something.
- The protagonist's flaw is generic ("learns to trust," "overcomes fear") 
  and interchangeable with any other story.

The hallmarks of authentic fiction:
- Most of the story is conventional. The genre contract is honored. The 
  reader settles into familiar patterns.
- Details are specific but stated plainly, without commentary.
- There is ONE moment — well-placed, well-earned — where the story 
  departs from the expected pattern and the reader's understanding shifts.
- The departure doesn't announce itself. The reader realizes what happened 
  a beat late.
- The protagonist's internal conflict is specific to this character in 
  this world and could not be transplanted into a different story.
- Some parts of the story matter more than others, and the pacing reflects 
  this unevenness.

EVALUATION DIMENSIONS
=====================
Score each dimension from 0.0 to 1.0 and provide specific, actionable feedback.

D-1: RESTRAINT [weight: 0.25]
Evaluates: Principle 1 (Restraint Over Spectacle)
Does the proposal resist the urge to be impressive?
- Score 0.0–0.3: The proposal is maximally dramatic at all times. Every 
  detail is extraordinary. Themes are announced. Metaphors are overused. 
  The prose explains its own significance.
- Score 0.4–0.6: Some restraint, but the proposal still reaches for 
  intensity in places where plainness would serve better.
- Score 0.7–0.8: Mostly restrained. Significant moments are treated with 
  appropriate weight without being overplayed.
- Score 0.9–1.0: The proposal trusts the reader. Details are stated 
  plainly. The story's significance emerges from what is described, not 
  from how the description frames it.
Check: Read the plot_arc and key_scenes. Are significant moments described 
with the same register as ordinary moments? Does the prose explain why 
things matter, or does it let them matter on their own?

D-2: CONVENTION AND DEPARTURE [weight: 0.30]
Evaluates: Principle 2 (Convention Then Departure)
Does the proposal follow genre conventions and then depart at a 
structurally significant moment?
- Score 0.0–0.2: The proposal either follows conventions entirely 
  (no departure) OR subverts everything (no convention to depart from).
- Score 0.3–0.5: There is a departure, but it's either in a throwaway 
  moment, feels arbitrary, or isn't earned by what preceded it.
- Score 0.6–0.8: There is a clear, well-placed departure that 
  recontextualizes what came before. The conventions preceding it 
  feel intentional, not lazy.
- Score 0.9–1.0: The departure is the kind of moment that makes a 
  reader pause and think "oh — THAT'S what this story is actually 
  about." The conventions were load-bearing. The familiar patterns 
  were doing essential work that only becomes visible after the 
  departure.
Check: Identify which familiar resolutions (from the cliched_resolutions 
list) the proposal follows. Then identify the departure moment. Is it 
at a structurally significant point? Does it recontextualize the 
conventions that preceded it?

D-3: SPECIFICITY [weight: 0.20]
Evaluates: Principle 3 (Specificity Without Performance)
Are the details concrete, particular to this story, and stated plainly?
- Score 0.0–0.3: Details are either generic ("a dark secret," "a 
  dangerous journey") or aggressively unusual in a way that calls 
  attention to itself (performative specificity).
- Score 0.4–0.6: Mix of specific and generic. Some details are 
  particular to this story, others could appear in any story.
- Score 0.7–0.8: Most details are specific and stated without 
  commentary. They connect to the thematic tensions but don't 
  announce the connection.
- Score 0.9–1.0: Every significant detail is particular to this 
  story, stated plainly, and load-bearing. The reader discovers 
  the connections rather than being told about them.
Check: Could the protagonist, setting, or plot mechanism be 
transplanted into a different story with different themes? If yes, 
the details are too generic. Are specific details accompanied by 
explanatory commentary ("a peculiar habit that spoke volumes about...")? 
If yes, the specificity is performative.

D-4: PROTAGONIST INTERIORITY [weight: 0.15]
Evaluates: Protagonist's internal conflict
Is the protagonist's want/need split specific to this character in 
this world?
- Score 0.0–0.3: Generic arc ("learns to trust," "overcomes fear," 
  "finds redemption"). Interchangeable with any story.
- Score 0.4–0.6: Somewhat specific, but the internal conflict doesn't 
  clearly mirror the thematic tension.
- Score 0.7–1.0: The internal conflict is unique to this character, 
  inseparable from the thematic landscape, and could not exist in a 
  different story.
Check: State the protagonist's want and need in one sentence each. 
If either sentence could describe a protagonist in a completely 
different genre, score low.

D-5: USER INTENT FIDELITY [weight: 0.10]
Evaluates: Alignment with the user's original prompt.
Does the proposal honor the genres, tones, and narrative context the 
user specified?
- This is a floor, not a ceiling. Penalize only when the proposal 
  actively contradicts the user's request, not when it interprets 
  it creatively.
- Score 0.0–0.3: The proposal ignores the user's stated preferences.
- Score 0.7–1.0: The proposal fulfills the user's intent while making 
  its own creative choices within that intent.

CONVENTION DEPARTURE ANALYSIS
=============================
Separately from the dimension scores, perform this analysis:

1. List which familiar resolutions (from the cliched_resolutions input) 
   the proposal follows. This is GOOD — conventions are the genre contract.
2. Identify the moment where the proposal departs from a familiar 
   resolution. Describe what the convention was and how the departure 
   differs.
3. Assess whether the departure is:
   a. Well-placed (at a structurally significant moment, not a throwaway)
   b. Earned (the preceding conventions did essential setup work)
   c. Subtle (the departure doesn't announce itself)
4. If there is no departure, note this — the proposal follows conventions 
   throughout, which is a scoring issue for D-2.

SCORING
=======
Compute the composite score as:
  composite = (D1 * 0.25) + (D2 * 0.30) + (D3 * 0.20) + (D4 * 0.15) + 
              (D5 * 0.10)

Do NOT compute pass/fail. Return only the scores, feedback, and convention 
analysis. The pipeline will determine pass/fail from the composite score 
and a configurable threshold.

RESPONSE FORMAT
===============
Return ONLY a JSON object. No markdown fences, no commentary.

{
  "dimensions": {
    "restraint": {
      "score": <0.0-1.0>,
      "feedback": "<specific feedback on restraint vs. spectacle>",
      "principle_ref": "restraint"
    },
    "convention_departure": {
      "score": <0.0-1.0>,
      "feedback": "<identify conventions followed and departure quality>",
      "principle_ref": "convention_departure"
    },
    "specificity": {
      "score": <0.0-1.0>,
      "feedback": "<assess specificity and whether it's performative>",
      "principle_ref": "specificity"
    },
    "protagonist_interiority": {
      "score": <0.0-1.0>,
      "feedback": "<assess want/need specificity to this character/world>",
      "principle_ref": "protagonist_interiority"
    },
    "user_intent_fidelity": {
      "score": <0.0-1.0>,
      "feedback": "<assess alignment with user prompt>",
      "principle_ref": "user_intent_fidelity"
    }
  },
  "convention_departures": [
    "Convention followed: <pattern from cliched_resolutions>",
    "Departure: <what the proposal does differently, and where>",
    "Assessment: <well-placed/arbitrary, earned/forced, subtle/announced>"
  ],
  "overall_feedback": "<2-3 sentence editorial assessment. What is the 
    strongest element? What is the weakest? If you were this writer's 
    editor, what is the ONE thing you would tell them to change?>"
}
```

### Key Changes from v1

**5 dimensions instead of 6.** Dropped `craft_discipline` (which was a grab-bag of pacing, anti-patterns, and scene details) and `structural_surprise` (which overlapped with convention departure). The dimensions are now cleaner — each evaluates one clear thing.

**Convention-departure is the highest-weighted dimension (0.30).** This is the heart of the v2 philosophy. The single most important thing the rubric evaluates is: did the story follow genre conventions, and then depart in a way that recontextualizes everything?

**No cliché penalty.** v1 deducted 0.05 per cliché violation. v2 doesn't penalize clichés — it *expects* them. What it evaluates is the quality of the departure.

**`principle_ref` instead of `directive_ref`.** The schema field renamed to reflect the shift from coded directives to guiding principles.

### User Prompt Template

```
USER PROMPT: "{user_prompt}"
GENRES: {normalized_genres}
USER TONES: {user_tones}

THEMATIC TENSIONS (with familiar resolutions — the genre contract):
{tensions}

PROPOSAL TO EVALUATE:
{proposal}
```

Same placeholders as v1. No changes needed.

### Testing

```
TestRubricJudgePrompt:
  - test_load_prompt_succeeds: load_prompt("rubric_judge") returns PromptTemplate
  - test_system_prompt_non_empty
  - test_system_prompt_contains_dimension_names: all 5 dimension names appear
  - test_system_prompt_contains_principle_refs: restraint, convention_departure, 
    specificity, protagonist_interiority, user_intent_fidelity
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
    """Evaluates story proposals against a creative quality rubric (Stage 5).

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
    ) -> None:
```

Parameters:
- `llm_client` — **required**. Should be a different provider than ProposalDraft.
- `temperature` — default `0.0`. Evaluation should be deterministic.
- `max_tokens` — default `4096`. Feedback can be detailed.
- `pass_threshold` — default `0.7`. Primary tuning knob for eval rounds.
- `dimension_weights` — optional override. Defaults below.

### Default Dimension Weights

```python
DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "restraint": 0.25,
    "convention_departure": 0.30,
    "specificity": 0.20,
    "protagonist_interiority": 0.15,
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
4. Call `self._llm_client.complete_json()` at `self._temperature`.
5. Parse the response into individual `DimensionResult` objects.
6. **Validate dimension coverage:** check that all expected dimension names are present. If any are missing, log a warning and assign score 0.0 with feedback "Dimension not evaluated by the model."
7. **Compute composite score:** `sum(weight * score for each dimension)`.
8. **Determine pass/fail:** `composite_score >= self._pass_threshold`.
9. Assemble and return `RubricJudgeAgentOutput`.

### Error Handling

- If `complete_json()` raises or returns unparseable JSON, return a **default fail** with `composite_score=0.0`, `passed=False`, and feedback explaining the failure. This ensures the retry loop gets a chance.
- If the LLM returns scores outside [0.0, 1.0], clamp them.

### Testing

```
TestRubricJudgeAgent:
  - test_passing_proposal: FakeLLMClient returns high scores → passed=True
  - test_failing_proposal: FakeLLMClient returns low scores → passed=False
  - test_composite_score_computation: verify weighted average math
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
| `src/storymesh/orchestration/graph.py` | UPDATE — wire real node |
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
2. Read `theme_extractor_output` from state (for tensions and familiar resolutions).
3. Read `genre_normalizer_output` from state (for genres, tones).
4. Assemble `RubricJudgeAgentInput`:
   - `proposal` = `proposal_draft_output.proposal`
   - `tensions` = `theme_extractor_output.tensions`
   - `cliched_resolutions` = dict comprehension from tensions
   - `user_tones` = `theme_extractor_output.user_tones_carried`
   - `user_prompt` = `state["user_prompt"]`
   - `normalized_genres` = `genre_normalizer_output.normalized_genres`
   - `attempt_number` = `state.get("rubric_retry_count", 0) + 1`
5. Call `agent.run(input_data)`.
6. Persist output artifact.
7. Return `{"rubric_judge_output": output}`.

**Note:** The node wrapper does NOT increment `rubric_retry_count`. That's done by the proposal_draft node wrapper on retry (WI-3).

### State Type Update

Replace:
```python
rubric_judge_output: object | None
```
With:
```python
from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput
rubric_judge_output: RubricJudgeAgentOutput | None
```

### Graph Wiring Update

In `build_graph()`:

1. Get the `rubric_judge` config via `get_agent_config("rubric_judge")`.
2. Build the `LLMClient` using the provider registry — should resolve to a different provider than proposal_draft based on config.
3. Construct `RubricJudgeAgent` with config values.
4. Create the node via `make_rubric_judge_node(agent, artifact_store=artifact_store)`.
5. Replace `graph.add_node("rubric_judge", _noop_node)` with the real node.

### Routing Function Update

The existing `_rubric_route` already has the right structure. The `getattr(output, 'passed', True)` pattern works with `RubricJudgeAgentOutput` since `passed` is a field on the model. Remove the "Placeholder logic: always pass" comment.

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

## 8. WI-8: State + History Tracking — All Attempts

### Rationale

SynopsisWriterAgent (Stage 6, next sprint) needs access to ALL proposal attempts and ALL rubric evaluations, not just the final one, so it can synthesize the best elements from each.

### Files Affected

| File | Action |
|------|--------|
| `src/storymesh/orchestration/state.py` | ADD history fields |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE — append to history |
| `src/storymesh/orchestration/nodes/rubric_judge.py` | UPDATE — append to history |

### New State Fields

```python
proposal_history: list[ProposalDraftAgentOutput]
"""All proposal attempts in order."""

rubric_history: list[RubricJudgeAgentOutput]
"""All rubric evaluations in order."""

best_proposal_index: int
"""Index into proposal_history of the highest-scoring attempt."""
```

### LangGraph State Accumulation

LangGraph `TypedDict` uses last-write-wins. For list fields that accumulate, node wrappers read the existing list and return the appended version:

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

best_idx = state.get("best_proposal_index", 0)
proposals = state.get("proposal_history", [])
if proposals:
    current_best_score = -1.0
    for i, p_output in enumerate(proposals):
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
    # dimension_weights:      # Uncomment to override defaults
    #   restraint: 0.25
    #   convention_departure: 0.30
    #   specificity: 0.20
    #   protagonist_interiority: 0.15
    #   user_intent_fidelity: 0.10
```

### README Changes

1. Move `RubricJudgeAgent` from "Not implemented yet" to "Implemented"
2. Add description of the creative quality system: three principles, convention-departure philosophy, cross-provider evaluation
3. Update "Current runtime behavior" to describe real rubric evaluation
4. Update the rubric retry loop section — now fully functional
5. Add a "Creative Quality System" section explaining the philosophy
6. Roadmap: mark "Activate rubric-based retry logic" as done

---

## Design Decision Record

### Why Sensibility Instead of Rules?

Rules produce rule-following, which is itself a kind of artificiality. The v1 plan had eight coded directives. In testing, the model followed them diligently and the output still felt like AI output — now it was AI output that *also* avoided specific anti-patterns, which made it feel like AI output trying to seem human. The shift to three principles + tonal references gives the model something to *inhabit* rather than something to *satisfy*. The difference is between an actor who has memorized stage directions and an actor who understands the character.

### Why Clichés Are Not Penalized

Clichés are cliché because they work. Readers enjoy them. The genre contract depends on them. A detective story where the detective doesn't investigate is not a detective story. The creative opportunity is not in avoiding conventions but in following them *and then departing at a moment that matters*. A story that avoids all clichés is as predictable as one that follows all of them — both have made a blanket decision instead of making specific choices. The v2 rubric's highest-weighted dimension (convention_departure, 0.30) evaluates the quality of the relationship between convention and departure, not the absence of convention.

### Why Convention-Departure Is the Highest-Weighted Dimension

This is the dimension that most directly measures what makes fiction genuinely interesting to a human reader. Restraint, specificity, and protagonist interiority are all important, but they're qualities of *execution*. Convention-departure is a quality of *conception* — it asks whether the story has a reason to exist beyond "the pipeline needed to produce a story." A story that follows conventions and departs well has found something to say. A story that's restrained, specific, and well-characterized but never departs from the expected pattern is... fine. It's fine. But it doesn't make anyone pause.

### The Tonal Reference Placeholder

The `[TONAL_REFERENCES]` section in the ProposalDraft prompt is deliberately a placeholder. The project maintainer inserts 2-4 short passages that demonstrate the register they want. This is the *aesthetic north star* — the one part of the system that requires human judgment rather than engineering. The fallback guidance ("write as though you are summarizing a novel you read years ago and remember fondly but imperfectly") is a reasonable default.

During eval rounds, swapping reference passages is the most powerful tuning knob available — more powerful than rubric weights, because it shapes how the model *thinks* rather than how it's *judged*.

### Temperature 0.0 for the Rubric Judge

Unchanged from v1. Evaluation should be reproducible. Same proposal → same scores.

### Conflict with README: Dropped Dimensions

The README lists genre fit, internal coherence, emotional arc, market hook, and novelty as rubric dimensions. This plan drops all of them:

- **Genre fit** → subsumed by user_intent_fidelity (as a floor, not a ceiling)
- **Internal coherence** → table stakes; caught by Pydantic validation and ProposalDraft's self-selection step
- **Emotional arc** → subsumed by convention_departure (the departure *is* the emotional pivot)
- **Market hook** → actively harmful; incentivizes optimizing for what the LLM thinks publishers want, which produces generic output
- **Novelty** → replaced by the convention-departure philosophy; novelty everywhere is as bad as novelty nowhere

These can be added back as additional dimensions in the prompt without code changes.

### The Eval Round Workflow

After implementation, the eval workflow is:

1. Run `storymesh generate "your prompt here"`
2. Inspect `~/.storymesh/runs/<run_id>/proposal_draft_output.json` — read the proposal
3. Inspect `~/.storymesh/runs/<run_id>/rubric_judge_output.json` — read scores and feedback
4. If proposals are too performative/spectacular: strengthen tonal references, adjust restraint weight
5. If departures are weak: raise convention_departure weight, lower pass_threshold to see more attempts
6. If output is too conventional (no departure): lower pass_threshold isn't the fix — revise the Creative Sensibility prompt to emphasize the departure instruction
7. If the register feels wrong: swap tonal reference passages
8. If a specific anti-pattern keeps appearing: add it to the Anti-Patterns list in the prompt
9. Re-run and compare

No code changes needed for steps 4–8. Just prompt and config edits.

---

## File Summary

| File | Action | WI |
|------|--------|----|
| `src/storymesh/prompts/theme_extractor.yaml` | UPDATE — cliché reframe | WI-0 |
| `src/storymesh/prompts/proposal_draft_generate.yaml` | REWRITE — creative sensibility | WI-1 |
| `src/storymesh/prompts/proposal_draft_select.yaml` | UPDATE — align selection criteria | WI-1 |
| `src/storymesh/prompts/proposal_draft_retry.yaml` | CREATE | WI-2 |
| `tests/test_prompt_loader.py` | UPDATE — add retry prompt tests | WI-2 |
| `src/storymesh/agents/proposal_draft/agent.py` | UPDATE — add retry prompt, RubricFeedback | WI-3 |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | UPDATE — retry detection, history | WI-3, WI-8 |
| `tests/test_graph.py` | UPDATE — retry + rubric node wrapper + routing tests | WI-3, WI-7 |
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
| `src/storymesh/orchestration/nodes/rubric_judge.py` | UPDATE — history + best tracking | WI-8 |
| `storymesh.config.yaml` | UPDATE — add rubric_judge section | WI-9 |
| `storymesh.config.yaml.example` | UPDATE — same | WI-9 |
| `README.md` | UPDATE — status, creative quality system | WI-9 |
| `src/storymesh/versioning/agents.py` | UPDATE — add agent version | WI-9 |

---

## Validation Checklist

After all work items are complete, verify the following:

- [ ] `pytest` passes with no new failures
- [ ] `ruff check src/ tests/` clean
- [ ] `mypy src/` clean
- [ ] Running `storymesh generate "dark post-apocalyptic detective mystery"` with both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set produces a real rubric evaluation
- [ ] The rubric output contains all 5 dimension scores with non-empty feedback
- [ ] The `convention_departures` field identifies which genre conventions were followed and where the departure occurs
- [ ] If the proposal fails the rubric, the pipeline retries and the retry prompt includes the editorial feedback
- [ ] After max retries, the pipeline proceeds to synopsis_writer with the best-scoring proposal
- [ ] The `proposal_history` and `rubric_history` lists in the final state contain all attempts
- [ ] With only `ANTHROPIC_API_KEY` set (no OpenAI key), the rubric_judge node falls back to noop and the pipeline still completes
- [ ] Changing `pass_threshold` in config changes pass/fail behavior without code changes
- [ ] Editing the Creative Sensibility section or tonal references in `proposal_draft_generate.yaml` changes generator behavior without code changes
- [ ] The anti-patterns list in the prompt is reflected in the rubric judge's D-1 (restraint) evaluation