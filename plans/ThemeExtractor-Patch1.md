# Patch: Add `cliched_resolutions` to ThematicTension

**Date:** 2026-03-28
**Applies to:** ThemeExtractorAgent-Plan.md (all affected work items)
**Scope:** Add a `cliched_resolutions` field to `ThematicTension` and propagate changes through prompt, agent, and tests

---

## Rationale

Testing of the ThemeExtractorAgent shows that the LLM tends to fall into predictable narrative valleys — clichéd resolutions to creative tensions. Rather than relying on vague prompt instructions like "be original," we explicitly model the anti-patterns. Each `ThematicTension` now carries a list of clichéd resolutions — the predictable, tropey approaches a lazy writer would default to.

This structured data serves two downstream purposes:

- **ProposalDraftAgent** treats them as exclusions: "explore this tension but do NOT use any of these approaches."
- **RubricJudgeAgent** treats them as evaluation criteria: penalize proposals that resemble any listed cliché, and identify specific clichéd elements to revise on retry.

The LLM is much better at avoiding specifically named patterns than it is at being generically "original."

---

## Changes Required

### 1. Schema: `src/storymesh/schemas/theme_extractor.py`

Add a new field to `ThematicTension`, after `intensity`:

```python
cliched_resolutions: list[str] = Field(
    min_length=1,
    description=(
        "Predictable, tropey narrative resolutions to this tension that "
        "the LLM identifies as clichéd. Downstream agents use these as "
        "exclusions (ProposalDraft) and evaluation criteria (RubricJudge)."
    ),
)
```

`min_length=1` is enforced — if the LLM can identify a tension, it can identify at least one clichéd way to resolve it.

---

### 2. Prompt: `src/storymesh/prompts/theme_extractor.yaml`

#### System prompt additions

In the system prompt, where the LLM is instructed on how to construct tensions, add instruction along these lines (adapt to match the existing prompt's voice and style):

- For each tension, identify 2–4 predictable or clichéd narrative resolutions — the approaches a lazy or unimaginative writer would default to when resolving this particular contradiction.
- Clichéd resolutions should be **specific and recognizable**, not vague platitudes. For example, "the hero saves the day" is too vague. "A lone detective rebuilds justice single-handedly through sheer determination" is specific enough to be useful as an exclusion.

#### Response format update

In the system prompt's JSON response format specification, update the tension object to include the new field. Example:

```json
{
  "tension_id": "T1",
  "cluster_a": "mystery",
  "assumption_a": "Truth is discoverable through investigation",
  "cluster_b": "post_apocalyptic",
  "assumption_b": "Institutional knowledge systems have collapsed",
  "creative_question": "What does investigation look like when there are no records, no witnesses, and no institutions to consult?",
  "intensity": 0.85,
  "cliched_resolutions": [
    "A lone detective rebuilds justice single-handedly through sheer determination",
    "The investigation reveals the apocalypse was caused by a conspiracy that can be 'solved'",
    "A hidden bunker contains preserved records that make the case trivially solvable"
  ]
}
```

If the system prompt contains a description of each field in the response schema, add an entry for `cliched_resolutions` that matches the description from the Pydantic field above.

---

### 3. Agent: `src/storymesh/agents/theme_extractor/agent.py`

No logic changes needed in the agent's `run()` method — the field flows through Pydantic parsing automatically. The LLM returns it in the JSON, Pydantic validates it when constructing `ThematicTension`, and it appears in the output.

Confirm that the agent does not do any manual field-by-field construction of `ThematicTension` objects that would need updating. If it does (rather than passing the parsed dict directly to the Pydantic constructor), add `cliched_resolutions` to that construction.

---

### 4. Tests: `tests/test_schemas_theme_extractor.py`

#### Update existing test helpers

Any helper function that constructs a `ThematicTension` for testing (e.g., `_thematic_tension(**overrides)`) must include `cliched_resolutions` in its defaults:

```python
def _thematic_tension(**overrides: object) -> ThematicTension:
    defaults: dict[str, object] = dict(
        tension_id="T1",
        cluster_a="mystery",
        assumption_a="Truth is discoverable",
        cluster_b="post_apocalyptic",
        assumption_b="Institutions have collapsed",
        creative_question="What does justice look like when no one enforces it?",
        intensity=0.85,
        cliched_resolutions=[
            "A lone hero rebuilds civilization single-handedly",
        ],
    )
    return ThematicTension(**(defaults | overrides))
```

#### Add new test cases

```
TestThematicTensionClichedResolutions:
  - test_valid_with_single_cliche: construction succeeds with one clichéd resolution
  - test_valid_with_multiple_cliches: construction succeeds with 3 clichéd resolutions
  - test_empty_cliched_resolutions_rejected: [] is rejected by min_length=1
  - test_cliched_resolutions_frozen: cannot mutate the list after construction
```

---

### 5. Tests: `tests/test_theme_extractor_agent.py`

#### Update FakeLLMClient response fixtures

Any pre-canned JSON response that the `FakeLLMClient` returns must include `cliched_resolutions` in each tension object. Find all test response fixtures that contain `"tensions"` and add the field. Example:

```python
# BEFORE (in test fixture JSON)
{
    "tension_id": "T1",
    "cluster_a": "mystery",
    "assumption_a": "Truth is discoverable",
    "cluster_b": "post_apocalyptic",
    "assumption_b": "Institutions collapsed",
    "creative_question": "What does investigation look like?",
    "intensity": 0.8
}

# AFTER
{
    "tension_id": "T1",
    "cluster_a": "mystery",
    "assumption_a": "Truth is discoverable",
    "cluster_b": "post_apocalyptic",
    "assumption_b": "Institutions collapsed",
    "creative_question": "What does investigation look like?",
    "intensity": 0.8,
    "cliched_resolutions": [
        "A lone detective rebuilds justice single-handedly"
    ]
}
```

#### Add new test case

```
TestClichedResolutions:
  - test_cliched_resolutions_present_in_output: verify each tension in the agent output has a non-empty cliched_resolutions list
```

---

### 6. No changes needed

The following files do **not** need changes for this patch:

- `src/storymesh/agents/theme_extractor/__init__.py` — no exports change
- `src/storymesh/orchestration/nodes/theme_extractor.py` — node wrapper passes data through, field flows automatically
- `src/storymesh/orchestration/graph.py` — no wiring changes
- `src/storymesh/orchestration/state.py` — type annotation unchanged
- `src/storymesh/versioning/schemas.py` — `THEMEPACK_SCHEMA_VERSION` stays at `"1.0"` since this is being added before the first release of the schema
- `storymesh.config.yaml` / `storymesh.config.yaml.example` — no config changes
- `README.md` — will be updated in WI-8 of the main plan which has not been executed yet

---

## Validation

After applying this patch:

```bash
# 1. All tests pass
pytest

# 2. Type checking
mypy src/

# 3. Lint
ruff check src/ tests/

# 4. If API keys are available, run end-to-end and inspect the artifact:
storymesh generate "dark post-apocalyptic detective mystery"
# Open ~/.storymesh/runs/<run_id>/theme_extractor_output.json
# Verify each tension object contains a non-empty "cliched_resolutions" array
```