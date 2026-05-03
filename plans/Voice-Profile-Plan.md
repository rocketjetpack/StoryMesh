# StoryMesh Implementation Plan — Voice Profile Differentiation

**Date:** 2026-05-03
**Scope:** Genre/tone-conditional prose voice via a small fixed set of voice profiles, plus a stylometric counter, logging cleanup, and documentation refresh.
**Target Version:** next minor

This plan addresses voice homogenization across the StoryMesh portfolio. Five recent runs were reviewed (`The Floor of the Sky`, `The Patient Frequency`, `The Body of the Place`, `The Golden Coaster`, `StarTrek_Fanfic`/Tuesday Variable). Despite covering literary SF, Supernatural fanfic, Star Trek fanfic, and a children's bedtime story, all five share a single recognizable prose voice. Inspection of `*_llm_calls.jsonl` confirmed the cause is structural: the prose-draft prompt is byte-identical across all five runs, the few-shot exemplars are uniformly dark-literary, and the prompt's avoid list mechanically generates the specific tics observed in output.

This plan introduces a small fixed set of **voice profiles**, a classifier that selects one per run, and a prompt-overlay mechanism that produces genre-appropriate prose without disturbing the existing literary-restraint default. It also adds an offline stylometric counter, cleans up logging hygiene, and updates documentation.

**Rubric-stage aesthetic bias is explicitly out of scope for this plan** and is recorded as a known limitation for a future follow-up.

---

## Table of Contents

1. [Design Overview](#1-design-overview)
2. [Out of Scope and Known Limitations](#2-out-of-scope-and-known-limitations)
3. [Conflicts with README and CLAUDE.md](#3-conflicts-with-readme-and-claudemd)
4. [Work Item Ordering](#4-work-item-ordering)
5. [WI-1: VoiceProfile Schema and Data Files](#5-wi-1-voiceprofile-schema-and-data-files)
6. [WI-2: VoiceProfileSelector Agent](#6-wi-2-voiceprofileselector-agent)
7. [WI-3: Voice-Profile-Aware StoryWriter Prompts](#7-wi-3-voice-profile-aware-storywriter-prompts)
8. [WI-4: Logging Hygiene — Distinct Agent Identities](#8-wi-4-logging-hygiene--distinct-agent-identities)
9. [WI-5: Stylometric Counter (Offline Diagnostic)](#9-wi-5-stylometric-counter-offline-diagnostic)
10. [WI-6: Documentation Refresh](#10-wi-6-documentation-refresh)
11. [Validation Checklist](#11-validation-checklist)

---

## 1. Design Overview

### The Diagnosis

Three things in the current pipeline jointly produce voice convergence:

1. **One prose prompt for all genres.** `src/storymesh/prompts/story_writer_draft.yaml` has the same craft principles, avoid list, and few-shot exemplars regardless of input. A bedtime story for a seven-year-old runs the same prompt as a literary-SF wall-mystery.
2. **Mechanical tic generation from the avoid list.** Forbidding "She felt / he realized" and "as if / as though" simultaneously routes the model to a single nearby substitute — `"the way X is when Y"` — which then becomes a portfolio-wide tic. The `"cognitive friction"` instruction similarly routes to cascading "`which was X, which was Y`" clauses.
3. **Uniformly dark-literary few-shot exemplars.** Both the scene-outliner prompt and the proposal-draft prompt use exemplars like "salvaged tax forms" and "tribunal in a parking garage." Exemplar mimicry is more robust than instruction-following, so even a cozy bedtime story is primed by these.

### The Approach

Voice doesn't map cleanly to genre. Literary horror and pulp horror share a genre but want opposite prose. Bedtime fantasy and grimdark fantasy share a genre and want opposite voices. The right unit of variation is **voice profile**, not genre.

This plan introduces three voice profiles:

| Profile | For | Allows | Forbids |
|---|---|---|---|
| `literary_restraint` | dark, mystery, literary, dread, eerie, conspiratorial, psychological, slow | current behavior | current avoid list |
| `cozy_warmth` | bedtime, cute, comfy, gentle, comforting, quietly wondrous | direct emotion-naming, ritualistic repetition, soft anthropomorphism, pattern fulfillment | feeling-by-proxy similes, dark imagery, philosophical paragraph cappers |
| `genre_active` | action, adventure, pulp, high-energy fanfic | kinetic prose, dialogue-forward scenes, faster pacing, less interiority | over-interior reflection, cascading subordinate clauses, slow paragraph cappers |

Three is small enough to keep the design tractable for a single semester, broad enough to cover the genres that have shipped through the pipeline so far, and structured to leave room for future additions without redesigning the routing layer.

### Routing

A new node, `voice_profile_selector`, runs after `genre_normalizer` and selects exactly one voice profile per run. It uses a small classifier prompt (Haiku, T=0, similar in spirit to genre_normalizer). The selected profile flows through `StoryMeshState` and is consumed by `StoryWriterAgent` and `ResonanceReviewerAgent`.

### Prompt overlay strategy

Rather than three full prose prompts, the existing `story_writer_draft.yaml` becomes a **base** prompt with placeholders for profile-specific sections. Each `VoiceProfile` carries an overlay that fills those placeholders with profile-appropriate craft principles, avoid items, and exemplars. This keeps the structural backbone of the prose prompt single-sourced and makes it easy to compare profiles side-by-side.

---

## 2. Out of Scope and Known Limitations

The following are deliberately deferred:

- **Rubric-judge genre conditioning.** The current rubric rewards literary-fiction qualities (restraint, convention departure, plain unornamented detail). For non-literary voice profiles, this rubric will systematically score proposals lower than they deserve. The retry loop will pull cozy proposals back toward literary aesthetic. **This means v1 will improve prose voice but not the underlying narrative shape.** A bedtime-story proposal will still be thesis-driven and philosophical underneath; only its prose surface will be cozy. This is acceptable for v1 and will need a v2 plan that introduces per-profile rubric rules or an alternative proposal-stage path.
- **ThemeExtractorAgent's dialectical-synthesis framework.** The framework assumes every story argues something via tension between genre traditions. Bedtime stories don't argue. Cozy fiction doesn't argue. A future plan should consider per-profile theme extraction or a lighter alternative for non-literary profiles.
- **Resonance reviewer voice-locking.** The reviewer's `"match existing voice exactly"` rule cements voice within a story. With three profiles this is now a benefit rather than a hazard, so no change is needed in this plan — but it's worth noting the rule is intentionally one-way and does not soften prose tics that the writer produced.
- **A formal eval system.** Bulk human review remains the qualitative check. The stylometric counter (WI-5) is a quantitative sanity check, not an eval.
- **Cross-portfolio diversity checks.** No machinery to detect that two unrelated runs produced near-identical prose. Per-run voice profiles substantially reduce this risk.

---

## 3. Conflicts with README and CLAUDE.md

These should be discussed before implementation lands:

1. **`CLAUDE.md` asserts: "the README.md in the root of the repository... is always up to date."** This is currently false — the README still describes `SynopsisWriterAgent` as a no-op placeholder while `StoryWriterAgent` and `ResonanceReviewerAgent` are both fully implemented. WI-6 corrects the README, but the assertion in `CLAUDE.md` itself should either be re-validated as a process commitment or softened to a goal.
2. **`CLAUDE.md`: "Communication is controlled by tightly binding Pydantic contracts at each step."** Adding a `VoiceProfile` to `StoryMeshState` and to the `StoryWriterAgentInput` extends contracts; this is consistent with the principle but worth flagging because it's a contract change.
3. **`CLAUDE.md`: "Prompts should ALWAYS reside in dedicated data files."** The voice-profile data is prompt-adjacent (overlay text, exemplars). This plan keeps it in `src/storymesh/prompts/voice_profiles/*.yaml` rather than agent code, consistent with the principle. Worth noting because it expands the prompts directory layout.
4. **The existing `Stylistic-Update-Plan.md` is a v2 redesign of the rubric.** This plan is partly a *third* iteration of related work. The relationship should be made explicit in v1 documentation: the stylistic-update plan addresses *what literary fiction should look like*; this plan addresses *which fiction should be literary*.
5. **One-agent-one-tool.** The new `voice_profile_selector` is a thin classifier — close to but not strictly conforming to the one-agent-one-tool pattern (it's a one-tool node, but very small). An alternative is folding selection into `genre_normalizer`, accepting that `genre_normalizer` then does two things. WI-2 below proposes the new node as the default with the alternative noted.

---

## 4. Work Item Ordering

```
WI-1 (schema + profiles)
   │
   ├──► WI-2 (selector)
   │      │
   │      ▼
   ├──► WI-3 (storywriter integration)
   │
   ├──► WI-4 (logging hygiene)             [independent]
   ├──► WI-5 (stylometric counter)         [independent]
   └──► WI-6 (docs)                        [last; reflects WI-1..5]
```

WI-4 and WI-5 are independent and can be done at any point, but landing WI-4 before WI-1 makes future log analysis cleaner.

---

## 5. WI-1: VoiceProfile Schema and Data Files

### Rationale

The voice profile is the new contract. It defines what differentiates one voice from another in a way the prose-writer prompt can consume. Schema-first ensures the rest of the work items have a stable target.

### Files Affected

| File | Action |
|---|---|
| `src/storymesh/schemas/voice_profile.py` | CREATE |
| `src/storymesh/versioning/schemas.py` | ADD `VOICE_PROFILE_SCHEMA_VERSION` |
| `src/storymesh/prompts/voice_profiles/literary_restraint.yaml` | CREATE |
| `src/storymesh/prompts/voice_profiles/cozy_warmth.yaml` | CREATE |
| `src/storymesh/prompts/voice_profiles/genre_active.yaml` | CREATE |
| `tests/test_schemas_voice_profile.py` | CREATE |
| `tests/test_voice_profile_loader.py` | CREATE |

### Schema Sketch (for discussion, not implementation)

```python
VOICE_PROFILE_SCHEMA_VERSION = "1.0"

class VoiceProfile(BaseModel):
    """A named voice profile that conditions prose generation.

    Loaded from src/storymesh/prompts/voice_profiles/<id>.yaml.
    Carried through StoryMeshState and consumed by prose-stage agents.
    """
    model_config = {"frozen": True}

    id: str = Field(min_length=1)                # e.g. "cozy_warmth"
    description: str = Field(min_length=10)      # one-paragraph human-readable
    tone_keywords: list[str] = Field(min_length=1)
    genre_keywords: list[str] = Field(default_factory=list)

    # Overlay text injected into story_writer_draft.yaml at three placeholders:
    craft_overlay: str = Field(min_length=10)    # additions/replacements to CRAFT PRINCIPLES
    avoid_overlay: str = Field(min_length=10)    # additions/replacements to AVOID
    exemplars: list[str] = Field(min_length=2)   # 2–4 exemplar `opens_with` sentences in this voice

    schema_version: str = VOICE_PROFILE_SCHEMA_VERSION
```

### Profile content notes

Each YAML profile is a complete data record. The `craft_overlay` and `avoid_overlay` are not full prompt rewrites — they are surgical additions and overrides. Examples (illustrative, not final text):

- `literary_restraint.craft_overlay`: empty (this is the current default).
- `literary_restraint.avoid_overlay`: empty (the current avoid list is preserved).
- `cozy_warmth.craft_overlay`: includes "Direct emotion-naming is welcome — 'Wren felt warm' is a complete sentence and does not need to be performed by an inanimate comparator." and "Repetition is part of the form — repeating a refrain or sensory anchor is comforting, not a tic."
- `cozy_warmth.avoid_overlay`: extends to include 'the way X is when Y' template, philosophical paragraph cappers, dark imagery (decay, ruin, sterility).
- `genre_active.craft_overlay`: includes "Kinetic over interior. If a moment can be conveyed through movement or dialogue, prefer that over interior observation."
- `genre_active.avoid_overlay`: extends to include cascading subordinate-clause structures, sentence-fragment paragraphs used as philosophical beats.

### Testing

```
TestVoiceProfileSchema:
  - test_valid_construction
  - test_frozen
  - test_id_pattern: id must be lowercase_snake_case
  - test_exemplars_min_length

TestVoiceProfileLoader:
  - test_loads_all_three_profiles
  - test_unknown_profile_raises
  - test_overlay_fields_non_empty (literary_restraint may be empty by design — assert explicitly)
```

---

## 6. WI-2: VoiceProfileSelector Agent

### Rationale

A new pipeline node, `voice_profile_selector`, runs after `genre_normalizer` and selects exactly one `VoiceProfile` per run. The classifier sees the user prompt, normalized genres, and tone tokens, and selects the best-fit profile. Defaults to `literary_restraint` on classifier failure or ambiguity, since that is the existing observed behavior and preserves backward compatibility.

### Alternative: fold into genre_normalizer

This is viable. It keeps node count down and avoids a new agent. The trade-off is `genre_normalizer` ends up with two responsibilities. **Default proposal: new node, for cleaner contracts and easier ablation.** Decision flag for review.

### Files Affected

| File | Action |
|---|---|
| `src/storymesh/agents/voice_profile_selector/__init__.py` | CREATE |
| `src/storymesh/agents/voice_profile_selector/agent.py` | CREATE |
| `src/storymesh/schemas/voice_profile_selector.py` | CREATE (input/output schemas) |
| `src/storymesh/prompts/voice_profile_selector.yaml` | CREATE |
| `src/storymesh/orchestration/graph.py` | EDIT (insert node between genre_normalizer and book_fetcher) |
| `src/storymesh/orchestration/state.py` | EDIT (add `voice_profile: VoiceProfile \| None`) |
| `src/storymesh/config.py` | EDIT (default agent config block) |
| `storymesh.config.yaml` | EDIT (config block) |
| `tests/test_voice_profile_selector_agent.py` | CREATE |

### Behavior

Input: user_prompt (str), normalized_genres (list[str]), user_tones (list[str]), available_profile_ids (list[str]).
Output: VoiceProfileSelectorAgentOutput with `selected_profile_id: str` and `rationale: str` (1–2 sentences for auditability).

The classifier prompt explicitly enumerates the available profiles with their descriptions and tone/genre keywords, and asks for a single profile id back. Temperature 0 for deterministic selection.

Failure handling: on parse failure or unknown profile id, log a warning and default to `literary_restraint`. The pipeline must not crash on selector failure.

### Testing

```
TestVoiceProfileSelectorAgent:
  - test_obvious_cozy_input_selects_cozy_warmth
  - test_obvious_dark_input_selects_literary_restraint
  - test_obvious_action_input_selects_genre_active
  - test_unknown_profile_id_defaults_to_literary_restraint
  - test_llm_failure_defaults_to_literary_restraint
  - test_rationale_is_recorded
```

A small offline fixture set of (input → expected_profile) pairs covers regression. Real-API tests behind the `real_api` marker.

---

## 7. WI-3: Voice-Profile-Aware StoryWriter Prompts

### Rationale

The existing `story_writer_draft.yaml` becomes the **base** prompt. Three placeholder regions accept overlay text from the active `VoiceProfile`. Same approach for `story_writer_outline.yaml` (for exemplar overlays) and a more limited overlay for `story_writer_summary.yaml` (back-cover copy register).

### Files Affected

| File | Action |
|---|---|
| `src/storymesh/prompts/story_writer_draft.yaml` | EDIT (introduce overlay placeholders) |
| `src/storymesh/prompts/story_writer_outline.yaml` | EDIT (exemplars region becomes a placeholder) |
| `src/storymesh/prompts/story_writer_summary.yaml` | EDIT (small register overlay) |
| `src/storymesh/agents/story_writer/agent.py` | EDIT (accept VoiceProfile in input, render overlays into prompts) |
| `src/storymesh/schemas/story_writer.py` | EDIT (add `voice_profile: VoiceProfile` to `StoryWriterAgentInput`) |
| `src/storymesh/orchestration/graph.py` | EDIT (story_writer node reads voice_profile from state) |
| `src/storymesh/agents/resonance_reviewer/agent.py` | EDIT (accept voice_profile, pass through to revision-pass prompt) |
| `tests/test_story_writer_agent.py` | EDIT (add cases per profile) |

### Placeholder design

In `story_writer_draft.yaml`:

```yaml
system: |
  You are StoryWriterAgent. ...
  ...
  CRAFT PRINCIPLES
    1. Concrete over generic. ...
    [...existing principles 1-9...]

    {craft_overlay}    # ← injected from VoiceProfile.craft_overlay; may be empty

  AVOID
    [...existing avoid items...]

    {avoid_overlay}    # ← injected from VoiceProfile.avoid_overlay; may be empty
```

Empty overlay strings produce a prompt byte-equivalent to today's. This is critical: `literary_restraint` runs must produce **identical** prompts to current runs to confirm no regression in the existing voice.

### Backward-compatibility test

```
TestStoryWriterAgent_BackwardCompat:
  - test_literary_restraint_prompt_md5_equals_current_md5
    # Loads the prompt rendered with literary_restraint and asserts byte-equality
    # to a snapshot of the current prompt. This protects the existing literary
    # voice from accidental drift during refactor.
```

### Testing

```
TestStoryWriterAgent_VoiceProfiles:
  - test_cozy_warmth_overlays_present_in_rendered_prompt
  - test_genre_active_overlays_present_in_rendered_prompt
  - test_outline_exemplars_use_profile_exemplars
  - test_resonance_reviewer_revision_passes_profile_through
```

Real-API smoke test (behind `real_api`) generates short drafts under each profile and asserts the stylometric counter (WI-5) flags fewer "the way X when Y" similes for `cozy_warmth` than for the current default. Exact thresholds tuned during eval rounds.

---

## 8. WI-4: Logging Hygiene — Distinct Agent Identities

### Rationale

In `*_llm_calls.jsonl`, three distinct roles all log under `agent: "story_writer"`:

- Scene outliner (T=0.5)
- Prose drafter (T=0.8)
- Back-cover copy writer (T=0.4)

This forces analysis to filter on temperature, which is fragile (any future temperature tuning breaks downstream tooling). Distinct agent identifiers fix it.

### Files Affected

| File | Action |
|---|---|
| `src/storymesh/agents/story_writer/agent.py` | EDIT (use distinct agent_name per LLM call) |
| `src/storymesh/llm/base.py` or `current_run` ContextVar | VERIFY (agent_name override per call is supported; per `LLM-Call-Logging.md` it should be) |
| `tests/test_story_writer_agent.py` | EDIT (assert correct agent_name in mock LLMClient calls) |

### Proposed names

| Old | New |
|---|---|
| `story_writer` (T=0.5) | `story_writer_outline` |
| `story_writer` (T=0.8) | `story_writer_draft` |
| `story_writer` (T=0.4) | `story_writer_summary` |

This matches the existing prompt file names, which is a useful coincidence — anyone reading logs and grepping for the prompt now has a direct path.

### Backward-compatibility note

Any existing log-analysis tooling that filters on `agent == "story_writer"` will break. The plan should include an `inspect-run` CLI update if such tooling exists.

### Testing

```
TestStoryWriterAgent_AgentNames:
  - test_outline_call_logs_as_story_writer_outline
  - test_draft_call_logs_as_story_writer_draft
  - test_summary_call_logs_as_story_writer_summary
```

---

## 9. WI-5: Stylometric Counter (Offline Diagnostic)

### Rationale

Per-run sanity check on tic frequency. No LLM calls, no pipeline integration, no pass/fail. Output is informational — a human reviewing a draft can glance at the counts to decide whether the voice profile produced what was intended.

### Files Affected

| File | Action |
|---|---|
| `src/storymesh/diagnostics/__init__.py` | CREATE |
| `src/storymesh/diagnostics/stylometric_counter.py` | CREATE |
| `src/storymesh/cli.py` | EDIT (add `storymesh stylometrics <run_id>` subcommand) |
| `tests/test_stylometric_counter.py` | CREATE |

### Counted tics

Each tic has a name, a regex (or rule), and a count. The output is a JSON record per run:

| Tic name | Detection rule (sketch) |
|---|---|
| `cascading_which_was` | sentences containing two or more `,\s*which\s+(was\|were\|had\|is\|are)` clauses |
| `proxy_feeling_simile` | `\bthe\s+way\s+(\w+\s+){0,4}(when\|that\|something)\b` (with refinement) |
| `negation_triplet` | three consecutive sentences each starting with `Not\b` or containing `\bnot\s+\w+,\s+not\s+\w+` |
| `sentence_fragment_paragraph_rate` | paragraphs whose word count is below 6, as a fraction of all paragraphs |
| `as_if_as_though` | count of `\bas\s+(if\|though)\b` |
| `numerical_precision_atmospheric` | bare numerals (digits or spelled-out small integers) appearing as standalone declarative sentences (`^[A-Z][^.]*\b\d+\b[^.]*\.$` with refinement) |
| `which_was_chain_depth_max` | maximum count of `, which (was/were/...)` in any single sentence |

The list is intentionally small. Each tic is a string-pattern question the user already knows is interesting.

### Output

```json
{
  "run_id": "...",
  "story_title": "...",
  "voice_profile": "literary_restraint",
  "word_count": 3120,
  "tics": {
    "cascading_which_was": {"count": 14, "per_1000_words": 4.49},
    "proxy_feeling_simile": {"count": 7, "per_1000_words": 2.24},
    "negation_triplet": {"count": 3, "per_1000_words": 0.96},
    "sentence_fragment_paragraph_rate": {"value": 0.18},
    "as_if_as_though": {"count": 2, "per_1000_words": 0.64},
    "numerical_precision_atmospheric": {"count": 9, "per_1000_words": 2.88},
    "which_was_chain_depth_max": {"value": 3}
  }
}
```

### CLI usage

```
storymesh stylometrics <run_id>            # prints JSON to stdout
storymesh stylometrics <run_id> --pretty   # human-readable table
storymesh stylometrics --all               # bulk over all runs in ~/.storymesh/runs
```

### Testing

```
TestStylometricCounter:
  - test_cascading_which_was_detects_known_examples
  - test_proxy_feeling_simile_detects_known_examples
  - test_proxy_feeling_simile_does_not_match_innocent_uses
  - test_word_count_normalization
  - test_handles_empty_draft
  - test_handles_unicode_punctuation (em-dashes, smart quotes)
```

Each detection rule is unit-tested against a small fixture corpus drawn from the five existing PDFs (positive cases) and from a few hand-written innocent paragraphs (negative cases).

### Explicit non-goals

- No pipeline integration. The counter never blocks a run and never feeds back into the model.
- No pass/fail thresholds. Numbers are informational.
- No semantic detection. Pattern matching only — false positives and false negatives are expected and acceptable.

---

## 10. WI-6: Documentation Refresh

### Rationale

The README is materially out of date and `CLAUDE.md` over-promises its currency. WI-6 fixes both and records the voice-profile architecture decision.

### Files Affected

| File | Action |
|---|---|
| `README.md` | EDIT (status, architecture, roadmap) |
| `.claude/CLAUDE.md` | EDIT (soften the "always up to date" claim, add voice-profile callout) |
| `plans/Voice-Profile-Plan.md` | EXISTS (this doc) |

### README changes

1. Move `StoryWriterAgent` from "scaffolded but not implemented" to "Implemented" with a description of the three-pass design.
2. Add `ResonanceReviewerAgent` as Stage 6b.
3. Add `VoiceProfileSelector` as Stage 0.5 (or wherever the new node lands), with description.
4. Update the example pipeline-output stage table.
5. Add a "Voice Profiles" section explaining the three profiles, when each is selected, and how to override (config flag).
6. Roadmap: add "Per-profile rubric judging" as a known follow-up.

### CLAUDE.md changes

1. Replace `"This file is always up to date"` with `"This file is the canonical reference for component scope; verify against the source tree if anything looks stale."` — accurate to what the README is for, without claiming a property that isn't enforced.
2. Add a one-paragraph note about voice profiles, pointing readers to this plan.

### Testing

Documentation has no unit tests. A reviewer should:

- Run `storymesh generate "a bedtime story about a friendly cloud"` and confirm `voice_profile=cozy_warmth` in the run artifacts.
- Run `storymesh generate "dark post-apocalyptic detective mystery"` and confirm `voice_profile=literary_restraint`.
- Spot-check that the README's pipeline diagram matches the actual graph wiring.

---

## 11. Validation Checklist

Before merging:

- [ ] `literary_restraint` profile produces a prose-draft prompt byte-identical to the current prompt (snapshot test).
- [ ] All three voice profiles load without error and pass schema validation.
- [ ] `voice_profile_selector` correctly classifies a fixture set of 12+ inputs across all three profiles.
- [ ] `voice_profile_selector` defaults to `literary_restraint` on LLM failure.
- [ ] `StoryWriterAgent` accepts a `VoiceProfile` and renders profile overlays into all three prompts.
- [ ] `ResonanceReviewerAgent` accepts and respects voice_profile during revision.
- [ ] LLM call logs distinguish `story_writer_outline`, `story_writer_draft`, `story_writer_summary` as distinct agents.
- [ ] Stylometric counter produces stable output on the five reference PDFs and matches hand-counted ground-truth within tolerance.
- [ ] `storymesh stylometrics` CLI subcommand works on past runs.
- [ ] README accurately describes implemented stages and the voice-profile mechanism.
- [ ] `CLAUDE.md` claim about README currency is corrected.
- [ ] All new tests pass; existing tests still pass; `mypy --strict` clean; `ruff` clean.
- [ ] Real-API smoke run confirms cozy generation reads cozy and not literary-restraint-with-bedtime-trappings.

---

## Open Questions

The following are flagged for discussion rather than decided in this plan:

1. **Where does `voice_profile_selector` live in the graph?** New node (proposed) vs. fold into `genre_normalizer`.
2. **Should the user be able to override the selector?** A config flag `voice_profile_override: cozy_warmth` would let the user force a profile for testing or stylistic preference. Low effort, high debug value. Probably yes, but worth confirming.
3. **Should `ThemeExtractorAgent` adapt to voice profile?** Out of scope here, but a small change (skip dialectical synthesis for `cozy_warmth` and produce a softer thematic frame instead) might be worth a half-day spike. Defer to v2.
4. **What's the right home for `VoiceProfile` schema and YAMLs?** Schema in `src/storymesh/schemas/voice_profile.py` (proposed) vs. inside `prompts/`. The split between schema and data is consistent with other parts of the repo, but the YAMLs are arguably *prompt* data, so they live under `prompts/voice_profiles/`. Confirm.