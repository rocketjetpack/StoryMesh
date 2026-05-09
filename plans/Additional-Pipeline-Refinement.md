# Tone Compliance: Implementation Plan

## Problem

The pipeline fails prompts with explicit tonal specifications (silly, high energy, playful,
tense, dark) because it treats tone as reinterpretable. A prompt for a "silly, high energy
bedtime story" produces a quiet, atmospheric, literary piece — technically accomplished but
tonally wrong. The craft is there; the contract with the user is not honoured.

The tonal data already flows through the pipeline correctly:
- `user_tones` is extracted by GenreNormalizerAgent
- ThemePack carries `user_tones_carried` downstream
- Narrative seeds have a `tonal_direction` field
- ProposalDraft receives `USER TONES` in the user prompt

The infrastructure exists. The prompts treat it as a suggestion at every stage. The fix is
three targeted prompt changes, no schema work required.

---

## Evidence

Run `61e90dcf6a05436f93debeb90560cbb9` — prompt: "A bedtime story about a turtle who learns
to skateboard. Cute, fun, high energy, and silly."

StoryMesh produced a quiet, meditative, dreamlike story. Beautiful prose. Completely wrong
tone. The baseline single-call output was demonstrably sillier, higher energy, and more
faithful to what the user asked for, despite weaker writing. D-5 (User Intent Fidelity)
permitted this because its "creative reinterpretation" carve-out applies to tone, which it
should not.

---

## Fix 1 — RubricJudge D-5: Tone Markers Are Non-Negotiable

**File:** `src/storymesh/prompts/styles/default/rubric_judge.yaml`

**What to change:**

The current D-5 ACCEPTABLE tier says "penalize only when the proposal actively contradicts
the user's request, not when it interprets it creatively." This carve-out must not apply
to tone. A proposal that substitutes artistic preference for the user's stated tone has
failed D-5 regardless of how good the writing is.

Specific changes:

1. Add a FAIL condition for tone substitution:
   > The proposal substitutes a different emotional register for the user's specified tone.
   > "Silly" reinterpreted as "whimsical restraint" is a failure. "High energy" reinterpreted
   > as "quiet intensity" is a failure. Tone markers are not aesthetic suggestions — they are
   > the emotional contract of the piece. Creative latitude applies to premise, structure, and
   > setting. It does not apply to the tonal register the user specified.

2. Add a D-5 check:
   > State the user's specified tones. State the proposal's actual emotional register. If they
   > are not the same thing — if you would need to stretch the definition of the user's words
   > to call the proposal compliant — score 0.

3. Preserve the existing creative latitude for genre and narrative context (setting, time
   period, premise angle) — only tone is now non-negotiable.

**Also update:** `styles/slim/rubric_judge.yaml` and `styles/bare_minimum/rubric_judge.yaml`
with proportionally shorter versions of the same principle.

---

## Fix 2 — ProposalDraft: Tone as Architectural Requirement

**Files:**
- `src/storymesh/prompts/styles/default/proposal_draft_generate.yaml`
- `src/storymesh/prompts/styles/default/proposal_draft_retry.yaml`

**What to change:**

Add explicit doctrine after the existing PRINCIPLE 2 (Restraint Where Restraint Serves)
that draws a hard line between aesthetic restraint and tone compliance:

> TONE IS NOT REINTERPRETABLE
> When the user specifies a tonal register — silly, playful, tense, high energy, dark,
> heartwarming — this is the emotional contract of the piece. It is not a suggestion.
> Do not substitute a subtler, more controlled, or more literary version.
>
> "Silly" means demonstrably silly: physical comedy, absurdist logic, character behaviour
> that produces laughter. A restrained story with occasional wry observations is not silly.
>
> "High energy" means kinetic pacing, escalating situations, active characters doing things.
> A quiet atmospheric piece with a single meaningful act is not high energy.
>
> Restraint is a craft tool. Apply it within the requested tone, not instead of it. A silly
> story can have restraint in how it lands a joke. It cannot have restraint in whether it is
> funny.

Also add a tone compliance check to STEP 1:
> Before developing the seed, state the user's specified tones and confirm that the seed's
> tonal_direction honours them directly. If the seed reframes the tone, choose a different
> angle.

**Also update:** `styles/slim/proposal_draft_generate.yaml`,
`styles/slim/proposal_draft_retry.yaml`, and the `bare_minimum` equivalents with
proportionally shorter versions.

---

## Fix 3 — ThemeExtractor: Tonal Direction Is Binding on Seeds

**File:** `src/storymesh/prompts/styles/default/theme_extractor.yaml`

**What to change:**

The current STEP 5 (seed generation) says seeds should "lean into the identified tones if
provided." That phrasing is the problem — "lean into" reads as optional flavouring.

Replace with:

> Each seed's `tonal_direction` must faithfully reflect the user's specified tones. If the
> user asked for "silly and high energy," every seed's tonal_direction must be silly and
> high energy — not "playful with melancholic undertones" or "energetic but grounded." The
> seeds may differ in premise, protagonist, and tension, but they must all operate within
> the requested tonal register.
>
> Tone is not a dial to balance against literary ambition. It is the form the story takes.

**Also update:** `styles/bare_minimum/theme_extractor.yaml` with a shorter version.

---

## Scope and Ordering

All three fixes are prompt-only. No schema changes, no new agents, no new stages. Suggested
order:

1. **Fix 2 first** (ProposalDraft) — highest leverage, catches tone failures before the
   rubric sees them. Retry loop reinforces correct tone on revision.

2. **Fix 1 second** (RubricJudge D-5) — makes the rubric actively reject tone failures
   rather than permitting creative reinterpretation. Will increase retry frequency for
   tone-misaligned proposals, which is correct behaviour.

3. **Fix 3 last** (ThemeExtractor) — tightens the earliest stage of the pipeline. Lower
   urgency because Fixes 1 and 2 will catch seeds that drift from tone, but this prevents
   the drift from happening at all.

---

## What This Does Not Address

The anti-pattern list in `proposal_draft_generate.yaml` was designed for adult literary
fiction and may suppress genre-appropriate elements in other forms. "The protagonist having
an epiphany in the climax" is a legitimate ending for a children's story. "Characters who
are haunted by something" is essential to horror. The anti-pattern list should eventually
be scoped to style, not to form — but that is a separate piece of work.

The Resonance Reviewer's scope (near-miss literary depth) is correct and should not be
changed as part of this work. It operates after the tonal contract is established.
