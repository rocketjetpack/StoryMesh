# Proposal: Redesigning the StoryMesh Prose Pipeline for More Human-Style, High-Quality Fiction

## Executive summary

The current pipeline is already strong at producing coherent, controlled, literary-leaning prose. The main weakness is not raw quality; it is **over-optimization**. Across the pipeline, the system tends to decide too early what the story means, how it resolves, and which details matter. That produces prose that is polished, structurally sound, and thematically legible, but sometimes slightly too engineered.

The most important fix is not to keep adding more style instructions to the final prose prompt. The important fix is to **move ambiguity, exploratory pressure, and interpretive space upstream**, so the later stages can write as if they are discovering the story rather than merely executing it.

The strongest outputs so far suggest that the system works best when it leans on:
- concrete procedural detail
- specific physical routine
- restrained but not over-explained emotion
- unresolved tension
- small irregularities in rhythm and interpretation

The weakest outputs tend to happen when the pipeline produces:
- clean thesis statements too early
- over-explicit thematic summaries
- symmetrical scene logic
- prose that explains itself one layer too much

This document proposes a staged redesign, in priority order, that preserves the strengths of the current system while reducing the “too clean” effect.

---

## What the sample outputs teach us

### What works

The ant story succeeds most when it is doing procedure and perception: measuring tunnels, tracking changes, recording discrepancies, and using physical notation to hold uncertainty in place. The best moments are those where the story’s logic is embodied in action rather than explained in abstraction.

The ship story succeeds most when it grounds the crisis in ordinary work: counting canisters, logging data, watching a meeting on a small maintenance screen, cleaning a room, adjusting a cart wheel, and finally sitting in the chair that she has always cleaned around. These details make the story feel inhabited.

### What still feels over-managed

Both stories occasionally turn observation into a clean interpretive sentence. The prose knows too exactly what is happening, what it means, and how to phrase that meaning. The result is literary competence with a slightly synthetic edge.

That is the key diagnosis:

> The pipeline is already good at clarity. It needs help generating *humanly uneven clarity*.

---

## Core design principle for the revision

### Stop over-constraining meaning before the prose stage

A human writer often discovers a story while writing it. Your current pipeline tends to solve the story before the prose stage begins. That makes the final writer model function like a renderer, not a discoverer.

The best redesign is therefore to:
1. keep structure,
2. reduce premature interpretation,
3. preserve concrete anchors,
4. allow small local uncertainties,
5. and let the prose stage carry more of the act of discovery.

---

## Recommended order of changes

### 1. `proposal_draft_generate.yaml`
This is the highest-leverage file.

Why first:
- It currently establishes the story’s philosophical shape too early.
- It asks for a “thematic_thesis” as a clean answer.
- It asks for a plot arc that already knows its best shape.
- It encourages the story to behave like a deliberate argument rather than an experience.

### 2. `rubric_judge.yaml`
This is the optimizer.

Why second:
- It selects for polished, legible, restrained output.
- That is useful, but it can accidentally eliminate residue, drift, and exploratory texture.
- If the judge favors clean structure too heavily, the rest of the pipeline will keep producing the same kind of prose.

### 3. `story_writer_outline.yaml`
This is where interpretation becomes scene-level law.

Why third:
- The outline should scaffold scenes, not explain them.
- If each scene already contains a tight thematic reading, the final prose stage has little room to breathe.

### 4. `story_writer_draft.yaml`
This is already closer to the right direction.

Why last:
- It already contains useful ideas: controlled imperfection, cognitive friction, and anti-compression.
- More complexity here will not solve upstream over-clarity.
- The final draft prompt should be the lightest touch, not the main place where the system fights itself.

---

## File-by-file recommendations

# 1) `proposal_draft_generate.yaml`

## Current role
The proposal generator is doing too much:
- developing the seed
- assigning the protagonist
- shaping the arc
- articulating the thesis
- and implicitly deciding the story’s meaning

This is too much certainty too early.

## Problems to address

### A. `thematic_thesis` is too declarative
The current framing treats the thesis as the story’s philosophical answer. That is excellent for a pitch document, but not ideal for prose generation. It encourages the pipeline to write from a conclusion.

#### Why this is a problem
A clear thesis is not the same as a human story engine. The more the system understands the story as an argument, the more it will produce elegant, complete, interpretive prose that feels authored by a planning process.

#### Proposed change
Replace or soften `thematic_thesis` with something like `thematic_pressure`, `thematic_drift`, or `situational_claim`.

The field should describe:
- what the story keeps circling around
- what kind of contradiction it cannot fully settle
- what pressure acts on the story

It should **not** read like a final philosophical verdict.

#### Why this helps
It preserves thematic coherence while reducing over-closure. The prose can still point toward meaning, but it will not be forced to summarize meaning in advance.

---

### B. `proposed_answer` is likely too clean
Any field that asks for the story’s “answer” pushes the model toward resolution thinking.

#### Proposed change
Rename or redefine this as a `provisional_direction` or `possible_reading`.

It should feel tentative, partial, or situational.

#### Why this helps
It permits stories that lean toward a meaning without fully locking into one. That is closer to how many human stories feel: not indecisive, but not over-explained.

---

### C. Add explicit unknowns
The pipeline currently rewards what the story knows. It should also preserve what the story does not know.

#### Proposed change
Add an `unknowns` field or section containing 2–3 unresolved questions or incomplete pressures.

Examples:
- what the protagonist cannot verify
- what the world resists explaining
- what remains ambiguous even after the story ends

#### Why this helps
Human stories often retain residue. Unknowns create residue without requiring sloppiness. They are controlled ambiguity, which is a major ingredient in natural-feeling prose.

---

### D. Reduce the mandatory “3-act plot arc” feel
The current prompt asks for a clean arc with named scenes and concrete turning points.

#### Proposed change
Keep structure, but allow the proposal to emphasize:
- accumulation
- pressure shifts
- reversals of attention
- procedural problem-solving
- local surprise

The proposal should not read like a beat sheet that already knows its ending.

#### Why this helps
It makes the story feel less like a designed machine and more like an event that unfolds under constraint.

---

### E. Keep the cover-image prompt separate from narrative development
The image prompt instructions are good, but they can distract from story shaping.

#### Proposed change
Keep the image prompt requirements, but make sure they remain clearly downstream from the story’s narrative logic.

#### Why this helps
It prevents cover design constraints from influencing the story’s conceptual shape too strongly.

---

## Recommended wording direction for `proposal_draft_generate.yaml`

The key shift is:
- from **“develop the seed into a complete story proposal with thesis and answer”**
- to **“develop the seed into a story proposal with pressure, direction, and unresolved questions”**

That single move would reduce a lot of downstream neatness.

---

# 2) `rubric_judge.yaml`

## Current role
The rubric judge is a major shaping force. It determines which candidate proposals survive, which means it determines which style of story the system learns to prefer.

## Likely failure mode
The judge appears to reward:
- restraint
- specificity
- convention followed by one meaningful departure
- structure that reads cleanly and intelligently

Those are all good. The issue is balance. When judged too heavily, they create stories that are tasteful but over-controlled.

---

## Problems to address

### A. It may reward over-clarity
A proposal that neatly states its argument is often easier to judge as “strong,” even if it produces slightly synthetic fiction.

#### Proposed change
Add an explicit penalty for overdetermination.

The rubric should penalize proposals that:
- state what the story means too directly
- resolve the central question too completely
- sound like a philosophical conclusion rather than a fiction premise

#### Why this helps
It shifts the evaluator away from preferring perfect intellectual closure and toward preferring interpretive space.

---

### B. It likely over-rewards clean structural novelty
The current system may be rewarding the story for having one elegant departure from convention.

#### Proposed change
Reframe novelty as something that can emerge from:
- situation
- perspective
- texture
- procedural detail
- awkwardness in social interaction

Do not require a single “big clever twist” as the main source of originality.

#### Why this helps
Big structural cleverness often reads as engineered. Small local originality feels more human.

---

### C. Add a texture / residue category
This is one of the most important changes.

#### Proposed change
Add a rubric category that asks whether the proposal allows for:
- non-functional detail
- unresolved observation
- a little unevenness
- moments that are not fully optimized for plot or thesis

#### Why this helps
This directly protects the kind of residue that makes prose feel authored rather than generated.

---

### D. Reward instability of thought, not just structure
The judge should notice whether a proposal supports hesitation, partial knowledge, or procedural discovery.

#### Proposed change
Add a criterion for cognitive friction:
- re-checking
- revisiting
- uncertain interpretation
- delayed understanding
- attention that moves away and comes back

#### Why this helps
This is one of the strongest human-feeling signals in the stories you already produced.

---

## Recommended wording direction for `rubric_judge.yaml`

The rubric should ask not only:
- Is this coherent?
- Is this specific?
- Is this structurally satisfying?

It should also ask:
- Does this leave room to sound lived-in?
- Does this permit small irregularities?
- Does this avoid explaining itself into flatness?

---

# 3) `story_writer_outline.yaml`

## Current role
The outline is likely the place where a lot of interpretation becomes locked into scene-level requirements.

## Problems to address

### A. `thematic_function` may be too interpretive
This field is useful, but it may be too strong.

#### Proposed change
Replace or soften `thematic_function` into a lighter field such as:
- `narrative_pressure`
- `scene_force`
- `pressure_point`

The field should describe what is happening in the scene and what pressure it applies, but not summarize what it means.

#### Why this helps
The prose writer gets a scaffold, not a pre-decoded reading.

---

### B. Scene summaries may be doing too much interpretation
If summaries are physically accurate but also secretly analytical, they reduce the discovery space later.

#### Proposed change
Keep summaries concrete and behavioral. Avoid embedding a fully worked-out reading of the scene.

#### Why this helps
The prose stage can then interpret the scene through language, rhythm, and detail rather than being handed the interpretation in advance.

---

### C. Add an observational anchor
This is a useful addition.

#### Proposed change
Each scene outline should include a concrete `observational_anchor`:
- an object
- a repeated procedure
- a room feature
- a physical routine
- a sensory detail

#### Why this helps
It gives the prose a stable physical object to return to. That tends to improve human-feeling continuity and prevents scenes from floating into pure abstraction.

---

### D. Add flexibility notes
An outline should be a scaffold, not a prison.

#### Proposed change
Add a short `flexibility_note` or `drift_allowance` section that tells the prose model it may:
- linger longer than expected
- compress unimportant material
- add minor local detail
- shift rhythm slightly

#### Why this helps
This encourages the final prose to breathe.

---

## Recommended wording direction for `story_writer_outline.yaml`

The outline should answer:
- What is physically happening?
- What pressure is this scene under?
- What concrete detail grounds it?

It should avoid becoming a miniature essay about what the scene means.

---

# 4) `story_writer_draft.yaml`

## Current role
This is the most successful prompt stage so far. It already contains many of the right ideas:
- controlled imperfection
- cognitive friction
- anti-compression
- concrete detail
- variation in rhythm

## Recommendation
Do **not** make this much longer or more complicated.

### Why
The remaining issues appear to be mostly upstream. If the final prose prompt becomes too crowded, it may just fight the consequences of earlier over-clarity instead of solving them.

---

## What to keep
The following sections are worth keeping:
- priority order
- controlled imperfection
- cognitive friction
- leaving space
- avoiding clean philosophical summaries
- using procedure for instability

These are all useful.

---

## What to simplify or tighten

### A. Reduce repeated restatements
The prompt currently says many versions of the same thing: don’t be too clean, don’t over-explain, vary rhythm, leave space.

#### Why this matters
Repeated instruction can make the model over-literal and self-conscious.

#### Proposed change
Compress related principles into fewer, sharper statements.

---

### B. Keep the “human” guidance, but make it less self-conscious
Too much explicit anti-AI rhetoric can sometimes make the model overperform human style.

#### Proposed change
Keep the guidance focused on craft rather than on diagnosing machine behavior in every paragraph.

#### Why this helps
It gives the model a target behavior rather than a fear of failure.

---

### C. Preserve the existing strengths
The current prompt already does a good job of:
- discouraging summary prose
- discouraging clean closures
- encouraging concrete detail
- encouraging procedural grounding

These are good instincts and should remain.

---

## Suggested final role for `story_writer_draft.yaml`
This prompt should be the smallest and most focused of the major writing prompts.
Its job is not to rescue the story from upstream over-clarity.
Its job is to render the scene list in prose while preserving irregularity, texture, and unresolved pressure.

---

## Proposed change sequence

### Phase 1: Rewrite `proposal_draft_generate.yaml`
This will produce the biggest structural improvement.

### Phase 2: Rewrite `rubric_judge.yaml`
This will change what kinds of proposals survive.

### Phase 3: Rewrite `story_writer_outline.yaml`
This will loosen how much interpretation gets baked into each scene.

### Phase 4: Trim `story_writer_draft.yaml`
This will keep the final prose stage from overfitting to earlier structure.

---

## What success should look like after the changes

A better pipeline should produce stories where:
- the prose feels observed rather than engineered
- the story does not explain its meaning too quickly
- scenes retain physical grounding
- characters sometimes stall, revisit, or slightly misread things
- not every paragraph feels equally architected
- the final draft has some residue left in it

The goal is not sloppiness.
The goal is **human unevenness under control**.

---

## What not to do

Do not solve this by adding more rules to the final prose prompt alone.
That will likely make the prose more self-aware and not less synthetic.

Do not make every stage more “literary.”
That usually just increases polish and decreases life.

Do not force every story to have a clever departure.
One of the easiest ways to make writing feel generated is to make every piece of writing perform originality in the same way.

---

## Bottom line

The current system is already good at producing coherent, restrained, polished prose. The next level is not more polish. It is permitting the system to remain a little less certain, a little less resolved, and a little more discoverable at every stage before the final prose is written.

The best place to start is `proposal_draft_generate.yaml`.
That is where the pipeline most strongly converts fiction into argument.

