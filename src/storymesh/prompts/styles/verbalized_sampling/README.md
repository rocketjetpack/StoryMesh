# StoryMesh
## Prompt Style: Verbalized Sampling

Verbalized Sampling (VS) is a prompt-engineering strategy that steers LLMs
toward more diverse creative output by asking them to reason about the
*probability* of their own responses and bias toward low-probability ones.

## Source

Primary reference: <https://arxiv.org/abs/2510.01171>
([PDF](https://arxiv.org/pdf/2510.01171))

## How StoryMesh implements VS

The paper proposes a *strict* form: the model emits N candidates as a list,
each tagged with a `response_prob` field, and the agent selects the
lowest-probability candidate. We implement a *loose* form instead — the
probability lives only in the model's reasoning, never in the JSON, and it
is applied at the **generation** step rather than the selection step.

Concretely, in this style:

- **ThemeExtractor (Stage 3)** is instructed to generate the maximum number
  of narrative seeds (`max_seeds`, default 5), ordered from least to most
  probable, with each seed required to have less than 10% probability of
  being another writer's first idea for the prompt. Tensions get the same
  treatment so the seeds inherit a more diverse contradiction space.

- **ProposalDraftAgent (Stage 4) — generate mode.** Each of the N candidate
  calls is instructed to take the least-probable *structural* angle for
  developing its assigned seed (protagonist archetype, plot shape, point of
  entry, pivotal scenes, image direction). Tone, narrative context, and
  genre intent are fixed — VS surprises the reader at the level of *what
  happens*, not at the level of *what register the story is in*. The Stage 4
  critic and Stage 5 rubric judge then select on craft, not probability.

- **ProposalDraftAgent (Stage 4) — retry mode.** VS still applies, but
  *editorial feedback outranks it*. If the rubric explicitly asks for
  grounding, restraint, or more conventional structure, the retry prompt
  treats that feedback as authoritative and chooses the angle that addresses
  it over the angle that is merely surprising. If the feedback asks for more
  specificity or distance from the default interpretation, VS and the
  feedback are aligned and VS applies in full.

- **StoryWriterAgent (Stage 6) — draft pass.** VS applies *only* at
  structural choice points: how to enter a scene, which beat to dramatise vs.
  compress, which sensory anchor to use, where to leave silence. VS
  explicitly does **not** apply at the sentence or syntax level. Sentence-
  level surprise — rare-word combinations, ornate syntax, lexical fireworks
  — is purple prose, and the existing craft guardrails forbid it. The result:
  scenes can land on unexpected moments, but the prose itself stays
  idiomatic and human.

- **No other stage uses VS.** RubricJudge is an *evaluator*, not a generator;
  applying VS there would corrupt scoring. The resonance reviewer, cover-art
  agent, and book assembler are also untouched. Across the pipeline, surprise
  is the *input*, quality is the *output*. A surprising-but-broken seed,
  proposal, or scene will lose to a more conventional sibling at the critic
  and rubric stages, exactly as it should.

- **The Pydantic contract is unchanged.** `NarrativeSeed`, `StoryProposal`,
  and the story-writer schemas have no probability field — VS is a
  generation-time bias, not a schema-level construct.

## Guardrails wired into the prompts

Every VS-bearing prompt explicitly says that "low probability" means
*structurally surprising*, not *tonally wrong* and not *lexically ornate*.
The model is told it must still respect:

- the user's stated tones (a "silly" prompt yields silly *and* low-probability
  seeds, never literary-restraint seeds dressed up as surprises),
- narrative context tokens (settings, eras, etc.),
- genre intent,
- and at the prose stage, the existing sentence-level craft guardrails
  (idiomatic English, no purple prose, no decorative oddness).

The retry prompt adds one more guardrail: **editorial feedback outranks VS.**
If the rubric judge asks for grounding or restraint, the retry takes that
direction over the default low-probability bias. This prevents a feedback
loop where every revision tries to be more surprising than the last.

Without these guardrails the technique drifts into weirdness-for-its-own-sake,
which the rubric judge will (correctly) penalise downstream.

## Why loose VS rather than strict VS

1. **Cost.** Strict VS requires the model to emit N full ThemePacks in a
   single response. Loose VS produces one ThemePack with N diverse seeds —
   roughly the same token output, much cheaper to generate.
2. **The model cannot meaningfully assign probabilities.** Both forms use
   probability as a steering instrument; only the strict form pretends the
   numbers are real. The loose form is more honest about what is happening.
3. **Downstream selection is already strong.** StoryMesh's Stage 4 critic
   and Stage 5 rubric judge are designed to choose on craft. We do not need
   probability as a tiebreaker.

If the strict form is ever wanted, it would slot in as a small
`unwrap_verbalized_sampling()` helper at the boundary of `LLMClient.complete_json`
that detects `{"candidates": [{"response_prob": …, …}]}` and returns the
lowest-probability candidate before pydantic construction. It is intentionally
not implemented today.
