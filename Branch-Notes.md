Mostly placeholder for things related to this branch.

I considered the state of the code at this point to be roughly "Complete", in that I have constructed an end-to-end pipeline with extensive observability and production of deliverable outputs.

I began this project wanting to produce short story proposals, and have meandered my way into medium length (10+ page) short fiction. I have learned much, but I think there are some very interesting areas that this project could explore or even look for publishing opportunities:

---

The following content was formatted by Claude.

**Date:** 2026-05-03
**Status:** Exploratory — for author's planning use, not an implementation plan

This document summarizes four candidate research directions identified during review of the StoryMesh pipeline and its first portfolio of generated stories (`The Floor of the Sky`, `The Patient Frequency`, `The Body of the Place`, `The Golden Coaster`, `StarTrek_Fanfic`/Tuesday Variable). Each direction is evaluated by what already exists in the project, what would still need to be built or collected, expected effort, and the realistic publication outcome.

The four are sorted roughly by ratio of contribution-to-effort given the project's remaining time. A recommendation is given at the end.

---

## Table of Contents

1. [Stylistic Homogenization Testbed](#1-stylistic-homogenization-testbed-strongest-candidate)
2. [Dialectical-Tensions Ablation](#2-dialectical-tensions-ablation)
3. [Restraint-vs-Avoidance as LLM Critical Reading](#3-restraint-vs-avoidance-as-llm-critical-reading)
4. [Voice Convergence as a Short Note](#4-voice-convergence-as-a-short-note)
5. [Recommendation](#5-recommendation)
6. [What to Avoid](#6-what-to-avoid)

---

## 1. Stylistic Homogenization Testbed (strongest candidate)

**What it is.** Use the pipeline as a measurement instrument. Systematically vary specific prompt-design decisions — avoid-list contents, few-shot exemplar register, craft-principle phrasings, single-shot vs scene-local prose generation — and measure their effect on cross-output stylistic diversity. The voice convergence finding (Section 4) becomes the motivating example; the systematic study becomes the contribution.

**What the project already has.** Full LLM-call logs at every stage, structured artifacts, Pydantic contracts that make swapping components clean, multi-provider support, a stylometric counter (in plan), and a portfolio of comparable runs.

**What still needs to be built or collected.**

- An experimental design — pick 3 prompt-design decisions to vary
- N ≥ 5 stories per condition, ideally across a fixed set of 5 seed prompts to control for input variation
- Additional stylometric metrics beyond what's planned: lexical diversity, sentence-length distribution, simile-construction variety
- A writeup framing the result as "which prompt-design decisions induce attractor states in output space"

**Effort.** 2–3 weeks for the smaller version; 4–6 weeks if per-scene generation is included as a treatment condition.

**Contribution.** Documents a phenomenon — cross-output convergence in long-form creative generation — that the field is increasingly susceptible to as prompts get more sophisticated, with a reproducible methodology. Methodologically novel because almost nobody evaluates LLM creative writing across outputs rather than per output.

**Realistic outcome.** Workshop paper (NeurIPS Creativity & Design, ACL/EMNLP creative-NLG tracks, CHI's creative-AI venues), or a high-quality blog post that gets attention. Strong thesis chapter regardless.

---

## 2. Dialectical-Tensions Ablation

**What it is.** Empirical validation of whether `ThemeExtractorAgent`'s dialectical-synthesis framework actually produces measurably different stories than a single-tradition baseline. The current architecture asserts that it does; this would test the assertion.

**What the project already has.** The framework as one configurable path, the rubric infrastructure to score outputs.

**What still needs to be built or collected.**

- A control prompt that produces proposals from a single genre tradition without the contradiction-finding step
- Paired runs across the same seeds, varying only the theme-extraction path
- A comparison framework — the existing rubric plus structural metrics (specificity density, protagonist-flaw uniqueness)

**Effort.** 1–2 weeks.

**Contribution.** Modest but solid — validates an existing architectural choice with evidence. Less novel than #1 because it confirms a hypothesis rather than opening a new question.

**Realistic outcome.** Thesis chapter; possibly bundled with #1 as the "validation" half of a longer writeup.

---

## 3. Restraint-vs-Avoidance as LLM Critical Reading

**What it is.** A study of whether LLMs can reliably distinguish earned silence (restraint) from missed opportunity (avoidance) in fiction, and where they systematically disagree with humans. The `ResonanceReviewerAgent` already encodes this distinction; this study would test whether the encoding is sound.

**What the project already has.** An agent that already does the discrimination, prompts that articulate the distinction.

**What still needs to be built or collected.**

- A hand-annotated dataset of 50–100 fiction passages from published work (ideally not LLM-generated), each labeled as restraint / avoidance / neither by 2–3 human raters
- Inter-rater agreement analysis among humans, then between humans and the model
- A few prompt variants of the discriminator to study sensitivity

**Effort.** 3–4 weeks. The annotation is the bottleneck and not negotiable — without human ground truth there is no contribution.

**Contribution.** New finding about LLM critical-reading capability. This is a *different paper* than the others — evaluation-methodology research rather than creative-pipeline research. Different audience, different venues.

**Realistic outcome.** NLP venue interested in evaluation (EMNLP findings, ACL findings, evaluation workshops). Standalone work.

---

## 4. Voice Convergence as a Short Note

**What it is.** The specific mechanical finding from this conversation: avoid-list bans on direct emotion-naming plus "as if" simultaneously route the model into a third construction (`"the way X is when Y"`) that becomes a portfolio-wide tic. Reproducible, generalizable, undocumented.

**What the project already has.** The finding, the evidence (five PDFs and the byte-identical prompt diagnosis), and the prompt-engineering specifics.

**What still needs to be built or collected.**

- A clean writeup
- One or two controlled reproductions on a different seed prompt or a different model, to show the mechanism generalizes

**Effort.** About a week.

**Contribution.** Small but real — a reproducible empirical observation about prompt-engineering side effects.

**Realistic outcome.** Blog post, arXiv preprint, or short workshop note. Not a thesis on its own but a useful artifact and a strong supporting piece.

---
