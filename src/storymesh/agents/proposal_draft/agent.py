"""ProposalDraftAgent — Stage 4 of the StoryMesh pipeline.

Develops narrative seeds into fully realised story proposals using a
multi-sample with self-selection architecture:

1. Generate N candidate proposals, each steered toward a different narrative
   seed, using independent stateless LLM calls at elevated temperature.
2. Evaluate and select the best candidate via a separate low-temperature
   critic call that checks candidates against clichéd resolutions, thematic
   coherence, and tonal alignment.

This implements a propose-evaluate-select decision cycle grounded in the
CoALA framework (Sumers et al., 2024), producing structurally diverse
creative output that a single LLM call cannot achieve.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.proposal_draft import (
    ProposalDraftAgentInput,
    ProposalDraftAgentOutput,
    SelectionRationale,
    StoryProposal,
)

logger = logging.getLogger(__name__)

_ALTERNATE_ANGLE_NOTE = (
    "\nNOTE: This seed was already assigned to a previous candidate. "
    "You MUST take a distinctly different creative angle — same seed, "
    "fundamentally different story premise, protagonist, and plot structure."
)


class ProposalDraftAgent:
    """Develops narrative seeds into full story proposals (Stage 4).

    Uses a multi-sample with self-selection architecture: generates N
    candidate proposals from different narrative seeds, then uses a
    critic call to select the strongest one. This produces structurally
    diverse creative output that a single LLM call cannot achieve.

    Each candidate call is fully independent and stateless — no conversational
    history is shared between calls. Divergence is enforced at the prompt level
    via seed-steering, candidate indexing, and an anti-overlap instruction.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 1.2,
        max_tokens: int = 4096,
        num_candidates: int = 3,
        selection_temperature: float = 0.2,
        selection_max_tokens: int = 2048,
    ) -> None:
        """Construct the agent.

        Args:
            llm_client: LLM client instance. Required — proposal drafting is a
                creative generation task with no deterministic fallback.
            temperature: Sampling temperature for candidate generation. Default
                1.2 maximises creative variance; the selection step acts as a
                quality filter so higher variance is affordable.
            max_tokens: Maximum tokens per candidate proposal call.
            num_candidates: Number of candidate proposals to generate. When
                num_candidates exceeds the number of available seeds, seeds are
                reused in order with an alternate-angle instruction added.
            selection_temperature: Sampling temperature for the selection critic
                call. Low value (default 0.2) produces consistent, analytical
                evaluation rather than creative output.
            selection_max_tokens: Maximum tokens for the selection call.
        """
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_candidates = num_candidates
        self._selection_temperature = selection_temperature
        self._selection_max_tokens = selection_max_tokens

        # Load both prompts eagerly so misconfiguration is caught at
        # construction time, not mid-pipeline.
        self._generate_prompt = load_prompt("proposal_draft_generate")
        self._select_prompt = load_prompt("proposal_draft_select")

    def run(self, input_data: ProposalDraftAgentInput) -> ProposalDraftAgentOutput:
        """Generate candidate proposals and select the strongest one.

        Args:
            input_data: Assembled input from multiple upstream pipeline stages.

        Returns:
            A frozen ProposalDraftAgentOutput with the selected proposal,
            all valid candidates, selection rationale, and debug metadata.

        Raises:
            RuntimeError: If all candidate proposals fail to parse.
        """
        seeds = input_data.narrative_seeds
        num_candidates = self._num_candidates

        logger.info(
            "ProposalDraftAgent starting | seeds=%d candidates=%d temperature=%.1f",
            len(seeds),
            num_candidates,
            self._temperature,
        )

        # Map candidate index → seed_id for debug output.
        # Keys are strings so the dict serialises cleanly to JSON (orjson
        # requires all dict keys to be str).
        seed_assignments: dict[str, str] = {
            str(i): seeds[i % len(seeds)].seed_id for i in range(num_candidates)
        }

        candidates: list[StoryProposal] = []
        parse_failures = 0

        for i in range(num_candidates):
            assigned_seed = seeds[i % len(seeds)]
            additional_seeds = [s for s in seeds if s.seed_id != assigned_seed.seed_id]
            # Candidates that wrap around get an explicit alternate-angle instruction
            # so the model knows the seed is being reused and must diverge.
            alternate_angle_note = _ALTERNATE_ANGLE_NOTE if i >= len(seeds) else ""

            user_prompt_text = self._generate_prompt.format_user(
                candidate_index=i + 1,
                total_candidates=num_candidates,
                alternate_angle_note=alternate_angle_note,
                user_prompt=input_data.user_prompt,
                normalized_genres=input_data.normalized_genres,
                user_tones=input_data.user_tones,
                narrative_context=input_data.narrative_context,
                assigned_seed=orjson.dumps(assigned_seed.model_dump()).decode(),
                additional_seeds=orjson.dumps(
                    [s.model_dump() for s in additional_seeds]
                ).decode(),
                tensions=orjson.dumps(
                    [t.model_dump() for t in input_data.tensions]
                ).decode(),
                genre_clusters=orjson.dumps(
                    [c.model_dump() for c in input_data.genre_clusters]
                ).decode(),
            )

            try:
                response = self._llm_client.complete_json(
                    user_prompt_text,
                    system_prompt=self._generate_prompt.system,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                proposal = StoryProposal(**response)
                candidates.append(proposal)
                logger.debug(
                    "Candidate %d/%d generated | seed=%s title=%r",
                    i + 1,
                    num_candidates,
                    assigned_seed.seed_id,
                    proposal.title,
                )
            except Exception as exc:
                parse_failures += 1
                logger.warning(
                    "Candidate %d/%d failed | seed=%s error=%s",
                    i + 1,
                    num_candidates,
                    assigned_seed.seed_id,
                    exc,
                )

        if not candidates:
            raise RuntimeError(
                "ProposalDraftAgent: all candidate proposals failed parsing."
            )

        # Successful generate calls (each attempted call that raised is a failure;
        # calls not yet attempted never incremented the counter).
        generate_calls = num_candidates  # all were attempted
        total_llm_calls = generate_calls  # start here; add 1 if selection runs

        if len(candidates) < 2:
            logger.warning(
                "ProposalDraftAgent: only 1 valid candidate; skipping selection step."
            )
            rationale = SelectionRationale(
                selected_index=0,
                rationale="Only one valid candidate was generated; selected by default.",
            )
            selected = candidates[0]
        else:
            total_llm_calls += 1
            candidates_json = orjson.dumps(
                [c.model_dump() for c in candidates]
            ).decode()
            tensions_json = orjson.dumps(
                [t.model_dump() for t in input_data.tensions]
            ).decode()

            select_user = self._select_prompt.format_user(
                user_prompt=input_data.user_prompt,
                user_tones=input_data.user_tones,
                tensions=tensions_json,
                candidates=candidates_json,
            )

            try:
                sel_response = self._llm_client.complete_json(
                    select_user,
                    system_prompt=self._select_prompt.system,
                    temperature=self._selection_temperature,
                    max_tokens=self._selection_max_tokens,
                )
                rationale = SelectionRationale(**sel_response)
                if rationale.selected_index >= len(candidates):
                    logger.warning(
                        "ProposalDraftAgent: selected_index %d out of range "
                        "(num_candidates=%d); clamping to 0.",
                        rationale.selected_index,
                        len(candidates),
                    )
                    rationale = SelectionRationale(
                        selected_index=0,
                        rationale=(
                            f"Clamped from out-of-range index "
                            f"{rationale.selected_index}. "
                            f"Original rationale: {rationale.rationale}"
                        ),
                        cliche_violations=rationale.cliche_violations,
                        runner_up_index=None,
                    )
                selected = candidates[rationale.selected_index]
            except Exception as exc:
                logger.warning(
                    "ProposalDraftAgent: selection call failed (%s); "
                    "falling back to candidate 0.",
                    exc,
                )
                rationale = SelectionRationale(
                    selected_index=0,
                    rationale="Selection call failed; candidate 0 selected as fallback.",
                )
                selected = candidates[0]

        logger.info(
            "ProposalDraftAgent complete | valid=%d failures=%d selected=%r",
            len(candidates),
            parse_failures,
            selected.title,
        )

        debug: dict[str, Any] = {
            "num_candidates_requested": num_candidates,
            "num_valid_candidates": len(candidates),
            "num_parse_failures": parse_failures,
            "draft_temperature": self._temperature,
            "selection_temperature": self._selection_temperature,
            "seed_assignments": seed_assignments,
            "total_llm_calls": total_llm_calls,
        }

        return ProposalDraftAgentOutput(
            proposal=selected,
            all_candidates=candidates,
            selection_rationale=rationale,
            debug=debug,
        )
