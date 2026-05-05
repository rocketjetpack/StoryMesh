"""ProposalReaderAgent — Stage 4.5 of the StoryMesh pipeline (retry path only).

Reads a story proposal as a simulated human reader and produces structured
reader-perspective feedback. This is intentionally distinct from the rubric
judge's craft evaluation: the vocabulary is non-technical, the perspective
is "do I want to read this?" rather than "does this satisfy craft principles?".

Cross-provider design: this agent uses a different model than the proposal
draft agent (e.g. GPT-4o while the drafter uses Claude) to surface blind
spots that a shared model family cannot see in its own output.

Runs only on the retry path, after RubricJudgeAgent has evaluated and before
ProposalDraftAgent attempts a directed revision.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.proposal_reader import (
    ProposalReaderAgentInput,
    ProposalReaderAgentOutput,
    ProposalReaderFeedback,
)
from storymesh.versioning.schemas import PROPOSAL_READER_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class ProposalReaderAgent:
    """Evaluates a story proposal from a reader's perspective (Stage 4.5, retry path).

    Single LLM call using a cross-provider model to produce reader-perspective
    feedback that complements the rubric judge's craft evaluation. Both
    feedback forms are then passed to ProposalDraftAgent for directed revision.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> None:
        """Construct the agent.

        Args:
            llm_client: LLM client instance. Required — reader evaluation has
                no deterministic fallback.
            temperature: Sampling temperature. Default 0.4 gives analytical
                reactions with natural variation.
            max_tokens: Maximum output tokens for the feedback call.
        """
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._prompt = load_prompt("proposal_reader_feedback")

    def run(self, input_data: ProposalReaderAgentInput) -> ProposalReaderAgentOutput:
        """Evaluate the proposal and return structured reader feedback.

        Args:
            input_data: Assembled input containing the best-scoring proposal
                and context from upstream pipeline stages.

        Returns:
            A frozen ProposalReaderAgentOutput with structured reader feedback.

        Raises:
            RuntimeError: If the LLM call or response parsing fails.
        """
        proposal_json = orjson.dumps(input_data.proposal.model_dump()).decode()

        logger.debug(
            "ProposalReaderAgent starting | title=%r",
            input_data.proposal.title,
        )

        user_prompt_text = self._prompt.format_user(
            user_prompt=input_data.user_prompt,
            normalized_genres=input_data.normalized_genres,
            user_tones=input_data.user_tones,
            proposal=proposal_json,
        )

        try:
            response = self._llm_client.complete_json(
                user_prompt_text,
                system_prompt=self._prompt.system,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            feedback = ProposalReaderFeedback(**response)
        except Exception as exc:
            raise RuntimeError(
                f"ProposalReaderAgent evaluation failed: {exc}"
            ) from exc

        logger.debug(
            "ProposalReaderAgent complete | engaged=%r",
            feedback.what_engaged_me[:80],
        )

        debug: dict[str, Any] = {
            "temperature": self._temperature,
            "schema_version": PROPOSAL_READER_SCHEMA_VERSION,
        }

        return ProposalReaderAgentOutput(feedback=feedback, debug=debug)
