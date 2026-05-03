"""RubricJudgeAgent — Stage 5 of the StoryMesh pipeline.

Evaluates a StoryProposal against a 5-dimension craft-quality rubric using an
LLM on a DIFFERENT provider than ProposalDraftAgent. The cross-provider design
prevents the evaluator from inheriting the generator's blind spots.

Uses three-tier scoring (0=fail, 1=acceptable, 2=strong) per dimension. The
composite score is the simple sum of all tier scores (max 10). Pass/fail is
computed here in Python so the decision is deterministic, auditable, and tunable
without LLM involvement.

If the LLM call fails entirely, the agent returns a default-fail output so the
retry loop gets a chance to improve the proposal rather than crashing.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.rubric_judge import (
    EXPECTED_DIMENSIONS,
    DimensionResult,
    RubricJudgeAgentInput,
    RubricJudgeAgentOutput,
)

logger = logging.getLogger(__name__)

_FALLBACK_FEEDBACK = "Dimension not evaluated by the model."


class RubricJudgeAgent:
    """Evaluates story proposals against a craft-quality rubric (Stage 5).

    Uses a DIFFERENT LLM provider than ProposalDraftAgent to give an
    independent editorial evaluation free from the generator's blind spots.
    Computes pass/fail from a sum of three-tier scores (0/1/2) against a
    configurable threshold — the LLM never makes the pass/fail decision.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        pass_threshold: int = 6,
    ) -> None:
        """Construct the agent.

        Args:
            llm_client: LLM client instance. Should be a different provider
                than ProposalDraftAgent for independent evaluation.
            temperature: Default 0.0 — evaluation should be reproducible.
            max_tokens: Maximum tokens for the rubric feedback response.
            pass_threshold: Minimum sum of tier scores to pass (max 10).
                Default 6 ("standard" quality). Presets: draft=5, standard=6,
                high=8.
        """
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._pass_threshold = pass_threshold
        self._prompt = load_prompt("rubric_judge")

    def run(self, input_data: RubricJudgeAgentInput) -> RubricJudgeAgentOutput:
        """Evaluate a story proposal and return a scored rubric output.

        Args:
            input_data: Proposal, tensions, and user context to evaluate.

        Returns:
            A frozen RubricJudgeAgentOutput with per-dimension scores,
            feedback, composite score, and pass/fail determination.
            On LLM failure, returns a default-fail output so the retry
            loop can attempt to produce a better proposal.
        """
        logger.info(
            "RubricJudgeAgent starting | attempt=%d threshold=%d",
            input_data.attempt_number,
            self._pass_threshold,
        )

        proposal_json = orjson.dumps(input_data.proposal.model_dump()).decode()
        tensions_json = orjson.dumps(
            [t.model_dump() for t in input_data.tensions]
        ).decode()

        user_prompt_text = self._prompt.format_user(
            user_prompt=input_data.user_prompt,
            normalized_genres=input_data.normalized_genres,
            user_tones=input_data.user_tones,
            tensions=tensions_json,
            proposal=proposal_json,
        )

        try:
            raw = self._llm_client.complete_json(
                user_prompt_text,
                system_prompt=self._prompt.system,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.warning("RubricJudgeAgent: LLM call failed (%s) — returning default fail.", exc)
            return self._default_fail(input_data, error=str(exc))

        return self._build_output(raw, input_data)

    def _build_output(
        self,
        raw: dict[str, Any],
        input_data: RubricJudgeAgentInput,
    ) -> RubricJudgeAgentOutput:
        """Parse LLM response, compute composite score, and build the output."""
        raw_dims: dict[str, Any] = raw.get("dimensions", {})
        dimensions: dict[str, DimensionResult] = {}

        for dim_name in EXPECTED_DIMENSIONS:
            if dim_name not in raw_dims:
                logger.warning(
                    "RubricJudgeAgent: dimension '%s' missing from LLM response — assigning 0.",
                    dim_name,
                )
                dimensions[dim_name] = DimensionResult(
                    score=0,
                    feedback=_FALLBACK_FEEDBACK,
                    principle_ref="N/A",
                )
            else:
                dim_data = raw_dims[dim_name]
                # Parse and clamp to valid tier range [0, 2].
                try:
                    score = int(dim_data.get("score", 0))
                except (TypeError, ValueError):
                    score = 0
                score = max(0, min(2, score))
                feedback = str(dim_data.get("feedback", _FALLBACK_FEEDBACK))
                if len(feedback) < 10:
                    feedback = f"{feedback} (no further detail provided)"
                dimensions[dim_name] = DimensionResult(
                    score=score,
                    feedback=feedback,
                    principle_ref=str(dim_data.get("principle_ref", dim_name)),
                )

        creative_direction: str = str(raw.get("creative_direction", ""))

        overall_feedback: str = raw.get(
            "overall_feedback",
            "No overall feedback provided.",
        )
        if len(overall_feedback) < 10:
            overall_feedback = f"Evaluation complete. {overall_feedback}"

        composite = self._compute_composite(dimensions)
        passed = composite >= self._pass_threshold

        logger.info(
            "RubricJudgeAgent complete | composite=%d threshold=%d passed=%s",
            composite,
            self._pass_threshold,
            passed,
        )

        debug: dict[str, Any] = {
            "threshold": self._pass_threshold,
            "raw_scores": {k: v.score for k, v in dimensions.items()},
            "attempt_number": input_data.attempt_number,
            "total_llm_calls": 1,
        }

        return RubricJudgeAgentOutput(
            passed=passed,
            composite_score=composite,
            pass_threshold=self._pass_threshold,
            dimensions=dimensions,
            creative_direction=creative_direction,
            overall_feedback=overall_feedback,
            debug=debug,
        )

    @staticmethod
    def _compute_composite(dimensions: dict[str, DimensionResult]) -> int:
        """Compute composite score as sum of all dimension tier scores."""
        return sum(dim.score for dim in dimensions.values())

    def _default_fail(
        self,
        input_data: RubricJudgeAgentInput,
        error: str = "",
    ) -> RubricJudgeAgentOutput:
        """Return a default-fail output when the LLM call fails entirely."""
        feedback_msg = (
            f"Rubric evaluation failed: {error}. Treating as fail to trigger retry."
            if error
            else "Rubric evaluation failed. Treating as fail to trigger retry."
        )
        dimensions = {
            name: DimensionResult(score=0, feedback=feedback_msg, principle_ref="N/A")
            for name in EXPECTED_DIMENSIONS
        }
        return RubricJudgeAgentOutput(
            passed=False,
            composite_score=0,
            pass_threshold=self._pass_threshold,
            dimensions=dimensions,
            creative_direction="",
            overall_feedback=feedback_msg,
            debug={
                "threshold": self._pass_threshold,
                "attempt_number": input_data.attempt_number,
                "total_llm_calls": 1,
                "llm_error": error,
            },
        )
