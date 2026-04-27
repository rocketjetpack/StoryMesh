"""LangGraph node wrapper for RubricJudgeAgent (Stage 5).

Assembles RubricJudgeAgentInput from upstream pipeline state, runs the agent,
persists the output artifact, and returns a partial state update containing
rubric_judge_output.

The node does NOT increment rubric_retry_count — that is done by the
proposal_draft node wrapper on retry. This node only evaluates and returns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.rubric_judge.agent import RubricJudgeAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.rubric_judge import RubricJudgeAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_rubric_judge_node(
    agent: RubricJudgeAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for RubricJudgeAgent (Stage 5).

    Args:
        agent: A fully constructed ``RubricJudgeAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def rubric_judge_node(state: StoryMeshState) -> dict[str, Any]:
        """Evaluate the current proposal and write the rubric output into state.

        Args:
            state: Current pipeline state. Must contain ``proposal_draft_output``,
                ``theme_extractor_output``, and ``genre_normalizer_output``.

        Returns:
            Partial state update dict with ``rubric_judge_output`` set.

        Raises:
            RuntimeError: If required upstream outputs are missing from state.
        """
        proposal_draft_output = state.get("proposal_draft_output")
        if proposal_draft_output is None:
            msg = "rubric_judge_node requires proposal_draft_output but it is None"
            raise RuntimeError(msg)

        theme_extractor_output = state.get("theme_extractor_output")
        if theme_extractor_output is None:
            msg = "rubric_judge_node requires theme_extractor_output but it is None"
            raise RuntimeError(msg)

        genre_normalizer_output = state.get("genre_normalizer_output")
        if genre_normalizer_output is None:
            msg = "rubric_judge_node requires genre_normalizer_output but it is None"
            raise RuntimeError(msg)

        retry_count: int = state.get("rubric_retry_count", 0)

        input_data = RubricJudgeAgentInput(
            proposal=proposal_draft_output.proposal,
            tensions=theme_extractor_output.tensions,
            cliched_resolutions={
                t.tension_id: t.cliched_resolutions
                for t in theme_extractor_output.tensions
            },
            user_tones=theme_extractor_output.user_tones_carried,
            user_prompt=state["user_prompt"],
            normalized_genres=genre_normalizer_output.normalized_genres,
            attempt_number=retry_count + 1,
        )

        token = current_run_id.set(state.get("run_id", ""))
        try:
            output = agent.run(input_data)
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "rubric_judge", output
            )

        rubric_history: list[Any] = list(state.get("rubric_history") or [])
        rubric_history.append(output)

        proposal_history: list[Any] = list(state.get("proposal_history") or [])
        best_idx: int = state.get("best_proposal_index", 0)
        if proposal_history and rubric_history:
            best_score = -1.0
            for i, rub in enumerate(rubric_history):
                if rub.composite_score > best_score:
                    best_score = rub.composite_score
                    best_idx = i

        return {
            "rubric_judge_output": output,
            "rubric_history": rubric_history,
            "best_proposal_index": best_idx,
        }

    return rubric_judge_node
