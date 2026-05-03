"""LangGraph node wrapper for ProposalDraftAgent (Stage 4).

Assembles ProposalDraftAgentInput from two upstream stages
(ThemeExtractorAgentOutput and GenreNormalizerAgentOutput), runs the agent,
persists the output artifact, and writes the result back into pipeline state.

On retry (when rubric_judge_output is present and failed), the wrapper builds
a RubricFeedback carrier from the evaluator's critique and passes it to the
agent so the retry prompt receives targeted editorial direction.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.proposal_draft.agent import ProposalDraftAgent, RubricFeedback
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.proposal_draft import ProposalDraftAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore
    from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput


def _format_feedback(rubric_output: RubricJudgeAgentOutput) -> str:
    """Format dimension-by-dimension rubric scores and feedback as readable text."""
    lines: list[str] = []
    dimensions = getattr(rubric_output, "dimensions", {})
    for dim_name, dim_result in dimensions.items():
        score = getattr(dim_result, "score", "N/A")
        feedback = getattr(dim_result, "feedback", "")
        ref = getattr(dim_result, "principle_ref", "")
        lines.append(f"[{dim_name}] ({ref}) score={score}: {feedback}")
    creative_direction = getattr(rubric_output, "creative_direction", "")
    if creative_direction:
        lines.append(f"\nCREATIVE DIRECTION: {creative_direction}")
    overall = getattr(rubric_output, "overall_feedback", "")
    if overall:
        lines.append(f"\nOVERALL: {overall}")
    return "\n".join(lines)


def _format_reader_feedback(reader_output: object) -> str:
    """Format ProposalReaderAgentOutput feedback fields as a readable text block."""
    feedback = getattr(reader_output, "feedback", None)
    if feedback is None:
        return ""
    lines = [
        f"What engaged them:  {getattr(feedback, 'what_engaged_me', '')}",
        f"What fell flat:     {getattr(feedback, 'what_fell_flat', '')}",
        f"Protagonist gap:    {getattr(feedback, 'protagonist_gap', '')}",
        f"Premise question:   {getattr(feedback, 'premise_question', '')}",
        f"Reader direction:   {getattr(feedback, 'reader_direction', '')}",
    ]
    return "\n".join(lines)


def _format_scores(rubric_output: RubricJudgeAgentOutput) -> str:
    """Format rubric scores as a compact summary."""
    lines: list[str] = []
    dimensions = getattr(rubric_output, "dimensions", {})
    for dim_name, dim_result in dimensions.items():
        score = getattr(dim_result, "score", "N/A")
        lines.append(f"  {dim_name}: {score}")
    composite = getattr(rubric_output, "composite_score", "N/A")
    threshold = getattr(rubric_output, "pass_threshold", "N/A")
    lines.append(f"  COMPOSITE: {composite}")
    lines.append(f"  THRESHOLD: {threshold}")
    return "\n".join(lines)


def make_proposal_draft_node(
    agent: ProposalDraftAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for ProposalDraftAgent (Stage 4).

    Assembles ``ProposalDraftAgentInput`` from ``theme_extractor_output`` and
    ``genre_normalizer_output``, runs the agent, persists the artifact (when an
    ``ArtifactStore`` is provided), and returns a partial state dict containing
    ``proposal_draft_output`` and, on retry, an incremented ``rubric_retry_count``.

    Args:
        agent: A fully constructed ``ProposalDraftAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def proposal_draft_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 4 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain
                ``theme_extractor_output``, ``genre_normalizer_output``,
                ``user_prompt``, and ``run_id``.

        Returns:
            Partial state update dict with ``proposal_draft_output`` set.
            On retry also sets ``rubric_retry_count`` incremented by 1.

        Raises:
            RuntimeError: If either upstream output is missing from state.
        """
        theme_extractor_output = state.get("theme_extractor_output")
        if theme_extractor_output is None:
            msg = (
                "proposal_draft_node requires theme_extractor_output but it is None"
            )
            raise RuntimeError(msg)

        genre_normalizer_output = state.get("genre_normalizer_output")
        if genre_normalizer_output is None:
            msg = (
                "proposal_draft_node requires genre_normalizer_output but it is None"
            )
            raise RuntimeError(msg)

        # Detect retry: rubric_judge has run (output is present). This covers
        # both failed proposals AND passing proposals routed back by min_retries.
        rubric_output = state.get("rubric_judge_output")
        is_retry = rubric_output is not None

        rubric_feedback: RubricFeedback | None = None
        reader_feedback_text: str | None = None
        if is_retry:
            assert rubric_output is not None
            # Use the best-scoring proposal rather than the most recent one —
            # revision should improve what is already strongest.
            proposal_history: list[Any] = list(state.get("proposal_history") or [])
            best_idx: int = state.get("best_proposal_index", 0)
            if proposal_history and 0 <= best_idx < len(proposal_history):
                best_proposal_json = json.dumps(
                    proposal_history[best_idx].proposal.model_dump(), indent=2
                )
            else:
                prev_proposal_output = state.get("proposal_draft_output")
                best_proposal_json = (
                    json.dumps(prev_proposal_output.proposal.model_dump(), indent=2)
                    if prev_proposal_output is not None
                    else "{}"
                )
            attempt_number = state.get("rubric_retry_count", 0) + 2  # +1 for base, +1 for this retry
            rubric_feedback = RubricFeedback(
                previous_proposal_json=best_proposal_json,
                feedback_text=_format_feedback(rubric_output),
                scores_text=_format_scores(rubric_output),
                attempt_number=attempt_number,
            )
            # Pass reader feedback when the proposal_reader node has run.
            reader_output = state.get("proposal_reader_output")
            if reader_output is not None:
                reader_feedback_text = _format_reader_feedback(reader_output)

        input_data = ProposalDraftAgentInput(
            narrative_seeds=theme_extractor_output.narrative_seeds,
            tensions=theme_extractor_output.tensions,
            genre_clusters=theme_extractor_output.genre_clusters,
            normalized_genres=genre_normalizer_output.normalized_genres,
            user_tones=theme_extractor_output.user_tones_carried,
            narrative_context=genre_normalizer_output.narrative_context,
            user_prompt=state["user_prompt"],
        )

        token = current_run_id.set(state.get("run_id", ""))
        try:
            output = agent.run(
                input_data,
                rubric_feedback=rubric_feedback,
                reader_feedback_text=reader_feedback_text,
            )
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "proposal_draft", output
            )

        history: list[Any] = list(state.get("proposal_history") or [])
        history.append(output)

        result: dict[str, Any] = {
            "proposal_draft_output": output,
            "proposal_history": history,
        }
        if is_retry:
            result["rubric_retry_count"] = state.get("rubric_retry_count", 0) + 1
        return result

    return proposal_draft_node
