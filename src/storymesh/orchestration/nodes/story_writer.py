"""LangGraph node wrapper for StoryWriterAgent (Stage 6).

Assembles StoryWriterAgentInput from upstream pipeline state, runs the agent,
persists the output artifact, and writes the result back into pipeline state.

The node selects the *best* proposal from the run's proposal history (as ranked
by composite rubric score) rather than the most recent one. When no history
exists the current ``proposal_draft_output`` is used directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.story_writer.agent import StoryWriterAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.story_writer import StoryWriterAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_story_writer_node(
    agent: StoryWriterAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for StoryWriterAgent (Stage 6).

    Assembles ``StoryWriterAgentInput`` from ``proposal_draft_output``,
    ``theme_extractor_output``, ``genre_normalizer_output``, and optionally
    ``rubric_judge_output``. Runs the agent, persists the artifact (when an
    ``ArtifactStore`` is provided), and returns a partial state dict containing
    ``story_writer_output``.

    The node uses the highest-scoring proposal from ``proposal_history`` when
    that history exists (populated by ``rubric_judge_node``), falling back to
    the current ``proposal_draft_output.proposal`` otherwise.

    Args:
        agent: A fully constructed ``StoryWriterAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def story_writer_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 6 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain
                ``proposal_draft_output``, ``theme_extractor_output``,
                ``genre_normalizer_output``, and ``run_id``.

        Returns:
            Partial state update dict with ``story_writer_output`` set.

        Raises:
            RuntimeError: If any required upstream output is missing from state.
        """
        proposal_draft_output = state.get("proposal_draft_output")
        if proposal_draft_output is None:
            msg = "story_writer_node requires proposal_draft_output but it is None"
            raise RuntimeError(msg)

        theme_extractor_output = state.get("theme_extractor_output")
        if theme_extractor_output is None:
            msg = "story_writer_node requires theme_extractor_output but it is None"
            raise RuntimeError(msg)

        genre_normalizer_output = state.get("genre_normalizer_output")
        if genre_normalizer_output is None:
            msg = "story_writer_node requires genre_normalizer_output but it is None"
            raise RuntimeError(msg)

        # Use the best proposal from the rubric history when available —
        # on retries this may differ from the most recent proposal_draft_output.
        proposal_history: list[Any] = list(state.get("proposal_history") or [])
        best_idx: int = state.get("best_proposal_index", 0)
        if proposal_history and 0 <= best_idx < len(proposal_history):
            best_proposal = proposal_history[best_idx].proposal
        else:
            best_proposal = proposal_draft_output.proposal

        rubric_output = state.get("rubric_judge_output")

        vps_output = state.get("voice_profile_selector_output")
        voice_profile = vps_output.voice_profile if vps_output is not None else None

        input_data = StoryWriterAgentInput(
            proposal=best_proposal,
            tensions=theme_extractor_output.tensions,
            rubric_feedback=rubric_output,
            user_prompt=state["user_prompt"],
            normalized_genres=genre_normalizer_output.normalized_genres,
            user_tones=theme_extractor_output.user_tones_carried,
            voice_profile=voice_profile,
        )

        token = current_run_id.set(state.get("run_id", ""))
        try:
            output = agent.run(input_data)
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "story_writer", output
            )

        return {"story_writer_output": output}

    return story_writer_node
