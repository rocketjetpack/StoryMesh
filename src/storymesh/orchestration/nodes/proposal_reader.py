"""LangGraph node wrapper for ProposalReaderAgent (Stage 4.5, retry path only).

Reads the best-scoring proposal from proposal_history, evaluates it from a
reader's perspective, and stores the output in state so ProposalDraftAgent
can use both forms of feedback for a directed revision.

This node only runs on the retry path — the graph routes rubric_judge failures
through here before returning to proposal_draft.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.proposal_reader.agent import ProposalReaderAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.proposal_reader import ProposalReaderAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_proposal_reader_node(
    agent: ProposalReaderAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for ProposalReaderAgent (Stage 4.5).

    Reads the best-scoring proposal from ``proposal_history`` (using
    ``best_proposal_index``), runs the reader feedback agent, and returns a
    partial state dict containing ``proposal_reader_output``.

    Args:
        agent: A fully constructed ``ProposalReaderAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def proposal_reader_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 4.5 and write reader feedback into pipeline state.

        Selects the best-scoring proposal from ``proposal_history`` using
        ``best_proposal_index``, falling back to the current
        ``proposal_draft_output`` when history is unavailable.

        Args:
            state: Current pipeline state. Must contain ``genre_normalizer_output``,
                ``theme_extractor_output``, and at least one of
                ``proposal_history`` or ``proposal_draft_output``.

        Returns:
            Partial state update dict with ``proposal_reader_output`` set.

        Raises:
            RuntimeError: If required upstream outputs are missing from state.
        """
        genre_normalizer_output = state.get("genre_normalizer_output")
        if genre_normalizer_output is None:
            msg = "proposal_reader_node requires genre_normalizer_output but it is None"
            raise RuntimeError(msg)

        theme_extractor_output = state.get("theme_extractor_output")
        if theme_extractor_output is None:
            msg = "proposal_reader_node requires theme_extractor_output but it is None"
            raise RuntimeError(msg)

        # Use the best-scoring proposal from history — same index the story_writer uses.
        proposal_history: list[Any] = list(state.get("proposal_history") or [])
        best_idx: int = state.get("best_proposal_index", 0)

        if proposal_history and 0 <= best_idx < len(proposal_history):
            best_proposal = proposal_history[best_idx].proposal
        else:
            proposal_draft_output = state.get("proposal_draft_output")
            if proposal_draft_output is None:
                msg = "proposal_reader_node: no proposal available in state"
                raise RuntimeError(msg)
            best_proposal = proposal_draft_output.proposal

        input_data = ProposalReaderAgentInput(
            proposal=best_proposal,
            user_prompt=state["user_prompt"],
            normalized_genres=genre_normalizer_output.normalized_genres,
            user_tones=theme_extractor_output.user_tones_carried,
        )

        token = current_run_id.set(state.get("run_id", ""))
        try:
            output = agent.run(input_data)
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "proposal_reader", output
            )

        return {"proposal_reader_output": output}

    return proposal_reader_node
