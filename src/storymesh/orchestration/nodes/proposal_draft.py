"""LangGraph node wrapper for ProposalDraftAgent (Stage 4).

Assembles ProposalDraftAgentInput from two upstream stages
(ThemeExtractorAgentOutput and GenreNormalizerAgentOutput), runs the agent,
persists the output artifact, and writes the result back into pipeline state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.proposal_draft.agent import ProposalDraftAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.proposal_draft import ProposalDraftAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_proposal_draft_node(
    agent: ProposalDraftAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for ProposalDraftAgent (Stage 4).

    Assembles ``ProposalDraftAgentInput`` from ``theme_extractor_output`` and
    ``genre_normalizer_output``, runs the agent, persists the artifact (when an
    ``ArtifactStore`` is provided), and returns a partial state dict containing
    only ``proposal_draft_output``. LangGraph merges this into the full state.

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
            output = agent.run(input_data)
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "proposal_draft", output
            )

        return {"proposal_draft_output": output}

    return proposal_draft_node
