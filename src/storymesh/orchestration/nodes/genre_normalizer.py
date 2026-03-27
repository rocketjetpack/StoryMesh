"""LangGraph node wrapper for GenreNormalizerAgent (Stage 0).

The node factory pattern used here (``make_genre_normalizer_node``) injects
a pre-built agent instance at graph-construction time. This keeps the node
function itself free of config or environment dependencies, making it
straightforward to test with a pre-configured agent (fake store, no LLM).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.genre_normalizer.agent import GenreNormalizerAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_genre_normalizer_node(
    agent: GenreNormalizerAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for GenreNormalizerAgent.

    The returned callable reads ``user_prompt`` from the pipeline state,
    runs the agent, persists the output artifact (if an ``ArtifactStore``
    is provided), and returns a partial state dict containing only the
    ``genre_normalizer_output`` key. LangGraph merges this into the full
    state automatically.

    Args:
        agent: A fully constructed ``GenreNormalizerAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def genre_normalizer_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 0 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``user_prompt`` and ``run_id``.

        Returns:
            Partial state update dict with ``genre_normalizer_output`` set.
        """
        input_data = GenreNormalizerAgentInput(
            raw_genre=state["user_prompt"],
            allow_llm_fallback=True,
        )
        output = agent.run(input_data)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(artifact_store, state["run_id"], "genre_normalizer", output)

        return {"genre_normalizer_output": output}

    return genre_normalizer_node
