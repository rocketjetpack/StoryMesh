"""LangGraph node wrapper for GenreNormalizerAgent (Stage 0).

The node factory pattern used here (``make_genre_normalizer_node``) injects
a pre-built agent instance at graph-construction time. This keeps the node
function itself free of config or environment dependencies, making it
straightforward to test with a pre-configured agent (fake store, no LLM).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from storymesh.agents.genre_normalizer.agent import GenreNormalizerAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentInput


def make_genre_normalizer_node(
    agent: GenreNormalizerAgent,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for GenreNormalizerAgent.

    The returned callable reads ``input_genre`` from the pipeline state,
    runs the agent, and returns a partial state dict containing only the
    ``genre_normalizer_output`` key. LangGraph merges this into the
    full state automatically.

    Args:
        agent: A fully constructed ``GenreNormalizerAgent`` instance.

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def genre_normalizer_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 0 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``input_genre``.

        Returns:
            Partial state update dict with ``genre_normalizer_output`` set.
        """
        input_data = GenreNormalizerAgentInput(
            raw_genre=state["input_genre"],
            allow_llm_fallback=True,
        )
        output = agent.run(input_data)
        return {"genre_normalizer_output": output}

    return genre_normalizer_node
