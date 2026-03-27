"""LangGraph node wrapper for BookFetcherAgent (Stage 1).

The node factory pattern used here (``make_book_fetcher_node``) injects a
pre-built agent instance at graph-construction time. This keeps the node
function itself free of config or environment dependencies, making it
straightforward to test with a pre-configured agent and mock client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from storymesh.agents.book_fetcher.agent import BookFetcherAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.book_fetcher import BookFetcherAgentInput


def make_book_fetcher_node(
    agent: BookFetcherAgent,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for BookFetcherAgent (Stage 1).

    Reads ``genre_normalizer_output`` from the pipeline state, constructs the
    BookFetcherAgentInput, runs the agent, and returns a partial state dict
    containing only ``book_fetcher_output``. LangGraph merges this into the
    full state automatically.

    Args:
        agent: A fully constructed ``BookFetcherAgent`` instance.

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def book_fetcher_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 1 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``genre_normalizer_output``.

        Returns:
            Partial state update dict with ``book_fetcher_output`` set.
        """
        genre_output = state["genre_normalizer_output"]
        if genre_output is None:
            msg = "book_fetcher_node requires genre_normalizer_output but it is None"
            raise RuntimeError(msg)
        input_data = BookFetcherAgentInput(
            normalized_genres=genre_output.normalized_genres,
        )
        output = agent.run(input_data)
        return {"book_fetcher_output": output}

    return book_fetcher_node
