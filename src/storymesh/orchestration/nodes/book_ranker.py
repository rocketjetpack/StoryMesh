"""LangGraph node wrapper for BookRankerAgent (Stage 2).

The node factory pattern used here (``make_book_ranker_node``) injects a
pre-built agent instance at graph-construction time. This keeps the node
function itself free of config or environment dependencies, making it
straightforward to test with a pre-configured agent and mock client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.book_ranker.agent import BookRankerAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.book_ranker import BookRankerAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_book_ranker_node(
    agent: BookRankerAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for BookRankerAgent (Stage 2).

    Reads ``book_fetcher_output`` and ``user_prompt`` from the pipeline state,
    constructs the BookRankerAgentInput, runs the agent, persists the output
    artifact (if an ``ArtifactStore`` is provided), and returns a partial state
    dict containing only ``book_ranker_output``. LangGraph merges this into the
    full state automatically.

    Args:
        agent: A fully constructed ``BookRankerAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def book_ranker_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 2 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``book_fetcher_output``,
                ``user_prompt``, and ``run_id``.

        Returns:
            Partial state update dict with ``book_ranker_output`` set.
        """
        book_fetcher_output = state.get("book_fetcher_output")
        if book_fetcher_output is None:
            msg = "book_ranker_node requires book_fetcher_output but it is None"
            raise RuntimeError(msg)

        input_data = BookRankerAgentInput(
            books=book_fetcher_output.books,
            user_prompt=state["user_prompt"],
            total_genres_queried=len(book_fetcher_output.queries_executed),
        )
        output = agent.run(input_data)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(artifact_store, state["run_id"], "book_ranker", output)

        return {"book_ranker_output": output}

    return book_ranker_node
