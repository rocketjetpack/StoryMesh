"""LangGraph node wrapper for BookFetcherAgent (Stage 1).

The node factory pattern used here (``make_book_fetcher_node``) injects a
pre-built agent instance at graph-construction time. This keeps the node
function itself free of config or environment dependencies, making it
straightforward to test with a pre-configured agent and mock client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.book_fetcher.agent import BookFetcherAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.book_fetcher import BookFetcherAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_book_fetcher_node(
    agent: BookFetcherAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for BookFetcherAgent (Stage 1).

    Reads ``genre_normalizer_output`` from the pipeline state, constructs the
    BookFetcherAgentInput, runs the agent, persists the output artifact (if an
    ``ArtifactStore`` is provided), and returns a partial state dict containing
    only ``book_fetcher_output``. LangGraph merges this into the full state
    automatically.

    Args:
        agent: A fully constructed ``BookFetcherAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def book_fetcher_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 1 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``genre_normalizer_output``
                and ``run_id``.

        Returns:
            Partial state update dict with ``book_fetcher_output`` set.
        """
        genre_output = state["genre_normalizer_output"]
        if genre_output is None:
            msg = "book_fetcher_node requires genre_normalizer_output but it is None"
            raise RuntimeError(msg)
        from storymesh.agents.book_fetcher.subject_map import resolve_subjects  # noqa: PLC0415

        # Combine explicit genres (Passes 1–3) with Pass 4 inferred genres,
        # then translate canonical names to Open Library subject strings.
        # Genres with no OL equivalent (e.g. workplace_fiction) are dropped;
        # unknown genres fall back to underscore→space normalisation.
        genres_to_query = genre_output.normalized_genres + [
            ig.canonical_genre for ig in genre_output.inferred_genres
        ]
        input_data = BookFetcherAgentInput(
            normalized_genres=resolve_subjects(genres_to_query),
        )
        output = agent.run(input_data)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(artifact_store, state["run_id"], "book_fetcher", output)

        return {"book_fetcher_output": output}

    return book_fetcher_node
