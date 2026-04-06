"""LangGraph node wrapper for ThemeExtractorAgent (Stage 3).

The node factory pattern used here (``make_theme_extractor_node``) injects a
pre-built agent instance at graph-construction time. This is the first node
that assembles its input from multiple upstream stages — both
``genre_normalizer_output`` and ``book_ranker_output`` are required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.theme_extractor.agent import ThemeExtractorAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.theme_extractor import ThemeExtractorAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore


def make_theme_extractor_node(
    agent: ThemeExtractorAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for ThemeExtractorAgent (Stage 3).

    This is the first node that draws from multiple upstream stages. It reads
    ``genre_normalizer_output`` and ``book_ranker_output`` from the pipeline
    state, assembles a self-contained ``ThemeExtractorAgentInput``, runs the
    agent, persists the output artifact (if an ``ArtifactStore`` is provided),
    and returns a partial state dict containing only ``theme_extractor_output``.
    LangGraph merges this into the full state automatically.

    Args:
        agent: A fully constructed ``ThemeExtractorAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
            Pass ``None`` (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def theme_extractor_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 3 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``genre_normalizer_output``,
                ``book_ranker_output``, ``user_prompt``, and ``run_id``.

        Returns:
            Partial state update dict with ``theme_extractor_output`` set.

        Raises:
            RuntimeError: If either upstream output is missing from state.
        """
        genre_normalizer_output = state.get("genre_normalizer_output")
        if genre_normalizer_output is None:
            msg = "theme_extractor_node requires genre_normalizer_output but it is None"
            raise RuntimeError(msg)

        book_ranker_output = state.get("book_ranker_output")
        if book_ranker_output is None:
            msg = "theme_extractor_node requires book_ranker_output but it is None"
            raise RuntimeError(msg)

        input_data = ThemeExtractorAgentInput(
            ranked_summaries=book_ranker_output.ranked_summaries,
            normalized_genres=genre_normalizer_output.normalized_genres,
            subgenres=genre_normalizer_output.subgenres,
            user_tones=genre_normalizer_output.user_tones,
            tone_override=genre_normalizer_output.tone_override,
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

            persist_node_output(artifact_store, state["run_id"], "theme_extractor", output)

        return {"theme_extractor_output": output}

    return theme_extractor_node
