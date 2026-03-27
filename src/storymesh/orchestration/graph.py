"""Compile the StoryMesh LangGraph StateGraph.

All pipeline topology is defined here. Adding a new agent means:
  1. Implementing the agent and its node wrapper under orchestration/nodes/.
  2. Instantiating the agent below with its config.
  3. Replacing the relevant ``_noop_node`` with the real node function.
  4. Adjusting edges (or adding conditional edges) as needed.

LangSmith node-level tracing is automatic when the environment variables
``LANGCHAIN_TRACING_V2=true`` and ``LANGCHAIN_API_KEY`` are present. No code
changes are required here. Individual LLM call tracing is handled by the
``@_traceable`` decorator in storymesh.llm.base.

Mypy note: LangGraph 0.2+ ships with py.typed but ``CompiledStateGraph``
generics are not fully resolvable under mypy strict. The ``# type: ignore``
comments on the langgraph imports suppress the resulting cascade. All node
functions in this module are fully annotated.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from storymesh.config import get_agent_config
from storymesh.llm.base import LLMClient
from storymesh.orchestration.nodes.genre_normalizer import make_genre_normalizer_node
from storymesh.orchestration.state import StoryMeshState

logger = logging.getLogger(__name__)

# Future node imports (uncomment as agents are implemented):
# from storymesh.orchestration.nodes.book_ranker import make_book_ranker_node
# from storymesh.orchestration.nodes.theme_extractor import make_theme_extractor_node
# from storymesh.orchestration.nodes.proposal_draft import make_proposal_draft_node
# from storymesh.orchestration.nodes.rubric_judge import make_rubric_judge_node
# from storymesh.orchestration.nodes.synopsis_writer import make_synopsis_writer_node


def _build_llm_client(agent_cfg: dict[str, Any]) -> LLMClient | None:
    """Instantiate the correct LLMClient subclass from an agent config dict.

    Returns ``None`` with a warning if the required API key environment
    variable is not set, allowing agents that support it to run in
    static-only mode. Raises ``ValueError`` for unknown provider names.

    Args:
        agent_cfg: Resolved agent config dict from ``get_agent_config()``.

    Returns:
        A concrete ``LLMClient`` instance, or ``None`` if the API key is absent.

    Raises:
        ValueError: If ``agent_cfg["provider"]`` is not a recognised value.
    """
    provider: str | None = agent_cfg.get("provider")
    model: str | None = agent_cfg.get("model")

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning(
                "ANTHROPIC_API_KEY is not set — the agent will run in "
                "static-only mode (no LLM fallback)."
            )
            return None
        from storymesh.llm.anthropic import AnthropicClient  # noqa: PLC0415

        return AnthropicClient(model=model)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY is not set — the agent will run in "
                "static-only mode (no LLM fallback)."
            )
            return None
        # Placeholder: uncomment when OpenAIClient is implemented.
        # from storymesh.llm.openai import OpenAIClient  # noqa: PLC0415
        # return OpenAIClient(model=model)
        raise NotImplementedError("OpenAIClient is not yet implemented.")

    if provider is None:
        return None

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Valid values: 'anthropic', 'openai'."
    )


def _noop_node(state: StoryMeshState) -> dict[str, Any]:
    """Placeholder node for pipeline stages not yet implemented.

    Returns an empty dict so LangGraph makes no state changes. Replace
    with the real node function as each agent is built.

    Args:
        state: Current pipeline state (unused).

    Returns:
        Empty dict — no state update.
    """
    return {}


def build_graph() -> Any:  # noqa: ANN401  # CompiledStateGraph generics not resolvable under mypy strict.
    """Construct and compile the StoryMesh pipeline StateGraph.

    Stage topology (linear for now; Stage 5 will become a conditional edge
    once RubricJudgeAgent is implemented):

        START → genre_normalizer → book_fetcher → book_ranker
              → theme_extractor → proposal_draft → rubric_judge
              → synopsis_writer → END

    Future topology notes:
    - Stage 5 (RubricJudge): replace the direct edge to ``synopsis_writer``
      with ``add_conditional_edges`` routing failed proposals back to
      ``proposal_draft`` for revision.
    - Checkpointer: pass ``checkpointer=MemorySaver()`` to ``compile()``
      when HITL or run-persistence is needed.

    Returns:
        A compiled LangGraph StateGraph ready for ``.invoke()`` or
        ``.astream_events()``.
    """
    # ── Stage 0: GenreNormalizerAgent ──────────────────────────────────────
    genre_cfg = get_agent_config("genre_normalizer")
    genre_llm = _build_llm_client(genre_cfg)

    from storymesh.agents.genre_normalizer.agent import GenreNormalizerAgent  # noqa: PLC0415

    genre_agent = GenreNormalizerAgent(
        llm_client=genre_llm,
        temperature=genre_cfg.get("temperature", 0.0),
        max_tokens=genre_cfg.get("max_tokens", 1024),
    )
    genre_node = make_genre_normalizer_node(genre_agent)

    # ── Stage 1: BookFetcherAgent ──────────────────────────────────────────
    from storymesh.agents.book_fetcher.agent import BookFetcherAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.book_fetcher import (  # noqa: PLC0415
        make_book_fetcher_node,
    )

    book_fetcher_agent = BookFetcherAgent()
    book_fetcher_node = make_book_fetcher_node(book_fetcher_agent)

    # ── Build the graph ────────────────────────────────────────────────────
    graph: Any = StateGraph(StoryMeshState)

    graph.add_node("genre_normalizer", genre_node)
    graph.add_node("book_fetcher", book_fetcher_node)
    graph.add_node("book_ranker", _noop_node)       # Stage 2 — placeholder
    graph.add_node("theme_extractor", _noop_node)   # Stage 3 — placeholder (LLM)
    graph.add_node("proposal_draft", _noop_node)    # Stage 4 — placeholder (LLM)
    graph.add_node("rubric_judge", _noop_node)      # Stage 5 — placeholder (LLM, conditional)
    graph.add_node("synopsis_writer", _noop_node)   # Stage 6 — placeholder (LLM)

    # ── Wire edges (linear) ────────────────────────────────────────────────
    graph.add_edge(START, "genre_normalizer")
    graph.add_edge("genre_normalizer", "book_fetcher")
    graph.add_edge("book_fetcher", "book_ranker")
    graph.add_edge("book_ranker", "theme_extractor")
    graph.add_edge("theme_extractor", "proposal_draft")
    graph.add_edge("proposal_draft", "rubric_judge")
    graph.add_edge("rubric_judge", "synopsis_writer")
    graph.add_edge("synopsis_writer", END)

    return graph.compile()
