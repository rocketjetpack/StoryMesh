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
from storymesh.core.artifacts import ArtifactStore
from storymesh.llm.base import LLMClient
from storymesh.orchestration.nodes.genre_normalizer import make_genre_normalizer_node
from storymesh.orchestration.state import StoryMeshState

logger = logging.getLogger(__name__)

# Future node imports (uncomment as agents are implemented):
# from storymesh.orchestration.nodes.proposal_draft import make_proposal_draft_node
# from storymesh.orchestration.nodes.rubric_judge import make_rubric_judge_node
# from storymesh.orchestration.nodes.synopsis_writer import make_synopsis_writer_node


# Maps provider names to the environment variable names that hold their API keys.
# Mirrors _PROVIDER_KEY_MAP in config.py; both should be kept in sync.
_PROVIDER_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Maps provider names to their implementing module, so that the module-level
# register_provider() call executes before the registry is consulted.
_PROVIDER_MODULE_MAP: dict[str, str] = {
    "anthropic": "storymesh.llm.anthropic",
    "openai": "storymesh.llm.openai",
}


def _ensure_provider_imported(provider: str) -> None:
    """Import the provider module so its ``register_provider()`` call executes.

    Args:
        provider: Provider name string (e.g. ``'anthropic'``).
    """
    import importlib  # noqa: PLC0415

    module_name = _PROVIDER_MODULE_MAP.get(provider)
    if module_name:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.warning(
                "Provider module '%s' could not be imported. "
                "Install the corresponding extra: pip install storymesh[%s]",
                module_name,
                provider,
            )


def _build_llm_client(agent_cfg: dict[str, Any]) -> LLMClient | None:
    """Instantiate the correct LLMClient subclass from an agent config dict.

    Uses the provider registry in ``storymesh.llm.base``. Returns ``None``
    with a warning if the required API key is not set, allowing agents to
    run in static-only mode.

    Args:
        agent_cfg: Resolved agent config dict from ``get_agent_config()``.

    Returns:
        A concrete ``LLMClient`` instance, or ``None`` if the API key is absent.

    Raises:
        ValueError: If ``agent_cfg["provider"]`` is not registered.
    """
    provider: str | None = agent_cfg.get("provider")
    model: str | None = agent_cfg.get("model")

    if provider is None:
        return None

    env_key = _PROVIDER_KEY_MAP.get(provider)
    if env_key and not os.environ.get(env_key):
        logger.warning(
            "%s is not set — the agent will run in static-only mode (no LLM fallback).",
            env_key,
        )
        return None

    _ensure_provider_imported(provider)

    from storymesh.llm.base import get_provider_class  # noqa: PLC0415

    cls = get_provider_class(provider)
    return cls(model=model)


MAX_RUBRIC_RETRIES: int = 2
"""Maximum number of times rubric_judge may route back to proposal_draft."""


def _genre_normalizer_route(state: StoryMeshState) -> str:
    """Route to book_fetcher if genre normalization succeeded, otherwise END.

    Genre normalization sets ``genre_normalizer_output`` to ``None`` and writes
    to ``errors`` when it cannot resolve any genres. Short-circuiting here
    prevents downstream nodes from running against an empty genre output.

    Args:
        state: Current pipeline state.

    Returns:
        ``'book_fetcher'`` on success, ``END`` on failure.
    """
    return "book_fetcher" if state.get("genre_normalizer_output") is not None else END


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


def _rubric_route(state: StoryMeshState) -> str:
    """Conditional routing function after rubric_judge.

    Routes to ``'proposal_draft'`` when the rubric fails and the retry
    budget has not been exhausted, otherwise routes to ``'synopsis_writer'``.

    The real ``RubricJudgeAgent`` should:

    * Set ``rubric_judge_output`` with a pass/fail signal accessible via
      ``getattr(output, 'passed', True)``.
    * Increment ``rubric_retry_count`` in its return dict when failing.

    The noop placeholder leaves ``rubric_judge_output`` as ``None``, which
    is treated as a pass so the pipeline always progresses normally.

    Args:
        state: Current pipeline state.

    Returns:
        Name of the next node to execute.
    """
    retry_count: int = state.get("rubric_retry_count", 0)
    rubric_output = state.get("rubric_judge_output")

    # None means the noop placeholder ran — treat as passed.
    passed: bool = rubric_output is None or bool(getattr(rubric_output, "passed", True))

    if passed or retry_count >= MAX_RUBRIC_RETRIES:
        return "synopsis_writer"
    return "proposal_draft"


def build_graph(artifact_store: ArtifactStore | None = None) -> Any:  # noqa: ANN401  # CompiledStateGraph generics not resolvable under mypy strict.
    """Construct and compile the StoryMesh pipeline StateGraph.

    Stage topology::

        START
          → genre_normalizer
          → book_fetcher
          → book_ranker
          → theme_extractor
          → proposal_draft
          → rubric_judge
          → [conditional]
              ├── PASS → synopsis_writer → END
              └── FAIL → proposal_draft (max 2 retries)

    Notes:
    - Stages 3–6 are noop placeholders until their agents are implemented.
    - The rubric retry loop is wired via ``_rubric_route``; the noop always
      passes so the pipeline progresses linearly until real agents are added.
    - Checkpointer: pass ``checkpointer=MemorySaver()`` to ``compile()``
      when HITL or run-persistence is needed.

    Args:
        artifact_store: Optional store passed to node factories for per-node
            artifact persistence. Pass ``None`` to skip persistence.

    Returns:
        A compiled LangGraph StateGraph ready for ``.stream()`` or
        ``.invoke()``.
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
    genre_node = make_genre_normalizer_node(genre_agent, artifact_store=artifact_store)

    # ── Stage 1: BookFetcherAgent ──────────────────────────────────────────
    from storymesh.agents.book_fetcher.agent import BookFetcherAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.book_fetcher import (  # noqa: PLC0415
        make_book_fetcher_node,
    )

    book_fetcher_agent = BookFetcherAgent()
    book_fetcher_node = make_book_fetcher_node(book_fetcher_agent, artifact_store=artifact_store)

    # ── Stage 2: BookRankerAgent ──────────────────────────────────────────
    from storymesh.agents.book_ranker.agent import BookRankerAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.book_ranker import (  # noqa: PLC0415
        make_book_ranker_node,
    )

    book_ranker_cfg = get_agent_config("book_ranker")
    book_ranker_llm = _build_llm_client(book_ranker_cfg) if book_ranker_cfg.get("llm_rerank") else None

    book_ranker_agent = BookRankerAgent(
        top_n=book_ranker_cfg.get("top_n", 10),
        weights=book_ranker_cfg.get("weights"),
        rating_confidence_threshold=book_ranker_cfg.get("rating_confidence_threshold", 50),
        llm_rerank=book_ranker_cfg.get("llm_rerank", False),
        llm_client=book_ranker_llm,
        temperature=book_ranker_cfg.get("temperature", 0.0),
        max_tokens=book_ranker_cfg.get("max_tokens", 1024),
        mmr_lambda=book_ranker_cfg.get("mmr_lambda", 0.6),
        mmr_candidates=book_ranker_cfg.get("mmr_candidates", 30),
    )
    book_ranker_node = make_book_ranker_node(book_ranker_agent, artifact_store=artifact_store)

    # ── Stage 3: ThemeExtractorAgent ──────────────────────────────────────
    from storymesh.agents.theme_extractor.agent import ThemeExtractorAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.theme_extractor import (  # noqa: PLC0415
        make_theme_extractor_node,
    )

    theme_cfg = get_agent_config("theme_extractor")
    theme_llm = _build_llm_client(theme_cfg)

    if theme_llm is None:
        logger.warning(
            "ThemeExtractorAgent: no LLM client available — stage 3 will run as noop."
        )
        theme_extractor_node: Any = _noop_node
    else:
        theme_agent = ThemeExtractorAgent(
            llm_client=theme_llm,
            temperature=theme_cfg.get("temperature", 0.6),
            max_tokens=theme_cfg.get("max_tokens", 4096),
            max_seeds=theme_cfg.get("max_seeds", 5),
        )
        theme_extractor_node = make_theme_extractor_node(
            theme_agent, artifact_store=artifact_store
        )

    # ── Build the graph ────────────────────────────────────────────────────
    graph: Any = StateGraph(StoryMeshState)

    graph.add_node("genre_normalizer", genre_node)
    graph.add_node("book_fetcher", book_fetcher_node)
    graph.add_node("book_ranker", book_ranker_node)
    graph.add_node("theme_extractor", theme_extractor_node)
    graph.add_node("proposal_draft", _noop_node)    # Stage 4 — placeholder (LLM)
    graph.add_node("rubric_judge", _noop_node)      # Stage 5 — placeholder (LLM, conditional)
    graph.add_node("synopsis_writer", _noop_node)   # Stage 6 — placeholder (LLM)

    # ── Wire edges (linear) ────────────────────────────────────────────────
    graph.add_edge(START, "genre_normalizer")
    graph.add_conditional_edges(
        "genre_normalizer",
        _genre_normalizer_route,
        {"book_fetcher": "book_fetcher", END: END},
    )
    graph.add_edge("book_fetcher", "book_ranker")
    graph.add_edge("book_ranker", "theme_extractor")
    graph.add_edge("theme_extractor", "proposal_draft")
    graph.add_edge("proposal_draft", "rubric_judge")
    graph.add_conditional_edges(
        "rubric_judge",
        _rubric_route,
        {
            "synopsis_writer": "synopsis_writer",
            "proposal_draft": "proposal_draft",
        },
    )
    graph.add_edge("synopsis_writer", END)

    return graph.compile()
