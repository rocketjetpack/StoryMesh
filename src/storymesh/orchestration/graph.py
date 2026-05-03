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
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from storymesh.config import get_agent_config
from storymesh.core.artifacts import ArtifactStore
from storymesh.llm.base import LLMClient
from storymesh.llm.image_base import ImageClient
from storymesh.orchestration.nodes.genre_normalizer import make_genre_normalizer_node
from storymesh.orchestration.state import StoryMeshState

logger = logging.getLogger(__name__)

# Future node imports (uncomment as agents are implemented):
# (story_writer is imported lazily inside build_graph below)


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

# Maps image provider names to the environment variable names for their API keys.
_IMAGE_PROVIDER_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
}

# Maps image provider names to their implementing module.
_IMAGE_PROVIDER_MODULE_MAP: dict[str, str] = {
    "openai": "storymesh.llm.openai_image",
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


def _build_llm_client(
    agent_cfg: dict[str, Any],
    agent_name: str = "unknown",
    artifact_store: ArtifactStore | None = None,
) -> LLMClient | None:
    """Instantiate the correct LLMClient subclass from an agent config dict.

    Uses the provider registry in ``storymesh.llm.base``. Returns ``None``
    with a warning if the required API key is not set, allowing agents to
    run in static-only mode.

    Args:
        agent_cfg: Resolved agent config dict from ``get_agent_config()``.
        agent_name: Label used in LLM call records (matches the LangGraph node name).
        artifact_store: If provided, wires ``ArtifactStore.log_llm_call`` as the
            ``on_call`` handler so every LLM call is appended to llm_calls.jsonl.

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
    on_call = artifact_store.log_llm_call if artifact_store is not None else None
    return cls(model=model, agent_name=agent_name, on_call=on_call)


def _ensure_image_provider_imported(provider: str) -> None:
    """Import the image provider module so its ``register_image_provider()`` call executes.

    Args:
        provider: Provider name string (e.g. ``'openai'``).
    """
    import importlib  # noqa: PLC0415

    module_name = _IMAGE_PROVIDER_MODULE_MAP.get(provider)
    if module_name:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.warning(
                "Image provider module '%s' could not be imported. "
                "Install the corresponding extra: pip install storymesh[%s]",
                module_name,
                provider,
            )


def _build_image_client(
    agent_cfg: dict[str, Any],
    agent_name: str = "unknown",
) -> ImageClient | None:
    """Instantiate the correct ImageClient from an agent config dict.

    Returns None with a warning if the required API key is not set.

    Args:
        agent_cfg: Resolved agent config dict from ``get_agent_config()``.
        agent_name: Label used in log output.

    Returns:
        A concrete ``ImageClient`` instance, or ``None`` if the API key is absent.

    Raises:
        ValueError: If ``agent_cfg["image_provider"]`` is not registered.
    """
    from storymesh.llm.image_base import get_image_provider_class  # noqa: PLC0415

    provider: str | None = agent_cfg.get("image_provider")
    model: str | None = agent_cfg.get("image_model")

    if provider is None:
        return None

    env_key = _IMAGE_PROVIDER_KEY_MAP.get(provider)
    if env_key and not os.environ.get(env_key):
        logger.warning(
            "%s is not set — the cover_art stage will run as noop.",
            env_key,
        )
        return None

    _ensure_image_provider_imported(provider)
    cls = get_image_provider_class(provider)
    return cls(model=model, agent_name=agent_name)  # type: ignore[arg-type]


_DEFAULT_MAX_RUBRIC_RETRIES: int = 2
"""Default maximum number of times rubric_judge may route back to proposal_draft."""


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


def _make_rubric_route(max_retries: int, min_retries: int = 0) -> Callable[[StoryMeshState], str]:
    """Create a rubric routing function with a configurable retry budget.

    Args:
        max_retries: Maximum times rubric_judge may route back to proposal_draft.
        min_retries: Minimum retries before a passing proposal can proceed.
            When > 0, even passing proposals are routed back for revision
            until ``retry_count >= min_retries``.

    Returns:
        A routing function suitable for ``add_conditional_edges``.
    """

    def _rubric_route(state: StoryMeshState) -> str:
        """Conditional routing function after rubric_judge.

        Routes to ``'proposal_reader_feedback'`` (which feeds into
        ``'proposal_draft'``) when:
        - The minimum retry count has not been met (even if passed), OR
        - The rubric failed and the max retry budget is not exhausted.

        Otherwise routes to ``'story_writer'``.

        Args:
            state: Current pipeline state.

        Returns:
            Name of the next node to execute.
        """
        retry_count: int = state.get("rubric_retry_count", 0)
        rubric_output = state.get("rubric_judge_output")

        # None means the noop placeholder ran (no LLM key) — treat as passed.
        passed: bool = rubric_output is None or bool(rubric_output.passed)

        # Enforce minimum retries: even passing proposals must go through
        # at least min_retries editorial revision cycles.
        if retry_count < min_retries and retry_count < max_retries:
            return "proposal_reader_feedback"

        if passed or retry_count >= max_retries:
            return "story_writer"
        return "proposal_reader_feedback"

    return _rubric_route


def build_graph(
    artifact_store: ArtifactStore | None = None,
    *,
    pass_threshold: int | None = None,
    max_retries: int | None = None,
    min_retries: int = 0,
    skip_resonance_review: bool = True,
) -> Any:  # noqa: ANN401  # CompiledStateGraph generics not resolvable under mypy strict.
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
              ├── PASS → story_writer → resonance_reviewer → cover_art → book_assembler → END
              └── FAIL / min_retries not met → proposal_draft (up to *max_retries*)

    Notes:
    - The rubric retry loop is wired via ``_make_rubric_route``; without an
      LLM key the rubric node runs as noop and always passes.
    - Agents without API keys degrade gracefully to noop nodes.
    - The resonance reviewer is skipped (noop) when ``skip_resonance_review``
      is True (default). Quality presets ``high`` and ``very_high`` set it
      to False.
    - Checkpointer: pass ``checkpointer=MemorySaver()`` to ``compile()``
      when HITL or run-persistence is needed.

    Args:
        artifact_store: Optional store passed to node factories for per-node
            artifact persistence. Pass ``None`` to skip persistence.
        pass_threshold: Override rubric pass threshold. When ``None`` the
            value from ``storymesh.config.yaml`` (or the default of 6) is used.
        max_retries: Override rubric retry budget. When ``None`` the module
            default (``_DEFAULT_MAX_RUBRIC_RETRIES``) is used.
        min_retries: Minimum editorial revision cycles before a passing
            proposal can proceed. Default 0 (no mandatory retries).
        skip_resonance_review: When True (default), the resonance reviewer
            node passes through without making LLM calls. Set to False by
            ``high`` and ``very_high`` quality presets.

    Returns:
        A compiled LangGraph StateGraph ready for ``.stream()`` or
        ``.invoke()``.
    """
    resolved_max_retries = max_retries if max_retries is not None else _DEFAULT_MAX_RUBRIC_RETRIES
    # ── Stage 0: GenreNormalizerAgent ──────────────────────────────────────
    genre_cfg = get_agent_config("genre_normalizer")
    genre_llm = _build_llm_client(genre_cfg, agent_name="genre_normalizer", artifact_store=artifact_store)

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
    book_ranker_llm = (
        _build_llm_client(book_ranker_cfg, agent_name="book_ranker", artifact_store=artifact_store)
        if book_ranker_cfg.get("llm_rerank")
        else None
    )

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
    theme_llm = _build_llm_client(theme_cfg, agent_name="theme_extractor", artifact_store=artifact_store)

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

    # ── Stage 4: ProposalDraftAgent ───────────────────────────────────────
    from storymesh.agents.proposal_draft.agent import ProposalDraftAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.proposal_draft import (  # noqa: PLC0415
        make_proposal_draft_node,
    )

    proposal_cfg = get_agent_config("proposal_draft")
    proposal_llm = _build_llm_client(
        proposal_cfg, agent_name="proposal_draft", artifact_store=artifact_store
    )

    if proposal_llm is None:
        logger.warning(
            "ProposalDraftAgent: no LLM client available — stage 4 will run as noop."
        )
        proposal_draft_node: Any = _noop_node
    else:
        proposal_agent = ProposalDraftAgent(
            llm_client=proposal_llm,
            temperature=proposal_cfg.get("temperature", 1.2),
            max_tokens=proposal_cfg.get("max_tokens", 4096),
            num_candidates=proposal_cfg.get("num_candidates", 3),
            selection_temperature=proposal_cfg.get("selection_temperature", 0.2),
            selection_max_tokens=proposal_cfg.get("selection_max_tokens", 2048),
            revision_temperature=proposal_cfg.get("revision_temperature", 0.5),
        )
        proposal_draft_node = make_proposal_draft_node(
            proposal_agent, artifact_store=artifact_store
        )

    # ── Stage 4.5: ProposalReaderAgent (retry path only) ──────────────────
    from storymesh.agents.proposal_reader.agent import ProposalReaderAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.proposal_reader import (  # noqa: PLC0415
        make_proposal_reader_node,
    )

    reader_cfg = get_agent_config("proposal_reader")
    reader_llm = _build_llm_client(
        reader_cfg, agent_name="proposal_reader", artifact_store=artifact_store
    )

    if reader_llm is None:
        logger.warning(
            "ProposalReaderAgent: no LLM client available — stage 4.5 will run as noop."
        )
        proposal_reader_node: Any = _noop_node
    else:
        reader_agent = ProposalReaderAgent(
            llm_client=reader_llm,
            temperature=reader_cfg.get("temperature", 0.4),
            max_tokens=reader_cfg.get("max_tokens", 1024),
        )
        proposal_reader_node = make_proposal_reader_node(
            reader_agent, artifact_store=artifact_store
        )

    # ── Stage 5: RubricJudgeAgent ─────────────────────────────────────────
    from storymesh.agents.rubric_judge.agent import RubricJudgeAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.rubric_judge import (  # noqa: PLC0415
        make_rubric_judge_node,
    )

    rubric_cfg = get_agent_config("rubric_judge")
    rubric_llm = _build_llm_client(
        rubric_cfg, agent_name="rubric_judge", artifact_store=artifact_store
    )

    if rubric_llm is None:
        logger.warning(
            "RubricJudgeAgent: no LLM client available — stage 5 will run as noop."
        )
        rubric_judge_node: Any = _noop_node
    else:
        resolved_threshold = (
            pass_threshold
            if pass_threshold is not None
            else rubric_cfg.get("pass_threshold", 6)
        )
        rubric_agent = RubricJudgeAgent(
            llm_client=rubric_llm,
            temperature=rubric_cfg.get("temperature", 0.0),
            max_tokens=rubric_cfg.get("max_tokens", 4096),
            pass_threshold=resolved_threshold,
        )
        rubric_judge_node = make_rubric_judge_node(rubric_agent, artifact_store=artifact_store)

    # ── Stage 6: StoryWriterAgent ─────────────────────────────────────────
    from storymesh.agents.story_writer.agent import StoryWriterAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.story_writer import (  # noqa: PLC0415
        make_story_writer_node,
    )

    story_cfg = get_agent_config("story_writer")
    story_llm = _build_llm_client(
        story_cfg, agent_name="story_writer", artifact_store=artifact_store
    )

    if story_llm is None:
        logger.warning(
            "StoryWriterAgent: no LLM client available — stage 6 will run as noop."
        )
        story_writer_node: Any = _noop_node
    else:
        story_agent = StoryWriterAgent(
            llm_client=story_llm,
            outline_temperature=story_cfg.get("outline_temperature", 0.5),
            draft_temperature=story_cfg.get("draft_temperature", 0.8),
            summary_temperature=story_cfg.get("summary_temperature", 0.4),
            outline_max_tokens=story_cfg.get("outline_max_tokens", 4096),
            draft_max_tokens=story_cfg.get("draft_max_tokens", 8000),
            summary_max_tokens=story_cfg.get("summary_max_tokens", 1024),
            target_words=story_cfg.get("target_words", 3000),
        )
        story_writer_node = make_story_writer_node(story_agent, artifact_store=artifact_store)

    # ── Stage 6b: ResonanceReviewerAgent ─────────────────────────────────
    from storymesh.agents.resonance_reviewer.agent import ResonanceReviewerAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.resonance_reviewer import (  # noqa: PLC0415
        make_resonance_reviewer_node,
    )

    resonance_cfg = get_agent_config("resonance_reviewer")
    review_llm = _build_llm_client(
        {
            "provider": resonance_cfg.get("review_provider"),
            "model": resonance_cfg.get("review_model"),
        },
        agent_name="resonance_reviewer_review",
        artifact_store=artifact_store,
    )
    revision_llm = _build_llm_client(
        {
            "provider": resonance_cfg.get("revision_provider"),
            "model": resonance_cfg.get("revision_model"),
        },
        agent_name="resonance_reviewer_revision",
        artifact_store=artifact_store,
    )

    if review_llm is None or revision_llm is None or skip_resonance_review:
        if not skip_resonance_review:
            logger.warning(
                "ResonanceReviewerAgent: missing LLM client(s) — stage 6b will run as noop."
            )
        resonance_reviewer_node: Any = _noop_node
    else:
        resonance_agent = ResonanceReviewerAgent(
            review_llm_client=review_llm,
            revision_llm_client=revision_llm,
            review_temperature=resonance_cfg.get("review_temperature", 0.4),
            revision_temperature=resonance_cfg.get("revision_temperature", 0.7),
            summary_temperature=resonance_cfg.get("summary_temperature", 0.4),
            review_max_tokens=resonance_cfg.get("review_max_tokens", 4096),
            revision_max_tokens=resonance_cfg.get("revision_max_tokens", 8000),
            summary_max_tokens=resonance_cfg.get("summary_max_tokens", 1024),
        )
        resonance_reviewer_node = make_resonance_reviewer_node(
            resonance_agent, artifact_store=artifact_store
        )

    # ── Stage 7: CoverArtAgent ────────────────────────────────────────────
    from storymesh.agents.cover_art.agent import CoverArtAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.cover_art import make_cover_art_node  # noqa: PLC0415

    cover_cfg = get_agent_config("cover_art")
    cover_image_client = _build_image_client(cover_cfg, agent_name="cover_art")

    if cover_image_client is None:
        logger.warning("CoverArtAgent: no image client available — stage 7 will run as noop.")
        cover_art_node: Any = _noop_node
    else:
        cover_agent = CoverArtAgent(
            image_client=cover_image_client,
            image_size=cover_cfg.get("image_size", "1024x1792"),
            image_quality=cover_cfg.get("image_quality", "auto"),
        )
        cover_art_node = make_cover_art_node(cover_agent, artifact_store=artifact_store)

    # ── Stage 8: BookAssemblerAgent ───────────────────────────────────────
    from storymesh.agents.book_assembler.agent import BookAssemblerAgent  # noqa: PLC0415
    from storymesh.orchestration.nodes.book_assembler import (  # noqa: PLC0415
        make_book_assembler_node,
    )

    assembler_cfg = get_agent_config("book_assembler")
    assembler_agent = BookAssemblerAgent(
        output_formats=assembler_cfg.get("output_formats", ["pdf", "epub"]),
    )
    book_assembler_node: Any = make_book_assembler_node(
        assembler_agent, artifact_store=artifact_store
    )

    # ── Build the graph ────────────────────────────────────────────────────
    graph: Any = StateGraph(StoryMeshState)

    graph.add_node("genre_normalizer", genre_node)
    graph.add_node("book_fetcher", book_fetcher_node)
    graph.add_node("book_ranker", book_ranker_node)
    graph.add_node("theme_extractor", theme_extractor_node)
    graph.add_node("proposal_draft", proposal_draft_node)
    graph.add_node("rubric_judge", rubric_judge_node)
    graph.add_node("proposal_reader_feedback", proposal_reader_node)  # Stage 4.5
    graph.add_node("story_writer", story_writer_node)            # Stage 6
    graph.add_node("resonance_reviewer", resonance_reviewer_node)  # Stage 6b
    graph.add_node("cover_art", cover_art_node)                    # Stage 7
    graph.add_node("book_assembler", book_assembler_node)          # Stage 8

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
        _make_rubric_route(resolved_max_retries, min_retries=min_retries),
        {
            "story_writer": "story_writer",
            "proposal_reader_feedback": "proposal_reader_feedback",
        },
    )
    graph.add_edge("proposal_reader_feedback", "proposal_draft")
    graph.add_edge("story_writer", "resonance_reviewer")
    graph.add_edge("resonance_reviewer", "cover_art")
    graph.add_edge("cover_art", "book_assembler")
    graph.add_edge("book_assembler", END)

    return graph.compile()
