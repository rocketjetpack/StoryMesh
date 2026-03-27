"""Unit tests for LangGraph node wrappers and StoryMeshState.

These tests exercise the node wrapper and state structure in isolation.
They use a pre-built MappingStore with a minimal in-memory taxonomy and
no LLM client, so no config file or API keys are required. The graph
compilation path (build_graph / get_config) is intentionally not exercised
here — that path is covered by test_generate.py which runs against the
committed storymesh.config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from storymesh.agents.genre_normalizer.agent import GenreNormalizerAgent
from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.orchestration.nodes.genre_normalizer import make_genre_normalizer_node
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentOutput

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def minimal_store(tmp_path: Path) -> MappingStore:
    """Build a MappingStore from a single-entry taxonomy for fast tests."""
    genre_map = {
        "fantasy": {
            "alternates": ["fantasie"],
            "genres": ["fantasy"],
            "subgenres": [],
            "default_tones": ["wondrous", "adventurous"],
        }
    }
    tone_map = {
        "dark": {
            "alternates": ["grim"],
            "normalized_tones": ["dark"],
        }
    }
    (tmp_path / "genre_map.json").write_bytes(orjson.dumps(genre_map))
    (tmp_path / "tone_map.json").write_bytes(orjson.dumps(tone_map))
    return MappingStore(
        genre_map_path=tmp_path / "genre_map.json",
        tone_map_path=tmp_path / "tone_map.json",
    )


@pytest.fixture()
def genre_agent(minimal_store: MappingStore) -> GenreNormalizerAgent:
    """GenreNormalizerAgent with a minimal store and no LLM client."""
    return GenreNormalizerAgent(store=minimal_store)


# ── TestGenreNormalizerNode ────────────────────────────────────────────────────


class TestGenreNormalizerNode:
    """Tests for the make_genre_normalizer_node factory and resulting node function."""

    def test_returns_genre_normalizer_output_type(
        self, genre_agent: GenreNormalizerAgent
    ) -> None:
        """The node must return a GenreNormalizerAgentOutput under the correct key."""
        node = make_genre_normalizer_node(genre_agent)
        state: StoryMeshState = {"user_prompt": "fantasy", "pipeline_version": "test"}

        result = node(state)

        assert "genre_normalizer_output" in result
        assert isinstance(result["genre_normalizer_output"], GenreNormalizerAgentOutput)

    def test_resolves_genre_from_state(
        self, genre_agent: GenreNormalizerAgent
    ) -> None:
        """The node must populate normalized_genres from the state's user_prompt."""
        node = make_genre_normalizer_node(genre_agent)
        state: StoryMeshState = {"user_prompt": "fantasy", "pipeline_version": "test"}

        result = node(state)

        output = result["genre_normalizer_output"]
        assert isinstance(output, GenreNormalizerAgentOutput)
        assert "fantasy" in output.normalized_genres

    def test_only_returns_own_key(
        self, genre_agent: GenreNormalizerAgent
    ) -> None:
        """The node must return a partial state dict with exactly one key."""
        node = make_genre_normalizer_node(genre_agent)
        state: StoryMeshState = {"user_prompt": "fantasy", "pipeline_version": "test"}

        result = node(state)

        assert set(result.keys()) == {"genre_normalizer_output"}

    def test_does_not_mutate_input_state(
        self, genre_agent: GenreNormalizerAgent
    ) -> None:
        """The node must treat input state as read-only."""
        node = make_genre_normalizer_node(genre_agent)
        state: StoryMeshState = {"user_prompt": "fantasy", "pipeline_version": "test"}
        original_keys = set(state.keys())

        node(state)

        assert set(state.keys()) == original_keys
        assert state["user_prompt"] == "fantasy"

    def test_different_agents_produce_independent_closures(
        self, minimal_store: MappingStore
    ) -> None:
        """Each make_genre_normalizer_node call must return an independent node."""
        agent_a = GenreNormalizerAgent(store=minimal_store)
        agent_b = GenreNormalizerAgent(store=minimal_store)
        node_a = make_genre_normalizer_node(agent_a)
        node_b = make_genre_normalizer_node(agent_b)

        assert node_a is not node_b

    def test_output_is_frozen(
        self, genre_agent: GenreNormalizerAgent
    ) -> None:
        """GenreNormalizerAgentOutput must be immutable (frozen Pydantic model)."""
        node = make_genre_normalizer_node(genre_agent)
        state: StoryMeshState = {"user_prompt": "fantasy", "pipeline_version": "test"}
        result = node(state)
        output = result["genre_normalizer_output"]

        assert isinstance(output, GenreNormalizerAgentOutput)
        with pytest.raises(Exception):  # noqa: B017
            output.normalized_genres = []  # type: ignore[misc]


# ── TestStoryMeshState ─────────────────────────────────────────────────────────


class TestStoryMeshState:
    """Tests for StoryMeshState TypedDict structure and partial construction."""

    def test_partial_state_is_valid(self) -> None:
        """total=False means a state dict with only some keys is valid."""
        state: StoryMeshState = {"user_prompt": "thriller"}
        assert state["user_prompt"] == "thriller"

    def test_pipeline_input_keys(self) -> None:
        """The two pipeline-level keys must be readable from state."""
        state: StoryMeshState = {
            "user_prompt": "mystery",
            "pipeline_version": "0.3.0",
        }
        assert state["user_prompt"] == "mystery"
        assert state["pipeline_version"] == "0.3.0"

    def test_full_initial_state_is_valid(self) -> None:
        """The full initial state dict used by StoryMeshPipeline must be constructable."""
        state: StoryMeshState = {
            "user_prompt": "cozy mystery",
            "pipeline_version": "0.4.0",
            "run_id": "abc123def456",
            "rubric_retry_count": 0,
            "genre_normalizer_output": None,
            "book_fetcher_output": None,
            "book_ranker_output": None,
            "theme_extractor_output": None,
            "proposal_draft_output": None,
            "rubric_judge_output": None,
            "synopsis_writer_output": None,
            "errors": [],
        }
        assert state["user_prompt"] == "cozy mystery"
        assert state["run_id"] == "abc123def456"
        assert state["rubric_retry_count"] == 0
        assert state["errors"] == []
        assert state["genre_normalizer_output"] is None

    def test_errors_field_accepts_list_of_strings(self) -> None:
        """The errors field must accept a list of strings."""
        state: StoryMeshState = {
            "user_prompt": "fantasy",
            "errors": ["seed_fetcher: timeout", "seed_ranker: no results"],
        }
        assert len(state["errors"]) == 2


# ── TestRubricRetryTopology ────────────────────────────────────────────────────


class TestRubricRetryTopology:
    """Tests for _rubric_route and the conditional edge in the compiled graph."""

    def test_routes_to_synopsis_writer_when_output_is_none(self) -> None:
        """Noop rubric_judge (output=None) must always route to synopsis_writer."""
        from storymesh.orchestration.graph import _rubric_route

        state: StoryMeshState = {"rubric_retry_count": 0, "rubric_judge_output": None}
        assert _rubric_route(state) == "synopsis_writer"

    def test_routes_to_proposal_draft_when_failed_and_retries_remain(self) -> None:
        """A failed rubric with retries remaining routes back to proposal_draft."""
        from storymesh.orchestration.graph import _rubric_route

        class _FailResult:
            passed = False

        state: StoryMeshState = {
            "rubric_retry_count": 0,
            "rubric_judge_output": _FailResult(),  # type: ignore[typeddict-item]
        }
        assert _rubric_route(state) == "proposal_draft"

    def test_routes_to_synopsis_writer_when_retry_budget_exhausted(self) -> None:
        """A failed rubric at MAX_RUBRIC_RETRIES forces progression to synopsis_writer."""
        from storymesh.orchestration.graph import MAX_RUBRIC_RETRIES, _rubric_route

        class _FailResult:
            passed = False

        state: StoryMeshState = {
            "rubric_retry_count": MAX_RUBRIC_RETRIES,
            "rubric_judge_output": _FailResult(),  # type: ignore[typeddict-item]
        }
        assert _rubric_route(state) == "synopsis_writer"

    def test_routes_to_synopsis_writer_when_passed(self) -> None:
        """A passing rubric result routes to synopsis_writer regardless of retry count."""
        from storymesh.orchestration.graph import _rubric_route

        class _PassResult:
            passed = True

        state: StoryMeshState = {
            "rubric_retry_count": 1,
            "rubric_judge_output": _PassResult(),  # type: ignore[typeddict-item]
        }
        assert _rubric_route(state) == "synopsis_writer"

    def test_compiled_graph_accepts_conditional_edge(self) -> None:
        """build_graph() must compile without errors with the conditional rubric edge."""
        from storymesh.orchestration.graph import build_graph

        graph = build_graph(artifact_store=None)
        assert graph is not None
