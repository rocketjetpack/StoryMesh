"""Unit tests for LangGraph node wrappers and StoryMeshState.

These tests exercise the node wrapper and state structure in isolation.
They use a pre-built MappingStore with a minimal in-memory taxonomy and
no LLM client, so no config file or API keys are required. The graph
compilation path (build_graph / get_config) is intentionally not exercised
here — that path is covered by test_generate.py which runs against the
committed storymesh.config.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import orjson
import pytest

from storymesh.agents.book_ranker.agent import BookRankerAgent
from storymesh.agents.genre_normalizer.agent import GenreNormalizerAgent
from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.proposal_draft.agent import ProposalDraftAgent
from storymesh.agents.theme_extractor.agent import ThemeExtractorAgent
from storymesh.llm.base import FakeLLMClient
from storymesh.orchestration.nodes.book_ranker import make_book_ranker_node
from storymesh.orchestration.nodes.genre_normalizer import make_genre_normalizer_node
from storymesh.orchestration.nodes.proposal_draft import make_proposal_draft_node
from storymesh.orchestration.nodes.theme_extractor import make_theme_extractor_node
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.book_fetcher import BookFetcherAgentOutput, BookRecord
from storymesh.schemas.book_ranker import BookRankerAgentOutput, RankedBookSummary
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentOutput
from storymesh.schemas.proposal_draft import ProposalDraftAgentOutput
from storymesh.schemas.theme_extractor import (
    GenreCluster,
    NarrativeSeed,
    ThematicTension,
    ThemeExtractorAgentOutput,
)

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


# ── TestBookRankerNode ─────────────────────────────────────────────────────────


def _make_book_fetcher_output(books: list[BookRecord]) -> BookFetcherAgentOutput:
    return BookFetcherAgentOutput(
        books=books,
        queries_executed=["mystery"],
    )


def _make_book_record(work_key: str) -> BookRecord:
    return BookRecord(
        work_key=work_key,
        title=f"Book {work_key}",
        source_genres=["mystery"],
    )


class TestBookRankerNode:
    """Tests for the make_book_ranker_node factory and resulting node function."""

    def test_returns_book_ranker_output_type(self, tmp_path: Path) -> None:
        """The node must return a BookRankerAgentOutput under the correct key."""
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            agent = BookRankerAgent()

        node = make_book_ranker_node(agent)
        book_fetcher_output = _make_book_fetcher_output([_make_book_record("/works/OL1W")])
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "book_fetcher_output": book_fetcher_output,
        }

        result = node(state)

        assert "book_ranker_output" in result
        assert isinstance(result["book_ranker_output"], BookRankerAgentOutput)

    def test_only_returns_own_key(self, tmp_path: Path) -> None:
        """The node must return a partial state dict with exactly one key."""
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            agent = BookRankerAgent()

        node = make_book_ranker_node(agent)
        book_fetcher_output = _make_book_fetcher_output([_make_book_record("/works/OL1W")])
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "book_fetcher_output": book_fetcher_output,
        }

        result = node(state)

        assert set(result.keys()) == {"book_ranker_output"}

    def test_none_book_fetcher_output_raises_runtime_error(self) -> None:
        """Node raises RuntimeError when book_fetcher_output is None."""
        agent = BookRankerAgent()
        node = make_book_ranker_node(agent)
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "book_fetcher_output": None,
        }

        with pytest.raises(RuntimeError, match="book_fetcher_output.*None"):
            node(state)  # type: ignore[arg-type]

    def test_ranked_books_count_matches_input(self, tmp_path: Path) -> None:
        """Output ranked_books length must not exceed top_n (default 10)."""
        with patch(
            "storymesh.agents.book_fetcher.agent.get_cache_dir",
            return_value=tmp_path,
        ):
            agent = BookRankerAgent(top_n=2)

        node = make_book_ranker_node(agent)
        books = [_make_book_record(f"/works/OL{i}W") for i in range(5)]
        book_fetcher_output = _make_book_fetcher_output(books)
        state: StoryMeshState = {
            "user_prompt": "mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "book_fetcher_output": book_fetcher_output,
        }

        result = node(state)
        output: Any = result["book_ranker_output"]
        assert len(output.ranked_books) == 2


# ── TestThemeExtractorNode ────────────────────────────────────────────────────


def _valid_theme_response() -> str:
    return json.dumps({
        "genre_clusters": [
            {
                "genre": "mystery",
                "books": ["Book 1"],
                "thematic_assumptions": ["Truth is discoverable"],
                "dominant_tropes": [],
            }
        ],
        "tensions": [
            {
                "tension_id": "T1",
                "cluster_a": "mystery",
                "assumption_a": "Truth is discoverable",
                "cluster_b": "mystery",
                "assumption_b": "Some truths destroy the investigator",
                "creative_question": "What does solving a case cost the solver?",
                "intensity": 0.7,
                "cliched_resolutions": [
                    "The detective solves the case and walks away unscathed",
                ],
            }
        ],
        "narrative_seeds": [
            {
                "seed_id": "S1",
                "concept": "A former detective must reinvent investigation from first principles.",
                "tensions_used": ["T1"],
                "tonal_direction": [],
                "narrative_context_used": [],
            }
        ],
    })


def _make_genre_normalizer_output() -> GenreNormalizerAgentOutput:
    return GenreNormalizerAgentOutput(
        raw_input="dark mystery",
        normalized_genres=["mystery"],
        subgenres=[],
        user_tones=["dark"],
        tone_override=False,
        narrative_context=[],
        debug={},
    )


def _make_book_ranker_output() -> BookRankerAgentOutput:
    summary = RankedBookSummary(
        work_key="/works/OL1W",
        title="Book 1",
        authors=["Author"],
        source_genres=["mystery"],
        composite_score=0.9,
        rank=1,
    )
    return BookRankerAgentOutput(
        ranked_books=[],
        ranked_summaries=[summary],
        dropped_count=0,
        llm_reranked=False,
        debug={},
    )


def _make_theme_agent(response: str = "") -> ThemeExtractorAgent:
    client = FakeLLMClient(responses=[response or _valid_theme_response()])
    return ThemeExtractorAgent(llm_client=client)


class TestThemeExtractorNode:
    """Tests for the make_theme_extractor_node factory and resulting node function."""

    def test_returns_theme_extractor_output_type(self) -> None:
        """The node must return a ThemeExtractorAgentOutput under the correct key."""
        node = make_theme_extractor_node(_make_theme_agent())
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "genre_normalizer_output": _make_genre_normalizer_output(),
            "book_ranker_output": _make_book_ranker_output(),
        }

        result = node(state)

        assert "theme_extractor_output" in result
        assert isinstance(result["theme_extractor_output"], ThemeExtractorAgentOutput)

    def test_only_returns_own_key(self) -> None:
        """The node must return a partial state dict with exactly one key."""
        node = make_theme_extractor_node(_make_theme_agent())
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "genre_normalizer_output": _make_genre_normalizer_output(),
            "book_ranker_output": _make_book_ranker_output(),
        }

        result = node(state)

        assert set(result.keys()) == {"theme_extractor_output"}

    def test_none_genre_output_raises_runtime_error(self) -> None:
        """Node raises RuntimeError when genre_normalizer_output is None."""
        node = make_theme_extractor_node(_make_theme_agent())
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "genre_normalizer_output": None,
            "book_ranker_output": _make_book_ranker_output(),
        }

        with pytest.raises(RuntimeError, match="genre_normalizer_output.*None"):
            node(state)  # type: ignore[arg-type]

    def test_none_book_ranker_output_raises_runtime_error(self) -> None:
        """Node raises RuntimeError when book_ranker_output is None."""
        node = make_theme_extractor_node(_make_theme_agent())
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "genre_normalizer_output": _make_genre_normalizer_output(),
            "book_ranker_output": None,
        }

        with pytest.raises(RuntimeError, match="book_ranker_output.*None"):
            node(state)  # type: ignore[arg-type]

    def test_assembles_input_from_multiple_stages(self) -> None:
        """The assembled input carries data from both upstream outputs."""
        captured_inputs: list[object] = []

        class _RecordingAgent(ThemeExtractorAgent):
            def run(self, input_data: object) -> ThemeExtractorAgentOutput:  # type: ignore[override]
                captured_inputs.append(input_data)
                return super().run(input_data)  # type: ignore[arg-type]

        client = FakeLLMClient(responses=[_valid_theme_response()])
        agent = _RecordingAgent(llm_client=client)
        node = make_theme_extractor_node(agent)

        genre_out = _make_genre_normalizer_output()
        state: StoryMeshState = {
            "user_prompt": "dark mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "genre_normalizer_output": genre_out,
            "book_ranker_output": _make_book_ranker_output(),
        }
        node(state)

        assert len(captured_inputs) == 1
        inp = captured_inputs[0]
        assert hasattr(inp, "normalized_genres")
        assert hasattr(inp, "ranked_summaries")


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


# ── TestGenreNormalizerRoute ───────────────────────────────────────────────────


class TestGenreNormalizerRoute:
    """Tests for _genre_normalizer_route and the node's error handling."""

    def test_routes_to_book_fetcher_when_output_is_present(self) -> None:
        """A populated genre_normalizer_output routes to book_fetcher."""
        from storymesh.orchestration.graph import _genre_normalizer_route

        state: StoryMeshState = {
            "genre_normalizer_output": object(),  # type: ignore[typeddict-item]
        }
        assert _genre_normalizer_route(state) == "book_fetcher"

    def test_routes_to_end_when_output_is_none(self) -> None:
        """genre_normalizer_output=None must route to END."""
        from langgraph.graph import END

        from storymesh.orchestration.graph import _genre_normalizer_route

        state: StoryMeshState = {"genre_normalizer_output": None}
        assert _genre_normalizer_route(state) == END

    def test_routes_to_end_when_output_key_absent(self) -> None:
        """Missing genre_normalizer_output key must also route to END."""
        from langgraph.graph import END

        from storymesh.orchestration.graph import _genre_normalizer_route

        state: StoryMeshState = {}
        assert _genre_normalizer_route(state) == END

    def test_node_catches_genre_resolution_error(
        self, minimal_store: MappingStore
    ) -> None:
        """Node must catch GenreResolutionError and write to the errors state key."""
        agent = GenreNormalizerAgent(store=minimal_store)
        node = make_genre_normalizer_node(agent)
        state: StoryMeshState = {
            "user_prompt": "xyzzy frobb glorp",
            "pipeline_version": "test",
        }

        result = node(state)

        assert result["genre_normalizer_output"] is None
        assert len(result["errors"]) == 1
        assert "No genres could be resolved" in result["errors"][0]

    def test_node_error_returns_two_keys(
        self, minimal_store: MappingStore
    ) -> None:
        """On error the node returns exactly genre_normalizer_output and errors."""
        agent = GenreNormalizerAgent(store=minimal_store)
        node = make_genre_normalizer_node(agent)
        state: StoryMeshState = {"user_prompt": "xyzzy frobb", "pipeline_version": "test"}

        result = node(state)

        assert set(result.keys()) == {"genre_normalizer_output", "errors"}


# ── TestProposalDraftNode ──────────────────────────────────────────────────────


_PROPOSAL_PLOT_ARC = (
    "Act 1: Mara Voss finds a body arranged with deliberate symbolism in the flooded "
    "district. Act 2: She must rebuild the infrastructure of investigation from scratch. "
    "Act 3: She convenes a community tribunal that delivers a verdict she must enforce alone."
)


def _make_theme_extractor_output_for_proposal() -> ThemeExtractorAgentOutput:
    cluster = GenreCluster(
        genre="mystery",
        books=["The Big Sleep"],
        thematic_assumptions=["Truth is recoverable through investigation"],
    )
    tension = ThematicTension(
        tension_id="T1",
        cluster_a="mystery",
        assumption_a="Truth is recoverable",
        cluster_b="post_apocalyptic",
        assumption_b="Records and institutions no longer exist",
        creative_question="What does investigation mean without infrastructure?",
        intensity=0.9,
        cliched_resolutions=["A lone detective rebuilds justice through sheer determination"],
    )
    seed = NarrativeSeed(
        seed_id="S1",
        concept="A scavenger detective reinvents investigation in a collapsed city.",
        tensions_used=["T1"],
    )
    return ThemeExtractorAgentOutput(
        genre_clusters=[cluster],
        tensions=[tension],
        narrative_seeds=[seed],
        user_tones_carried=["dark"],
    )


def _valid_proposal_json_for_node() -> str:
    return json.dumps({
        "seed_id": "S1",
        "title": "The Last Inquest",
        "protagonist": (
            "Mara Voss — former homicide detective whose faith in due process "
            "survived the collapse even as the process itself did not."
        ),
        "setting": (
            "A flooded mid-21st-century city-state where municipal records "
            "were lost in the first year of collapse."
        ),
        "plot_arc": _PROPOSAL_PLOT_ARC,
        "thematic_thesis": (
            "Justice does not require institutions to be meaningful, "
            "but meaning without institutions cannot produce justice."
        ),
        "key_scenes": [
            "Mara finds the arranged body and recognises the signature.",
            "She convenes a community tribunal with no legal authority.",
        ],
        "tensions_addressed": ["T1"],
        "tone": ["dark"],
        "genre_blend": ["mystery", "post_apocalyptic"],
    })


def _make_proposal_agent(num_candidates: int = 1) -> ProposalDraftAgent:
    """Build a ProposalDraftAgent with enough fake responses to complete a run."""
    proposals = [_valid_proposal_json_for_node() for _ in range(num_candidates)]
    rationale = json.dumps({
        "selected_index": 0,
        "rationale": "Only one valid candidate; selected by default.",
        "cliche_violations": {},
        "runner_up_index": None,
    })
    client = FakeLLMClient(responses=proposals + ([rationale] if num_candidates > 1 else []))
    return ProposalDraftAgent(
        llm_client=client,
        num_candidates=num_candidates,
    )


class TestProposalDraftNode:
    """Tests for the make_proposal_draft_node factory and resulting node function."""

    def _base_state(self) -> StoryMeshState:
        return {
            "user_prompt": "dark post-apocalyptic mystery",
            "pipeline_version": "test",
            "run_id": "abc123",
            "genre_normalizer_output": _make_genre_normalizer_output(),
            "theme_extractor_output": _make_theme_extractor_output_for_proposal(),
        }

    def test_returns_proposal_draft_output_type(self) -> None:
        """The node must return a ProposalDraftAgentOutput under the correct key."""
        node = make_proposal_draft_node(_make_proposal_agent())
        result = node(self._base_state())
        assert "proposal_draft_output" in result
        assert isinstance(result["proposal_draft_output"], ProposalDraftAgentOutput)

    def test_only_returns_own_key(self) -> None:
        """The node must return a partial state dict with exactly one key."""
        node = make_proposal_draft_node(_make_proposal_agent())
        result = node(self._base_state())
        assert set(result.keys()) == {"proposal_draft_output"}

    def test_missing_theme_extractor_output_raises(self) -> None:
        """RuntimeError when theme_extractor_output is None."""
        node = make_proposal_draft_node(_make_proposal_agent())
        state = self._base_state()
        state["theme_extractor_output"] = None
        with pytest.raises(RuntimeError, match="theme_extractor_output.*None"):
            node(state)  # type: ignore[arg-type]

    def test_missing_genre_normalizer_output_raises(self) -> None:
        """RuntimeError when genre_normalizer_output is None."""
        node = make_proposal_draft_node(_make_proposal_agent())
        state = self._base_state()
        state["genre_normalizer_output"] = None
        with pytest.raises(RuntimeError, match="genre_normalizer_output.*None"):
            node(state)  # type: ignore[arg-type]

    def test_assembles_input_from_multiple_stages(self) -> None:
        """Input must carry narrative_seeds from theme_extractor and narrative_context
        from genre_normalizer."""
        captured_inputs: list[object] = []

        class _RecordingAgent(ProposalDraftAgent):
            def run(self, input_data: object) -> ProposalDraftAgentOutput:  # type: ignore[override]
                captured_inputs.append(input_data)
                return super().run(input_data)  # type: ignore[arg-type]

        client = FakeLLMClient(responses=[_valid_proposal_json_for_node()])
        agent = _RecordingAgent(llm_client=client, num_candidates=1)
        node = make_proposal_draft_node(agent)
        node(self._base_state())

        assert len(captured_inputs) == 1
        inp = captured_inputs[0]
        assert hasattr(inp, "narrative_seeds")
        assert hasattr(inp, "narrative_context")
        assert hasattr(inp, "user_tones")

    def test_output_key_is_proposal_draft_output(self) -> None:
        """The returned dict key must be exactly 'proposal_draft_output'."""
        node = make_proposal_draft_node(_make_proposal_agent())
        result = node(self._base_state())
        assert "proposal_draft_output" in result

    def test_current_run_id_set_during_execution(self) -> None:
        """ContextVar current_run_id must be set to state['run_id'] during agent.run()."""
        from storymesh.llm.base import current_run_id as _crid

        observed_run_ids: list[str] = []

        class _ObservingAgent(ProposalDraftAgent):
            def run(self, input_data: object) -> ProposalDraftAgentOutput:  # type: ignore[override]
                observed_run_ids.append(_crid.get())
                return super().run(input_data)  # type: ignore[arg-type]

        client = FakeLLMClient(responses=[_valid_proposal_json_for_node()])
        agent = _ObservingAgent(llm_client=client, num_candidates=1)
        node = make_proposal_draft_node(agent)
        node(self._base_state())

        assert observed_run_ids == ["abc123"]
