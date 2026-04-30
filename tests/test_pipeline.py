"""Unit tests for storymesh.orchestration.pipeline.StoryMeshPipeline.

These tests exercise ``StoryMeshPipeline.generate()`` end-to-end without
compiling a real LangGraph or hitting any LLM providers. ``build_graph``
is replaced with a fake that yields pre-recorded stream chunks, and
``warn_missing_provider_keys`` is stubbed so no config file or API keys
are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import orjson
import pytest

from storymesh.core.artifacts import ArtifactStore
from storymesh.orchestration.pipeline import StoryMeshPipeline
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentOutput
from storymesh.schemas.result import GenerationResult

# ---------------------------------------------------------------------------
# Helpers — fake compiled graph and genre output
# ---------------------------------------------------------------------------


def _make_genre_output() -> GenreNormalizerAgentOutput:
    """Build a minimal, valid GenreNormalizerAgentOutput for success-path tests."""
    return GenreNormalizerAgentOutput(
        raw_input="mystery",
        normalized_genres=["mystery"],
        subgenres=[],
        user_tones=[],
        tone_override=False,
        narrative_context=[],
        debug={},
    )


class _FakeGraph:
    """Stand-in for a compiled LangGraph ``StateGraph``.

    ``stream`` yields the chunks provided at construction time so tests
    can simulate any partial or full pipeline execution.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self.stream_calls: list[tuple[dict[str, Any], str]] = []

    def stream(self, state: dict[str, Any], *, stream_mode: str) -> list[dict[str, Any]]:
        self.stream_calls.append((state, stream_mode))
        return self._chunks


@pytest.fixture()
def tmp_store(tmp_path: Path) -> ArtifactStore:
    """ArtifactStore rooted in a per-test tmp directory."""
    return ArtifactStore(root=tmp_path)


def _make_pipeline(
    store: ArtifactStore,
    chunks: list[dict[str, Any]],
) -> tuple[StoryMeshPipeline, _FakeGraph]:
    """Construct a pipeline wired to a tmp ArtifactStore and a fake graph."""
    pipeline = StoryMeshPipeline()
    pipeline._artifact_store = store  # type: ignore[attr-defined]
    fake_graph = _FakeGraph(chunks)
    pipeline._graph = fake_graph  # type: ignore[attr-defined]
    return pipeline, fake_graph


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_graph_not_built_at_construction(self) -> None:
        """The graph must be lazily built on the first ``generate()`` call."""
        pipeline = StoryMeshPipeline()
        assert pipeline._graph is None  # type: ignore[attr-defined]

    def test_artifact_store_instantiated_eagerly(self) -> None:
        """An ArtifactStore is built eagerly so node closures share one instance."""
        pipeline = StoryMeshPipeline()
        assert isinstance(pipeline._artifact_store, ArtifactStore)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# generate() — lazy graph build
# ---------------------------------------------------------------------------


class TestLazyGraphBuild:
    def test_build_graph_called_on_first_generate(
        self, tmp_store: ArtifactStore
    ) -> None:
        """The first ``generate()`` invocation triggers ``build_graph`` once."""
        pipeline = StoryMeshPipeline()
        pipeline._artifact_store = tmp_store  # type: ignore[attr-defined]

        fake_graph = _FakeGraph([
            {"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}},
        ])

        with (
            patch(
                "storymesh.orchestration.graph.build_graph", return_value=fake_graph
            ) as mock_build,
            patch("storymesh.config.warn_missing_provider_keys"),
            patch("storymesh.config.get_config", return_value={}),
        ):
            pipeline.generate("mystery")

        mock_build.assert_called_once()

    def test_build_graph_not_called_second_time(
        self, tmp_store: ArtifactStore
    ) -> None:
        """A second call reuses the cached graph and does not rebuild."""
        pipeline = StoryMeshPipeline()
        pipeline._artifact_store = tmp_store  # type: ignore[attr-defined]

        fake_graph = _FakeGraph([
            {"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}},
        ])

        with (
            patch(
                "storymesh.orchestration.graph.build_graph", return_value=fake_graph
            ) as mock_build,
            patch("storymesh.config.warn_missing_provider_keys"),
            patch("storymesh.config.get_config", return_value={}),
        ):
            pipeline.generate("mystery")
            pipeline.generate("fantasy")

        mock_build.assert_called_once()

    def test_warn_missing_provider_keys_invoked_on_first_build(
        self, tmp_store: ArtifactStore
    ) -> None:
        """``warn_missing_provider_keys`` runs exactly once alongside the first build."""
        pipeline = StoryMeshPipeline()
        pipeline._artifact_store = tmp_store  # type: ignore[attr-defined]

        fake_graph = _FakeGraph([
            {"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}},
        ])

        with (
            patch("storymesh.orchestration.graph.build_graph", return_value=fake_graph),
            patch(
                "storymesh.config.warn_missing_provider_keys",
            ) as mock_warn,
            patch("storymesh.config.get_config", return_value={"some": "config"}),
        ):
            pipeline.generate("mystery")
            pipeline.generate("fantasy")

        mock_warn.assert_called_once_with({"some": "config"})


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------


class TestGenerateHappyPath:
    def test_returns_generation_result(self, tmp_store: ArtifactStore) -> None:
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}}],
        )
        result = pipeline.generate("mystery")

        assert isinstance(result, GenerationResult)

    def test_synopsis_placeholder_includes_user_prompt(
        self, tmp_store: ArtifactStore
    ) -> None:
        """Until StoryWriterAgent is implemented, the placeholder echoes the prompt."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}}],
        )
        result = pipeline.generate("a dark mystery")

        assert "a dark mystery" in result.final_synopsis
        assert "StoryWriterAgent is not yet implemented" in result.final_synopsis

    def test_metadata_includes_run_id_and_version(
        self, tmp_store: ArtifactStore
    ) -> None:
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}}],
        )
        result = pipeline.generate("mystery")

        assert "run_id" in result.metadata
        assert isinstance(result.metadata["run_id"], str)
        assert len(result.metadata["run_id"]) == 32  # uuid4 hex
        assert "pipeline_version" in result.metadata
        assert "run_dir" in result.metadata
        assert "user_prompt" in result.metadata
        assert result.metadata["user_prompt"] == "mystery"

    def test_metadata_records_stage_timings_for_each_streamed_node(
        self, tmp_store: ArtifactStore
    ) -> None:
        """Every streamed chunk contributes a key to ``stage_timings``."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [
                {"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}},
                {"book_fetcher": {"book_fetcher_output": None}},
                {"book_ranker": {"book_ranker_output": None}},
            ],
        )
        result = pipeline.generate("mystery")

        timings = result.metadata["stage_timings"]
        assert set(timings.keys()) == {"genre_normalizer", "book_fetcher", "book_ranker"}
        for v in timings.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_run_metadata_json_written_twice_with_final_timings(
        self, tmp_store: ArtifactStore
    ) -> None:
        """``run_metadata.json`` exists after ``generate`` and contains final timings."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [
                {"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}},
                {"book_fetcher": {"book_fetcher_output": None}},
            ],
        )
        result = pipeline.generate("mystery")

        run_dir = tmp_store.runs_dir / result.metadata["run_id"]
        metadata_path = run_dir / "run_metadata.json"
        assert metadata_path.exists()
        loaded = orjson.loads(metadata_path.read_bytes())
        assert loaded["user_prompt"] == "mystery"
        assert loaded["run_id"] == result.metadata["run_id"]
        # The final (second) write includes stage_timings; the initial write did not.
        assert "stage_timings" in loaded
        assert set(loaded["stage_timings"].keys()) == {"genre_normalizer", "book_fetcher"}

    def test_stream_invoked_with_initial_state_containing_all_keys(
        self, tmp_store: ArtifactStore
    ) -> None:
        """The state passed to ``graph.stream`` has every StoryMeshState key initialised."""
        pipeline, fake_graph = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}}],
        )
        pipeline.generate("mystery")

        assert len(fake_graph.stream_calls) == 1
        initial_state, stream_mode = fake_graph.stream_calls[0]
        assert stream_mode == "updates"
        for key in (
            "user_prompt",
            "pipeline_version",
            "run_id",
            "rubric_retry_count",
            "genre_normalizer_output",
            "book_fetcher_output",
            "book_ranker_output",
            "theme_extractor_output",
            "proposal_draft_output",
            "rubric_judge_output",
            "story_writer_output",
            "errors",
        ):
            assert key in initial_state, f"missing initial state key: {key}"
        assert initial_state["rubric_retry_count"] == 0
        assert initial_state["errors"] == []

    def test_each_generate_call_gets_unique_run_id(
        self, tmp_store: ArtifactStore
    ) -> None:
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": _make_genre_output()}}],
        )
        r1 = pipeline.generate("mystery")
        r2 = pipeline.generate("fantasy")

        assert r1.metadata["run_id"] != r2.metadata["run_id"]


# ---------------------------------------------------------------------------
# generate() — failure path: genre normalization returned None
# ---------------------------------------------------------------------------


class TestGenerateGenreFailurePath:
    def test_result_surfaces_user_friendly_error_message(
        self, tmp_store: ArtifactStore
    ) -> None:
        """A run where ``genre_normalizer_output`` stays ``None`` returns a friendly message."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [
                {
                    "genre_normalizer": {
                        "genre_normalizer_output": None,
                        "errors": ["No genres could be resolved from the input."],
                    },
                },
            ],
        )
        result = pipeline.generate("xyzzy frobb")

        assert "Could not generate a synopsis" in result.final_synopsis
        assert "No genres could be resolved" in result.final_synopsis

    def test_errors_propagated_to_result(self, tmp_store: ArtifactStore) -> None:
        pipeline, _ = _make_pipeline(
            tmp_store,
            [
                {
                    "genre_normalizer": {
                        "genre_normalizer_output": None,
                        "errors": ["err1", "err2"],
                    },
                },
            ],
        )
        result = pipeline.generate("xyzzy")

        assert result.errors == ["err1", "err2"]

    def test_failure_with_no_recorded_errors_uses_default_message(
        self, tmp_store: ArtifactStore
    ) -> None:
        """If no error strings were written, the user still sees a generic fallback."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": None}}],
        )
        result = pipeline.generate("xyzzy")

        assert "Genre normalization failed" in result.final_synopsis

    def test_failure_still_writes_run_metadata(self, tmp_store: ArtifactStore) -> None:
        """Run metadata is persisted even on an early-exit failure path."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": None}}],
        )
        result = pipeline.generate("xyzzy")

        run_dir = tmp_store.runs_dir / result.metadata["run_id"]
        assert (run_dir / "run_metadata.json").exists()

    def test_failure_still_returns_metadata(self, tmp_store: ArtifactStore) -> None:
        """The returned metadata is populated even when the pipeline short-circuits."""
        pipeline, _ = _make_pipeline(
            tmp_store,
            [{"genre_normalizer": {"genre_normalizer_output": None}}],
        )
        result = pipeline.generate("xyzzy")

        assert result.metadata["user_prompt"] == "xyzzy"
        assert "run_id" in result.metadata
        assert "pipeline_version" in result.metadata
