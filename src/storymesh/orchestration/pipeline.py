"""StoryMesh pipeline entrypoint.

Wraps the compiled LangGraph StateGraph. The public interface
(``StoryMeshPipeline.generate``) is unchanged from the perspective of
callers in ``storymesh.__init__`` and the CLI.

The graph is built lazily on the first ``generate()`` call and cached on
the instance, so constructing ``StoryMeshPipeline()`` does not require a
config file or API keys to be present.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from storymesh.core.artifacts import ArtifactStore
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.result import GenerationResult
from storymesh.versioning.package import __version__ as storymesh_version


class StoryMeshPipeline:
    """Deterministic orchestration entrypoint for StoryMesh."""

    def __init__(self) -> None:
        # Graph is built on the first generate() call to defer config loading.
        self._graph: Any = None

    def generate(self, genre: str) -> GenerationResult:
        """Run the StoryMesh pipeline for the given genre.

        On the first call, compiles the LangGraph pipeline (which loads
        config and constructs agents). Subsequent calls reuse the cached
        compiled graph. Assembles ``GenerationResult`` from the final state.

        :param genre: The fiction genre to generate a synopsis for.
        :type genre: str
        :return: A GenerationResult containing the synopsis and metadata.
        :rtype: GenerationResult
        """
        if self._graph is None:
            from storymesh.orchestration.graph import build_graph  # noqa: PLC0415

            self._graph = build_graph()

        initial_state: StoryMeshState = {
            "input_genre": genre,
            "pipeline_version": storymesh_version,
            "genre_normalizer_output": None,
            "genre_seed_fetcher_output": None,
            "seed_ranker_output": None,
            "book_profile_synthesizer_output": None,
            "theme_aggregator_output": None,
            "proposal_output": None,
            "rubric_judge_output": None,
            "synthesis_writer_output": None,
            "errors": [],
        }

        final_state: dict[str, Any] = self._graph.invoke(initial_state)

        # Persist artifacts
        run_id = uuid.uuid4().hex
        artifact_store = ArtifactStore()

        artifact_store.save_run(run_id, {
            "input_genre": genre,
            "pipeline_version": storymesh_version,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
            "run_id": run_id
        })

        stage_outputs: dict[str, object | None] = {
            "genre_normalizer": final_state.get("genre_normalizer_output"),
            "seed_fetcher": final_state.get("genre_seed_fetcher_output"),
            "seed_ranker": final_state.get("seed_ranker_output"),
            "book_profile_synthesizer": final_state.get("book_profile_synthesizer_output"),
            "theme_aggregator": final_state.get("theme_aggregator_output"),
            "proposal": final_state.get("proposal_output"),
            "rubric_judge": final_state.get("rubric_judge_output"),
            "synthesis_writer": final_state.get("synthesis_writer_output"),
        }

        for stage_name, output in stage_outputs.items():
            if output is None:
                continue
            if hasattr(output, "model_dump"):
                artifact_store.save_run_file(run_id, f"{stage_name}_output.json", output.model_dump())
            elif isinstance(output, dict):
                artifact_store.save_run_file(run_id, f"{stage_name}_output.json", output)

        # TODO: When SynthesisWriterAgent is implemented, replace this block with:
        #   synthesis_out = final_state.get("synthesis_writer_output")
        #   final_synopsis = synthesis_out.final_synopsis
        final_synopsis = (
            f"Placeholder synopsis for genre '{genre}'. "
            "SynthesisWriterAgent is not yet implemented."
        )

        return GenerationResult(
            final_synopsis=final_synopsis,
            scores={},
            similarity_risk={},
            metadata={
                "input_genre": final_state.get("input_genre", genre),
                "pipeline_version": storymesh_version,
                "run_id": run_id
            },
        )
