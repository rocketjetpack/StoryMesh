"""StoryMesh pipeline entrypoint.

Wraps the compiled LangGraph StateGraph. The public interface
(``StoryMeshPipeline.generate``) is unchanged from the perspective of
callers in ``storymesh.__init__`` and the CLI.

The graph is built lazily on the first ``generate()`` call and cached on
the instance, so constructing ``StoryMeshPipeline()`` does not require a
config file or API keys to be present.
"""

from __future__ import annotations

import time
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
        # Single ArtifactStore instance shared with node closures via build_graph().
        self._artifact_store = ArtifactStore()

    def generate(self, user_prompt: str) -> GenerationResult:
        """Run the StoryMesh pipeline for the given prompt.

        On the first call, compiles the LangGraph pipeline (which loads
        config and constructs agents). Subsequent calls reuse the cached
        compiled graph. Assembles ``GenerationResult`` from the final state.

        :param user_prompt: Free-text description of the desired fiction.
        :type user_prompt: str
        :return: A GenerationResult containing the synopsis and metadata.
        :rtype: GenerationResult
        """
        if self._graph is None:
            from storymesh.config import get_config, warn_missing_provider_keys  # noqa: PLC0415
            from storymesh.orchestration.graph import build_graph  # noqa: PLC0415

            warn_missing_provider_keys(get_config())
            self._graph = build_graph(artifact_store=self._artifact_store)

        # Generate run_id before invocation so metadata is on disk even if the
        # graph crashes mid-run (enables partial-run post-mortem inspection).
        run_id = uuid.uuid4().hex
        # Capture timestamp before graph execution so both the initial write and
        # the post-run update reflect the run *start* time, not the end time.
        run_timestamp = datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017

        self._artifact_store.save_run(run_id, {
            "user_prompt": user_prompt,
            "pipeline_version": storymesh_version,
            "timestamp": run_timestamp,
            "run_id": run_id,
        })

        initial_state: StoryMeshState = {
            "user_prompt": user_prompt,
            "pipeline_version": storymesh_version,
            "run_id": run_id,
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

        # Stream the graph to collect per-node timing without touching node code.
        # stream_mode="updates" yields one {node_name: state_delta} dict per node.
        stage_timings: dict[str, float] = {}
        final_state: dict[str, Any] = dict(initial_state)

        prev_time = time.perf_counter()
        for chunk in self._graph.stream(initial_state, stream_mode="updates"):
            now = time.perf_counter()
            for node_name, node_update in chunk.items():
                stage_timings[node_name] = round(now - prev_time, 4)
                if isinstance(node_update, dict):
                    final_state.update(node_update)
            prev_time = now

        # Individual stage artifacts are now written by each node as it
        # completes (see persist_node_output in core/artifacts.py), so no
        # post-invocation artifact loop is needed here.

        # Update run_metadata.json with stage timings now that the graph has
        # finished. The initial save (before graph execution) records the run
        # for crash post-mortems; this second write adds timing data for the
        # run inspector and any future tooling.
        self._artifact_store.save_run(run_id, {
            "user_prompt": user_prompt,
            "pipeline_version": storymesh_version,
            "timestamp": run_timestamp,
            "run_id": run_id,
            "stage_timings": stage_timings,
        })

        base_metadata: dict[str, Any] = {
            "user_prompt": final_state.get("user_prompt", user_prompt),
            "pipeline_version": storymesh_version,
            "run_id": run_id,
            "stage_timings": stage_timings,
            "run_dir": str(self._artifact_store.runs_dir / run_id),
        }

        # If genre normalization failed, the graph short-circuits to END and
        # genre_normalizer_output remains None. Surface the error as a readable
        # result rather than propagating a crash.
        if not final_state.get("genre_normalizer_output"):
            errors: list[str] = final_state.get("errors") or []
            user_message = errors[0] if errors else "Genre normalization failed."
            return GenerationResult(
                final_synopsis=f"Could not generate a synopsis: {user_message}",
                errors=errors,
                metadata=base_metadata,
            )

        # TODO: When SynopsisWriterAgent is implemented, replace this block with:
        #   synopsis_out = final_state.get("synopsis_writer_output")
        #   final_synopsis = synopsis_out.final_synopsis
        final_synopsis = (
            f"Placeholder synopsis for '{user_prompt}'. "
            "SynopsisWriterAgent is not yet implemented."
        )

        return GenerationResult(
            final_synopsis=final_synopsis,
            scores={},
            similarity_risk={},
            metadata=base_metadata,
        )
