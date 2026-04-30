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

import orjson

from storymesh.core.artifacts import ArtifactStore, persist_node_output
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
            "cover_art_output": None,
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


def regenerate_cover_art(run_id: str | None = None) -> str:
    """Re-run CoverArtAgent for a previous pipeline run.

    Loads ``proposal_draft_output.json`` from the run directory, regenerates
    the cover image using the current ``storymesh.config.yaml`` settings, and
    overwrites ``cover_art.png`` and ``cover_art_output.json`` in the same run
    directory.

    Args:
        run_id: Run to regenerate for. Pass ``None`` (default) to target the
            most recent run.

    Returns:
        Absolute path to the regenerated ``cover_art.png``.

    Raises:
        RuntimeError: If no runs exist, or if ``proposal_draft_output.json`` is
            missing from the requested run.
        ValueError: If the cover art agent cannot be configured (no API key or
            unknown image provider).
    """
    from storymesh.agents.cover_art.agent import CoverArtAgent  # noqa: PLC0415
    from storymesh.config import get_agent_config  # noqa: PLC0415
    from storymesh.orchestration.graph import _build_image_client  # noqa: PLC0415
    from storymesh.schemas.cover_art import CoverArtAgentInput, CoverArtAgentOutput  # noqa: PLC0415
    from storymesh.schemas.proposal_draft import ProposalDraftAgentOutput  # noqa: PLC0415
    from storymesh.versioning.schemas import COVER_ART_SCHEMA_VERSION  # noqa: PLC0415

    store = ArtifactStore()

    # Resolve run_id — default to the most recent run.
    resolved_id: str
    if run_id is None:
        ids = store.list_run_ids()
        if not ids:
            raise RuntimeError("No runs found in the artifact store.")
        resolved_id = ids[0]
    else:
        resolved_id = run_id

    # Load and deserialise the proposal draft output.
    raw = store.load_run_file(resolved_id, "proposal_draft_output.json")
    if raw is None:
        raise RuntimeError(
            f"No proposal_draft_output.json found for run {resolved_id!r}. "
            "The proposal draft stage must have completed before cover art can be regenerated."
        )
    proposal_draft_output = ProposalDraftAgentOutput.model_validate(orjson.loads(raw))

    # Build the image client from current config.
    cover_cfg = get_agent_config("cover_art")
    image_client = _build_image_client(cover_cfg, agent_name="cover_art")
    if image_client is None:
        raise ValueError(
            "Cannot build image client — OPENAI_API_KEY is not set or image_provider "
            "is not configured in storymesh.config.yaml."
        )

    agent = CoverArtAgent(
        image_client=image_client,
        image_size=str(cover_cfg.get("image_size", "1024x1792")),
        image_quality=str(cover_cfg.get("image_quality", "auto")),
    )

    proposal = proposal_draft_output.proposal
    raw_result = agent.run(
        CoverArtAgentInput(
            image_prompt=proposal.image_prompt,
            title=proposal.title,
        )
    )

    # Persist — overwrite the previous PNG and JSON sidecar.
    store.save_run_binary(resolved_id, "cover_art.png", raw_result.image_bytes)
    image_path = str(store.runs_dir / resolved_id / "cover_art.png")

    output = CoverArtAgentOutput(
        image_path=image_path,
        image_prompt=raw_result.image_prompt,
        revised_prompt=raw_result.revised_prompt,
        model=raw_result.model,
        image_size=raw_result.image_size,
        image_quality=raw_result.image_quality,
        debug={
            "title": proposal.title,
            "latency_ms": raw_result.latency_ms,
            "source_image_prompt": proposal.image_prompt,
            "regenerated": True,
        },
        schema_version=COVER_ART_SCHEMA_VERSION,
    )
    persist_node_output(store, resolved_id, "cover_art", output)

    return image_path
