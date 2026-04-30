"""LangGraph node wrapper for CoverArtAgent (Stage 7)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.cover_art.agent import CoverArtAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.cover_art import CoverArtAgentInput, CoverArtAgentOutput
from storymesh.versioning.schemas import COVER_ART_SCHEMA_VERSION

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def make_cover_art_node(
    agent: CoverArtAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for CoverArtAgent (Stage 7).

    Args:
        agent: A fully constructed CoverArtAgent instance.
        artifact_store: Optional store for artifact persistence.
            Pass None (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature StoryMeshState -> dict[str, Any].
    """

    def cover_art_node(state: StoryMeshState) -> dict[str, Any]:
        """Generate a cover image from the selected proposal.

        If proposal_draft_output is absent (upstream nooped), returns an
        empty dict so the pipeline progresses without error.
        """
        proposal_draft_output = state.get("proposal_draft_output")
        if proposal_draft_output is None:
            logger.warning(
                "cover_art_node: proposal_draft_output is None — skipping cover art generation."
            )
            return {}

        proposal = proposal_draft_output.proposal
        input_data = CoverArtAgentInput(
            image_prompt=proposal.image_prompt,
            title=proposal.title,
        )

        raw = agent.run(input_data)

        # Persist PNG and set image_path; skip gracefully without a store.
        image_path = ""
        if artifact_store is not None:
            run_id: str = state.get("run_id", "")
            artifact_store.save_run_binary(run_id, "cover_art.png", raw.image_bytes)
            image_path = str(artifact_store.runs_dir / run_id / "cover_art.png")

        output = CoverArtAgentOutput(
            image_path=image_path,
            image_prompt=raw.image_prompt,
            revised_prompt=raw.revised_prompt,
            model=raw.model,
            image_size=raw.image_size,
            image_quality=raw.image_quality,
            debug={
                "title": proposal.title,
                "latency_ms": raw.latency_ms,
                "source_image_prompt": proposal.image_prompt,
            },
            schema_version=COVER_ART_SCHEMA_VERSION,
        )

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(artifact_store, state.get("run_id", ""), "cover_art", output)

        return {"cover_art_output": output}

    return cover_art_node
