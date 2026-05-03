"""LangGraph node wrapper for BookAssemblerAgent (Stage 8)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.book_assembler.agent import BookAssemblerAgent
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.book_assembler import BookAssemblerAgentInput, BookAssemblerAgentOutput
from storymesh.versioning.schemas import BOOK_ASSEMBLER_SCHEMA_VERSION

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def make_book_assembler_node(
    agent: BookAssemblerAgent,
    artifact_store: ArtifactStore | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for BookAssemblerAgent (Stage 8).

    Args:
        agent: A fully constructed BookAssemblerAgent instance.
        artifact_store: Optional store for artifact persistence.
            Pass None (default) to skip persistence (e.g. in unit tests).

    Returns:
        A node callable with signature StoryMeshState -> dict[str, Any].
    """

    def book_assembler_node(state: StoryMeshState) -> dict[str, Any]:
        """Assemble the PDF and EPUB from completed story and cover art.

        If ``story_writer_output`` is absent the node returns an empty dict,
        allowing the pipeline to reach END without error when the story writer
        ran as a noop.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update dict with ``book_assembler_output`` set,
            or an empty dict if required upstream outputs are missing.
        """
        story_writer_output = state.get("story_writer_output")
        if story_writer_output is None:
            logger.warning(
                "book_assembler_node: story_writer_output is None — skipping assembly."
            )
            return {}

        proposal_draft_output = state.get("proposal_draft_output")
        if proposal_draft_output is None:
            logger.warning(
                "book_assembler_node: proposal_draft_output is None — skipping assembly."
            )
            return {}

        run_id: str = state.get("run_id", "")
        cover_art_output = state.get("cover_art_output")

        input_data = BookAssemblerAgentInput(
            story_writer_output=story_writer_output,
            proposal=proposal_draft_output.proposal,
            cover_art_output=cover_art_output,
            run_id=run_id,
        )

        raw = agent.run(input_data)

        # Persist PDF and EPUB binaries; build output with the resulting paths.
        pdf_path = ""
        epub_path = ""
        if artifact_store is not None:
            if raw.pdf_bytes:
                artifact_store.save_run_binary(run_id, "output.pdf", raw.pdf_bytes)
                pdf_path = str(artifact_store.runs_dir / run_id / "output.pdf")
            if raw.epub_bytes:
                artifact_store.save_run_binary(run_id, "output.epub", raw.epub_bytes)
                epub_path = str(artifact_store.runs_dir / run_id / "output.epub")

        output = BookAssemblerAgentOutput(
            pdf_path=pdf_path,
            epub_path=epub_path,
            title=raw.title,
            word_count=raw.word_count,
            debug=raw.debug,
            schema_version=BOOK_ASSEMBLER_SCHEMA_VERSION,
        )

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(artifact_store, run_id, "book_assembler", output)

        return {"book_assembler_output": output}

    return book_assembler_node
