"""LangGraph node wrapper for BookAssemblerAgent (Stage 8)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from storymesh.agents.book_assembler.agent import BookAssemblerAgent
from storymesh.core.email_delivery import EmailConfig, deliver_book, title_to_filename
from storymesh.core.llm_usage import load_llm_usage_summary
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.book_assembler import BookAssemblerAgentInput, BookAssemblerAgentOutput
from storymesh.schemas.cover_art import CoverArtAgentOutput
from storymesh.versioning.schemas import BOOK_ASSEMBLER_SCHEMA_VERSION

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def make_book_assembler_node(
    agent: BookAssemblerAgent,
    artifact_store: ArtifactStore | None = None,
    email_config: EmailConfig | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for BookAssemblerAgent (Stage 8).

    Args:
        agent: A fully constructed BookAssemblerAgent instance.
        artifact_store: Optional store for artifact persistence.
            Pass None (default) to skip persistence (e.g. in unit tests).
        email_config: Optional resolved email configuration. When present and
            a recipient is configured (via config or per-run state override),
            the assembled book is emailed after file persistence.

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

        # Compute pipeline runtime up to this stage. The book_assembler stage
        # itself is fast (rendering only) and not yet finished here, so this
        # is a slight underestimate of total wall clock — but it's the best
        # value available before the page goes to PDF.
        runtime_seconds: float | None = None
        pipeline_start_time = state.get("pipeline_start_time")
        if pipeline_start_time is not None:
            runtime_seconds = time.perf_counter() - pipeline_start_time

        # Aggregate token totals from llm_calls.jsonl. All upstream LLM stages
        # have already flushed their records by the time this node runs.
        token_usage: dict[str, int] | None = None
        if artifact_store is not None and run_id:
            token_usage = load_llm_usage_summary(artifact_store, run_id)

        input_data = BookAssemblerAgentInput(
            story_writer_output=story_writer_output,
            proposal=proposal_draft_output.proposal,
            cover_art_output=cover_art_output,
            run_id=run_id,
            user_prompt=state.get("user_prompt"),
            runtime_seconds=runtime_seconds,
            token_usage=token_usage,
        )

        raw = agent.run(input_data)

        # Derive title-based filename stem for disk and email attachments.
        filename_stem = title_to_filename(raw.title)

        # Persist PDF and EPUB binaries; build output with the resulting paths.
        pdf_path = ""
        epub_path = ""
        if artifact_store is not None:
            if raw.pdf_bytes:
                pdf_filename = f"{filename_stem}.pdf"
                artifact_store.save_run_binary(run_id, pdf_filename, raw.pdf_bytes)
                pdf_path = str(artifact_store.runs_dir / run_id / pdf_filename)
            if raw.epub_bytes:
                epub_filename = f"{filename_stem}.epub"
                artifact_store.save_run_binary(run_id, epub_filename, raw.epub_bytes)
                epub_path = str(artifact_store.runs_dir / run_id / epub_filename)

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

        # Email delivery — fires when a recipient is configured.
        if email_config is not None and artifact_store is not None:
            recipient = (
                state.get("email_recipient_override") or email_config.recipient
            )
            if recipient:
                cover_image_bytes = _load_cover_image(cover_art_output)
                deliver_book(
                    title=raw.title,
                    synopsis=story_writer_output.back_cover_summary,
                    cover_image_bytes=cover_image_bytes,
                    pdf_path=pdf_path,
                    epub_path=epub_path,
                    recipient=recipient,
                    email_config=email_config,
                )

        return {"book_assembler_output": output}

    return book_assembler_node


def _load_cover_image(cover_art_output: CoverArtAgentOutput | None) -> bytes | None:
    """Read cover image bytes from disk given a CoverArtAgentOutput.

    Returns ``None`` when the output is absent or the image file cannot
    be read, so email delivery gracefully falls back to the typographic
    cover header.

    Args:
        cover_art_output: CoverArtAgentOutput instance or None.

    Returns:
        Raw PNG bytes, or None.
    """
    if cover_art_output is None:
        return None
    image_path: str = cover_art_output.image_path
    if not image_path:
        return None
    try:
        return Path(image_path).read_bytes()
    except OSError as exc:
        logger.warning("Could not read cover image for email: %s", exc)
        return None
