"""LangGraph node wrapper for ResonanceReviewerAgent (Stage 6b).

Assembles ResonanceReviewerAgentInput from the story writer output and
proposal, runs the agent, persists the output artifact, and replaces
``story_writer_output`` in the pipeline state with the revised draft so
downstream nodes (cover_art, book_assembler) consume the revised text
without any changes to their code.

When ``skip_resonance_review`` is True (set by quality presets below
``high``), the node returns immediately without making any LLM calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.resonance_reviewer.agent import ResonanceReviewerAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.resonance_reviewer import ResonanceReviewerAgentInput
from storymesh.schemas.story_writer import StoryWriterAgentOutput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def make_resonance_reviewer_node(
    agent: ResonanceReviewerAgent,
    artifact_store: ArtifactStore | None = None,
    *,
    skip: bool = False,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for ResonanceReviewerAgent.

    Assembles ``ResonanceReviewerAgentInput`` from ``story_writer_output``
    and ``proposal_draft_output``, runs the agent, persists the artifact,
    and replaces ``story_writer_output`` with the revised draft and summary.

    Args:
        agent: A fully constructed ``ResonanceReviewerAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
        skip: When True, the node returns immediately without running the
            agent. Set by quality presets below ``high``.

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def resonance_reviewer_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 6b and update the pipeline state.

        Args:
            state: Current pipeline state. Must contain
                ``story_writer_output`` and ``proposal_draft_output``.

        Returns:
            Partial state update dict with ``resonance_reviewer_output``
            set. When moments are expanded, ``story_writer_output`` is
            also replaced with the revised draft and summary.

        Raises:
            RuntimeError: If required upstream outputs are missing.
        """
        story_output = state.get("story_writer_output")
        if story_output is None:
            msg = "resonance_reviewer_node requires story_writer_output but it is None"
            raise RuntimeError(msg)

        if skip:
            logger.debug(
                "ResonanceReviewerAgent: skipped (quality preset below high)."
            )
            return {}

        proposal_output = state.get("proposal_draft_output")
        if proposal_output is None:
            msg = "resonance_reviewer_node requires proposal_draft_output but it is None"
            raise RuntimeError(msg)

        proposal = proposal_output.proposal

        # Build scene summary for structural context
        scene_summary = "\n".join(
            f"- {s.title}: {s.summary}" for s in story_output.scene_list
        )

        vps_output = state.get("voice_profile_selector_output")
        voice_profile = vps_output.voice_profile if vps_output is not None else None

        input_data = ResonanceReviewerAgentInput(
            full_draft=story_output.full_draft,
            proposal_title=proposal.title,
            thematic_thesis=proposal.thematic_thesis,
            scene_list_summary=scene_summary,
            user_prompt=state["user_prompt"],
            voice_profile=voice_profile,
        )

        token = current_run_id.set(state.get("run_id", ""))
        try:
            output = agent.run(input_data)
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "resonance_reviewer", output
            )

        result: dict[str, Any] = {"resonance_reviewer_output": output}

        # Replace story_writer_output with revised draft + summary when
        # moments were expanded, so downstream nodes consume revised text.
        if output.moments_expanded > 0:
            revised_story_output = StoryWriterAgentOutput(
                back_cover_summary=(
                    output.revised_summary
                    if output.revised_summary is not None
                    else story_output.back_cover_summary
                ),
                scene_list=story_output.scene_list,
                full_draft=output.revised_draft,
                word_count=len(output.revised_draft.split()),
                debug={
                    **story_output.debug,
                    "resonance_review_applied": True,
                    "moments_expanded": output.moments_expanded,
                    "revision_word_delta": output.revision_word_delta,
                },
                schema_version=story_output.schema_version,
            )
            result["story_writer_output"] = revised_story_output

        return result

    return resonance_reviewer_node
