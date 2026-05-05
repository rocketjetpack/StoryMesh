"""LangGraph node wrapper for VoiceProfileSelectorAgent (Stage 0.5).

Runs after genre_normalizer and before book_fetcher. Selects a voice profile
for the run based on normalized genres and tone keywords.

If ``voice_profile_override`` is configured, skips the LLM call entirely and
loads the specified profile directly. This is primarily useful for testing
and forcing a specific prose register.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from storymesh.agents.voice_profile_selector.agent import VoiceProfileSelectorAgent
from storymesh.llm.base import current_run_id
from storymesh.orchestration.state import StoryMeshState
from storymesh.schemas.voice_profile import BUILT_IN_PROFILE_IDS, load_voice_profile
from storymesh.schemas.voice_profile_selector import VoiceProfileSelectorAgentInput

if TYPE_CHECKING:
    from storymesh.core.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def make_voice_profile_selector_node(
    agent: VoiceProfileSelectorAgent,
    artifact_store: ArtifactStore | None = None,
    voice_profile_override: str | None = None,
) -> Callable[[StoryMeshState], dict[str, Any]]:
    """Return a LangGraph-compatible node function for VoiceProfileSelectorAgent.

    If ``voice_profile_override`` is provided and names a valid profile, the
    agent LLM call is skipped and the override profile is used directly. This
    is useful for testing or forcing a specific voice without burning tokens.

    Args:
        agent: A fully constructed ``VoiceProfileSelectorAgent`` instance.
        artifact_store: Optional store for per-node artifact persistence.
        voice_profile_override: Optional profile ID to force without LLM classification.

    Returns:
        A node callable with signature ``StoryMeshState -> dict[str, Any]``.
    """

    def voice_profile_selector_node(state: StoryMeshState) -> dict[str, Any]:
        """Execute Stage 0.5 and write the output into the pipeline state.

        Args:
            state: Current pipeline state. Must contain ``genre_normalizer_output``
                and ``run_id``.

        Returns:
            Partial state update dict with ``voice_profile_selector_output`` set.
        """
        genre_output = state.get("genre_normalizer_output")
        if genre_output is None:
            raise RuntimeError(
                "voice_profile_selector: genre_normalizer_output is missing from state."
            )

        # Handle user-specified override — skip LLM call entirely.
        if voice_profile_override and voice_profile_override in BUILT_IN_PROFILE_IDS:
            logger.debug(
                "VoiceProfileSelectorNode: using override profile %r",
                voice_profile_override,
            )
            profile = load_voice_profile(voice_profile_override)
            from storymesh.schemas.voice_profile_selector import VoiceProfileSelectorAgentOutput  # noqa: PLC0415

            output = VoiceProfileSelectorAgentOutput(
                selected_profile_id=profile.id,
                rationale=f"User-specified override: {voice_profile_override}",
                voice_profile=profile,
                debug={"defaulted_to_fallback": False, "override": True},
            )
            return {"voice_profile_selector_output": output}

        input_data = VoiceProfileSelectorAgentInput(
            user_prompt=state.get("user_prompt", ""),
            normalized_genres=genre_output.normalized_genres,
            user_tones=genre_output.user_tones,
            available_profile_ids=list(BUILT_IN_PROFILE_IDS),
        )

        token = current_run_id.set(state.get("run_id", ""))
        try:
            output = agent.run(input_data)
        finally:
            current_run_id.reset(token)

        if artifact_store is not None:
            from storymesh.core.artifacts import persist_node_output  # noqa: PLC0415

            persist_node_output(
                artifact_store, state["run_id"], "voice_profile_selector", output
            )

        return {"voice_profile_selector_output": output}

    return voice_profile_selector_node
