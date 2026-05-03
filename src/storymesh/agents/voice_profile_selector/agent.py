"""VoiceProfileSelectorAgent — Stage 0.5 of the StoryMesh pipeline.

Selects a voice profile for the run based on the user prompt, normalized genres,
and tone keywords. Uses a single deterministic LLM call (T=0) to classify the
input into one of the built-in voice profiles.

Failure handling: on LLM failure, unknown profile ID, or any parsing error,
logs a warning and defaults to ``literary_restraint``. The pipeline must not
crash on selector failure — the existing behavior is preserved as the fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.voice_profile import BUILT_IN_PROFILE_IDS, VoiceProfile, load_voice_profile
from storymesh.schemas.voice_profile_selector import (
    VoiceProfileSelectorAgentInput,
    VoiceProfileSelectorAgentOutput,
)
from storymesh.versioning.schemas import VOICE_PROFILE_SELECTOR_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_FALLBACK_PROFILE_ID = "literary_restraint"


class VoiceProfileSelectorAgent:
    """Classifies a run's voice profile from genres and tone keywords (Stage 0.5).

    Single deterministic LLM call. Defaults to ``literary_restraint`` on any
    failure, preserving backward-compatible behavior.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        """Construct the agent.

        Args:
            llm_client: LLM client instance.
            temperature: Sampling temperature. Default 0.0 for deterministic
                selection — the same input should always produce the same profile.
            max_tokens: Maximum output tokens. 256 is ample for the two-field JSON.
        """
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._prompt = load_prompt("voice_profile_selector")

    def run(self, input_data: VoiceProfileSelectorAgentInput) -> VoiceProfileSelectorAgentOutput:
        """Select a voice profile and return the result.

        Args:
            input_data: Genres, tones, and user prompt for classification.

        Returns:
            A frozen VoiceProfileSelectorAgentOutput with the selected profile.
            Never raises — falls back to literary_restraint on any failure.
        """
        logger.info(
            "VoiceProfileSelectorAgent starting | genres=%s tones=%s",
            input_data.normalized_genres,
            input_data.user_tones,
        )

        defaulted_to_fallback = False
        selected_id = _FALLBACK_PROFILE_ID
        rationale = "Defaulted to literary_restraint (fallback)."

        try:
            user_prompt_text = self._prompt.format_user(
                user_prompt=input_data.user_prompt,
                normalized_genres=input_data.normalized_genres,
                user_tones=input_data.user_tones,
            )
            response = self._llm_client.complete_json(
                user_prompt_text,
                system_prompt=self._prompt.system,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw_id = str(response.get("selected_profile_id", "")).strip()
            raw_rationale = str(response.get("rationale", "")).strip()

            if raw_id not in input_data.available_profile_ids:
                logger.warning(
                    "VoiceProfileSelectorAgent: unknown profile id %r — defaulting to %r.",
                    raw_id,
                    _FALLBACK_PROFILE_ID,
                )
                defaulted_to_fallback = True
            else:
                selected_id = raw_id
                rationale = raw_rationale or rationale

        except Exception as exc:
            logger.warning(
                "VoiceProfileSelectorAgent: LLM call failed (%s) — defaulting to %r.",
                exc,
                _FALLBACK_PROFILE_ID,
            )
            defaulted_to_fallback = True

        voice_profile = _load_profile_with_fallback(selected_id)

        logger.info(
            "VoiceProfileSelectorAgent complete | selected=%r defaulted=%s",
            voice_profile.id,
            defaulted_to_fallback,
        )

        debug: dict[str, Any] = {
            "temperature": self._temperature,
            "defaulted_to_fallback": defaulted_to_fallback,
            "schema_version": VOICE_PROFILE_SELECTOR_SCHEMA_VERSION,
        }

        return VoiceProfileSelectorAgentOutput(
            selected_profile_id=voice_profile.id,
            rationale=rationale,
            voice_profile=voice_profile,
            debug=debug,
        )


def _load_profile_with_fallback(profile_id: str) -> VoiceProfile:
    """Load a voice profile, falling back to literary_restraint on any error."""
    try:
        return load_voice_profile(profile_id)
    except Exception as exc:
        logger.warning(
            "VoiceProfileSelectorAgent: could not load profile %r (%s) — using fallback.",
            profile_id,
            exc,
        )
        return load_voice_profile(_FALLBACK_PROFILE_ID)


def list_available_profiles() -> list[str]:
    """Return the list of available built-in voice profile IDs."""
    return list(BUILT_IN_PROFILE_IDS)
