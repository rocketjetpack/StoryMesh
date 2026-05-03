"""StoryWriterAgent — Stage 6 of the StoryMesh pipeline.

Produces a complete short story in three passes:

1. **Outline pass** (low temperature): Expands the StoryProposal's key_scenes
   into 6–10 structured SceneOutline objects. Each outline carries an
   `opens_with` sentence that is used verbatim by the prose pass, preventing
   generic AI opening lines.

2. **Draft pass** (medium-high temperature): Writes the full prose using the
   scene outlines as structure. All craft principles (sentence rhythm, subtext,
   concrete detail, temporal irregularity) are enforced via the prompt.

3. **Summary pass** (low temperature): Writes ~300-word back-cover marketing
   copy from the completed draft. Writing the summary after the draft ensures
   it accurately reflects what was written rather than what was planned.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput
from storymesh.schemas.story_writer import (
    SceneOutline,
    StoryWriterAgentInput,
    StoryWriterAgentOutput,
)
from storymesh.schemas.voice_profile import VoiceProfile, load_voice_profile
from storymesh.versioning.schemas import STORY_WRITER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Craft notes section header inserted into prompts when rubric feedback is present.
_CRAFT_NOTES_HEADER = """\
CRAFT NOTES FROM PRIOR EVALUATION
===================================
The story proposal was evaluated before reaching you. The following dimensions
scored below the quality threshold. Address these weaknesses in your work.

"""

# Inserted into prompts when no rubric feedback is available.
_NO_CRAFT_NOTES = ""


def _format_craft_notes(rubric_output: RubricJudgeAgentOutput) -> str:
    """Format rubric dimension feedback into actionable craft notes.

    Only includes dimensions scoring below 2 (i.e., not "strong"). Dimensions
    at the top tier are working and should not be disrupted.

    Args:
        rubric_output: A RubricJudgeAgentOutput instance.

    Returns:
        Formatted craft notes string, or empty string if all scores are strong.
    """
    tier_labels = {0: "fail", 1: "acceptable", 2: "strong"}
    lines: list[str] = []
    dimensions = getattr(rubric_output, "dimensions", {})
    for dim_name, dim_result in dimensions.items():
        score = getattr(dim_result, "score", 2)
        if score < 2:
            feedback = getattr(dim_result, "feedback", "")
            label = tier_labels.get(score, str(score))
            lines.append(f"- {dim_name} ({label}): {feedback}")

    overall = getattr(rubric_output, "overall_feedback", "")
    composite = getattr(rubric_output, "composite_score", None)

    if not lines:
        return _NO_CRAFT_NOTES

    notes = _CRAFT_NOTES_HEADER
    if composite is not None:
        notes += f"Prior composite score: {composite}/10\n\n"
    notes += "\n".join(lines)
    if overall:
        notes += f"\n\nOverall editorial note: {overall}"
    return notes


def _format_profile_exemplars(profile: VoiceProfile) -> str:
    """Format voice profile exemplars into the opens_with examples format.

    Args:
        profile: The active voice profile.

    Returns:
        Multi-line string of exemplar sentences, formatted for the outline prompt.
    """
    return "\n".join(f'     - "{e}"' for e in profile.exemplars)


def _format_scene_list_for_prompt(scenes: list[SceneOutline]) -> str:
    """Render scene outlines as human-readable text for the draft prompt.

    Args:
        scenes: Ordered list of SceneOutline objects from the outline pass.

    Returns:
        Multi-line string with each scene's details formatted for the prompt.
    """
    parts: list[str] = []
    for i, scene in enumerate(scenes, start=1):
        parts.append(
            f"SCENE {i}: {scene.title}\n"
            f'Opens with (use verbatim): "{scene.opens_with}"\n'
            f"What happens: {scene.summary}\n"
            f"Narrative pressure: {scene.narrative_pressure}\n"
            f"Observational anchor: {scene.observational_anchor}"
        )
    return "\n\n".join(parts)


class StoryWriterAgent:
    """Writes a complete short story from a StoryProposal (Stage 6).

    Uses three sequential LLM passes: scene outline expansion, prose drafting,
    and back-cover summary generation.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        outline_temperature: float = 0.5,
        draft_temperature: float = 0.8,
        summary_temperature: float = 0.4,
        outline_max_tokens: int = 4096,
        draft_max_tokens: int = 6000,
        summary_max_tokens: int = 1024,
        target_words: int = 3000,
    ) -> None:
        """Construct the agent.

        Args:
            llm_client: LLM client instance. Required — all three passes are
                creative generation tasks with no deterministic fallback.
            outline_temperature: Temperature for scene outline generation.
                Low value (default 0.5) produces structured, consistent outlines.
            draft_temperature: Temperature for prose generation. Medium-high
                (default 0.8) balances creativity with coherence across a
                long-form output.
            summary_temperature: Temperature for back-cover copy. Low value
                (default 0.4) produces clean, controlled marketing copy.
            outline_max_tokens: Token budget for the outline pass.
            draft_max_tokens: Token budget for the prose pass. Must be large
                enough for the full story — default 6000 supports ~4500 words.
            summary_max_tokens: Token budget for the summary pass.
            target_words: Target word count for the full draft. Used in both
                the outline prompt (to calibrate scene count) and the draft
                prompt (to guide length).
        """
        self._llm_client = llm_client
        self._outline_temperature = outline_temperature
        self._draft_temperature = draft_temperature
        self._summary_temperature = summary_temperature
        self._outline_max_tokens = outline_max_tokens
        self._draft_max_tokens = draft_max_tokens
        self._summary_max_tokens = summary_max_tokens
        self._target_words = target_words

        # Load all prompts eagerly so misconfiguration is caught at
        # construction time, not mid-pipeline.
        self._outline_prompt = load_prompt("story_writer_outline")
        self._draft_prompt = load_prompt("story_writer_draft")
        self._summary_prompt = load_prompt("story_writer_summary")

    def run(self, input_data: StoryWriterAgentInput) -> StoryWriterAgentOutput:
        """Generate scene outlines, write prose draft, and produce back-cover summary.

        Args:
            input_data: Assembled input from the pipeline node wrapper.

        Returns:
            A frozen StoryWriterAgentOutput with back_cover_summary, scene_list,
            full_draft, word_count, and debug metadata.

        Raises:
            RuntimeError: If the outline or draft passes fail to produce valid output.
        """
        proposal = input_data.proposal

        # Resolve voice profile overlays — fall back to literary_restraint when None.
        voice_profile: VoiceProfile = (
            input_data.voice_profile
            if input_data.voice_profile is not None
            else load_voice_profile("literary_restraint")
        )
        craft_overlay = voice_profile.craft_overlay
        avoid_overlay = voice_profile.avoid_overlay
        summary_overlay = voice_profile.summary_overlay
        profile_exemplars = _format_profile_exemplars(voice_profile)

        craft_notes = (
            _format_craft_notes(input_data.rubric_feedback)
            if input_data.rubric_feedback is not None
            else _NO_CRAFT_NOTES
        )
        craft_notes_section = (
            f"CRAFT NOTES\n===========\n{craft_notes}" if craft_notes else ""
        )

        tensions_json = orjson.dumps(
            [t.model_dump() for t in input_data.tensions]
        ).decode()

        logger.info(
            "StoryWriterAgent starting | title=%r target_words=%d has_rubric=%s",
            proposal.title,
            self._target_words,
            input_data.rubric_feedback is not None,
        )

        # ── Pass 1: Scene Outline ──────────────────────────────────────────
        scene_list = self._run_outline_pass(
            proposal=proposal,
            tensions_json=tensions_json,
            craft_notes_section=craft_notes_section,
            profile_exemplars=profile_exemplars,
            user_prompt=input_data.user_prompt,
            normalized_genres=input_data.normalized_genres,
            user_tones=input_data.user_tones,
        )

        logger.info(
            "StoryWriterAgent outline complete | scenes=%d",
            len(scene_list),
        )

        # ── Pass 2: Prose Draft ────────────────────────────────────────────
        full_draft = self._run_draft_pass(
            proposal=proposal,
            scene_list=scene_list,
            tensions_json=tensions_json,
            craft_notes_section=craft_notes_section,
            craft_overlay=craft_overlay,
            avoid_overlay=avoid_overlay,
        )

        word_count = len(full_draft.split())
        logger.info(
            "StoryWriterAgent draft complete | words=%d",
            word_count,
        )

        # ── Pass 3: Back-Cover Summary ─────────────────────────────────────
        back_cover_summary = self._run_summary_pass(
            proposal=proposal,
            full_draft=full_draft,
            user_prompt=input_data.user_prompt,
            summary_overlay=summary_overlay,
        )

        logger.info("StoryWriterAgent summary complete")

        debug: dict[str, Any] = {
            "outline_temperature": self._outline_temperature,
            "draft_temperature": self._draft_temperature,
            "summary_temperature": self._summary_temperature,
            "target_words": self._target_words,
            "scene_count": len(scene_list),
            "word_count": word_count,
            "total_llm_calls": 3,
            "had_rubric_feedback": input_data.rubric_feedback is not None,
        }

        return StoryWriterAgentOutput(
            back_cover_summary=back_cover_summary,
            scene_list=scene_list,
            full_draft=full_draft,
            word_count=word_count,
            debug=debug,
            schema_version=STORY_WRITER_SCHEMA_VERSION,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _run_outline_pass(
        self,
        *,
        proposal: StoryProposal,
        tensions_json: str,
        craft_notes_section: str,
        profile_exemplars: str,
        user_prompt: str,
        normalized_genres: list[str],
        user_tones: list[str],
    ) -> list[SceneOutline]:
        """Execute Pass 1: expand key_scenes into structured scene outlines.

        Args:
            proposal: The selected StoryProposal.
            tensions_json: JSON-encoded list of ThematicTension objects.
            craft_notes_section: Formatted rubric craft notes, or empty string.
            profile_exemplars: Formatted voice-profile exemplar sentences.
            user_prompt: Original user input string.
            normalized_genres: Canonical genre names.
            user_tones: User-specified tone words.

        Returns:
            List of SceneOutline objects (at least 3).

        Raises:
            RuntimeError: If the outline call fails or produces no valid scenes.
        """
        self._llm_client.agent_name = "story_writer_outline"
        formatted_system = self._outline_prompt.system.format(
            craft_notes_section=craft_notes_section,
            profile_exemplars=profile_exemplars,
        )
        proposal_json = orjson.dumps(proposal.model_dump()).decode()
        user_prompt_text = self._outline_prompt.format_user(
            target_words=self._target_words,
            user_prompt=user_prompt,
            normalized_genres=normalized_genres,
            user_tones=user_tones,
            proposal=proposal_json,
            tensions=tensions_json,
            craft_notes_section=craft_notes_section,
        )

        try:
            response = self._llm_client.complete_json(
                user_prompt_text,
                system_prompt=formatted_system,
                temperature=self._outline_temperature,
                max_tokens=self._outline_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"StoryWriterAgent outline pass failed: {exc}"
            ) from exc

        raw_scenes = response.get("scenes", [])
        if not raw_scenes:
            raise RuntimeError(
                "StoryWriterAgent outline pass returned no scenes."
            )

        scenes: list[SceneOutline] = []
        for i, raw in enumerate(raw_scenes):
            try:
                scenes.append(SceneOutline(**raw))
            except Exception as exc:
                logger.warning(
                    "StoryWriterAgent: scene %d failed validation (%s) — skipping.",
                    i + 1,
                    exc,
                )

        if not scenes:
            raise RuntimeError(
                "StoryWriterAgent: all scene outlines failed validation."
            )

        return scenes

    def _run_draft_pass(
        self,
        *,
        proposal: StoryProposal,
        scene_list: list[SceneOutline],
        tensions_json: str,
        craft_notes_section: str,
        craft_overlay: str,
        avoid_overlay: str,
    ) -> str:
        """Execute Pass 2: write full prose draft from scene outlines.

        Args:
            proposal: The selected StoryProposal.
            scene_list: Validated scene outlines from Pass 1.
            tensions_json: JSON-encoded thematic tensions.
            craft_notes_section: Formatted rubric craft notes, or empty string.
            craft_overlay: Additional craft principles from the active voice profile.
            avoid_overlay: Additional avoid items from the active voice profile.

        Returns:
            Full prose draft as a string, scenes separated by SCENE_BREAK.

        Raises:
            RuntimeError: If the draft call fails or returns an empty string.
        """
        self._llm_client.agent_name = "story_writer_draft"
        formatted_system = self._draft_prompt.system.format(
            craft_overlay=craft_overlay,
            avoid_overlay=avoid_overlay,
        )
        scene_list_text = _format_scene_list_for_prompt(scene_list)
        unknowns_text = (
            "\n".join(f"- {u}" for u in proposal.unknowns)
            if proposal.unknowns
            else "(none)"
        )
        user_prompt_text = self._draft_prompt.format_user(
            title=proposal.title,
            target_words=self._target_words,
            protagonist=proposal.protagonist,
            thematic_thesis=proposal.thematic_thesis,
            tone=proposal.tone,
            tensions=tensions_json,
            scene_list=scene_list_text,
            unknowns=unknowns_text,
            craft_notes_section=craft_notes_section,
        )

        try:
            response = self._llm_client.complete_json(
                user_prompt_text,
                system_prompt=formatted_system,
                temperature=self._draft_temperature,
                max_tokens=self._draft_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"StoryWriterAgent draft pass failed: {exc}"
            ) from exc

        draft = str(response.get("full_draft", ""))
        if not draft or not draft.strip():
            raise RuntimeError(
                "StoryWriterAgent draft pass returned an empty draft."
            )

        return draft.strip()

    def _run_summary_pass(
        self,
        *,
        proposal: StoryProposal,
        full_draft: str,
        user_prompt: str,
        summary_overlay: str,
    ) -> str:
        """Execute Pass 3: write back-cover summary from completed draft.

        Args:
            proposal: The selected StoryProposal (for title and thesis).
            full_draft: Completed prose draft from Pass 2.
            user_prompt: Original user input string.
            summary_overlay: Register note from the active voice profile.

        Returns:
            Back-cover marketing copy as a string.

        Raises:
            RuntimeError: If the summary call fails or returns an empty string.
        """
        self._llm_client.agent_name = "story_writer_summary"
        formatted_system = self._summary_prompt.system.format(
            summary_overlay=summary_overlay,
        )
        user_prompt_text = self._summary_prompt.format_user(
            title=proposal.title,
            user_prompt=user_prompt,
            thematic_thesis=proposal.thematic_thesis,
            full_draft=full_draft,
        )

        try:
            response = self._llm_client.complete_json(
                user_prompt_text,
                system_prompt=formatted_system,
                temperature=self._summary_temperature,
                max_tokens=self._summary_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"StoryWriterAgent summary pass failed: {exc}"
            ) from exc

        summary = str(response.get("back_cover_summary", ""))
        if not summary or not summary.strip():
            raise RuntimeError(
                "StoryWriterAgent summary pass returned an empty summary."
            )

        return summary.strip()
