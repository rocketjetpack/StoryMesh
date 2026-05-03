"""ResonanceReviewerAgent — Stage 6b of the StoryMesh pipeline.

Reviews a completed prose draft for near-miss moments — places where the
story implies depth but retreats before engaging — and produces targeted
expansions. Uses three internal LLM passes:

1. **Review pass** (cross-provider): Identifies 0-3 near-miss moments,
   classifying each as restraint (earned silence) or avoidance (missed
   opportunity). Uses a different provider than the writer to avoid shared
   biases.

2. **Revision pass** (same provider as writer): Expands only avoidance
   moments within the existing draft, matching the original voice. Adds
   roughly 50-150 words per moment.

3. **Summary pass** (same provider as writer): Re-generates the back-cover
   summary from the revised draft so it reflects the final text.
"""

from __future__ import annotations

import logging
from typing import Any

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.resonance_reviewer import (
    NearMissMoment,
    ResonanceReviewerAgentInput,
    ResonanceReviewerAgentOutput,
)
from storymesh.versioning.schemas import RESONANCE_REVIEWER_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class ResonanceReviewerAgent:
    """Reviews a story draft for near-miss moments and produces targeted expansions.

    Two separate LLM clients enable cross-provider review: the review pass
    uses a different model (e.g. GPT-4o) to find blind spots the writer's
    model cannot see, while the revision pass uses the same provider as the
    writer (e.g. Claude) to maintain voice consistency.
    """

    def __init__(
        self,
        *,
        review_llm_client: LLMClient,
        revision_llm_client: LLMClient,
        review_temperature: float = 0.4,
        revision_temperature: float = 0.7,
        summary_temperature: float = 0.4,
        review_max_tokens: int = 4096,
        revision_max_tokens: int = 8000,
        summary_max_tokens: int = 1024,
    ) -> None:
        """Construct the agent.

        Args:
            review_llm_client: LLM client for the review pass. Should be a
                different provider than the story writer for cross-provider
                analysis (e.g. GPT-4o when the writer is Claude).
            revision_llm_client: LLM client for the revision and summary
                passes. Should be the same provider as the story writer to
                maintain voice consistency.
            review_temperature: Temperature for the review pass. Low value
                (default 0.4) produces consistent, analytical output.
            revision_temperature: Temperature for the revision pass. Medium
                (default 0.7) balances creativity with voice matching.
            summary_temperature: Temperature for the summary re-run.
            review_max_tokens: Token budget for the review pass.
            revision_max_tokens: Token budget for the revision pass. Must be
                large enough for the complete revised draft.
            summary_max_tokens: Token budget for the summary re-run.
        """
        self._review_llm = review_llm_client
        self._revision_llm = revision_llm_client
        self._review_temperature = review_temperature
        self._revision_temperature = revision_temperature
        self._summary_temperature = summary_temperature
        self._review_max_tokens = review_max_tokens
        self._revision_max_tokens = revision_max_tokens
        self._summary_max_tokens = summary_max_tokens

        self._review_prompt = load_prompt("resonance_reviewer_review")
        self._revise_prompt = load_prompt("resonance_reviewer_revise")
        self._summary_prompt = load_prompt("story_writer_summary")

    def run(self, input_data: ResonanceReviewerAgentInput) -> ResonanceReviewerAgentOutput:
        """Run review, revision, and optional summary re-generation passes.

        Args:
            input_data: Assembled input from the pipeline node wrapper.

        Returns:
            A frozen ResonanceReviewerAgentOutput with near-miss diagnostics,
            the revised draft, and optionally a new back-cover summary.
        """
        original_word_count = len(input_data.full_draft.split())

        logger.info(
            "ResonanceReviewerAgent starting | title=%r words=%d",
            input_data.proposal_title,
            original_word_count,
        )

        # ── Pass 1: Review (cross-provider) ───────────────────────────────
        all_moments = self._run_review_pass(input_data)
        logger.info(
            "ResonanceReviewerAgent review complete | moments_found=%d",
            len(all_moments),
        )

        # Filter to avoidance-only
        avoidance_moments = [m for m in all_moments if m.classification == "avoidance"]
        restraint_count = len(all_moments) - len(avoidance_moments)

        logger.info(
            "ResonanceReviewerAgent classification | avoidance=%d restraint=%d",
            len(avoidance_moments),
            restraint_count,
        )

        if not avoidance_moments:
            logger.info("ResonanceReviewerAgent: no avoidance moments — draft unchanged.")
            return ResonanceReviewerAgentOutput(
                near_miss_moments=[],
                revised_draft=input_data.full_draft,
                revised_summary=None,
                revision_word_delta=0,
                moments_found=len(all_moments),
                moments_expanded=0,
                debug={
                    "review_temperature": self._review_temperature,
                    "total_moments": len(all_moments),
                    "restraint_moments": restraint_count,
                    "avoidance_moments": 0,
                    "total_llm_calls": 1,
                    "skipped_revision": True,
                },
                schema_version=RESONANCE_REVIEWER_SCHEMA_VERSION,
            )

        # ── Pass 2: Revision (same provider as writer) ────────────────────
        revised_draft = self._run_revision_pass(input_data.full_draft, avoidance_moments)
        revised_word_count = len(revised_draft.split())
        word_delta = revised_word_count - original_word_count

        logger.info(
            "ResonanceReviewerAgent revision complete | word_delta=%+d",
            word_delta,
        )

        # ── Pass 3: Summary re-run (same provider as writer) ─────────────
        revised_summary = self._run_summary_pass(
            revised_draft,
            title=input_data.proposal_title,
            thematic_thesis=input_data.thematic_thesis,
            user_prompt=input_data.user_prompt,
        )

        logger.info("ResonanceReviewerAgent summary re-run complete.")

        debug: dict[str, Any] = {
            "review_temperature": self._review_temperature,
            "revision_temperature": self._revision_temperature,
            "summary_temperature": self._summary_temperature,
            "total_moments": len(all_moments),
            "restraint_moments": restraint_count,
            "avoidance_moments": len(avoidance_moments),
            "original_word_count": original_word_count,
            "revised_word_count": revised_word_count,
            "total_llm_calls": 3,
        }

        return ResonanceReviewerAgentOutput(
            near_miss_moments=avoidance_moments,
            revised_draft=revised_draft,
            revised_summary=revised_summary,
            revision_word_delta=word_delta,
            moments_found=len(all_moments),
            moments_expanded=len(avoidance_moments),
            debug=debug,
            schema_version=RESONANCE_REVIEWER_SCHEMA_VERSION,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _run_review_pass(
        self,
        input_data: ResonanceReviewerAgentInput,
    ) -> list[NearMissMoment]:
        """Execute Pass 1: identify near-miss moments (cross-provider).

        Args:
            input_data: The full agent input.

        Returns:
            List of NearMissMoment objects (both avoidance and restraint).

        Raises:
            RuntimeError: If the review LLM call fails.
        """
        user_prompt_text = self._review_prompt.format_user(
            title=input_data.proposal_title,
            thematic_thesis=input_data.thematic_thesis,
            scene_list_summary=input_data.scene_list_summary,
            full_draft=input_data.full_draft,
        )

        try:
            response = self._review_llm.complete_json(
                user_prompt_text,
                system_prompt=self._review_prompt.system,
                temperature=self._review_temperature,
                max_tokens=self._review_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ResonanceReviewerAgent review pass failed: {exc}"
            ) from exc

        raw_moments = response.get("moments", [])
        moments: list[NearMissMoment] = []
        for i, raw in enumerate(raw_moments):
            try:
                moments.append(NearMissMoment(**raw))
            except Exception as exc:
                logger.warning(
                    "ResonanceReviewerAgent: moment %d failed validation (%s) — skipping.",
                    i + 1,
                    exc,
                )

        return moments

    def _run_revision_pass(
        self,
        full_draft: str,
        avoidance_moments: list[NearMissMoment],
    ) -> str:
        """Execute Pass 2: expand avoidance moments in the draft.

        Args:
            full_draft: The original prose draft.
            avoidance_moments: Validated near-miss moments classified as avoidance.

        Returns:
            Complete revised draft with expansions applied.

        Raises:
            RuntimeError: If the revision LLM call fails or returns empty.
        """
        directives_parts: list[str] = []
        for i, moment in enumerate(avoidance_moments, start=1):
            directives_parts.append(
                f"MOMENT {i}:\n"
                f"Passage: \"{moment.passage_ref}\"\n"
                f"What it implies: {moment.what_it_implies}\n"
                f"What the reader wanted: {moment.what_the_reader_wanted}\n"
                f"What the story did instead: {moment.what_the_story_did}\n"
                f"Expansion directive: {moment.expansion_directive}"
            )

        near_miss_directives = "\n\n".join(directives_parts)

        user_prompt_text = self._revise_prompt.format_user(
            full_draft=full_draft,
            near_miss_directives=near_miss_directives,
        )

        try:
            response = self._revision_llm.complete_json(
                user_prompt_text,
                system_prompt=self._revise_prompt.system,
                temperature=self._revision_temperature,
                max_tokens=self._revision_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ResonanceReviewerAgent revision pass failed: {exc}"
            ) from exc

        revised = str(response.get("revised_draft", ""))
        if not revised or not revised.strip():
            raise RuntimeError(
                "ResonanceReviewerAgent revision pass returned an empty draft."
            )

        return revised.strip()

    def _run_summary_pass(
        self,
        revised_draft: str,
        *,
        title: str,
        thematic_thesis: str,
        user_prompt: str,
    ) -> str:
        """Execute Pass 3: re-generate back-cover summary from revised draft.

        Reuses the story_writer_summary prompt to maintain consistency.

        Args:
            revised_draft: The revised prose draft after expansions.
            title: Story title.
            thematic_thesis: Central thematic pressure.
            user_prompt: Original user input string.

        Returns:
            Back-cover marketing copy as a string.

        Raises:
            RuntimeError: If the summary call fails or returns empty.
        """
        user_prompt_text = self._summary_prompt.format_user(
            title=title,
            user_prompt=user_prompt,
            thematic_thesis=thematic_thesis,
            full_draft=revised_draft,
        )

        try:
            response = self._revision_llm.complete_json(
                user_prompt_text,
                system_prompt=self._summary_prompt.system,
                temperature=self._summary_temperature,
                max_tokens=self._summary_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ResonanceReviewerAgent summary pass failed: {exc}"
            ) from exc

        summary = str(response.get("back_cover_summary", ""))
        if not summary or not summary.strip():
            raise RuntimeError(
                "ResonanceReviewerAgent summary pass returned an empty summary."
            )

        return summary.strip()
