"""ResonanceReviewerAgent — Stage 6b of the StoryMesh pipeline.

Reviews a completed prose draft through four orthogonal lenses and produces
a single revised draft that addresses all actionable findings.

The four review lenses (each its own LLM call, all cross-provider):

1. **Near-miss** — places where the story implies depth and retreats.
   Findings classified as ``avoidance`` are expanded; ``restraint`` is left
   alone. Expansion budget: roughly 100-250 words per moment.

2. **Tone drift** — passages where the prose register diverges from the
   user's requested tones in ways the story's needs do not justify. Each
   finding is a passage *replacement* with a tone-specific rewrite directive.

3. **Ending verdict** — the closing 200-400 words. Singular finding: either
   the ending preserves the unresolved pressure named by the thematic_thesis
   or it delivers a verdict. Revision is a *cut*, not a rewrite.

4. **Slop marker** — high-confidence AI-tell phrases with verbatim quoted
   evidence. Each finding is a phrase *replacement* with concrete bodied
   prose.

After all four review passes, the agent merges actionable findings into a
single revision prompt and runs one revision LLM call (same provider as the
writer, for voice consistency), followed by a summary re-run.

Total LLM calls: up to 6 (4 review + 1 revision + 1 summary). When a review
pass fails the agent logs and continues — partial findings still produce
useful revisions.
"""

from __future__ import annotations

import logging
from typing import Any

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.resonance_reviewer import (
    EndingVerdictFinding,
    NearMissMoment,
    ResonanceReviewerAgentInput,
    ResonanceReviewerAgentOutput,
    SlopMarker,
    ToneDriftFinding,
)
from storymesh.versioning.schemas import RESONANCE_REVIEWER_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class ResonanceReviewerAgent:
    """Reviews a story draft through four orthogonal lenses and revises.

    Two separate LLM clients enable cross-provider review: every review pass
    uses ``review_llm_client`` (intended to be a different provider than the
    writer to surface blind spots), while the revision and summary passes
    use ``revision_llm_client`` (intended to match the writer for voice
    consistency).
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
            review_llm_client: LLM client for all four review passes. Should
                be a different provider than the story writer.
            revision_llm_client: LLM client for the revision and summary
                passes. Should match the writer's provider for voice
                consistency.
            review_temperature: Temperature for the review passes. Low value
                (default 0.4) produces consistent analytical output.
            revision_temperature: Temperature for the revision pass.
            summary_temperature: Temperature for the summary re-run.
            review_max_tokens: Token budget per review pass.
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

        self._near_miss_prompt = load_prompt("resonance_reviewer_review")
        self._tone_prompt = load_prompt("resonance_reviewer_tone_review")
        self._ending_prompt = load_prompt("resonance_reviewer_ending_review")
        self._slop_prompt = load_prompt("resonance_reviewer_slop_review")
        self._revise_prompt = load_prompt("resonance_reviewer_revise")
        self._summary_prompt = load_prompt("story_writer_summary")

    def run(self, input_data: ResonanceReviewerAgentInput) -> ResonanceReviewerAgentOutput:
        """Run all four review passes, then one revision and one summary pass.

        Args:
            input_data: Assembled input from the pipeline node wrapper.

        Returns:
            A frozen ResonanceReviewerAgentOutput with per-lens findings, the
            revised draft, and (when revision occurred) a new back-cover
            summary.
        """
        original_word_count = len(input_data.full_draft.split())

        logger.debug(
            "ResonanceReviewerAgent starting | title=%r words=%d tones=%s",
            input_data.proposal_title,
            original_word_count,
            input_data.requested_tones,
        )

        # ── Review passes (4×, cross-provider) ────────────────────────────
        near_miss_all, near_miss_failed = self._run_near_miss_review(input_data)
        tone_drifts, tone_failed = self._run_tone_review(input_data)
        ending_finding, ending_failed = self._run_ending_review(input_data)
        slop_markers, slop_failed = self._run_slop_review(input_data)

        # Filter near-miss to avoidance-only — restraint is reported, not revised.
        avoidance_moments = [m for m in near_miss_all if m.classification == "avoidance"]
        restraint_count = len(near_miss_all) - len(avoidance_moments)

        review_calls_attempted = 4
        review_calls_failed = sum(
            1 for f in (near_miss_failed, tone_failed, ending_failed, slop_failed) if f
        )

        findings_total = (
            len(avoidance_moments)
            + len(tone_drifts)
            + (1 if ending_finding is not None else 0)
            + len(slop_markers)
        )

        logger.debug(
            "ResonanceReviewerAgent reviews complete | "
            "avoidance=%d restraint=%d tone_drifts=%d ending_verdict=%s slop=%d failed_passes=%d",
            len(avoidance_moments),
            restraint_count,
            len(tone_drifts),
            "yes" if ending_finding else "no",
            len(slop_markers),
            review_calls_failed,
        )

        # If nothing is actionable, return early with the original draft.
        if findings_total == 0:
            logger.debug("ResonanceReviewerAgent: no actionable findings — draft unchanged.")
            return ResonanceReviewerAgentOutput(
                near_miss_moments=[],
                tone_drift_findings=[],
                ending_verdict_finding=None,
                slop_markers=[],
                revised_draft=input_data.full_draft,
                revised_summary=None,
                revision_word_delta=0,
                moments_found=len(near_miss_all),
                moments_expanded=0,
                findings_total=0,
                debug={
                    "review_temperature": self._review_temperature,
                    "review_calls_attempted": review_calls_attempted,
                    "review_calls_failed": review_calls_failed,
                    "near_miss_total": len(near_miss_all),
                    "near_miss_restraint": restraint_count,
                    "near_miss_avoidance": 0,
                    "tone_drifts": 0,
                    "ending_verdict": False,
                    "slop_markers": 0,
                    "total_llm_calls": review_calls_attempted - review_calls_failed,
                    "skipped_revision": True,
                },
                schema_version=RESONANCE_REVIEWER_SCHEMA_VERSION,
            )

        # Voice overlays for the revision + summary prompts.
        voice_register_note = (
            input_data.voice_profile.craft_overlay
            if input_data.voice_profile is not None
            else ""
        )
        summary_overlay = (
            input_data.voice_profile.summary_overlay
            if input_data.voice_profile is not None
            else ""
        )

        # ── Revision pass (single call, all findings merged) ──────────────
        revised_draft = self._run_revision_pass(
            input_data.full_draft,
            avoidance_moments=avoidance_moments,
            tone_drifts=tone_drifts,
            ending_finding=ending_finding,
            slop_markers=slop_markers,
            voice_register_note=voice_register_note,
        )
        revised_word_count = len(revised_draft.split())
        word_delta = revised_word_count - original_word_count

        logger.debug("ResonanceReviewerAgent revision complete | word_delta=%+d", word_delta)

        # ── Summary re-run ────────────────────────────────────────────────
        revised_summary = self._run_summary_pass(
            revised_draft,
            title=input_data.proposal_title,
            thematic_thesis=input_data.thematic_thesis,
            user_prompt=input_data.user_prompt,
            summary_overlay=summary_overlay,
        )
        logger.debug("ResonanceReviewerAgent summary re-run complete.")

        debug: dict[str, Any] = {
            "review_temperature": self._review_temperature,
            "revision_temperature": self._revision_temperature,
            "summary_temperature": self._summary_temperature,
            "review_calls_attempted": review_calls_attempted,
            "review_calls_failed": review_calls_failed,
            "near_miss_total": len(near_miss_all),
            "near_miss_restraint": restraint_count,
            "near_miss_avoidance": len(avoidance_moments),
            "tone_drifts": len(tone_drifts),
            "ending_verdict": ending_finding is not None,
            "slop_markers": len(slop_markers),
            "original_word_count": original_word_count,
            "revised_word_count": revised_word_count,
            "total_llm_calls": (review_calls_attempted - review_calls_failed) + 2,
        }

        return ResonanceReviewerAgentOutput(
            near_miss_moments=avoidance_moments,
            tone_drift_findings=tone_drifts,
            ending_verdict_finding=ending_finding,
            slop_markers=slop_markers,
            revised_draft=revised_draft,
            revised_summary=revised_summary,
            revision_word_delta=word_delta,
            moments_found=len(near_miss_all),
            moments_expanded=len(avoidance_moments),
            findings_total=findings_total,
            debug=debug,
            schema_version=RESONANCE_REVIEWER_SCHEMA_VERSION,
        )

    # ── Review-pass helpers (one per lens) ────────────────────────────────

    def _run_near_miss_review(
        self, input_data: ResonanceReviewerAgentInput
    ) -> tuple[list[NearMissMoment], bool]:
        """Near-miss review. Returns (moments, failed_bool)."""
        user_prompt_text = self._near_miss_prompt.format_user(
            title=input_data.proposal_title,
            thematic_thesis=input_data.thematic_thesis,
            scene_list_summary=input_data.scene_list_summary,
            full_draft=input_data.full_draft,
        )
        try:
            response = self._review_llm.complete_json(
                user_prompt_text,
                system_prompt=self._near_miss_prompt.system,
                temperature=self._review_temperature,
                max_tokens=self._review_max_tokens,
            )
        except Exception as exc:
            logger.warning("ResonanceReviewerAgent near-miss review failed: %s", exc)
            return [], True

        moments: list[NearMissMoment] = []
        for i, raw in enumerate(response.get("moments", [])):
            try:
                moments.append(NearMissMoment(**raw))
            except Exception as exc:
                logger.warning(
                    "ResonanceReviewerAgent: near-miss moment %d failed validation (%s) — skipping.",
                    i + 1,
                    exc,
                )
        return moments, False

    def _run_tone_review(
        self, input_data: ResonanceReviewerAgentInput
    ) -> tuple[list[ToneDriftFinding], bool]:
        """Tone-drift review. Skipped when no tones requested."""
        if not input_data.requested_tones:
            return [], False

        requested_tones_str = ", ".join(input_data.requested_tones)
        # Phrase used in the system prompt — e.g. "silly and high energy".
        requested_tones_phrase = " and ".join(input_data.requested_tones)
        system_prompt = self._tone_prompt.system.format(
            requested_tones=requested_tones_str,
            requested_tones_phrase=requested_tones_phrase,
        )
        user_prompt_text = self._tone_prompt.format_user(
            title=input_data.proposal_title,
            thematic_thesis=input_data.thematic_thesis,
            requested_tones=requested_tones_str,
            full_draft=input_data.full_draft,
        )
        try:
            response = self._review_llm.complete_json(
                user_prompt_text,
                system_prompt=system_prompt,
                temperature=self._review_temperature,
                max_tokens=self._review_max_tokens,
            )
        except Exception as exc:
            logger.warning("ResonanceReviewerAgent tone review failed: %s", exc)
            return [], True

        findings: list[ToneDriftFinding] = []
        for i, raw in enumerate(response.get("findings", [])):
            try:
                findings.append(ToneDriftFinding(**raw))
            except Exception as exc:
                logger.warning(
                    "ResonanceReviewerAgent: tone-drift finding %d failed validation (%s) — skipping.",
                    i + 1,
                    exc,
                )
        return findings, False

    def _run_ending_review(
        self, input_data: ResonanceReviewerAgentInput
    ) -> tuple[EndingVerdictFinding | None, bool]:
        """Ending-verdict review. Singular finding or None."""
        user_prompt_text = self._ending_prompt.format_user(
            title=input_data.proposal_title,
            thematic_thesis=input_data.thematic_thesis,
            full_draft=input_data.full_draft,
        )
        try:
            response = self._review_llm.complete_json(
                user_prompt_text,
                system_prompt=self._ending_prompt.system,
                temperature=self._review_temperature,
                max_tokens=self._review_max_tokens,
            )
        except Exception as exc:
            logger.warning("ResonanceReviewerAgent ending review failed: %s", exc)
            return None, True

        raw = response.get("finding")
        if raw is None:
            return None, False
        try:
            return EndingVerdictFinding(**raw), False
        except Exception as exc:
            logger.warning(
                "ResonanceReviewerAgent: ending-verdict finding failed validation (%s) — skipping.",
                exc,
            )
            return None, False

    def _run_slop_review(
        self, input_data: ResonanceReviewerAgentInput
    ) -> tuple[list[SlopMarker], bool]:
        """Slop / AI-tell review."""
        user_prompt_text = self._slop_prompt.format_user(
            title=input_data.proposal_title,
            full_draft=input_data.full_draft,
        )
        try:
            response = self._review_llm.complete_json(
                user_prompt_text,
                system_prompt=self._slop_prompt.system,
                temperature=self._review_temperature,
                max_tokens=self._review_max_tokens,
            )
        except Exception as exc:
            logger.warning("ResonanceReviewerAgent slop review failed: %s", exc)
            return [], True

        markers: list[SlopMarker] = []
        for i, raw in enumerate(response.get("markers", [])):
            try:
                markers.append(SlopMarker(**raw))
            except Exception as exc:
                logger.warning(
                    "ResonanceReviewerAgent: slop marker %d failed validation (%s) — skipping.",
                    i + 1,
                    exc,
                )
        return markers, False

    # ── Revision + summary helpers ────────────────────────────────────────

    def _run_revision_pass(
        self,
        full_draft: str,
        *,
        avoidance_moments: list[NearMissMoment],
        tone_drifts: list[ToneDriftFinding],
        ending_finding: EndingVerdictFinding | None,
        slop_markers: list[SlopMarker],
        voice_register_note: str = "",
    ) -> str:
        """Single revision pass that consumes the union of actionable findings."""
        revision_directives = self._format_revision_directives(
            avoidance_moments=avoidance_moments,
            tone_drifts=tone_drifts,
            ending_finding=ending_finding,
            slop_markers=slop_markers,
        )

        formatted_system = self._revise_prompt.system.format(
            voice_register_note=voice_register_note,
        )
        user_prompt_text = self._revise_prompt.format_user(
            full_draft=full_draft,
            revision_directives=revision_directives,
        )

        try:
            response = self._revision_llm.complete_json(
                user_prompt_text,
                system_prompt=formatted_system,
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

    @staticmethod
    def _format_revision_directives(
        *,
        avoidance_moments: list[NearMissMoment],
        tone_drifts: list[ToneDriftFinding],
        ending_finding: EndingVerdictFinding | None,
        slop_markers: list[SlopMarker],
    ) -> str:
        """Format the union of findings into a single labelled directives block.

        Each finding is prefixed with its kind so the revision prompt can
        apply the kind-specific rule. Order: near-miss, tone, ending, slop.
        """
        parts: list[str] = []
        for i, moment in enumerate(avoidance_moments, start=1):
            parts.append(
                f"[NEAR-MISS MOMENT {i}]\n"
                f"Passage: \"{moment.passage_ref}\"\n"
                f"What it implies: {moment.what_it_implies}\n"
                f"What the reader wanted: {moment.what_the_reader_wanted}\n"
                f"What the story did instead: {moment.what_the_story_did}\n"
                f"Expansion directive: {moment.expansion_directive}"
            )
        for i, drift in enumerate(tone_drifts, start=1):
            parts.append(
                f"[TONE DRIFT {i}]\n"
                f"Passage: \"{drift.passage_ref}\"\n"
                f"Requested tones: {', '.join(drift.requested_tones)}\n"
                f"Observed register: {drift.observed_register}\n"
                f"Why unearned: {drift.why_unearned}\n"
                f"Rewrite directive: {drift.rewrite_directive}"
            )
        if ending_finding is not None:
            parts.append(
                "[ENDING VERDICT]\n"
                f"Final passage: \"{ending_finding.final_passage}\"\n"
                f"Verdict named: {ending_finding.verdict_named}\n"
                f"Tension lost: {ending_finding.tension_lost}\n"
                f"Cut directive: {ending_finding.cut_directive}"
            )
        for i, marker in enumerate(slop_markers, start=1):
            parts.append(
                f"[SLOP MARKER {i}]\n"
                f"Quoted phrase: \"{marker.quoted_phrase}\"\n"
                f"Tell category: {marker.tell_category}\n"
                f"Why slop: {marker.why_slop}\n"
                f"Replacement directive: {marker.replacement_directive}"
            )
        return "\n\n".join(parts)

    def _run_summary_pass(
        self,
        revised_draft: str,
        *,
        title: str,
        thematic_thesis: str,
        user_prompt: str,
        summary_overlay: str = "",
    ) -> str:
        """Re-generate back-cover summary from revised draft."""
        formatted_system = self._summary_prompt.system.format(
            summary_overlay=summary_overlay,
        )
        user_prompt_text = self._summary_prompt.format_user(
            title=title,
            user_prompt=user_prompt,
            thematic_thesis=thematic_thesis,
            full_draft=revised_draft,
        )

        try:
            response = self._revision_llm.complete_json(
                user_prompt_text,
                system_prompt=formatted_system,
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
