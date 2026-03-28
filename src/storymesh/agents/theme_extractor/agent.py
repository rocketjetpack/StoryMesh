"""ThemeExtractorAgent — Stage 3 of the StoryMesh pipeline.

Extracts thematic tensions from ranked books across genre traditions.
Rather than listing themes shared across books, it identifies the thematic
assumptions each genre tradition takes for granted, finds contradictions
between traditions, and frames those contradictions as creative questions
for story generation.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson

from storymesh.llm.base import LLMClient
from storymesh.prompts.loader import load_prompt
from storymesh.schemas.theme_extractor import (
    GenreCluster,
    NarrativeSeed,
    ThematicTension,
    ThemeExtractorAgentInput,
    ThemeExtractorAgentOutput,
)

logger = logging.getLogger(__name__)


class ThemeExtractorAgent:
    """Extracts thematic tensions from ranked books across genre traditions (Stage 3).

    This is the creative engine of the pipeline. Rather than listing themes
    shared across books, it identifies the thematic assumptions each genre
    tradition takes for granted, finds contradictions between traditions,
    and frames those contradictions as creative questions for story generation.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_seeds: int = 5,
    ) -> None:
        """Construct the agent.

        Args:
            llm_client: LLM client instance. Required — theme extraction is a
                creative synthesis task with no deterministic fallback.
            temperature: LLM temperature. Default 0.6 balances creative
                interpretation with structured JSON output requirements.
            max_tokens: Maximum tokens for the LLM call.
            max_seeds: Maximum number of NarrativeSeeds to generate. Caps
                ProposalDraftAgent's selection workload.
        """
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_seeds = max_seeds

        # Load the prompt eagerly so misconfiguration is caught at construction.
        self._prompt_template = load_prompt("theme_extractor")

    def run(self, input_data: ThemeExtractorAgentInput) -> ThemeExtractorAgentOutput:
        """Extract thematic tensions and generate narrative seeds.

        Args:
            input_data: Assembled input from multiple upstream pipeline stages.

        Returns:
            A frozen ThemeExtractorAgentOutput (ThemePack) with genre clusters,
            thematic tensions, and narrative seeds.

        Raises:
            ValueError: If the LLM response is missing required fields or fails
                Pydantic validation.
            RuntimeError: If the LLM call fails (propagated from complete_json).
        """
        logger.info(
            "ThemeExtractorAgent starting | books=%d genres=%d max_seeds=%d",
            len(input_data.ranked_summaries),
            len(input_data.normalized_genres),
            self._max_seeds,
        )

        # Serialize the book list to a compact representation for the prompt.
        # Strip composite_score and work_key — the LLM does not need scoring internals.
        book_dicts = [
            {
                "title": s.title,
                "authors": s.authors,
                "source_genres": s.source_genres,
                "rank": s.rank,
            }
            for s in input_data.ranked_summaries
        ]
        book_list_json = orjson.dumps(book_dicts).decode()

        formatted_user = self._prompt_template.format_user(
            user_prompt=input_data.user_prompt,
            normalized_genres=input_data.normalized_genres,
            subgenres=input_data.subgenres,
            user_tones=input_data.user_tones,
            narrative_context=input_data.narrative_context,
            book_list=book_list_json,
            max_seeds=self._max_seeds,
        )

        response = self._llm_client.complete_json(
            formatted_user,
            system_prompt=self._prompt_template.system,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        genre_clusters = self._parse_genre_clusters(response)
        tensions = self._parse_tensions(response)
        narrative_seeds = self._parse_narrative_seeds(response)

        debug: dict[str, Any] = {
            "books_processed": len(input_data.ranked_summaries),
            "clusters_found": len(genre_clusters),
            "tensions_found": len(tensions),
            "seeds_generated": len(narrative_seeds),
        }

        logger.info(
            "ThemeExtractorAgent complete | clusters=%d tensions=%d seeds=%d",
            len(genre_clusters),
            len(tensions),
            len(narrative_seeds),
        )

        return ThemeExtractorAgentOutput(
            genre_clusters=genre_clusters,
            tensions=tensions,
            narrative_seeds=narrative_seeds,
            user_tones_carried=list(input_data.user_tones),
            debug=debug,
        )

    @staticmethod
    def _parse_genre_clusters(response: dict[str, Any]) -> list[GenreCluster]:
        """Parse and validate genre_clusters from the LLM response dict.

        Args:
            response: Raw JSON dict from the LLM.

        Returns:
            List of validated GenreCluster instances.

        Raises:
            ValueError: If the field is missing, empty, or fails Pydantic validation.
        """
        raw = response.get("genre_clusters", [])
        try:
            clusters = [GenreCluster(**c) for c in raw]
        except Exception as exc:
            raise ValueError(
                f"ThemeExtractorAgent: failed to parse genre_clusters: {exc}"
            ) from exc
        if not clusters:
            raise ValueError(
                "ThemeExtractorAgent: LLM response contained no genre_clusters."
            )
        return clusters

    @staticmethod
    def _parse_tensions(response: dict[str, Any]) -> list[ThematicTension]:
        """Parse and validate tensions from the LLM response dict.

        Args:
            response: Raw JSON dict from the LLM.

        Returns:
            List of validated ThematicTension instances.

        Raises:
            ValueError: If the field is missing, empty, or fails Pydantic validation.
        """
        raw = response.get("tensions", [])
        try:
            tensions = [ThematicTension(**t) for t in raw]
        except Exception as exc:
            raise ValueError(
                f"ThemeExtractorAgent: failed to parse tensions: {exc}"
            ) from exc
        if not tensions:
            raise ValueError(
                "ThemeExtractorAgent: LLM response contained no tensions."
            )
        return tensions

    @staticmethod
    def _parse_narrative_seeds(response: dict[str, Any]) -> list[NarrativeSeed]:
        """Parse and validate narrative_seeds from the LLM response dict.

        Args:
            response: Raw JSON dict from the LLM.

        Returns:
            List of validated NarrativeSeed instances.

        Raises:
            ValueError: If the field is missing, empty, or fails Pydantic validation.
        """
        raw = response.get("narrative_seeds", [])
        try:
            seeds = [NarrativeSeed(**s) for s in raw]
        except Exception as exc:
            raise ValueError(
                f"ThemeExtractorAgent: failed to parse narrative_seeds: {exc}"
            ) from exc
        if not seeds:
            raise ValueError(
                "ThemeExtractorAgent: LLM response contained no narrative_seeds."
            )
        return seeds
