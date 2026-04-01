"""
GenreNormalizerAgent - The public interface for normalizing genres from user prompts.

Takes the raw user input and produces a structured, validated output which contains
normalized genres, subgenres, user tones, and override information for downstream agents.
Resolution details and audit trails are available in the debug dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.genre_normalizer.resolver import resolve_all
from storymesh.agents.genre_normalizer.tone_merge import merge_tones
from storymesh.exceptions import GenreResolutionError
from storymesh.llm.base import LLMClient
from storymesh.schemas.genre_normalizer import (
    GenreNormalizerAgentInput,
    GenreNormalizerAgentOutput,
    InferredGenre,
)


class GenreNormalizerAgent:
    """Transforms raw input into a structured object.

    This is a hybrid agent that attempts to provide deterministic mapping via file lookups
    with fuzzy matching with the ability to fall back to an LLM for words which fail matching.
    """

    def __init__(
            self,
            store: MappingStore | None = None,
            genre_map_path: Path | None = None,
            tone_map_path: Path | None = None,
            fuzzy_threshold: float = 0.85,
            llm_client: LLMClient | None = None,
            temperature: float = 0.0,
            max_tokens: int = 1024
        ) -> None:
        """
        Initialize the agent.

        Args:
            store: Pre-built MappingStore. If provided, ignore path arguments
            genre_map_path: Path to the genre_map.json and tone_map.json files
            fuzzy_threshold: Minimum rapidfuzz confidence for fuzzy matching
        """

        if store is not None:
            self._store = store
        else:
            kwargs: dict[str, Path] = {}
            if genre_map_path is not None:
                kwargs["genre_map_path"] = genre_map_path
            if tone_map_path is not None:
                kwargs["tone_map_path"] = tone_map_path
            self._store = MappingStore(**kwargs)

        self._fuzzy_match_threshold = fuzzy_threshold

        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def run(self, input_data: GenreNormalizerAgentInput) -> GenreNormalizerAgentOutput:
        """
        Run the genre normalization pipeline.

        Args:
            input_data: The validated input contract.

        Returns:
            A frozen, versioned GenreNoralizerAgentOutput contract.
        """
        
        resolver_result = resolve_all(
            raw_input = input_data.raw_genre,
            store = self._store,
            fuzzy_threshold = self._fuzzy_match_threshold,
            allow_llm_fallback = input_data.allow_llm_fallback,
            llm_client = self._llm_client,
            temperature = self._temperature,
            max_tokens = self._max_tokens
        )

        tone_result = merge_tones(
            genre_resolutions = resolver_result.genre_resolutions,
            tone_resolutions = resolver_result.tone_resolutions
        )

        normalized_genres = _deduplicate_preserve_order([
            genre for resolution in resolver_result.genre_resolutions
                for genre in resolution.canonical_genres
        ])

        subgenres = _deduplicate_preserve_order([
            subgenre for resolution in resolver_result.genre_resolutions
                for subgenre in resolution.subgenres
        ])

        inferred_genres: list[InferredGenre] = resolver_result.inferred_genres

        # Option B: When Passes 1–3 find nothing but Pass 4 infers genres,
        # promote the inferred canonical genres into normalized_genres as a
        # last-resort fallback so the pipeline can continue. The full
        # InferredGenre objects remain in inferred_genres for downstream
        # consumers that want rationale and confidence data.
        if not normalized_genres and inferred_genres:
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger(__name__).warning(
                "Passes 1–3 found no explicit genres for %r. "
                "Promoting %d Pass 4 inferred genre(s) into normalized_genres "
                "as a last-resort fallback.",
                input_data.raw_genre,
                len(inferred_genres),
            )
            normalized_genres = _deduplicate_preserve_order(
                [ig.canonical_genre for ig in inferred_genres]
            )

        # Build the debug dict with full resolution and audit data.
        debug: dict[str, Any] = {
            **tone_result.debug,
            "genre_resolutions": [r.model_dump() for r in resolver_result.genre_resolutions],
            "tone_resolutions": [r.model_dump() for r in resolver_result.tone_resolutions],
            "narrative_context": resolver_result.narrative_context,
            "unresolved_tokens": resolver_result.unresolved_tokens,
            "inferred_genres": [ig.model_dump() for ig in inferred_genres],
        }

        if not normalized_genres:
            llm_configured = self._llm_client is not None
            llm_allowed = input_data.allow_llm_fallback

            if llm_configured and llm_allowed:
                detail = (
                    "LLM fallback was attempted but returned no recognizable genres. "
                    "Try rephrasing with explicit genre keywords."
                )
            elif llm_configured and not llm_allowed:
                detail = (
                    "LLM fallback is disabled. "
                    "Enable it or rephrase with explicit genre keywords."
                )
            elif not llm_configured and llm_allowed:
                detail = (
                    "No LLM client is configured. "
                    "Rephrase with explicit genre keywords (e.g. 'fantasy', 'thriller')."
                )
            else:
                detail = (
                    "No genres could be resolved. "
                    "Try including a genre keyword (e.g. 'fantasy', 'thriller')."
                )

            raise GenreResolutionError(
                f"No genres could be resolved from input: {input_data.raw_genre!r}. {detail}"
            )

        # Assemble the output contract.
        return GenreNormalizerAgentOutput(
            raw_input=input_data.raw_genre,
            normalized_genres=normalized_genres,
            subgenres=subgenres,
            user_tones=tone_result.user_tones,
            tone_override=tone_result.tone_override,
            override_note=tone_result.override_note,
            narrative_context=resolver_result.narrative_context,
            inferred_genres=inferred_genres,
            debug=debug,
        )

def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicates from a list while preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result