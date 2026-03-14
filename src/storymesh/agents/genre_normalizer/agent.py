"""
GenreNormalizerAgent - The public interface for normalizing genres from user prompts.

Takes the raw user input and produces a structured, validated output which contains
normalized genres, subgenres, tone profile, and full audit trails.
"""

from __future__ import annotations

from pathlib import Path

from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.genre_normalizer.resolver import resolve_all
from storymesh.agents.genre_normalizer.tone_merge import merge_tones
from storymesh.schemas.genre_normalizer import GenreNormalizerAgentInput, GenreNormalizerAgentOutput


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
            fuzzy_threshold: float = 0.85
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
            allow_llm_fallback = input_data.allow_llm_fallback
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

        # Assemble the output contract.
        return GenreNormalizerAgentOutput(
            raw_input = input_data.raw_genre,
            normalized_genres = normalized_genres,
            subgenres = subgenres,
            default_tones = tone_result.default_tones,
            explicit_tones = tone_result.explicit_tones,
            effective_tone = tone_result.effective_tone,
            tone_profile = tone_result.tone_profile,
            tone_conflicts = tone_result.tone_conflicts,
            genre_resolutions = resolver_result.genre_resolutions,
            tone_resolutions = resolver_result.tone_resolutions,
            unresolved_tokens = resolver_result.unresolved_tokens
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