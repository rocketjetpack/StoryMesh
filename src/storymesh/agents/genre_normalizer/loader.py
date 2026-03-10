"""
This is the file loader and index builder for the GenreNormalizerAgent.

This module loads genre_map.json and tone_map.json, validates each entry
against Pydantic schemas, and builds flat lookup indices that are keyed
by normalized strings. Collisions are handled by ensuring that no two
distinct entries resolve to the same normalized key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from storymesh.agents.genre_normalizer.normalize import normalize_text
from storymesh.schemas.genre_normalizer import GenreMapEntry, ToneMapEntry

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_GENRE_MAP_PATH = _DATA_DIR / "genre_map.json"
_DEFAULT_TONE_MAP_PATH = _DATA_DIR / "tone_map.json"

class MappingLoadError(Exception):
    """Raised when a map file fails to load or validate."""

class MappingStore:
    """
    Loads, validates, and indexes genre and tone map files.

    On initialization, reads both JSON files, validates each entry against
    Pydantic schemas, builds flat lookup indices with normalized keys, and
    finally checks for cross-entry and cross-file collisions.

    Attributes:
        genre_index: Flat lookup from normalized key to a GenreMapEntry.
        tone_index: Flat lookup from normalized key to a ToneMapEntry.
    """

    def __init__(
        self,
        genre_map_path: Path = _DEFAULT_GENRE_MAP_PATH,
        tone_map_path: Path = _DEFAULT_TONE_MAP_PATH,
    ) -> None:
        genre_entries = self._load_file(genre_map_path, "genre_map")
        tone_entries = self._load_file(tone_map_path, "tone_map")

        self.genre_index = self._build_genre_index(genre_entries)
        self.tone_index = self._build_tone_index(tone_entries)

        self.genre_max_words: int = (
            max((len(k.split()) for k in self.genre_index), default=0)
        )

        self.tone_max_words: int = ( 
            max((len(k.split()) for k in self.tone_index), default=0)
        )

        self._check_cross_file_collisions()

    def lookup_genre(self, key: str) -> GenreMapEntry | None:
        """
        Lookup a normalized key in the genre index.

        Args:
            key: The normalized key to lookup.

        Returns:
            The corresponding GenreMapEntry if found, else None.
        """

        return self.genre_index.get(normalize_text(key))

    def lookup_tone(self, key: str) -> ToneMapEntry | None:
        """
        Lookup a normalized key in the tone index.

        Args:
            key: The normalized key to lookup.

        Returns:
            The corresponding ToneMapEntry if found, else None.
        """

        return self.tone_index.get(normalize_text(key))

    @staticmethod
    def _load_file(path: Path, label: str) -> dict[str, dict[str, Any]]:
        """
        Read and parse a JSON mapping file.

        Args:
            path: Path to the JSON file.
            label: A human-readable label for error messages.

        Returns:
            The parsed JSON as a dictionary.

        Raises:
            MappingLoadError: If the file cannot be read or parsed.
        """

        if not path.exists():
            raise MappingLoadError(f"{label} file not found at {path}")

        try:
            raw = path.read_bytes()
            data = orjson.loads(raw)

            if not isinstance(data, dict):
                raise MappingLoadError(
                    f"{label} JSON must be an object at the top level, got {type(data).__name__}" # noqa: E501
                )
            return data
        except (OSError, orjson.JSONDecodeError) as e:
            raise MappingLoadError(
                f"Failed to read or parse {label} at {path}: {e}"
            ) from e

    @staticmethod
    def _build_genre_index(
            entries: dict[str, dict[str, Any]],
        ) -> dict[str, GenreMapEntry]:
        """
        Validate genre entries and build a flat normalized lookup index.

        Args:
            entries: Raw JSON entries keyed by canonical name.

        Returns:
            A flat dictionary mapping each normalized key and alternate
            value to a valid GenreMapEntry.

        Raises:
            MappingLoadError: if any entry fails validation or if two distinct
            entries produce the same normalized key.
        """
        index: dict[str, GenreMapEntry] = {}
        key_owners: dict[str, str] = {} # Maps normalized keys to canonical values

        for canonical_key, raw_entry in entries.items():
            try:
                entry = GenreMapEntry(**raw_entry)
            except ValidationError as e:
                raise MappingLoadError(
                    f"Invalid genre_map entry for: '{canonical_key}': {e}"
                ) from e

            # Combine the canonical key with all alternates into a list.
            all_forms = [canonical_key] + entry.alternates

            for form in all_forms:
                normalized = normalize_text(form)

                if normalized in index:
                    if index[normalized] is not entry:
                        raise MappingLoadError(
                            f"Genre index collision detected: normalized key '{normalized}' " # noqa: E501
                            f"is claimed by both '{key_owners[normalized]}' and '{canonical_key}'" # noqa: E501
                        )
                else:
                    index[normalized] = entry
                    key_owners[normalized] = canonical_key
        return index   

    @staticmethod
    def _build_tone_index(
            #self,
            entries: dict[str, dict[str, Any]],
        ) -> dict[str, ToneMapEntry]:
        """
        Validate tone entries and build a flat normalized lookup index.

        Args:
            entries: Raw JSON entries keyed by canonical name.

        Returns:
            A flat dictionary mapping each normalized key and alternate
            value to a valid ToneMapEntry.

        Raises:
            MappingLoadError: if any entry fails validation or if two distinct
            entries produce the same normalized key.
        """
        index: dict[str, ToneMapEntry] = {}
        key_owners: dict[str, str] = {} # Maps normalized keys to canonical values

        for canonical_key, raw_entry in entries.items():
            try:
                entry = ToneMapEntry(**raw_entry)
            except ValidationError as e:
                raise MappingLoadError(
                    f"Invalid tone_map entry for: '{canonical_key}': {e}"
                ) from e

            # Combine the canonical key with all alternates into a list.
            all_forms = [canonical_key] + entry.alternates

            for form in all_forms:
                normalized = normalize_text(form)

                if normalized in index:
                    if index[normalized] is not entry:
                        raise MappingLoadError(
                            f"Tone index collision detected: normalized key '{normalized}' " # noqa: E501
                            f"is claimed by both '{key_owners[normalized]}' and '{canonical_key}'" # noqa: E501
                        )
                else:
                    index[normalized] = entry
                    key_owners[normalized] = canonical_key
        return index

    def _check_cross_file_collisions(self) -> None:
        """
        Ensure that no normalized key appears in both genre and tone indices.

        Raises:
            MappingLoadError: if a collision is detected.
        """
        genre_keys = set(self.genre_index.keys())
        tone_keys = set(self.tone_index.keys())
        collisions = genre_keys.intersection(tone_keys)

        if collisions:
            keys_str = ", ".join(f"'{key}'" for key in collisions)
            raise MappingLoadError(
                f"Cross-file collision detected for normalized keys: {keys_str}"
            )