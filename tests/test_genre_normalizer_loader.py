"""
Unit tests for storymesh.agents.genre_normalizer.loader module.
"""

from pathlib import Path

import orjson
import pytest

from storymesh.agents.genre_normalizer.loader import MappingLoadError, MappingStore


# Helper functions
def _write_json(path: Path, data: object) -> Path:
    """Write a python object as JSON."""
    path.write_bytes(orjson.dumps(data))
    return path

def _minimal_genre_map() -> dict:
    """A small valid genre map."""
    return {
        "fantasy": {
            "alternates": [],
            "genres": ["fantasy"],
            "subgenres": [],
            "default_tones": [ "wondrous", "adventurous", "epic"]
        },
        "science fiction": {
            "alternates": ["sci-fi", "scifi"],
            "genres": ["science_fiction"],
            "subgenres": [],
            "default_tones": ["futuristic", "speculative", "thought-provoking"]
        }
    }

def _minimal_tone_map() -> dict:
    """A small valid tone map."""
    return {
        "optimistic": {
            "alternates": [],
            "normalized_tones": ["optimistic", "hopeful", "upbeat"]
        },
        "dark": {
            "alternates": ["gloomy", "bleak"],
            "normalized_tones": ["dark", "grim", "somber"]
        }
    }

def _make_store(
        tmp_path: Path,
        genre_data: dict | None = None,
        tone_data: dict | None = None,
    ) -> MappingStore:
    """Build a MappingStore from minimal test data."""
    genre_path = _write_json(
        tmp_path / "genre_map.json",
        genre_data or _minimal_genre_map()
    )

    tone_path = _write_json(
        tmp_path / "tone_map.json",
        tone_data or _minimal_tone_map()
    )

    return MappingStore(genre_map_path=genre_path, tone_map_path=tone_path)

# File Loading Tests
class TestFileLoading:
    def test_missing_genre_file_raises(self, tmp_path: Path) -> None:
        tone_path = _write_json(tmp_path / "tone_map.json", _minimal_tone_map())
        missing = tmp_path / "nonexistant-file.json"
        with pytest.raises(MappingLoadError, match="not found"):
            MappingStore(genre_map_path=missing, tone_map_path=tone_path)

    def test_missing_tone_file_raises(self, tmp_path: Path) -> None:
        genre_path = _write_json(tmp_path / "genre_map.json", _minimal_genre_map())
        missing = tmp_path / "nonexistant-file.json"
        with pytest.raises(MappingLoadError, match="not found"):
            MappingStore(genre_map_path=genre_path, tone_map_path=missing)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "genre_map.json"
        bad_file.write_text("not valid json")
        tone_map = _write_json(tmp_path / "tone_map.json", _minimal_tone_map())
        with pytest.raises(MappingLoadError, match="Failed to read or parse"):
            MappingStore(genre_map_path=bad_file, tone_map_path=tone_map)
    
    def test_json_array_raises(self, tmp_path: Path) -> None:
        array_file = _write_json(
            tmp_path / "genre_map.json",
            ["not", "a", "dictionary"]
        )
        tone_map = _write_json(tmp_path / "tone_map.json", _minimal_tone_map())
        with pytest.raises(MappingLoadError, match="JSON must be an object"):
            MappingStore(genre_map_path=array_file, tone_map_path=tone_map)

# Entry validation
class TestEntryValidation:
    def test_invalid_genre_entry_raises(self, tmp_path: Path) -> None:
        bad_genre_map = {
            "broken": {
                "alternates": [],
                "genres": [],
                "subgenres": [],
                "default_tones": []
            }
        }
        with pytest.raises(MappingLoadError, match="Invalid genre_map entry"):
            _make_store(tmp_path, genre_data=bad_genre_map)

    def test_invalid_tone_entry_raises(self, tmp_path: Path) -> None:
        bad_tone_map = {
            "broken": {
                "alternates": [],
                "normalized_tones": []
            }
        }
        with pytest.raises(MappingLoadError, match="Invalid tone_map entry"):
            _make_store(tmp_path, tone_data=bad_tone_map)

# Index tests

class TestIndex:
    def test_primary_key_lookup(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        entry = store.lookup_genre("fantasy")
        assert entry is not None
        assert "fantasy" in entry.genres

    def test_alternate_key_lookup(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        entry = store.lookup_genre("sci-fi")
        assert entry is not None
        assert "science_fiction" in entry.genres

    def test_normalized_alternate_lookup(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        entry = store.lookup_genre("SciFi")
        assert entry is not None
        assert "science_fiction" in entry.genres

    def test_genre_miss_returns_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.lookup_genre("nonexistent genre") is None
    
    def test_tone_primary_key_lookup(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        entry = store.lookup_tone("optimistic")
        assert entry is not None
        assert "optimistic" in entry.normalized_tones

    def test_tone_alternate_lookup(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        entry = store.lookup_tone("gloomy")
        assert entry is not None
        assert "dark" in entry.normalized_tones

    def test_tone_miss_returns_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.lookup_tone("nonexistent tone") is None

    def test_index_keys_are_normalized(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # The following lookups should succeed due to normalization:
        assert store.lookup_genre("FANTASY") is not None
        assert store.lookup_genre("Science Fiction") is not None
        assert store.lookup_tone("DARK") is not None
        assert store.lookup_tone("Gloomy") is not None

    def test_all_alternates_indexed(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # All alternates should be indexed and return the same entry:
        sci_fi_entry = store.lookup_genre("sci-fi")
        assert sci_fi_entry is not None
        assert store.lookup_genre("scifi") == sci_fi_entry
        assert store.lookup_genre("Science Fiction") == sci_fi_entry

    def test_genre_max_words(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.genre_max_words == 2

    def test_tone_max_words(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.tone_max_words == 1

# Deduplication
class TestEntryDedupe:
    def test_primary_and_alternate_normalize_to_same_key(self, tmp_path: Path) -> None:
        """
        If a primary key and alternate normalize the same no collision should
        be raised.
        """
        bad_genre_map = {
            "post-apocalyptic": {
                "alternates": ["post apocalyptic"],
                "genres": ["science_fiction"],
                "subgenres": ["post_apocalyptic"],
                "default_tones": ["tense", "bleak", "survivalist"]
            }
        }
        store = _make_store(tmp_path, genre_data=bad_genre_map)
        entry = store.lookup_genre("post apocalyptic")
        assert entry is not None
        assert "post_apocalyptic" in entry.subgenres

# Collision Detection
class TestCollisionDetection:
    def test_cross_entry_genre_collision_raises(self, tmp_path: Path) -> None:
        colliding_map = {
            "sci fi": {
                "alternates": [],
                "genres": ["science_fiction"],
                "subgenres": [],
                "default_tones": [ "speculative", "thought-provoking"]
            },
            "science fiction": {
                "alternates": ["sci fi"],
                "genres": ["science_fiction"],
                "subgenres": [],
                "default_tones": [ "speculative", "thought-provoking"]
            }
        }
        with pytest.raises(MappingLoadError, match="collision detected"):
            _make_store(tmp_path, genre_data=colliding_map)

    def test_cross_entry_tone_collision_raises(self, tmp_path: Path) -> None:
        colliding_map = {
            "gloomy": {
                "alternates": [],
                "normalized_tones": ["dark", "grim", "somber"]
            },
            "dark": {
                "alternates": ["gloomy"],
                "normalized_tones": ["dark", "grim", "somber"]
            }
        }
        with pytest.raises(MappingLoadError, match="collision detected"):
            _make_store(tmp_path, tone_data=colliding_map)

# Full mapping file validation
class TestFullIncludedMapFiles:
    def test_included_map_files_load(self) -> None:
        """
        Validate that the included genre_map.json and tone_map.json files load without
        any errors.
        """
        store = MappingStore()
        assert len(store.genre_index) > 0
        assert len(store.tone_index) > 0