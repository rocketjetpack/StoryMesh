"""Unit tests for storymesh.agents.book_fetcher.subject_map.resolve_subjects().

Tests cover the three code paths in resolve_subjects():
  1. Known genre with entries → first subject string used
  2. Known genre with empty list → silently dropped
  3. Unknown genre → underscore-to-space fallback
Plus deduplication and mixed-input scenarios.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

from storymesh.agents.book_fetcher.subject_map import resolve_subjects

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal map used in tests that patch the module-level dict so tests
# remain independent of genre_subject_map.json on disk.
_MOCK_MAP: dict[str, list[str]] = {
    "fantasy": ["fantasy"],
    "science_fiction": ["science fiction"],
    "thriller": ["thriller", "suspense"],
    "workplace_fiction": [],
}


def _with_mock_map(fn: Callable[..., None]) -> Callable[..., None]:
    """Decorator: patches _GENRE_SUBJECT_MAP for the duration of the test."""
    return patch(
        "storymesh.agents.book_fetcher.subject_map._GENRE_SUBJECT_MAP",
        _MOCK_MAP,
    )(fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveSubjectsKnownGenre:
    """Known genres in the map with entries."""

    @_with_mock_map
    def test_single_known_genre_returns_first_subject(self) -> None:
        result = resolve_subjects(["science_fiction"])
        assert result == ["science fiction"]

    @_with_mock_map
    def test_multi_entry_genre_returns_only_first(self) -> None:
        """thriller maps to ["thriller", "suspense"]; only "thriller" is used."""
        result = resolve_subjects(["thriller"])
        assert result == ["thriller"]

    @_with_mock_map
    def test_multiple_known_genres(self) -> None:
        result = resolve_subjects(["fantasy", "science_fiction"])
        assert result == ["fantasy", "science fiction"]


class TestResolveSubjectsEmptyMapping:
    """Genres that map to an empty list are silently dropped."""

    @_with_mock_map
    def test_empty_mapped_genre_is_dropped(self) -> None:
        result = resolve_subjects(["workplace_fiction"])
        assert result == []

    @_with_mock_map
    def test_empty_mapped_genre_among_others_is_dropped(self) -> None:
        result = resolve_subjects(["fantasy", "workplace_fiction", "thriller"])
        assert result == ["fantasy", "thriller"]


class TestResolveSubjectsUnknownGenre:
    """Genres absent from the map get underscore→space fallback."""

    @_with_mock_map
    def test_unknown_genre_underscore_to_space(self) -> None:
        result = resolve_subjects(["dark_fantasy"])
        assert result == ["dark fantasy"]

    @_with_mock_map
    def test_unknown_genre_no_underscore_passed_through(self) -> None:
        # "horror" not in _MOCK_MAP, so fallback: "horror".replace("_", " ") == "horror"
        result = resolve_subjects(["horror"])
        assert result == ["horror"]

    @_with_mock_map
    def test_unknown_and_known_genre_combined(self) -> None:
        result = resolve_subjects(["fantasy", "dark_fantasy"])
        assert result == ["fantasy", "dark fantasy"]


class TestResolveSubjectsDeduplication:
    """Duplicate subject strings are removed, preserving first-occurrence order."""

    @_with_mock_map
    def test_duplicate_genres_deduplicated(self) -> None:
        result = resolve_subjects(["fantasy", "fantasy"])
        assert result == ["fantasy"]

    @_with_mock_map
    def test_same_subject_from_two_genres_deduplicated(self) -> None:
        # Both "fantasy" (in map → "fantasy") and "fantasy_novel" (unknown,
        # fallback → "fantasy novel") are different subjects, but two identical
        # known genres should still deduplicate.
        result = resolve_subjects(["science_fiction", "science_fiction"])
        assert result == ["science fiction"]
        assert len(result) == 1


class TestResolveSubjectsMixedInput:
    """Mixed known, empty-mapped, unknown, and duplicate genres in one call."""

    @_with_mock_map
    def test_mixed_input_integration(self) -> None:
        genres = [
            "science_fiction",    # known → "science fiction"
            "workplace_fiction",  # empty mapping → dropped
            "dark_fantasy",       # unknown → "dark fantasy"
            "science_fiction",    # duplicate → dropped
            "fantasy",            # known → "fantasy"
        ]
        result = resolve_subjects(genres)
        assert result == ["science fiction", "dark fantasy", "fantasy"]

    @_with_mock_map
    def test_empty_input_returns_empty(self) -> None:
        assert resolve_subjects([]) == []
