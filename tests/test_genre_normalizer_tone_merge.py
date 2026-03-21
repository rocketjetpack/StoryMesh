"""
Unit tests for storymesh.agents.genre_normalizer.tone_merge.
"""

import pytest

from storymesh.agents.genre_normalizer.tone_merge import (
    ToneMergeResult,
    merge_tones,
)
from storymesh.schemas.genre_normalizer import (
    GenreResolution,
    ResolutionMethod,
    ToneResolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genre_resolution(**overrides: object) -> GenreResolution:
    defaults = dict(
        input_token="fantasy",
        canonical_genres=["fantasy"],
        default_tones=["adventurous", "epic"],
        method=ResolutionMethod.STATIC_EXACT,
        confidence=1.0,
    )
    return GenreResolution(**(defaults | overrides))


def _tone_resolution(**overrides: object) -> ToneResolution:
    defaults = dict(
        input_token="optimistic",
        normalized_tones=["optimistic", "hopeful"],
        method=ResolutionMethod.STATIC_EXACT,
        confidence=1.0,
    )
    return ToneResolution(**(defaults | overrides))


# ---------------------------------------------------------------------------
# Defaults Only (no explicit tones)
# ---------------------------------------------------------------------------

class TestDefaultsOnly:
    def test_single_genre(self) -> None:
        result = merge_tones(
            genre_resolutions=[_genre_resolution()],
            tone_resolutions=[],
        )
        # Contract fields
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.override_note is None
        # Debug fields
        assert result.debug["default_tones"] == ["adventurous", "epic"]
        assert result.debug["explicit_tones"] == []
        assert result.debug["tone_profile"] == ["adventurous", "epic"]
        assert result.debug["effective_tone"] == "adventurous"
        assert result.debug["tone_conflicts"] is None

    def test_multiple_genres_concatenated(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    input_token="post-apocalyptic",
                    canonical_genres=["science_fiction"],
                    default_tones=["bleak", "tense"],
                ),
                _genre_resolution(
                    input_token="mystery",
                    canonical_genres=["mystery"],
                    default_tones=["suspenseful", "cerebral"],
                ),
            ],
            tone_resolutions=[],
        )
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.debug["default_tones"] == ["bleak", "tense", "suspenseful", "cerebral"]
        assert result.debug["tone_profile"] == ["bleak", "tense", "suspenseful", "cerebral"]
        assert result.debug["effective_tone"] == "bleak"

    def test_duplicate_defaults_deduplicated(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    input_token="fantasy",
                    default_tones=["epic", "adventurous"],
                ),
                _genre_resolution(
                    input_token="epic fantasy",
                    default_tones=["epic", "grand"],
                ),
            ],
            tone_resolutions=[],
        )
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.debug["default_tones"] == ["epic", "adventurous", "grand"]
        assert result.debug["tone_profile"] == ["epic", "adventurous", "grand"]


# ---------------------------------------------------------------------------
# Explicits Only (no genre defaults)
# ---------------------------------------------------------------------------

class TestExplicitsOnly:
    def test_single_explicit(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(default_tones=[]),
            ],
            tone_resolutions=[_tone_resolution()],
        )
        assert result.user_tones == ["optimistic"]
        assert result.tone_override is False
        assert result.override_note is None
        assert result.debug["explicit_tones"] == ["optimistic", "hopeful"]
        assert result.debug["default_tones"] == []
        assert result.debug["tone_profile"] == ["optimistic", "hopeful"]
        assert result.debug["effective_tone"] == "optimistic"
        assert result.debug["tone_conflicts"] is None

    def test_multiple_explicits(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(default_tones=[]),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="gritty",
                    normalized_tones=["gritty", "raw"],
                ),
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic", "hopeful"],
                ),
            ],
        )
        assert result.user_tones == ["gritty", "optimistic"]
        assert result.tone_override is False
        assert result.debug["explicit_tones"] == ["gritty", "raw", "optimistic", "hopeful"]
        assert result.debug["effective_tone"] == "gritty"

    def test_duplicate_explicits_deduplicated(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(default_tones=[]),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="dark",
                    normalized_tones=["dark", "grim"],
                ),
                _tone_resolution(
                    input_token="gritty",
                    normalized_tones=["gritty", "dark"],
                ),
            ],
        )
        assert result.user_tones == ["dark", "gritty"]
        assert result.debug["explicit_tones"] == ["dark", "grim", "gritty"]


# ---------------------------------------------------------------------------
# Combined: Explicits Override Defaults
# ---------------------------------------------------------------------------

class TestCombinedPriority:
    def test_explicits_before_defaults(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    input_token="post-apocalyptic",
                    default_tones=["bleak", "tense", "survivalist"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic", "hopeful"],
                ),
            ],
        )
        assert result.user_tones == ["optimistic"]
        assert result.tone_override is True
        assert result.override_note is not None
        assert result.debug["tone_profile"] == [
            "optimistic", "hopeful",
            "bleak", "tense", "survivalist",
        ]
        assert result.debug["effective_tone"] == "optimistic"

    def test_shared_tone_appears_in_explicit_position(self) -> None:
        """If an explicit tone matches a default, it stays in explicit position only."""
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    default_tones=["dark", "brooding", "ominous"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="dark",
                    normalized_tones=["dark", "grim"],
                ),
            ],
        )
        assert result.user_tones == ["dark"]
        assert result.tone_override is True
        assert result.debug["tone_profile"] == ["dark", "grim", "brooding", "ominous"]
        assert result.debug["effective_tone"] == "dark"

    def test_multiple_genres_and_explicits(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    input_token="post-apocalyptic",
                    default_tones=["bleak", "tense"],
                ),
                _genre_resolution(
                    input_token="mystery",
                    default_tones=["suspenseful", "cerebral"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="gritty",
                    normalized_tones=["gritty", "raw"],
                ),
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic", "hopeful"],
                ),
            ],
        )
        assert result.user_tones == ["gritty", "optimistic"]
        assert result.tone_override is True
        assert result.debug["tone_profile"] == [
            "gritty", "raw", "optimistic", "hopeful",
            "bleak", "tense", "suspenseful", "cerebral",
        ]
        assert result.debug["effective_tone"] == "gritty"


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_conflict_when_explicit_not_in_defaults(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    default_tones=["bleak", "tense"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic"],
                ),
            ],
        )
        assert result.tone_override is True
        assert result.override_note is not None
        assert result.debug["tone_conflicts"] is not None
        assert len(result.debug["tone_conflicts"]) == 1
        assert "optimistic" in result.debug["tone_conflicts"][0]

    def test_no_conflict_when_explicit_subset_of_defaults(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    default_tones=["dark", "brooding", "ominous"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="dark",
                    normalized_tones=["dark"],
                ),
            ],
        )
        assert result.tone_override is False
        assert result.override_note is None
        assert result.debug["tone_conflicts"] is None

    def test_no_conflict_when_no_defaults(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(default_tones=[]),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic"],
                ),
            ],
        )
        assert result.tone_override is False
        assert result.debug["tone_conflicts"] is None

    def test_no_conflict_when_no_explicits(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(default_tones=["bleak", "tense"]),
            ],
            tone_resolutions=[],
        )
        assert result.tone_override is False
        assert result.debug["tone_conflicts"] is None

    def test_multiple_conflicts(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    default_tones=["bleak", "tense"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic"],
                ),
                _tone_resolution(
                    input_token="lighthearted",
                    normalized_tones=["lighthearted"],
                ),
            ],
        )
        assert result.tone_override is True
        assert result.user_tones == ["optimistic", "lighthearted"]
        assert result.debug["tone_conflicts"] is not None
        assert len(result.debug["tone_conflicts"]) == 2

    def test_partial_conflict(self) -> None:
        """One explicit matches defaults, another doesn't."""
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(
                    default_tones=["dark", "brooding"],
                ),
            ],
            tone_resolutions=[
                _tone_resolution(
                    input_token="dark",
                    normalized_tones=["dark"],
                ),
                _tone_resolution(
                    input_token="optimistic",
                    normalized_tones=["optimistic"],
                ),
            ],
        )
        assert result.tone_override is True
        assert result.user_tones == ["dark", "optimistic"]
        assert result.debug["tone_conflicts"] is not None
        assert len(result.debug["tone_conflicts"]) == 1
        assert "optimistic" in result.debug["tone_conflicts"][0]


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestFallback:
    def test_empty_both_sources(self) -> None:
        result = merge_tones(
            genre_resolutions=[
                _genre_resolution(default_tones=[]),
            ],
            tone_resolutions=[],
        )
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.debug["effective_tone"] == "neutral"
        assert result.debug["tone_profile"] == ["neutral"]
        assert result.debug["tone_conflicts"] is None

    def test_empty_resolutions(self) -> None:
        """Edge case: no resolutions at all. Shouldn't happen in practice
        but the merge logic should handle it gracefully."""
        result = merge_tones(
            genre_resolutions=[],
            tone_resolutions=[],
        )
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.debug["effective_tone"] == "neutral"
        assert result.debug["tone_profile"] == ["neutral"]


# ---------------------------------------------------------------------------
# ToneMergeResult
# ---------------------------------------------------------------------------

class TestToneMergeResult:
    def test_frozen(self) -> None:
        result = ToneMergeResult()
        with pytest.raises(AttributeError):
            result.user_tones = ["changed"]  # type: ignore[misc]

    def test_defaults(self) -> None:
        result = ToneMergeResult()
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.override_note is None
        assert result.debug == {}