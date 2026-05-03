"""Unit tests for load_voice_profile() in src/storymesh/schemas/voice_profile.py."""

from __future__ import annotations

import pytest

from storymesh.schemas.voice_profile import (
    BUILT_IN_PROFILE_IDS,
    VoiceProfile,
    load_voice_profile,
)

# ---------------------------------------------------------------------------
# Loading all built-in profiles
# ---------------------------------------------------------------------------


class TestLoadAllBuiltInProfiles:
    def test_loads_literary_restraint(self) -> None:
        profile = load_voice_profile("literary_restraint")
        assert isinstance(profile, VoiceProfile)
        assert profile.id == "literary_restraint"

    def test_loads_cozy_warmth(self) -> None:
        profile = load_voice_profile("cozy_warmth")
        assert isinstance(profile, VoiceProfile)
        assert profile.id == "cozy_warmth"

    def test_loads_genre_active(self) -> None:
        profile = load_voice_profile("genre_active")
        assert isinstance(profile, VoiceProfile)
        assert profile.id == "genre_active"

    def test_all_built_in_ids_load(self) -> None:
        for profile_id in BUILT_IN_PROFILE_IDS:
            profile = load_voice_profile(profile_id)
            assert profile.id == profile_id, f"id mismatch for {profile_id}"

    def test_all_profiles_are_frozen(self) -> None:
        from pydantic import ValidationError

        for profile_id in BUILT_IN_PROFILE_IDS:
            profile = load_voice_profile(profile_id)
            with pytest.raises(ValidationError):
                profile.id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unknown profile raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestUnknownProfileRaises:
    def test_unknown_profile_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Voice profile not found"):
            load_voice_profile("nonexistent_profile")

    def test_empty_id_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_voice_profile("")

    def test_path_traversal_does_not_load(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_voice_profile("../story_writer_draft")


# ---------------------------------------------------------------------------
# Overlay field content assertions
# ---------------------------------------------------------------------------


class TestOverlayFieldContent:
    def test_literary_restraint_craft_overlay_is_empty(self) -> None:
        """literary_restraint intentionally uses empty overlays for backward compat."""
        profile = load_voice_profile("literary_restraint")
        assert profile.craft_overlay == ""

    def test_literary_restraint_avoid_overlay_is_empty(self) -> None:
        profile = load_voice_profile("literary_restraint")
        assert profile.avoid_overlay == ""

    def test_literary_restraint_summary_overlay_is_empty(self) -> None:
        profile = load_voice_profile("literary_restraint")
        assert profile.summary_overlay == ""

    def test_cozy_warmth_craft_overlay_is_non_empty(self) -> None:
        profile = load_voice_profile("cozy_warmth")
        assert profile.craft_overlay.strip(), "cozy_warmth craft_overlay must be non-empty"

    def test_cozy_warmth_avoid_overlay_is_non_empty(self) -> None:
        profile = load_voice_profile("cozy_warmth")
        assert profile.avoid_overlay.strip(), "cozy_warmth avoid_overlay must be non-empty"

    def test_cozy_warmth_summary_overlay_is_non_empty(self) -> None:
        profile = load_voice_profile("cozy_warmth")
        assert profile.summary_overlay.strip(), "cozy_warmth summary_overlay must be non-empty"

    def test_genre_active_craft_overlay_is_non_empty(self) -> None:
        profile = load_voice_profile("genre_active")
        assert profile.craft_overlay.strip(), "genre_active craft_overlay must be non-empty"

    def test_genre_active_avoid_overlay_is_non_empty(self) -> None:
        profile = load_voice_profile("genre_active")
        assert profile.avoid_overlay.strip(), "genre_active avoid_overlay must be non-empty"

    def test_genre_active_summary_overlay_is_non_empty(self) -> None:
        profile = load_voice_profile("genre_active")
        assert profile.summary_overlay.strip(), "genre_active summary_overlay must be non-empty"


# ---------------------------------------------------------------------------
# Exemplar counts
# ---------------------------------------------------------------------------


class TestExemplarCounts:
    def test_all_profiles_have_at_least_two_exemplars(self) -> None:
        for profile_id in BUILT_IN_PROFILE_IDS:
            profile = load_voice_profile(profile_id)
            assert len(profile.exemplars) >= 2, (
                f"{profile_id} has fewer than 2 exemplars"
            )

    def test_literary_restraint_has_four_exemplars(self) -> None:
        """literary_restraint carries the four canonical dark-literary examples."""
        profile = load_voice_profile("literary_restraint")
        assert len(profile.exemplars) == 4


# ---------------------------------------------------------------------------
# Tone and genre keyword presence
# ---------------------------------------------------------------------------


class TestKeywordPresence:
    def test_literary_restraint_tone_keywords_include_dark(self) -> None:
        profile = load_voice_profile("literary_restraint")
        assert "dark" in profile.tone_keywords

    def test_cozy_warmth_tone_keywords_include_cozy(self) -> None:
        profile = load_voice_profile("cozy_warmth")
        assert "cozy" in profile.tone_keywords

    def test_genre_active_tone_keywords_include_action(self) -> None:
        profile = load_voice_profile("genre_active")
        assert "action" in profile.tone_keywords
