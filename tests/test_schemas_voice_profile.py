"""Unit tests for VoiceProfile schema (src/storymesh/schemas/voice_profile.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.voice_profile import VoiceProfile
from storymesh.versioning.schemas import VOICE_PROFILE_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_profile(**overrides: object) -> VoiceProfile:
    """Return a minimal valid VoiceProfile, with optional field overrides."""
    defaults: dict[str, object] = {
        "id": "test_profile",
        "description": "A test profile for unit testing purposes only.",
        "tone_keywords": ["dark", "cerebral"],
        "genre_keywords": ["mystery"],
        "craft_overlay": "",
        "avoid_overlay": "",
        "exemplars": [
            "The door was already open when she arrived.",
            "He counted the empty chairs before sitting down.",
        ],
        "summary_overlay": "",
    }
    defaults.update(overrides)
    return VoiceProfile(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction and schema version
# ---------------------------------------------------------------------------


class TestVoiceProfileConstruction:
    def test_valid_construction(self) -> None:
        profile = _valid_profile()
        assert profile.id == "test_profile"
        assert profile.schema_version == VOICE_PROFILE_SCHEMA_VERSION

    def test_schema_version_is_constant(self) -> None:
        profile = _valid_profile()
        assert profile.schema_version == "1.0"

    def test_minimal_fields(self) -> None:
        """Only required fields — optional fields use defaults."""
        profile = VoiceProfile(
            id="minimal_profile",
            description="Minimal profile for testing.",
            tone_keywords=["quiet"],
            exemplars=[
                "She sat by the window without opening it.",
                "The letter arrived on a Tuesday, which felt right.",
            ],
        )
        assert profile.craft_overlay == ""
        assert profile.avoid_overlay == ""
        assert profile.summary_overlay == ""
        assert profile.genre_keywords == []


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestVoiceProfileFrozen:
    def test_frozen(self) -> None:
        profile = _valid_profile()
        with pytest.raises(ValidationError):
            profile.id = "mutated"  # type: ignore[misc]

    def test_frozen_nested_fields(self) -> None:
        profile = _valid_profile()
        with pytest.raises(ValidationError):
            profile.craft_overlay = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# id validation
# ---------------------------------------------------------------------------


class TestVoiceProfileIdPattern:
    def test_valid_snake_case(self) -> None:
        profile = _valid_profile(id="cozy_warmth")
        assert profile.id == "cozy_warmth"

    def test_valid_single_word(self) -> None:
        profile = _valid_profile(id="literary")
        assert profile.id == "literary"

    def test_valid_with_numbers(self) -> None:
        profile = _valid_profile(id="profile_v2")
        assert profile.id == "profile_v2"

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase snake_case"):
            _valid_profile(id="Literary_Restraint")

    def test_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase snake_case"):
            _valid_profile(id="cozy-warmth")

    def test_leading_underscore_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase snake_case"):
            _valid_profile(id="_leading")

    def test_leading_digit_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase snake_case"):
            _valid_profile(id="1profile")

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_profile(id="")

    def test_space_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase snake_case"):
            _valid_profile(id="cozy warmth")


# ---------------------------------------------------------------------------
# exemplars min_length
# ---------------------------------------------------------------------------


class TestVoiceProfileExemplars:
    def test_two_exemplars_accepted(self) -> None:
        profile = _valid_profile(exemplars=["First sentence.", "Second sentence."])
        assert len(profile.exemplars) == 2

    def test_four_exemplars_accepted(self) -> None:
        profile = _valid_profile(
            exemplars=["A.", "B.", "C.", "D."]
        )
        assert len(profile.exemplars) == 4

    def test_one_exemplar_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_profile(exemplars=["Only one sentence here."])

    def test_empty_exemplars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_profile(exemplars=[])


# ---------------------------------------------------------------------------
# tone_keywords min_length
# ---------------------------------------------------------------------------


class TestVoiceProfileToneKeywords:
    def test_one_tone_keyword_accepted(self) -> None:
        profile = _valid_profile(tone_keywords=["dark"])
        assert profile.tone_keywords == ["dark"]

    def test_empty_tone_keywords_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_profile(tone_keywords=[])


# ---------------------------------------------------------------------------
# description min_length
# ---------------------------------------------------------------------------


class TestVoiceProfileDescription:
    def test_short_description_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_profile(description="Short.")

    def test_long_description_accepted(self) -> None:
        profile = _valid_profile(description="A sufficiently long description for the profile.")
        assert len(profile.description) >= 10
