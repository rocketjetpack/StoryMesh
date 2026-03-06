"""
Unit tests for storymesh.schemas.genre_normalizer.
"""

import pytest
from pydantic import ValidationError

from storymesh.schemas.genre_normalizer import (
    GenreMapEntry,
    GenreNormalizerAgentInput,
    GenreNormalizerAgentOutput,
    GenreResolution,
    ResolutionMethod,
    ToneMapEntry,
    ToneResolution,
)
from storymesh.versioning.schemas import GENRE_CONSTRAINT_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genre_resolution(**overrides: object) -> GenreResolution:
    defaults = dict(
        input_token="fantasy",
        canonical_genres=["Fantasy"],
        default_tones=["adventurous"],
        method=ResolutionMethod.STATIC_EXACT,
        confidence=1.0,
    )
    return GenreResolution(**(defaults | overrides))


def _tone_resolution(**overrides: object) -> ToneResolution:
    defaults = dict(
        input_token="dark",
        normalized_tones=["dark"],
        method=ResolutionMethod.STATIC_EXACT,
        confidence=1.0,
    )
    return ToneResolution(**(defaults | overrides))


def _output(**overrides: object) -> GenreNormalizerAgentOutput:
    defaults = dict(
        raw_input="dark fantasy",
        normalized_genres=["Fantasy"],
        default_tones=["adventurous"],
        explicit_tones=[],
        effective_tone="adventurous",
        genre_resolutions=[_genre_resolution()],
    )
    return GenreNormalizerAgentOutput(**(defaults | overrides))


# ---------------------------------------------------------------------------
# GenreMapEntry
# ---------------------------------------------------------------------------

class TestGenreMapEntry:
    def test_valid_genres_only(self) -> None:
        entry = GenreMapEntry(genres=["fantasy", "Fantasy"])
        assert entry.genres == ["fantasy", "Fantasy"]
        assert entry.subgenres == []
        assert entry.default_tones == []

    def test_valid_subgenres_only(self) -> None:
        entry = GenreMapEntry(subgenres=["dark fantasy"])
        assert entry.subgenres == ["dark fantasy"]

    def test_valid_both(self) -> None:
        entry = GenreMapEntry(
            genres=["Fantasy"],
            subgenres=["Dark Fantasy"],
            default_tones=["dark"]
        )
        assert entry.default_tones == ["dark"]

    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one genre or subgenre"):
            GenreMapEntry()

    def test_invalid_both_empty_lists(self) -> None:
        with pytest.raises(ValidationError, match="at least one genre or subgenre"):
            GenreMapEntry(genres=[], subgenres=[])


# ---------------------------------------------------------------------------
# ToneMapEntry
# ---------------------------------------------------------------------------

class TestToneMapEntry:
    def test_valid(self) -> None:
        entry = ToneMapEntry(normalized_tones=["dark", "gritty"])
        assert entry.normalized_tones == ["dark", "gritty"]

    def test_invalid_empty_list(self) -> None:
        with pytest.raises(ValidationError):
            ToneMapEntry(normalized_tones=[])

    def test_invalid_missing_field(self) -> None:
        with pytest.raises(ValidationError):
            ToneMapEntry()


# ---------------------------------------------------------------------------
# ResolutionMethod
# ---------------------------------------------------------------------------

class TestResolutionMethod:
    def test_values(self) -> None:
        assert ResolutionMethod.STATIC_EXACT == "static_exact"
        assert ResolutionMethod.STATIC_FUZZY == "static_fuzzy"
        assert ResolutionMethod.LLM_LIVE == "llm_live"
        assert ResolutionMethod.LLM_CACHED == "llm_cached"

    def test_is_str(self) -> None:
        assert isinstance(ResolutionMethod.STATIC_EXACT, str)


# ---------------------------------------------------------------------------
# GenreResolution
# ---------------------------------------------------------------------------

class TestGenreResolution:
    def test_valid(self) -> None:
        gr = _genre_resolution()
        assert gr.input_token == "fantasy"
        assert gr.method == ResolutionMethod.STATIC_EXACT
        assert gr.confidence == 1.0

    def test_frozen(self) -> None:
        gr = _genre_resolution()
        with pytest.raises(ValidationError):
            gr.input_token = "changed"  # type: ignore[misc]

    def test_empty_input_token_raises(self) -> None:
        with pytest.raises(ValidationError):
            _genre_resolution(input_token="")

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _genre_resolution(confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _genre_resolution(confidence=1.1)

    def test_confidence_boundary_values(self) -> None:
        assert _genre_resolution(confidence=0.0).confidence == 0.0
        assert _genre_resolution(confidence=1.0).confidence == 1.0

    def test_empty_canonical_genres_raises_validationerror(self) -> None:
        with pytest.raises(ValidationError):
            _genre_resolution(canonical_genres=[])

    def test_all_resolution_methods(self) -> None:
        for method in ResolutionMethod:
            gr = _genre_resolution(method=method)
            assert gr.method == method


# ---------------------------------------------------------------------------
# ToneResolution
# ---------------------------------------------------------------------------

class TestToneResolution:
    def test_valid(self) -> None:
        tr = _tone_resolution()
        assert tr.input_token == "dark"
        assert tr.normalized_tones == ["dark"]
        assert tr.is_override is True

    def test_frozen(self) -> None:
        tr = _tone_resolution()
        with pytest.raises(ValidationError):
            tr.input_token = "changed"  # type: ignore[misc]

    def test_empty_input_token_raises(self) -> None:
        with pytest.raises(ValidationError):
            _tone_resolution(input_token="")

    def test_empty_normalized_tones_raises(self) -> None:
        with pytest.raises(ValidationError):
            _tone_resolution(normalized_tones=[])

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _tone_resolution(confidence=-0.01)
        with pytest.raises(ValidationError):
            _tone_resolution(confidence=1.01)

    def test_is_override_default(self) -> None:
        tr = _tone_resolution()
        assert tr.is_override is True

    def test_is_override_settable(self) -> None:
        tr = _tone_resolution(is_override=False)
        assert tr.is_override is False


# ---------------------------------------------------------------------------
# GenreNormalizerAgentInput
# ---------------------------------------------------------------------------

class TestGenreNormalizerAgentInput:
    def test_valid(self) -> None:
        inp = GenreNormalizerAgentInput(raw_genre="dark fantasy")
        assert inp.raw_genre == "dark fantasy"
        assert inp.allow_llm_fallback is True

    def test_allow_llm_fallback_override(self) -> None:
        inp = GenreNormalizerAgentInput(raw_genre="thriller", allow_llm_fallback=False)
        assert inp.allow_llm_fallback is False

    def test_empty_raw_genre_raises(self) -> None:
        with pytest.raises(ValidationError):
            GenreNormalizerAgentInput(raw_genre="")


# ---------------------------------------------------------------------------
# GenreNormalizerAgentOutput
# ---------------------------------------------------------------------------

class TestGenreNormalizerAgentOutput:
    def test_valid_no_explicit_tones(self) -> None:
        out = _output()
        assert out.raw_input == "dark fantasy"
        assert out.normalized_genres == ["Fantasy"]
        assert out.effective_tone == "adventurous"
        assert out.schema_version == GENRE_CONSTRAINT_SCHEMA_VERSION

    def test_valid_explicit_tone_matches_default(self) -> None:
        # explicit subset of default => no conflict needed
        out = _output(
            default_tones=["adventurous", "dark"],
            explicit_tones=["dark"],
            effective_tone="dark",
            tone_conflicts=None,
        )
        assert out.effective_tone == "dark"

    def test_valid_with_tone_conflicts(self) -> None:
        out = _output(
            default_tones=["adventurous"],
            explicit_tones=["dark"],
            effective_tone="dark",
            tone_conflicts=["dark vs adventurous"],
        )
        assert out.tone_conflicts == ["dark vs adventurous"]

    def test_frozen(self) -> None:
        out = _output()
        with pytest.raises(ValidationError):
            out.raw_input = "changed"  # type: ignore[misc]

    def test_empty_raw_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            _output(raw_input="")

    def test_empty_normalized_genres_raises(self) -> None:
        with pytest.raises(ValidationError):
            _output(normalized_genres=[])

    def test_empty_genre_resolutions_raises(self) -> None:
        with pytest.raises(ValidationError):
            _output(genre_resolutions=[])

    def test_effective_tone_not_in_any_tone_raises(self) -> None:
        with pytest.raises(ValidationError, match="Effective tone must be derived"):
            _output(
                default_tones=["adventurous"],
                explicit_tones=[],
                effective_tone="unknown_tone",
            )

    def test_effective_tone_from_explicit_only(self) -> None:
        # no default tones; effective_tone comes from explicit
        out = _output(
            default_tones=[],
            explicit_tones=["dark"],
            effective_tone="dark",
        )
        assert out.effective_tone == "dark"

    def test_tone_conflict_required_when_explicit_not_subset_of_default(self) -> None:
        with pytest.raises(ValidationError, match="Tone conflicts must not be None"):
            _output(
                default_tones=["adventurous"],
                explicit_tones=["dark"],
                effective_tone="dark",
                tone_conflicts=None,  # should be required
            )

    def test_no_tone_conflict_when_explicit_subset_of_default(self) -> None:
        # explicit == {"adventurous"} which is a subset of default => no conflict needed
        out = _output(
            default_tones=["adventurous", "dark"],
            explicit_tones=["adventurous"],
            effective_tone="adventurous",
            tone_conflicts=None,
        )
        assert out.tone_conflicts is None

    def test_schema_version(self) -> None:
        out = _output()
        assert out.schema_version == GENRE_CONSTRAINT_SCHEMA_VERSION

    def test_optional_fields_default(self) -> None:
        out = _output()
        assert out.subgenres == []
        assert out.default_tones == ["adventurous"]
        assert out.explicit_tones == []
        assert out.tone_conflicts is None
        assert out.unresolved_tokens == []
        assert out.tone_resolutions == []

# ---------------------------------------------------------------------------
# Round Trip JSON Validation
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_input_roundtrip(self) -> None:
        original = GenreNormalizerAgentInput(
            raw_genre = "dark fantasy romance enemies-to-lovers",
            allow_llm_fallback = False,
        )

        json_str = original.model_dump_json()

        reconstructed = GenreNormalizerAgentInput.model_validate_json(json_str)

        assert reconstructed == original
