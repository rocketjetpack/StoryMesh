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
    InferredGenre,
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
    def test_valid_minimal(self) -> None:
        out = _output()
        assert out.raw_input == "dark fantasy"
        assert out.normalized_genres == ["Fantasy"]
        assert out.user_tones == []
        assert out.tone_override is False
        assert out.override_note is None
        assert out.debug == {}
        assert out.schema_version == GENRE_CONSTRAINT_SCHEMA_VERSION

    def test_valid_with_user_tones_no_override(self) -> None:
        out = _output(user_tones=["dark"])
        assert out.user_tones == ["dark"]
        assert out.tone_override is False
        assert out.override_note is None

    def test_valid_with_override(self) -> None:
        out = _output(
            user_tones=["dreamy", "cozy"],
            tone_override=True,
            override_note="User tones (dreamy, cozy) override typical genre defaults (epic, grand)",
        )
        assert out.tone_override is True
        assert out.override_note is not None

    def test_override_without_note_raises(self) -> None:
        with pytest.raises(ValidationError, match="override_note must be provided"):
            _output(tone_override=True, override_note=None)

    def test_override_with_empty_note_raises(self) -> None:
        with pytest.raises(ValidationError, match="override_note must be provided"):
            _output(tone_override=True, override_note="")

    def test_no_override_with_note_is_valid(self) -> None:
        """override_note present but tone_override False is allowed."""
        out = _output(
            tone_override=False,
            override_note="Some note",
        )
        assert out.override_note == "Some note"

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

    def test_schema_version(self) -> None:
        out = _output()
        assert out.schema_version == GENRE_CONSTRAINT_SCHEMA_VERSION

    def test_optional_fields_default(self) -> None:
        out = _output()
        assert out.subgenres == []
        assert out.user_tones == []
        assert out.tone_override is False
        assert out.override_note is None
        assert out.debug == {}

    def test_debug_dict_preserved(self) -> None:
        debug_data = {
            "default_tones": ["epic"],
            "genre_resolutions": [{"input_token": "fantasy"}],
        }
        out = _output(debug=debug_data)
        assert out.debug == debug_data

# ---------------------------------------------------------------------------
# Round Trip JSON Validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ResolutionMethod — LLM_INFERRED (Pass 4)
# ---------------------------------------------------------------------------


class TestResolutionMethodLlmInferred:
    def test_enum_value_exists(self) -> None:
        assert ResolutionMethod.LLM_INFERRED == "llm_inferred"

    def test_usable_in_genre_resolution_method_field(self) -> None:
        """LLM_INFERRED can be assigned to a GenreResolution.method field."""
        gr = GenreResolution(
            input_token="techno",
            canonical_genres=["science_fiction"],
            method=ResolutionMethod.LLM_INFERRED,
            confidence=0.7,
        )
        assert gr.method == ResolutionMethod.LLM_INFERRED


# ---------------------------------------------------------------------------
# InferredGenre (Pass 4)
# ---------------------------------------------------------------------------


def _inferred_genre(**overrides: object) -> InferredGenre:
    defaults: dict[str, object] = {
        "canonical_genre": "science_fiction",
        "rationale": "The prompt describes a programmer in a high-stakes optimization loop.",
    }
    return InferredGenre(**(defaults | overrides))


class TestInferredGenre:
    def test_valid_minimal(self) -> None:
        ig = _inferred_genre()
        assert ig.canonical_genre == "science_fiction"
        assert ig.rationale != ""
        assert ig.method == ResolutionMethod.LLM_INFERRED

    def test_default_confidence(self) -> None:
        ig = _inferred_genre()
        assert ig.confidence == pytest.approx(0.7)

    def test_confidence_override(self) -> None:
        ig = _inferred_genre(confidence=0.85)
        assert ig.confidence == pytest.approx(0.85)

    def test_confidence_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            _inferred_genre(confidence=-0.01)

    def test_confidence_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            _inferred_genre(confidence=1.01)

    def test_subgenres_default_empty(self) -> None:
        ig = _inferred_genre()
        assert ig.subgenres == []

    def test_default_tones_default_empty(self) -> None:
        ig = _inferred_genre()
        assert ig.default_tones == []

    def test_subgenres_populated(self) -> None:
        ig = _inferred_genre(subgenres=["techno_thriller", "workplace_fiction"])
        assert ig.subgenres == ["techno_thriller", "workplace_fiction"]

    def test_empty_canonical_genre_raises(self) -> None:
        with pytest.raises(ValidationError):
            _inferred_genre(canonical_genre="")

    def test_empty_rationale_raises(self) -> None:
        with pytest.raises(ValidationError):
            _inferred_genre(rationale="")

    def test_frozen(self) -> None:
        ig = _inferred_genre()
        with pytest.raises(ValidationError):
            ig.canonical_genre = "fantasy"  # type: ignore[misc]

    def test_method_always_llm_inferred(self) -> None:
        ig = _inferred_genre()
        assert ig.method == ResolutionMethod.LLM_INFERRED


# ---------------------------------------------------------------------------
# GenreNormalizerAgentOutput — inferred_genres field
# ---------------------------------------------------------------------------


class TestGenreNormalizerAgentOutputInferredGenres:
    def test_inferred_genres_defaults_to_empty(self) -> None:
        out = _output()
        assert out.inferred_genres == []

    def test_inferred_genres_accepted(self) -> None:
        ig = _inferred_genre()
        out = _output(inferred_genres=[ig])
        assert len(out.inferred_genres) == 1
        assert out.inferred_genres[0].canonical_genre == "science_fiction"

    def test_inferred_genres_multiple(self) -> None:
        igs = [
            _inferred_genre(canonical_genre="science_fiction"),
            _inferred_genre(
                canonical_genre="literary_fiction",
                rationale="Obsessive single-character focus implies literary fiction.",
            ),
        ]
        out = _output(inferred_genres=igs)
        assert len(out.inferred_genres) == 2

    def test_inferred_genres_round_trip(self) -> None:
        ig = _inferred_genre(subgenres=["techno_thriller"], default_tones=["cerebral"])
        out = _output(inferred_genres=[ig])
        json_str = out.model_dump_json()
        reconstructed = GenreNormalizerAgentOutput.model_validate_json(json_str)
        assert reconstructed.inferred_genres[0].subgenres == ["techno_thriller"]


class TestRoundTrip:
    def test_input_roundtrip(self) -> None:
        original = GenreNormalizerAgentInput(
            raw_genre = "dark fantasy romance enemies-to-lovers",
            allow_llm_fallback = False,
        )

        json_str = original.model_dump_json()

        reconstructed = GenreNormalizerAgentInput.model_validate_json(json_str)

        assert reconstructed == original

    def test_output_roundtrip(self) -> None:
        original = _output(
            user_tones=["dreamy", "cozy"],
            tone_override=True,
            override_note="User tones (dreamy, cozy) override typical genre defaults (epic)",
            debug={"default_tones": ["epic"]},
        )

        json_str = original.model_dump_json()

        reconstructed = GenreNormalizerAgentOutput.model_validate_json(json_str)

        assert reconstructed == original