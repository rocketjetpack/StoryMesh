"""Unit tests for storymesh.schemas.theme_extractor."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.book_ranker import RankedBookSummary
from storymesh.schemas.theme_extractor import (
    GenreCluster,
    NarrativeSeed,
    ThematicTension,
    ThemeExtractorAgentInput,
    ThemeExtractorAgentOutput,
)
from storymesh.versioning.schemas import THEMEPACK_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summary(rank: int = 1) -> RankedBookSummary:
    return RankedBookSummary(
        work_key=f"/works/OL{rank}W",
        title=f"Book {rank}",
        authors=["Author"],
        source_genres=["mystery"],
        composite_score=0.9,
        rank=rank,
    )


def _cluster(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "genre": "mystery",
        "books": ["The Big Sleep"],
        "thematic_assumptions": ["Truth is discoverable through investigation"],
    }
    return {**defaults, **overrides}


def _tension(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "tension_id": "T1",
        "cluster_a": "mystery",
        "assumption_a": "Truth is discoverable",
        "cluster_b": "post_apocalyptic",
        "assumption_b": "Institutional knowledge has collapsed",
        "creative_question": "What does investigation look like without records?",
        "intensity": 0.8,
        "cliched_resolutions": [
            "A lone detective rebuilds justice single-handedly through sheer determination",
        ],
    }
    return {**defaults, **overrides}


def _seed(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "seed_id": "S1",
        "concept": "A former detective must reinvent investigation from first principles.",
        "tensions_used": ["T1"],
    }
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# TestGenreCluster
# ---------------------------------------------------------------------------


class TestGenreCluster:
    def test_valid_construction(self) -> None:
        cluster = GenreCluster(**_cluster())
        assert cluster.genre == "mystery"
        assert cluster.books == ["The Big Sleep"]
        assert len(cluster.thematic_assumptions) == 1

    def test_frozen(self) -> None:
        cluster = GenreCluster(**_cluster())
        with pytest.raises(ValidationError):
            cluster.genre = "fantasy"  # type: ignore[misc]

    def test_empty_books_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GenreCluster(**_cluster(books=[]))

    def test_empty_assumptions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GenreCluster(**_cluster(thematic_assumptions=[]))

    def test_dominant_tropes_defaults_to_empty(self) -> None:
        cluster = GenreCluster(**_cluster())
        assert cluster.dominant_tropes == []

    def test_dominant_tropes_accepted(self) -> None:
        cluster = GenreCluster(**_cluster(dominant_tropes=["red herring", "unreliable narrator"]))
        assert len(cluster.dominant_tropes) == 2


# ---------------------------------------------------------------------------
# TestThematicTension
# ---------------------------------------------------------------------------


class TestThematicTension:
    def test_valid_construction(self) -> None:
        tension = ThematicTension(**_tension())
        assert tension.tension_id == "T1"
        assert tension.intensity == pytest.approx(0.8)

    def test_intensity_bounds_lower(self) -> None:
        with pytest.raises(ValidationError):
            ThematicTension(**_tension(intensity=-0.1))

    def test_intensity_bounds_upper(self) -> None:
        with pytest.raises(ValidationError):
            ThematicTension(**_tension(intensity=1.1))

    def test_intensity_boundary_zero(self) -> None:
        tension = ThematicTension(**_tension(intensity=0.0))
        assert tension.intensity == pytest.approx(0.0)

    def test_intensity_boundary_one(self) -> None:
        tension = ThematicTension(**_tension(intensity=1.0))
        assert tension.intensity == pytest.approx(1.0)

    def test_frozen(self) -> None:
        tension = ThematicTension(**_tension())
        with pytest.raises(ValidationError):
            tension.tension_id = "T2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestThematicTensionClichedResolutions
# ---------------------------------------------------------------------------


class TestThematicTensionClichedResolutions:
    def test_valid_with_single_cliche(self) -> None:
        tension = ThematicTension(**_tension(cliched_resolutions=["Hero saves the day via determination"]))
        assert len(tension.cliched_resolutions) == 1

    def test_valid_with_multiple_cliches(self) -> None:
        cliches = [
            "A lone detective rebuilds justice single-handedly",
            "A hidden bunker contains all the missing records",
            "The conspiracy is exposed and order is restored overnight",
        ]
        tension = ThematicTension(**_tension(cliched_resolutions=cliches))
        assert len(tension.cliched_resolutions) == 3

    def test_empty_cliched_resolutions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThematicTension(**_tension(cliched_resolutions=[]))

    def test_cliched_resolutions_frozen(self) -> None:
        tension = ThematicTension(**_tension())
        with pytest.raises(ValidationError):
            tension.cliched_resolutions = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestNarrativeSeed
# ---------------------------------------------------------------------------


class TestNarrativeSeed:
    def test_valid_construction(self) -> None:
        seed = NarrativeSeed(**_seed())
        assert seed.seed_id == "S1"
        assert seed.tensions_used == ["T1"]

    def test_concept_min_length(self) -> None:
        with pytest.raises(ValidationError):
            NarrativeSeed(**_seed(concept="short"))

    def test_empty_tensions_used_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NarrativeSeed(**_seed(tensions_used=[]))

    def test_frozen(self) -> None:
        seed = NarrativeSeed(**_seed())
        with pytest.raises(ValidationError):
            seed.seed_id = "S2"  # type: ignore[misc]

    def test_tonal_direction_defaults_to_empty(self) -> None:
        seed = NarrativeSeed(**_seed())
        assert seed.tonal_direction == []

    def test_narrative_context_used_defaults_to_empty(self) -> None:
        seed = NarrativeSeed(**_seed())
        assert seed.narrative_context_used == []


# ---------------------------------------------------------------------------
# TestThemeExtractorAgentInput
# ---------------------------------------------------------------------------


class TestThemeExtractorAgentInput:
    def test_valid_construction(self) -> None:
        inp = ThemeExtractorAgentInput(
            ranked_summaries=[_summary()],
            normalized_genres=["mystery"],
            user_prompt="dark mystery",
        )
        assert inp.user_prompt == "dark mystery"

    def test_empty_ranked_summaries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThemeExtractorAgentInput(
                ranked_summaries=[],
                normalized_genres=["mystery"],
                user_prompt="dark mystery",
            )

    def test_empty_genres_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThemeExtractorAgentInput(
                ranked_summaries=[_summary()],
                normalized_genres=[],
                user_prompt="dark mystery",
            )

    def test_defaults_for_optional_fields(self) -> None:
        inp = ThemeExtractorAgentInput(
            ranked_summaries=[_summary()],
            normalized_genres=["mystery"],
            user_prompt="dark mystery",
        )
        assert inp.subgenres == []
        assert inp.user_tones == []
        assert inp.narrative_context == []
        assert inp.tone_override is False


# ---------------------------------------------------------------------------
# TestThemeExtractorAgentOutput
# ---------------------------------------------------------------------------


def _valid_output() -> ThemeExtractorAgentOutput:
    return ThemeExtractorAgentOutput(
        genre_clusters=[GenreCluster(**_cluster())],
        tensions=[ThematicTension(**_tension())],
        narrative_seeds=[NarrativeSeed(**_seed())],
    )


class TestThemeExtractorAgentOutput:
    def test_valid_construction(self) -> None:
        output = _valid_output()
        assert len(output.genre_clusters) == 1
        assert len(output.tensions) == 1
        assert len(output.narrative_seeds) == 1

    def test_frozen(self) -> None:
        output = _valid_output()
        with pytest.raises(ValidationError):
            output.genre_clusters = []  # type: ignore[misc]

    def test_schema_version_matches(self) -> None:
        output = _valid_output()
        assert output.schema_version == THEMEPACK_SCHEMA_VERSION

    def test_empty_clusters_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThemeExtractorAgentOutput(
                genre_clusters=[],
                tensions=[ThematicTension(**_tension())],
                narrative_seeds=[NarrativeSeed(**_seed())],
            )

    def test_empty_tensions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThemeExtractorAgentOutput(
                genre_clusters=[GenreCluster(**_cluster())],
                tensions=[],
                narrative_seeds=[NarrativeSeed(**_seed())],
            )

    def test_empty_seeds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThemeExtractorAgentOutput(
                genre_clusters=[GenreCluster(**_cluster())],
                tensions=[ThematicTension(**_tension())],
                narrative_seeds=[],
            )

    def test_user_tones_carried_defaults_to_empty(self) -> None:
        output = _valid_output()
        assert output.user_tones_carried == []

    def test_debug_defaults_to_empty_dict(self) -> None:
        output = _valid_output()
        assert output.debug == {}
