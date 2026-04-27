"""Unit tests for storymesh.schemas.rubric_judge."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storymesh.schemas.rubric_judge import (
    EXPECTED_DIMENSIONS,
    DimensionResult,
    RubricJudgeAgentInput,
    RubricJudgeAgentOutput,
)
from storymesh.versioning.schemas import RUBRIC_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_dimension(**overrides: object) -> DimensionResult:
    defaults: dict[str, object] = {
        "score": 0.8,
        "feedback": "Good restraint across the full arc.",
        "principle_ref": "restraint",
    }
    defaults.update(overrides)
    return DimensionResult(**defaults)  # type: ignore[arg-type]


def _valid_dimensions() -> dict[str, DimensionResult]:
    return {
        "restraint": _valid_dimension(score=0.8, principle_ref="restraint"),
        "convention_departure": _valid_dimension(score=0.7, principle_ref="convention_departure"),
        "specificity": _valid_dimension(score=0.75, principle_ref="specificity"),
        "protagonist_interiority": _valid_dimension(score=0.6, principle_ref="protagonist_interiority"),
        "user_intent_fidelity": _valid_dimension(score=0.9, principle_ref="user_intent_fidelity"),
    }


def _valid_output(**overrides: object) -> RubricJudgeAgentOutput:
    defaults: dict[str, object] = {
        "passed": True,
        "composite_score": 0.75,
        "pass_threshold": 0.7,
        "dimensions": _valid_dimensions(),
        "convention_departures": [],
        "overall_feedback": "Strong proposal with specific details and unresolved tension.",
        "debug": {},
    }
    defaults.update(overrides)
    return RubricJudgeAgentOutput(**defaults)  # type: ignore[arg-type]


def _make_proposal() -> object:
    """Build a minimal StoryProposal for RubricJudgeAgentInput tests."""
    from storymesh.schemas.proposal_draft import StoryProposal
    return StoryProposal(
        seed_id="S1",
        title="The Last Inquest",
        protagonist=(
            "Mara Voss — former census-taker whose faith in due process "
            "survived the collapse even as the process itself did not."
        ),
        setting="A flooded city-state where municipal records were lost in year one.",
        plot_arc=(
            "Act 1: Mara finds a body arranged with deliberate symbolism. "
            "Act 2: She rebuilds investigation infrastructure from salvage. "
            "Act 3: A community tribunal convicts but enforcement is hers alone."
        ),
        thematic_thesis="Justice requires witnesses, not institutions.",
        key_scenes=[
            "Mara finds the arranged body at dawn, the smell of copper and mildew.",
            "The tribunal room, candlelit, thirty people deciding without authority.",
        ],
        tensions_addressed=["T1"],
        tone=["dark", "cerebral"],
        genre_blend=["mystery", "post_apocalyptic"],
    )


def _make_tension() -> object:
    """Build a minimal ThematicTension."""
    from storymesh.schemas.theme_extractor import ThematicTension
    return ThematicTension(
        tension_id="T1",
        cluster_a="mystery",
        assumption_a="Truth is discoverable and justice is possible.",
        cluster_b="post_apocalyptic",
        assumption_b="Systems have collapsed and order is impossible.",
        creative_question="What does justice mean without institutions?",
        intensity=0.9,
        cliched_resolutions=["love conquers all", "the hero restores order"],
    )


# ---------------------------------------------------------------------------
# DimensionResult
# ---------------------------------------------------------------------------

class TestDimensionResult:
    def test_valid_construction(self) -> None:
        dim = _valid_dimension()
        assert dim.score == 0.8
        assert dim.principle_ref == "restraint"

    def test_frozen(self) -> None:
        dim = _valid_dimension()
        with pytest.raises((TypeError, ValidationError)):
            dim.score = 0.5  # type: ignore[misc]

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_dimension(score=-0.1)

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_dimension(score=1.1)

    def test_score_at_boundaries_accepted(self) -> None:
        assert _valid_dimension(score=0.0).score == 0.0
        assert _valid_dimension(score=1.0).score == 1.0

    def test_short_feedback_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_dimension(feedback="short")

    def test_empty_principle_ref_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_dimension(principle_ref="")


# ---------------------------------------------------------------------------
# RubricJudgeAgentOutput
# ---------------------------------------------------------------------------

class TestRubricJudgeAgentOutput:
    def test_valid_construction(self) -> None:
        out = _valid_output()
        assert out.passed is True
        assert 0.0 <= out.composite_score <= 1.0

    def test_frozen(self) -> None:
        out = _valid_output()
        with pytest.raises((TypeError, ValidationError)):
            out.passed = False  # type: ignore[misc]

    def test_schema_version_matches_constant(self) -> None:
        out = _valid_output()
        assert out.schema_version == RUBRIC_SCHEMA_VERSION

    def test_convention_departures_defaults_empty(self) -> None:
        out = RubricJudgeAgentOutput(
            passed=True,
            composite_score=0.8,
            pass_threshold=0.7,
            dimensions=_valid_dimensions(),
            overall_feedback="Strong proposal with clear specificity and tension.",
        )
        assert out.convention_departures == []

    def test_dimensions_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            _valid_output(dimensions={})

    def test_composite_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _valid_output(composite_score=-0.1)
        with pytest.raises(ValidationError):
            _valid_output(composite_score=1.1)

    def test_pass_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _valid_output(pass_threshold=-0.1)
        with pytest.raises(ValidationError):
            _valid_output(pass_threshold=1.1)

    def test_overall_feedback_min_length(self) -> None:
        with pytest.raises(ValidationError):
            _valid_output(overall_feedback="Too short")

    def test_passed_true_when_score_above_threshold(self) -> None:
        out = _valid_output(passed=True, composite_score=0.75, pass_threshold=0.7)
        assert out.passed is True

    def test_passed_false_when_score_below_threshold(self) -> None:
        out = _valid_output(passed=False, composite_score=0.5, pass_threshold=0.7)
        assert out.passed is False

    def test_convention_departures_list(self) -> None:
        out = _valid_output(convention_departures=[
            "Convention followed: lone detective solves case",
            "Departure: tribunal convicts without legal authority",
        ])
        assert len(out.convention_departures) == 2

    def test_debug_defaults_empty(self) -> None:
        out = _valid_output()
        assert isinstance(out.debug, dict)


# ---------------------------------------------------------------------------
# RubricJudgeAgentInput
# ---------------------------------------------------------------------------

class TestRubricJudgeAgentInput:
    def test_valid_construction(self) -> None:
        inp = RubricJudgeAgentInput(
            proposal=_make_proposal(),  # type: ignore[arg-type]
            tensions=[_make_tension()],  # type: ignore[list-item]
            user_tones=["dark"],
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery", "post_apocalyptic"],
        )
        assert inp.attempt_number == 1

    def test_frozen(self) -> None:
        inp = RubricJudgeAgentInput(
            proposal=_make_proposal(),  # type: ignore[arg-type]
            tensions=[_make_tension()],  # type: ignore[list-item]
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery"],
        )
        with pytest.raises((TypeError, ValidationError)):
            inp.attempt_number = 2  # type: ignore[misc]

    def test_attempt_number_ge_one(self) -> None:
        with pytest.raises(ValidationError):
            RubricJudgeAgentInput(
                proposal=_make_proposal(),  # type: ignore[arg-type]
                tensions=[_make_tension()],  # type: ignore[list-item]
                user_prompt="dark post-apocalyptic mystery",
                normalized_genres=["mystery"],
                attempt_number=0,
            )

    def test_tensions_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            RubricJudgeAgentInput(
                proposal=_make_proposal(),  # type: ignore[arg-type]
                tensions=[],
                user_prompt="dark post-apocalyptic mystery",
                normalized_genres=["mystery"],
            )

    def test_user_tones_defaults_empty(self) -> None:
        inp = RubricJudgeAgentInput(
            proposal=_make_proposal(),  # type: ignore[arg-type]
            tensions=[_make_tension()],  # type: ignore[list-item]
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery"],
        )
        assert inp.user_tones == []

    def test_cliched_resolutions_defaults_empty(self) -> None:
        inp = RubricJudgeAgentInput(
            proposal=_make_proposal(),  # type: ignore[arg-type]
            tensions=[_make_tension()],  # type: ignore[list-item]
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery"],
        )
        assert inp.cliched_resolutions == {}

    def test_cliched_resolutions_accepts_dict(self) -> None:
        inp = RubricJudgeAgentInput(
            proposal=_make_proposal(),  # type: ignore[arg-type]
            tensions=[_make_tension()],  # type: ignore[list-item]
            user_prompt="dark post-apocalyptic mystery",
            normalized_genres=["mystery"],
            cliched_resolutions={"T1": ["love conquers all", "hero restores order"]},
        )
        assert inp.cliched_resolutions["T1"] == ["love conquers all", "hero restores order"]


# ---------------------------------------------------------------------------
# EXPECTED_DIMENSIONS constant
# ---------------------------------------------------------------------------

class TestExpectedDimensions:
    def test_all_five_dimensions_present(self) -> None:
        assert EXPECTED_DIMENSIONS == {
            "restraint",
            "convention_departure",
            "specificity",
            "protagonist_interiority",
            "user_intent_fidelity",
        }
