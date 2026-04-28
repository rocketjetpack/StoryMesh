"""Unit tests for RubricJudgeAgent."""

from __future__ import annotations

import json
from typing import Any

from storymesh.agents.rubric_judge.agent import (
    DEFAULT_DIMENSION_WEIGHTS,
    RubricJudgeAgent,
)
from storymesh.llm.base import FakeLLMClient
from storymesh.schemas.rubric_judge import (
    EXPECTED_DIMENSIONS,
    RubricJudgeAgentInput,
    RubricJudgeAgentOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(
    responses: list[str] | None = None,
    *,
    pass_threshold: float = 0.7,
    dimension_weights: dict[str, float] | None = None,
) -> RubricJudgeAgent:
    if responses is None:
        responses = [_high_score_response()]
    return RubricJudgeAgent(
        llm_client=FakeLLMClient(responses=responses),
        temperature=0.0,
        pass_threshold=pass_threshold,
        dimension_weights=dimension_weights,
    )


def _make_input(attempt_number: int = 1) -> RubricJudgeAgentInput:
    from storymesh.schemas.proposal_draft import StoryProposal
    from storymesh.schemas.theme_extractor import ThematicTension

    proposal = StoryProposal(
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

    tension = ThematicTension(
        tension_id="T1",
        cluster_a="mystery",
        assumption_a="Truth is discoverable and justice is possible.",
        cluster_b="post_apocalyptic",
        assumption_b="Systems have collapsed and order is impossible.",
        creative_question="What does justice mean without institutions?",
        intensity=0.9,
        cliched_resolutions=["love conquers all", "the hero restores order"],
    )

    return RubricJudgeAgentInput(
        proposal=proposal,
        tensions=[tension],
        user_tones=["dark"],
        user_prompt="dark post-apocalyptic mystery",
        normalized_genres=["mystery", "post_apocalyptic"],
        attempt_number=attempt_number,
    )


def _dim_response(
    score: float,
    feedback: str = "Adequate evaluation of this dimension.",
    ref: str = "restraint",
) -> dict[str, Any]:
    return {"score": score, "feedback": feedback, "principle_ref": ref}


def _full_response(scores: dict[str, float] | None = None) -> dict[str, Any]:
    default_scores = {d: 0.8 for d in EXPECTED_DIMENSIONS}
    if scores:
        default_scores.update(scores)
    dims = {
        name: _dim_response(score, f"Feedback for {name}.", name)
        for name, score in default_scores.items()
    }
    return {
        "dimensions": dims,
        "convention_departures": [],
        "overall_feedback": "Strong proposal with clear specificity and unresolved tension.",
    }


def _high_score_response() -> str:
    return json.dumps(_full_response())


def _low_score_response() -> str:
    return json.dumps(_full_response(scores={d: 0.3 for d in EXPECTED_DIMENSIONS}))


# ---------------------------------------------------------------------------
# Pass / Fail
# ---------------------------------------------------------------------------

class TestPassFail:
    def test_passing_proposal(self) -> None:
        agent = _make_agent([_high_score_response()], pass_threshold=0.7)
        out = agent.run(_make_input())
        assert isinstance(out, RubricJudgeAgentOutput)
        assert out.passed is True

    def test_failing_proposal(self) -> None:
        agent = _make_agent([_low_score_response()], pass_threshold=0.7)
        out = agent.run(_make_input())
        assert out.passed is False

    def test_custom_threshold_lower(self) -> None:
        agent = _make_agent([_low_score_response()], pass_threshold=0.2)
        out = agent.run(_make_input())
        assert out.passed is True

    def test_custom_threshold_higher(self) -> None:
        agent = _make_agent([_high_score_response()], pass_threshold=0.95)
        out = agent.run(_make_input())
        assert out.passed is False


# ---------------------------------------------------------------------------
# Composite score computation
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_composite_weighted_average(self) -> None:
        scores = {d: 1.0 for d in EXPECTED_DIMENSIONS}
        agent = _make_agent([json.dumps(_full_response(scores=scores))])
        out = agent.run(_make_input())
        assert abs(out.composite_score - 1.0) < 0.001

    def test_composite_zero_scores(self) -> None:
        scores = {d: 0.0 for d in EXPECTED_DIMENSIONS}
        agent = _make_agent([json.dumps(_full_response(scores=scores))])
        out = agent.run(_make_input())
        assert out.composite_score == 0.0

    def test_custom_weights_change_composite(self) -> None:
        custom_weights = {d: (1.0 if d == "restraint" else 0.0) for d in EXPECTED_DIMENSIONS}
        scores = {d: (1.0 if d == "restraint" else 0.0) for d in EXPECTED_DIMENSIONS}
        resp = json.dumps(_full_response(scores=scores))
        agent = _make_agent([resp], dimension_weights=custom_weights)
        out = agent.run(_make_input())
        assert abs(out.composite_score - 1.0) < 0.001

    def test_default_weights_sum_to_one(self) -> None:
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_convention_departure_is_highest_weight(self) -> None:
        max_weight = max(DEFAULT_DIMENSION_WEIGHTS.values())
        assert DEFAULT_DIMENSION_WEIGHTS["convention_departure"] == max_weight


# ---------------------------------------------------------------------------
# Missing dimension handling
# ---------------------------------------------------------------------------

class TestMissingDimension:
    def test_missing_dimension_gets_zero_score(self) -> None:
        partial_dims = {
            "restraint": _dim_response(0.8, "Good restraint.", "restraint"),
            # convention_departure and others omitted
        }
        resp = json.dumps({
            "dimensions": partial_dims,
            "convention_departures": [],
            "overall_feedback": "Partial evaluation — some dimensions missing.",
        })
        agent = _make_agent([resp])
        out = agent.run(_make_input())
        assert "convention_departure" in out.dimensions
        assert out.dimensions["convention_departure"].score == 0.0

    def test_all_expected_dimensions_present_in_output(self) -> None:
        agent = _make_agent([_high_score_response()])
        out = agent.run(_make_input())
        assert set(out.dimensions.keys()) >= EXPECTED_DIMENSIONS

    def test_out_of_range_score_clamped(self) -> None:
        dims = {d: _dim_response(2.5, f"Feedback for {d}.", d) for d in EXPECTED_DIMENSIONS}
        resp = json.dumps({
            "dimensions": dims,
            "convention_departures": [],
            "overall_feedback": "Scores out of range — should be clamped.",
        })
        agent = _make_agent([resp])
        out = agent.run(_make_input())
        for dim in out.dimensions.values():
            assert 0.0 <= dim.score <= 1.0


# ---------------------------------------------------------------------------
# LLM failure handling
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_llm_exception_returns_default_fail(self) -> None:
        client = FakeLLMClient(responses=["not json at all }{"])
        agent = RubricJudgeAgent(llm_client=client, pass_threshold=0.7)
        out = agent.run(_make_input())
        assert out.passed is False
        assert out.composite_score == 0.0
        assert all(d.score == 0.0 for d in out.dimensions.values())

    def test_default_fail_has_all_dimensions(self) -> None:
        client = FakeLLMClient(responses=["bad json }{"])
        agent = RubricJudgeAgent(llm_client=client, pass_threshold=0.7)
        out = agent.run(_make_input())
        assert set(out.dimensions.keys()) == EXPECTED_DIMENSIONS

    def test_default_fail_feedback_mentions_failure(self) -> None:
        client = FakeLLMClient(responses=["bad json }{"])
        agent = RubricJudgeAgent(llm_client=client, pass_threshold=0.7)
        out = agent.run(_make_input())
        assert "fail" in out.overall_feedback.lower()


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_temperature_zero_by_default(self) -> None:
        calls: list[float] = []
        original_complete_json = FakeLLMClient.complete_json

        class _TrackingClient(FakeLLMClient):
            def complete_json(self, *args: object, temperature: float = 0.0, **kwargs: object) -> dict[str, object]:
                calls.append(temperature)
                return original_complete_json(self, *args, temperature=temperature, **kwargs)  # type: ignore[arg-type]

        agent = RubricJudgeAgent(
            llm_client=_TrackingClient(responses=[_high_score_response()]),
            temperature=0.0,
        )
        agent.run(_make_input())
        assert calls == [0.0]


# ---------------------------------------------------------------------------
# Debug metadata
# ---------------------------------------------------------------------------

class TestDebugMetadata:
    def test_debug_contains_weights(self) -> None:
        agent = _make_agent()
        out = agent.run(_make_input())
        assert "weights_used" in out.debug
        assert "threshold" in out.debug

    def test_debug_contains_raw_scores(self) -> None:
        agent = _make_agent()
        out = agent.run(_make_input())
        assert "raw_scores" in out.debug
        assert set(out.debug["raw_scores"].keys()) >= EXPECTED_DIMENSIONS

    def test_debug_contains_attempt_number(self) -> None:
        agent = _make_agent()
        out = agent.run(_make_input(attempt_number=2))
        assert out.debug["attempt_number"] == 2
