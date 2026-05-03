"""Unit tests for RubricJudgeAgent."""

from __future__ import annotations

import json
from typing import Any

from storymesh.agents.rubric_judge.agent import (
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
    pass_threshold: int = 6,
) -> RubricJudgeAgent:
    if responses is None:
        responses = [_high_score_response()]
    return RubricJudgeAgent(
        llm_client=FakeLLMClient(responses=responses),
        temperature=0.0,
        pass_threshold=pass_threshold,
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
        image_prompt=(
            "A rain-slicked street in a flooded cityscape at dusk, a lone figure "
            "silhouetted against pale ruins of a collapsed civic tower. "
            "Gritty noir ink wash style, muted greys and a single amber light source."
        ),
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
    score: int,
    feedback: str = "Adequate evaluation of this dimension.",
    ref: str = "restraint",
) -> dict[str, Any]:
    return {"score": score, "feedback": feedback, "principle_ref": ref}


def _full_response(scores: dict[str, int] | None = None) -> dict[str, Any]:
    default_scores = {d: 2 for d in EXPECTED_DIMENSIONS}
    if scores:
        default_scores.update(scores)
    dims = {
        name: _dim_response(score, f"Feedback for {name}.", name)
        for name, score in default_scores.items()
    }
    return {
        "dimensions": dims,
        "overall_feedback": "Strong proposal with clear specificity and unresolved tension.",
    }


def _high_score_response() -> str:
    return json.dumps(_full_response())


def _low_score_response() -> str:
    return json.dumps(_full_response(scores={d: 0 for d in EXPECTED_DIMENSIONS}))


# ---------------------------------------------------------------------------
# Pass / Fail
# ---------------------------------------------------------------------------

class TestPassFail:
    def test_passing_proposal(self) -> None:
        agent = _make_agent([_high_score_response()], pass_threshold=6)
        out = agent.run(_make_input())
        assert isinstance(out, RubricJudgeAgentOutput)
        assert out.passed is True

    def test_failing_proposal(self) -> None:
        agent = _make_agent([_low_score_response()], pass_threshold=6)
        out = agent.run(_make_input())
        assert out.passed is False

    def test_custom_threshold_lower(self) -> None:
        agent = _make_agent([_low_score_response()], pass_threshold=0)
        out = agent.run(_make_input())
        assert out.passed is True

    def test_custom_threshold_higher(self) -> None:
        agent = _make_agent([_high_score_response()], pass_threshold=10)
        out = agent.run(_make_input())
        assert out.passed is True  # All scores are 2, sum=10 which equals threshold

    def test_threshold_above_max_fails(self) -> None:
        """A threshold of 10 with acceptable (1) scores should fail."""
        scores = {d: 1 for d in EXPECTED_DIMENSIONS}
        resp = json.dumps(_full_response(scores=scores))
        agent = _make_agent([resp], pass_threshold=10)
        out = agent.run(_make_input())
        assert out.passed is False


# ---------------------------------------------------------------------------
# Composite score computation
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_composite_sum_all_strong(self) -> None:
        scores = {d: 2 for d in EXPECTED_DIMENSIONS}
        agent = _make_agent([json.dumps(_full_response(scores=scores))])
        out = agent.run(_make_input())
        assert out.composite_score == 10

    def test_composite_zero_scores(self) -> None:
        scores = {d: 0 for d in EXPECTED_DIMENSIONS}
        agent = _make_agent([json.dumps(_full_response(scores=scores))])
        out = agent.run(_make_input())
        assert out.composite_score == 0

    def test_composite_mixed_scores(self) -> None:
        scores = {"restraint": 2, "story_serving_choices": 1, "specificity": 0,
                  "protagonist_interiority": 2, "user_intent_fidelity": 1}
        resp = json.dumps(_full_response(scores=scores))
        agent = _make_agent([resp])
        out = agent.run(_make_input())
        assert out.composite_score == 6

    def test_composite_all_acceptable(self) -> None:
        scores = {d: 1 for d in EXPECTED_DIMENSIONS}
        agent = _make_agent([json.dumps(_full_response(scores=scores))])
        out = agent.run(_make_input())
        assert out.composite_score == 5


# ---------------------------------------------------------------------------
# Missing dimension handling
# ---------------------------------------------------------------------------

class TestMissingDimension:
    def test_missing_dimension_gets_zero_score(self) -> None:
        partial_dims = {
            "restraint": _dim_response(2, "Good restraint.", "restraint"),
            # story_serving_choices and others omitted
        }
        resp = json.dumps({
            "dimensions": partial_dims,
            "overall_feedback": "Partial evaluation — some dimensions missing.",
        })
        agent = _make_agent([resp])
        out = agent.run(_make_input())
        assert "story_serving_choices" in out.dimensions
        assert out.dimensions["story_serving_choices"].score == 0

    def test_all_expected_dimensions_present_in_output(self) -> None:
        agent = _make_agent([_high_score_response()])
        out = agent.run(_make_input())
        assert set(out.dimensions.keys()) >= EXPECTED_DIMENSIONS

    def test_out_of_range_score_clamped(self) -> None:
        dims = {d: _dim_response(5, f"Feedback for {d}.", d) for d in EXPECTED_DIMENSIONS}
        resp = json.dumps({
            "dimensions": dims,
            "overall_feedback": "Scores out of range — should be clamped.",
        })
        agent = _make_agent([resp])
        out = agent.run(_make_input())
        for dim in out.dimensions.values():
            assert 0 <= dim.score <= 2


# ---------------------------------------------------------------------------
# LLM failure handling
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_llm_exception_returns_default_fail(self) -> None:
        client = FakeLLMClient(responses=["not json at all }{"])
        agent = RubricJudgeAgent(llm_client=client, pass_threshold=6)
        out = agent.run(_make_input())
        assert out.passed is False
        assert out.composite_score == 0
        assert all(d.score == 0 for d in out.dimensions.values())

    def test_default_fail_has_all_dimensions(self) -> None:
        client = FakeLLMClient(responses=["bad json }{"])
        agent = RubricJudgeAgent(llm_client=client, pass_threshold=6)
        out = agent.run(_make_input())
        assert set(out.dimensions.keys()) == EXPECTED_DIMENSIONS

    def test_default_fail_feedback_mentions_failure(self) -> None:
        client = FakeLLMClient(responses=["bad json }{"])
        agent = RubricJudgeAgent(llm_client=client, pass_threshold=6)
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
    def test_debug_contains_threshold(self) -> None:
        agent = _make_agent()
        out = agent.run(_make_input())
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
