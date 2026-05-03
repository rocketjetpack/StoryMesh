"""Unit tests for ProposalReaderAgent."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from storymesh.agents.proposal_reader.agent import ProposalReaderAgent
from storymesh.llm.base import FakeLLMClient
from storymesh.schemas.proposal_draft import StoryProposal
from storymesh.schemas.proposal_reader import (
    ProposalReaderAgentInput,
    ProposalReaderAgentOutput,
    ProposalReaderFeedback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(responses: list[str] | None = None) -> ProposalReaderAgent:
    if responses is None:
        responses = [_valid_feedback_response()]
    return ProposalReaderAgent(
        llm_client=FakeLLMClient(responses=responses),
        temperature=0.4,
        max_tokens=1024,
    )


def _make_proposal() -> StoryProposal:
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
        image_prompt=(
            "A rain-slicked street in a flooded cityscape at dusk, a lone figure "
            "silhouetted against pale ruins of a collapsed civic tower. "
            "Gritty noir ink wash style, muted greys and a single amber light source."
        ),
    )


def _make_input() -> ProposalReaderAgentInput:
    return ProposalReaderAgentInput(
        proposal=_make_proposal(),
        user_prompt="A mystery set after the collapse of civilization.",
        normalized_genres=["mystery", "post_apocalyptic"],
        user_tones=["dark", "cerebral"],
    )


def _valid_feedback_response() -> str:
    return json.dumps({
        "what_engaged_me": (
            "The image of rebuilding investigation infrastructure from salvage — "
            "a portable record system, cultivating informants — is specific and "
            "makes the world feel real in a way that pure destruction narratives rarely do."
        ),
        "what_fell_flat": (
            "The tribunal scene feels inevitable from the setup. I know it's coming "
            "before I get there. The surprise should be something else."
        ),
        "protagonist_gap": (
            "Mara's faith in due process is named but I don't feel it. What does "
            "she actually do when the process fails? What is the cost of keeping that faith?"
        ),
        "premise_question": (
            "Who arranged the body and why does the symbolism matter? "
            "The proposal gestures at communication-through-murder but doesn't "
            "give me enough to know if this is interesting or just dark window dressing."
        ),
        "reader_direction": (
            "Give me one specific detail about Mara — something she does or keeps "
            "— that makes her faith in process feel like a character trait rather than a label."
        ),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_returns_valid_output() -> None:
    """Agent returns a well-formed ProposalReaderAgentOutput on valid LLM response."""
    agent = _make_agent()
    output = agent.run(_make_input())

    assert isinstance(output, ProposalReaderAgentOutput)
    assert isinstance(output.feedback, ProposalReaderFeedback)
    assert len(output.feedback.what_engaged_me) >= 10
    assert len(output.feedback.what_fell_flat) >= 10
    assert len(output.feedback.protagonist_gap) >= 10
    assert len(output.feedback.premise_question) >= 10
    assert len(output.feedback.reader_direction) >= 10


def test_output_is_frozen() -> None:
    """ProposalReaderAgentOutput and its nested feedback are frozen (immutable)."""
    agent = _make_agent()
    output = agent.run(_make_input())

    with pytest.raises(ValidationError):
        output.feedback = output.feedback  # type: ignore[misc]


def test_schema_version_present() -> None:
    """Output carries a non-empty schema_version string."""
    agent = _make_agent()
    output = agent.run(_make_input())

    assert isinstance(output.schema_version, str)
    assert output.schema_version


def test_debug_contains_temperature() -> None:
    """Debug dict includes the temperature used for the call."""
    agent = _make_agent()
    output = agent.run(_make_input())

    assert "temperature" in output.debug
    assert output.debug["temperature"] == 0.4


def test_llm_called_exactly_once() -> None:
    """Agent makes exactly one LLM call per run."""
    fake_llm = FakeLLMClient(responses=[_valid_feedback_response()])
    agent = ProposalReaderAgent(
        llm_client=fake_llm,
        temperature=0.4,
        max_tokens=1024,
    )
    agent.run(_make_input())

    assert fake_llm.call_count == 1


def test_parse_failure_raises_runtime_error() -> None:
    """Agent wraps LLM parse failures in RuntimeError."""
    agent = _make_agent(responses=["not valid json {{{"])

    with pytest.raises(RuntimeError, match="ProposalReaderAgent evaluation failed"):
        agent.run(_make_input())


def test_missing_field_raises_runtime_error() -> None:
    """Agent raises RuntimeError when a required feedback field is absent."""
    incomplete = json.dumps({
        "what_engaged_me": "Something interesting.",
        # what_fell_flat, protagonist_gap, premise_question, reader_direction missing
    })
    agent = _make_agent(responses=[incomplete])

    with pytest.raises(RuntimeError, match="ProposalReaderAgent evaluation failed"):
        agent.run(_make_input())


def test_user_tones_optional() -> None:
    """Agent runs without error when user_tones is empty."""
    input_data = ProposalReaderAgentInput(
        proposal=_make_proposal(),
        user_prompt="A mystery set after the collapse of civilization.",
        normalized_genres=["mystery"],
        user_tones=[],
    )
    agent = _make_agent()
    output = agent.run(input_data)

    assert isinstance(output, ProposalReaderAgentOutput)


def test_feedback_fields_are_strings() -> None:
    """All five feedback fields are non-empty strings."""
    agent = _make_agent()
    output = agent.run(_make_input())
    fb = output.feedback

    for field_name in (
        "what_engaged_me",
        "what_fell_flat",
        "protagonist_gap",
        "premise_question",
        "reader_direction",
    ):
        value = getattr(fb, field_name)
        assert isinstance(value, str), f"{field_name} should be a str"
        assert value.strip(), f"{field_name} should not be empty"
