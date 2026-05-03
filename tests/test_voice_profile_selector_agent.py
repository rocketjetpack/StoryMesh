"""Unit tests for VoiceProfileSelectorAgent (Stage 0.5)."""

from __future__ import annotations

import json

import pytest

from storymesh.agents.voice_profile_selector.agent import VoiceProfileSelectorAgent
from storymesh.llm.base import FakeLLMClient
from storymesh.schemas.voice_profile import VoiceProfile
from storymesh.schemas.voice_profile_selector import (
    VoiceProfileSelectorAgentInput,
    VoiceProfileSelectorAgentOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(responses: list[str] | None = None) -> VoiceProfileSelectorAgent:
    if responses is None:
        responses = [_response("literary_restraint", "Dark input maps clearly to literary_restraint.")]
    return VoiceProfileSelectorAgent(
        llm_client=FakeLLMClient(responses=responses),
        temperature=0.0,
        max_tokens=256,
    )


def _response(profile_id: str, rationale: str) -> str:
    return json.dumps({"selected_profile_id": profile_id, "rationale": rationale})


def _make_input(
    user_prompt: str = "a dark mystery set after the collapse",
    normalized_genres: list[str] | None = None,
    user_tones: list[str] | None = None,
) -> VoiceProfileSelectorAgentInput:
    return VoiceProfileSelectorAgentInput(
        user_prompt=user_prompt,
        normalized_genres=normalized_genres or ["mystery", "post_apocalyptic"],
        user_tones=user_tones or ["dark", "cerebral"],
        available_profile_ids=["literary_restraint", "cozy_warmth", "genre_active"],
    )


# ---------------------------------------------------------------------------
# Happy path — each profile selectable
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_selects_literary_restraint(self) -> None:
        agent = _make_agent([_response("literary_restraint", "Dark input maps clearly.")])
        output = agent.run(_make_input())

        assert output.selected_profile_id == "literary_restraint"
        assert isinstance(output.voice_profile, VoiceProfile)
        assert output.voice_profile.id == "literary_restraint"

    def test_selects_cozy_warmth(self) -> None:
        agent = _make_agent([_response("cozy_warmth", "Bedtime story matches cozy_warmth.")])
        output = agent.run(_make_input(
            user_prompt="a bedtime story about a friendly cloud",
            normalized_genres=["children"],
            user_tones=["cozy", "warm"],
        ))

        assert output.selected_profile_id == "cozy_warmth"
        assert output.voice_profile.id == "cozy_warmth"

    def test_selects_genre_active(self) -> None:
        agent = _make_agent([_response("genre_active", "Action input matches genre_active.")])
        output = agent.run(_make_input(
            user_prompt="a fast-paced heist adventure",
            normalized_genres=["action", "adventure"],
            user_tones=["exciting", "fast_paced"],
        ))

        assert output.selected_profile_id == "genre_active"
        assert output.voice_profile.id == "genre_active"

    def test_returns_valid_output_type(self) -> None:
        agent = _make_agent()
        output = agent.run(_make_input())

        assert isinstance(output, VoiceProfileSelectorAgentOutput)

    def test_output_is_frozen(self) -> None:
        from pydantic import ValidationError

        agent = _make_agent()
        output = agent.run(_make_input())

        with pytest.raises(ValidationError):
            output.selected_profile_id = "mutated"  # type: ignore[misc]

    def test_rationale_is_recorded(self) -> None:
        rationale = "Dark post-apocalyptic input clearly maps to literary_restraint."
        agent = _make_agent([_response("literary_restraint", rationale)])
        output = agent.run(_make_input())

        assert output.rationale == rationale

    def test_voice_profile_loaded(self) -> None:
        agent = _make_agent([_response("cozy_warmth", "Cozy.")])
        output = agent.run(_make_input())

        assert isinstance(output.voice_profile, VoiceProfile)
        assert len(output.voice_profile.exemplars) >= 2

    def test_debug_contains_temperature(self) -> None:
        agent = _make_agent()
        output = agent.run(_make_input())

        assert "temperature" in output.debug
        assert output.debug["temperature"] == 0.0

    def test_debug_defaulted_false_on_success(self) -> None:
        agent = _make_agent([_response("literary_restraint", "Clear match.")])
        output = agent.run(_make_input())

        assert output.debug["defaulted_to_fallback"] is False


# ---------------------------------------------------------------------------
# Failure handling — default to literary_restraint
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_unknown_profile_id_defaults_to_literary_restraint(self) -> None:
        agent = _make_agent([_response("nonexistent_profile", "Something.")])
        output = agent.run(_make_input())

        assert output.selected_profile_id == "literary_restraint"
        assert output.debug["defaulted_to_fallback"] is True

    def test_llm_failure_defaults_to_literary_restraint(self) -> None:
        agent = VoiceProfileSelectorAgent(
            llm_client=FakeLLMClient(responses=[RuntimeError("provider error")]),
        )
        output = agent.run(_make_input())

        assert output.selected_profile_id == "literary_restraint"
        assert output.debug["defaulted_to_fallback"] is True

    def test_invalid_json_defaults_to_literary_restraint(self) -> None:
        agent = _make_agent(["not valid json {{{"])
        output = agent.run(_make_input())

        assert output.selected_profile_id == "literary_restraint"
        assert output.debug["defaulted_to_fallback"] is True

    def test_missing_profile_id_field_defaults_to_literary_restraint(self) -> None:
        agent = _make_agent([json.dumps({"rationale": "Missing profile id."})])
        output = agent.run(_make_input())

        assert output.selected_profile_id == "literary_restraint"
        assert output.debug["defaulted_to_fallback"] is True

    def test_never_raises(self) -> None:
        """Agent must not raise even on total LLM failure — fallback silently."""
        agent = VoiceProfileSelectorAgent(
            llm_client=FakeLLMClient(responses=[RuntimeError("crash")]),
        )
        output = agent.run(_make_input())

        assert output is not None

    def test_llm_call_count_is_one_on_success(self) -> None:
        fake_llm = FakeLLMClient(responses=[_response("literary_restraint", "Clear.")])
        agent = VoiceProfileSelectorAgent(llm_client=fake_llm)
        agent.run(_make_input())

        assert fake_llm.call_count == 1
