"""Unit tests for storymesh.agents.proposal_draft.agent."""

from __future__ import annotations

import json
from typing import Any

import pytest

from storymesh.agents.proposal_draft.agent import ProposalDraftAgent
from storymesh.llm.base import FakeLLMClient
from storymesh.schemas.proposal_draft import (
    ProposalDraftAgentInput,
    ProposalDraftAgentOutput,
    SelectionRationale,
    StoryProposal,
)
from storymesh.schemas.theme_extractor import (
    GenreCluster,
    NarrativeSeed,
    ThematicTension,
)
from storymesh.versioning.schemas import PROPOSAL_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# CapturingFakeLLMClient — records arguments per call for assertion
# ---------------------------------------------------------------------------


class CapturingFakeLLMClient(FakeLLMClient):
    """FakeLLMClient that records each call's arguments for test assertions."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses=responses)
        self.captured_calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.captured_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return super().complete(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_PLOT_ARC = (
    "Act 1: Mara Voss finds a body arranged with deliberate symbolism in the "
    "flooded district — someone is communicating through murder. Act 2: She must "
    "rebuild the infrastructure of investigation from scratch: informants, witnesses, "
    "a portable record system. The killer destroys each piece of her work. "
    "Act 3: She convenes a community tribunal with no legal standing but undeniable "
    "moral authority and delivers a verdict she must enforce alone."
)


def _cluster(**overrides: object) -> GenreCluster:
    defaults: dict[str, object] = {
        "genre": "mystery",
        "books": ["The Big Sleep"],
        "thematic_assumptions": ["Truth is recoverable through investigation"],
    }
    return GenreCluster(**{**defaults, **overrides})


def _tension(**overrides: object) -> ThematicTension:
    defaults: dict[str, object] = {
        "tension_id": "T1",
        "cluster_a": "mystery",
        "assumption_a": "Truth is recoverable",
        "cluster_b": "post_apocalyptic",
        "assumption_b": "Records and institutions no longer exist",
        "creative_question": "What does investigation mean without infrastructure?",
        "intensity": 0.9,
        "cliched_resolutions": [
            "A lone detective rebuilds justice through sheer determination",
        ],
    }
    return ThematicTension(**{**defaults, **overrides})


def _seed(seed_id: str = "S1", **overrides: object) -> NarrativeSeed:
    defaults: dict[str, object] = {
        "seed_id": seed_id,
        "concept": "A scavenger detective reinvents investigation in a collapsed city.",
        "tensions_used": ["T1"],
    }
    return NarrativeSeed(**{**defaults, **overrides})


def _valid_proposal_dict(seed_id: str = "S1", title: str = "The Last Inquest") -> dict:
    return {
        "seed_id": seed_id,
        "title": title,
        "protagonist": (
            "Mara Voss — former homicide detective whose faith in due process "
            "survived the collapse even as the process itself did not."
        ),
        "setting": (
            "A flooded mid-21st-century city-state where municipal records "
            "were lost in the first year of collapse."
        ),
        "plot_arc": _LONG_PLOT_ARC,
        "thematic_thesis": (
            "Justice does not require institutions to be meaningful, "
            "but meaning without institutions cannot produce justice."
        ),
        "key_scenes": [
            "Mara finds the arranged body and recognises the killer's signature.",
            "She convenes a community tribunal with no legal authority.",
            "The tribunal reaches a verdict she must choose whether to enforce.",
        ],
        "tensions_addressed": ["T1"],
        "tone": ["dark", "cerebral"],
        "genre_blend": ["mystery", "post_apocalyptic"],
        "image_prompt": (
            "A rain-slicked street in a flooded cityscape at dusk, a lone figure "
            "silhouetted against pale ruins of a collapsed civic tower. "
            "Gritty noir ink wash style, muted greys and a single amber light source."
        ),
    }


def _valid_proposal_json(seed_id: str = "S1", title: str = "The Last Inquest") -> str:
    return json.dumps(_valid_proposal_dict(seed_id=seed_id, title=title))


def _valid_rationale_json(selected_index: int = 0) -> str:
    return json.dumps({
        "selected_index": selected_index,
        "rationale": (
            "Candidate 0 avoids all flagged clichés and has the sharpest "
            "thematic thesis."
        ),
        "cliche_violations": {"0": [], "1": [], "2": []},
        "runner_up_index": 1,
    })


def _input(
    seeds: list[NarrativeSeed] | None = None,
    **overrides: object,
) -> ProposalDraftAgentInput:
    return ProposalDraftAgentInput(
        narrative_seeds=seeds or [_seed("S1"), _seed("S2"), _seed("S3")],
        tensions=[_tension()],
        genre_clusters=[_cluster()],
        normalized_genres=["mystery", "post_apocalyptic"],
        user_prompt="dark post-apocalyptic mystery",
        **overrides,  # type: ignore[arg-type]
    )


def _agent_with_responses(
    responses: list[str],
    num_candidates: int = 3,
    temperature: float = 1.2,
    selection_temperature: float = 0.2,
) -> ProposalDraftAgent:
    """Return an agent backed by a basic FakeLLMClient."""
    client = FakeLLMClient(responses=responses)
    return ProposalDraftAgent(
        llm_client=client,
        num_candidates=num_candidates,
        temperature=temperature,
        selection_temperature=selection_temperature,
    )


def _capturing_agent(
    responses: list[str],
    num_candidates: int = 3,
    temperature: float = 1.2,
    selection_temperature: float = 0.2,
) -> tuple[ProposalDraftAgent, CapturingFakeLLMClient]:
    """Return an agent and its CapturingFakeLLMClient for call-argument assertions."""
    client = CapturingFakeLLMClient(responses=responses)
    agent = ProposalDraftAgent(
        llm_client=client,
        num_candidates=num_candidates,
        temperature=temperature,
        selection_temperature=selection_temperature,
    )
    return agent, client


# ---------------------------------------------------------------------------
# TestBasicGeneration
# ---------------------------------------------------------------------------


class TestBasicGeneration:
    """Happy-path tests with 3 valid candidates + 1 selection response."""

    def _responses(self) -> list[str]:
        return [
            _valid_proposal_json("S1", "Title A"),
            _valid_proposal_json("S2", "Title B"),
            _valid_proposal_json("S3", "Title C"),
            _valid_rationale_json(selected_index=0),
        ]

    def test_returns_proposal_draft_output_type(self) -> None:
        agent = _agent_with_responses(self._responses())
        output = agent.run(_input())
        assert isinstance(output, ProposalDraftAgentOutput)

    def test_proposal_is_story_proposal(self) -> None:
        agent = _agent_with_responses(self._responses())
        output = agent.run(_input())
        assert isinstance(output.proposal, StoryProposal)

    def test_all_candidates_populated(self) -> None:
        agent = _agent_with_responses(self._responses())
        output = agent.run(_input())
        assert len(output.all_candidates) == 3

    def test_selection_rationale_populated(self) -> None:
        agent = _agent_with_responses(self._responses())
        output = agent.run(_input())
        assert isinstance(output.selection_rationale, SelectionRationale)

    def test_schema_version_set(self) -> None:
        agent = _agent_with_responses(self._responses())
        output = agent.run(_input())
        assert output.schema_version == PROPOSAL_SCHEMA_VERSION

    def test_selected_proposal_in_candidates(self) -> None:
        agent = _agent_with_responses(self._responses())
        output = agent.run(_input())
        assert output.proposal in output.all_candidates


# ---------------------------------------------------------------------------
# TestSeedSteering
# ---------------------------------------------------------------------------


class TestSeedSteering:
    def test_each_candidate_gets_different_seed(self) -> None:
        """With 3 seeds and 3 candidates, each call should target a different seed."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(responses, num_candidates=3)
        agent.run(_input(seeds=[_seed("S1"), _seed("S2"), _seed("S3")]))

        assert len(client.captured_calls) == 4
        prompts = [c["prompt"] for c in client.captured_calls[:3]]
        # Each generate call should contain a different seed_id as the assigned seed.
        # orjson produces compact JSON so the key:value has no spaces.
        assert any('"seed_id":"S1"' in p for p in prompts)
        assert any('"seed_id":"S2"' in p for p in prompts)
        assert any('"seed_id":"S3"' in p for p in prompts)

    def test_more_candidates_than_seeds_wraps(self) -> None:
        """With 2 seeds and 3 candidates, the third candidate wraps to seed S1."""
        seeds = [_seed("S1"), _seed("S2")]
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S1"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(responses, num_candidates=3)
        agent.run(_input(seeds=seeds))

        # Third generate prompt should contain the alternate-angle note
        third_prompt = client.captured_calls[2]["prompt"]
        assert "different creative angle" in third_prompt or "alternate" in third_prompt.lower()

    def test_single_seed_all_candidates_get_same_seed(self) -> None:
        """With 1 seed and 3 candidates, all candidates develop the same seed."""
        seeds = [_seed("S1")]
        responses = [
            _valid_proposal_json("S1", "Title A"),
            _valid_proposal_json("S1", "Title B"),
            _valid_proposal_json("S1", "Title C"),
            _valid_rationale_json(),
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        output = agent.run(_input(seeds=seeds))
        assert len(output.all_candidates) == 3


# ---------------------------------------------------------------------------
# TestLLMInteraction
# ---------------------------------------------------------------------------


class TestLLMInteraction:
    def test_num_candidates_llm_calls(self) -> None:
        """3 candidates + 1 selection = 4 total LLM calls."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(responses, num_candidates=3)
        agent.run(_input())
        assert client.call_count == 4

    def test_draft_temperature_used(self) -> None:
        """Each draft call uses the configured draft temperature."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(
            responses, num_candidates=3, temperature=1.2
        )
        agent.run(_input())
        draft_calls = client.captured_calls[:3]
        for call in draft_calls:
            assert call["temperature"] == pytest.approx(1.2)

    def test_selection_temperature_used(self) -> None:
        """The selection (last) call uses the configured selection temperature."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(
            responses, num_candidates=3, selection_temperature=0.2
        )
        agent.run(_input())
        selection_call = client.captured_calls[-1]
        assert selection_call["temperature"] == pytest.approx(0.2)

    def test_system_prompt_passed_to_drafts(self) -> None:
        """Each draft call includes a non-None system prompt."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(responses)
        agent.run(_input())
        for call in client.captured_calls[:3]:
            assert call["system_prompt"] is not None
            assert len(call["system_prompt"]) > 0

    def test_system_prompt_passed_to_selection(self) -> None:
        """The selection call includes a non-None system prompt."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent, client = _capturing_agent(responses)
        agent.run(_input())
        selection_call = client.captured_calls[-1]
        assert selection_call["system_prompt"] is not None
        assert len(selection_call["system_prompt"]) > 0


# ---------------------------------------------------------------------------
# TestParseFailures
# ---------------------------------------------------------------------------


class TestParseFailures:
    def test_one_candidate_fails_others_succeed(self) -> None:
        """3 candidates attempted, 1 returns bad JSON → 2 valid candidates, selection runs."""
        responses = [
            _valid_proposal_json("S1"),
            "not valid json {{{{",
            _valid_proposal_json("S3"),
            _valid_rationale_json(selected_index=0),
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        output = agent.run(_input())
        assert len(output.all_candidates) == 2
        assert output.debug["num_parse_failures"] == 1

    def test_all_candidates_fail_raises_runtime_error(self) -> None:
        """All candidates return bad JSON → RuntimeError."""
        responses = [
            "bad json",
            "bad json",
            "bad json",
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        with pytest.raises(RuntimeError, match="all candidate proposals failed"):
            agent.run(_input())

    def test_single_valid_candidate_skips_selection(self) -> None:
        """Only 1 candidate survives → no selection call, synthetic rationale used."""
        responses = [
            _valid_proposal_json("S1"),
            "bad json",
            "bad json",
        ]
        agent, client = _capturing_agent(responses, num_candidates=3)
        output = agent.run(_input())
        # 3 generate calls only — no selection call.
        assert client.call_count == 3
        assert len(output.all_candidates) == 1
        assert "Only one valid candidate" in output.selection_rationale.rationale

    def test_selection_call_fails_falls_back_to_first(self) -> None:
        """Selection returns bad JSON → candidate 0 selected with synthetic rationale."""
        responses = [
            _valid_proposal_json("S1", "Title A"),
            _valid_proposal_json("S2", "Title B"),
            _valid_proposal_json("S3", "Title C"),
            "bad json selection",
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        output = agent.run(_input())
        assert output.selection_rationale.selected_index == 0
        assert "fallback" in output.selection_rationale.rationale.lower()

    def test_selected_index_out_of_range_clamped(self) -> None:
        """Selection returns index 99 → clamped to 0."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            json.dumps({
                "selected_index": 99,
                "rationale": "Candidate 99 is best (out-of-range test).",
                "cliche_violations": {},
                "runner_up_index": None,
            }),
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        output = agent.run(_input())
        assert output.selection_rationale.selected_index == 0
        assert output.proposal == output.all_candidates[0]


# ---------------------------------------------------------------------------
# TestDebugMetadata
# ---------------------------------------------------------------------------


class TestDebugMetadata:
    def _run(self, num_candidates: int = 3) -> ProposalDraftAgentOutput:
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent = _agent_with_responses(responses, num_candidates=num_candidates)
        return agent.run(_input())

    def test_debug_contains_num_candidates_requested(self) -> None:
        output = self._run()
        assert output.debug["num_candidates_requested"] == 3

    def test_debug_contains_num_valid_candidates(self) -> None:
        output = self._run()
        assert output.debug["num_valid_candidates"] == 3

    def test_debug_contains_num_parse_failures(self) -> None:
        output = self._run()
        assert output.debug["num_parse_failures"] == 0

    def test_debug_contains_seed_assignments(self) -> None:
        output = self._run()
        assignments = output.debug["seed_assignments"]
        # Keys are stringified indices (required for JSON serialisation).
        assert "0" in assignments
        assert "1" in assignments
        assert "2" in assignments
        assert assignments["0"] == "S1"
        assert assignments["1"] == "S2"
        assert assignments["2"] == "S3"

    def test_debug_contains_total_llm_calls(self) -> None:
        output = self._run()
        # 3 generate + 1 selection
        assert output.debug["total_llm_calls"] == 4

    def test_debug_contains_temperatures(self) -> None:
        output = self._run()
        assert output.debug["draft_temperature"] == pytest.approx(1.2)
        assert output.debug["selection_temperature"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_seed_input(self) -> None:
        """Agent works correctly with only 1 narrative seed."""
        seeds = [_seed("S1")]
        responses = [
            _valid_proposal_json("S1", "A"),
            _valid_proposal_json("S1", "B"),
            _valid_proposal_json("S1", "C"),
            _valid_rationale_json(),
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        output = agent.run(_input(seeds=seeds))
        assert isinstance(output, ProposalDraftAgentOutput)

    def test_empty_user_tones_ok(self) -> None:
        """Agent works without user tones."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent = _agent_with_responses(responses)
        output = agent.run(_input(user_tones=[]))
        assert isinstance(output, ProposalDraftAgentOutput)

    def test_empty_narrative_context_ok(self) -> None:
        """Agent works without narrative context."""
        responses = [
            _valid_proposal_json("S1"),
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(),
        ]
        agent = _agent_with_responses(responses)
        output = agent.run(_input(narrative_context=[]))
        assert isinstance(output, ProposalDraftAgentOutput)

    def test_two_candidates_minimum_selection_runs(self) -> None:
        """With exactly 2 valid candidates, the selection step runs normally."""
        responses = [
            _valid_proposal_json("S1", "Title A"),
            _valid_proposal_json("S2", "Title B"),
            _valid_rationale_json(selected_index=1),
        ]
        agent, client = _capturing_agent(responses, num_candidates=2)
        output = agent.run(_input(seeds=[_seed("S1"), _seed("S2")]))
        # 2 generate + 1 selection = 3 calls
        assert client.call_count == 3
        assert len(output.all_candidates) == 2
        assert output.proposal == output.all_candidates[1]

    def test_pydantic_validation_failure_in_candidate_counted_as_failure(self) -> None:
        """A Pydantic ValidationError during candidate parsing is treated as a failure."""
        # Valid JSON but missing required fields → ValidationError
        bad_candidate = json.dumps({"seed_id": "S1", "title": "Incomplete"})
        responses = [
            bad_candidate,
            _valid_proposal_json("S2"),
            _valid_proposal_json("S3"),
            _valid_rationale_json(selected_index=0),
        ]
        agent = _agent_with_responses(responses, num_candidates=3)
        output = agent.run(_input())
        assert output.debug["num_parse_failures"] == 1
        assert output.debug["num_valid_candidates"] == 2
