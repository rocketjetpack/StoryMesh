"""Unit tests for storymesh.agents.story_writer.agent and the Stage 6 node wrapper.

Covers:
- StoryWriterAgent: happy path (all three passes succeed)
- StoryWriterAgent: per-pass failure modes (empty/exception/invalid scenes)
- StoryWriterAgent: craft notes formatting from RubricJudgeAgentOutput
- StoryWriterAgent: temperature and token params forwarded correctly
- StoryWriterAgent: voice profile overlays injected into system prompts (WI-3)
- make_story_writer_node: state assembly, best-proposal selection, artifact persistence
- make_story_writer_node: voice_profile read from voice_profile_selector_output (WI-3)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from storymesh.agents.story_writer.agent import (
    StoryWriterAgent,
    _format_craft_notes,
    _format_scene_list_for_prompt,
)
from storymesh.llm.base import FakeLLMClient
from storymesh.orchestration.nodes.story_writer import make_story_writer_node
from storymesh.schemas.proposal_draft import ProposalDraftAgentOutput, StoryProposal
from storymesh.schemas.rubric_judge import RubricJudgeAgentOutput
from storymesh.schemas.story_writer import (
    SceneOutline,
    StoryWriterAgentInput,
    StoryWriterAgentOutput,
)
from storymesh.schemas.theme_extractor import ThematicTension
from storymesh.schemas.voice_profile import load_voice_profile
from storymesh.versioning.schemas import STORY_WRITER_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# CapturingFakeLLMClient
# ---------------------------------------------------------------------------


class CapturingFakeLLMClient(FakeLLMClient):
    """FakeLLMClient that records each call's arguments for assertion."""

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
                "agent_name": self.agent_name,
            }
        )
        return super().complete(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_OUTLINE_RESPONSE = json.dumps({
    "scenes": [
        {
            "scene_id": "scene_01",
            "title": "The Third Complaint",
            "summary": (
                "Mara finds the body arranged in the old courthouse. "
                "She photographs each detail with a cracked phone."
            ),
            "narrative_pressure": (
                "Inhabits the tension between order and absence: the body is arranged "
                "as if evidence still matters in a city with no court."
            ),
            "observational_anchor": "The cracked phone screen she uses to photograph evidence.",
            "opens_with": (
                "The body had been arranged with the careful deliberation of someone "
                "who expected it to be found."
            ),
        },
        {
            "scene_id": "scene_02",
            "title": "The Canvass",
            "summary": (
                "Mara interviews three witnesses in a flooded block. "
                "None of them saw anything. One of them is lying."
            ),
            "narrative_pressure": (
                "Pressurises the tension between investigation and futility: "
                "her tools are intact but the infrastructure they require is gone."
            ),
            "observational_anchor": "The salvaged tax forms she uses as notebook paper.",
            "opens_with": (
                "She had run out of notebook paper two weeks ago and was using "
                "the backs of salvaged tax forms."
            ),
        },
        {
            "scene_id": "scene_03",
            "title": "The Verdict",
            "summary": (
                "Mara convenes a community tribunal with no legal standing. "
                "They reach a verdict she must decide whether to enforce."
            ),
            "narrative_pressure": (
                "Lands on the tension unresolved: the tribunal has moral authority "
                "but no mechanism for enforcement."
            ),
            "observational_anchor": "The parking garage's oil-stained concrete floor.",
            "opens_with": (
                "The tribunal met in what had been a parking garage because it was the "
                "only space large enough and because no one had claimed it."
            ),
        },
    ]
})

_DRAFT_RESPONSE = json.dumps({
    "full_draft": (
        "The body had been arranged with the careful deliberation of someone "
        "who expected it to be found.\n\nShe documented it twice. "
        "The second pass was habit.\n\n---\n\n"
        "She had run out of notebook paper two weeks ago and was using "
        "the backs of salvaged tax forms.\n\nThe first witness said he was at "
        "his sister's. The second said the same thing.\n\n---\n\n"
        "The tribunal met in what had been a parking garage because it was the "
        "only space large enough and because no one had claimed it.\n\n"
        "They reached a verdict. She folded the paper and put it in her coat."
    )
})

_SUMMARY_RESPONSE = json.dumps({
    "back_cover_summary": (
        "In the weeks after the flood receded, Mara Voss returned to what she knew.\n\n"
        "The records were gone. The precinct was a waterline stain on brick. "
        "She kept working anyway.\n\n"
        "When a body turns up arranged like evidence, Mara is the only person "
        "in the district who knows how to read what it is saying. "
        "She has no authority, no institution, and no one to deliver the case to "
        "when it is finished. She works anyway."
    )
})


def _make_tension(tension_id: str = "T1") -> ThematicTension:
    return ThematicTension(
        tension_id=tension_id,
        cluster_a="mystery",
        assumption_a="Truth is recoverable through investigation",
        cluster_b="post_apocalyptic",
        assumption_b="Records and institutions no longer exist",
        creative_question="What does investigation mean without infrastructure?",
        intensity=0.9,
        cliched_resolutions=["A lone detective rebuilds justice through sheer determination"],
    )


def _make_proposal() -> StoryProposal:
    return StoryProposal(
        seed_id="S1",
        title="The Last Inquest",
        protagonist=(
            "Mara Voss — former homicide detective whose faith in due process "
            "survived the collapse even as the process itself did not."
        ),
        setting=(
            "A flooded mid-21st-century city-state where municipal records "
            "were lost in the first year of collapse."
        ),
        plot_arc=(
            "Act 1: Mara finds a body arranged with deliberate symbolism. "
            "Act 2: She rebuilds the infrastructure of investigation. "
            "Act 3: She convenes a tribunal with no legal standing."
        ),
        thematic_thesis=(
            "Justice does not require institutions to be meaningful, "
            "but meaning without institutions cannot produce justice."
        ),
        key_scenes=[
            "Mara finds the arranged body.",
            "She interviews the last witness.",
            "The tribunal reaches a verdict.",
        ],
        tensions_addressed=["T1"],
        tone=["dark", "cerebral"],
        genre_blend=["mystery", "post_apocalyptic"],
        image_prompt="A rain-slicked street in a flooded cityscape at dusk.",
    )


def _make_input(
    rubric_feedback: RubricJudgeAgentOutput | None = None,
    voice_profile: object | None = None,
) -> StoryWriterAgentInput:
    return StoryWriterAgentInput(
        proposal=_make_proposal(),
        tensions=[_make_tension()],
        rubric_feedback=rubric_feedback,
        user_prompt="dark post-apocalyptic mystery",
        normalized_genres=["mystery", "post_apocalyptic"],
        user_tones=["dark", "cerebral"],
        voice_profile=voice_profile,  # type: ignore[arg-type]
    )


def _agent_with_responses(
    responses: list[str],
    outline_temperature: float = 0.5,
    draft_temperature: float = 0.8,
    summary_temperature: float = 0.4,
    outline_max_tokens: int = 4096,
    draft_max_tokens: int = 6000,
    summary_max_tokens: int = 1024,
    target_words: int = 3000,
) -> StoryWriterAgent:
    client = FakeLLMClient(responses=responses)
    return StoryWriterAgent(
        llm_client=client,
        outline_temperature=outline_temperature,
        draft_temperature=draft_temperature,
        summary_temperature=summary_temperature,
        outline_max_tokens=outline_max_tokens,
        draft_max_tokens=draft_max_tokens,
        summary_max_tokens=summary_max_tokens,
        target_words=target_words,
    )


def _capturing_agent(
    responses: list[str],
    outline_temperature: float = 0.5,
    draft_temperature: float = 0.8,
    summary_temperature: float = 0.4,
    outline_max_tokens: int = 4096,
    draft_max_tokens: int = 6000,
    summary_max_tokens: int = 1024,
    target_words: int = 3000,
) -> tuple[StoryWriterAgent, CapturingFakeLLMClient]:
    client = CapturingFakeLLMClient(responses=responses)
    return StoryWriterAgent(
        llm_client=client,
        outline_temperature=outline_temperature,
        draft_temperature=draft_temperature,
        summary_temperature=summary_temperature,
        outline_max_tokens=outline_max_tokens,
        draft_max_tokens=draft_max_tokens,
        summary_max_tokens=summary_max_tokens,
        target_words=target_words,
    ), client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestStoryWriterAgentHappyPath:
    def test_returns_story_writer_agent_output(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert isinstance(result, StoryWriterAgentOutput)

    def test_back_cover_summary_populated(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert "Mara Voss" in result.back_cover_summary

    def test_scene_list_count(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert len(result.scene_list) == 3

    def test_full_draft_contains_content(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert "body had been arranged" in result.full_draft

    def test_word_count_positive(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert result.word_count > 0

    def test_word_count_matches_draft(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        import json as _json
        expected_draft = _json.loads(_DRAFT_RESPONSE)["full_draft"].strip()
        assert result.word_count == len(expected_draft.split())

    def test_scene_list_opens_with_preserved(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert result.scene_list[0].opens_with.startswith(
            "The body had been arranged"
        )

    def test_schema_version_set(self) -> None:
        from storymesh.versioning.schemas import STORY_WRITER_SCHEMA_VERSION

        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert result.schema_version == STORY_WRITER_SCHEMA_VERSION

    def test_debug_contains_expected_keys(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        for key in (
            "outline_temperature",
            "draft_temperature",
            "summary_temperature",
            "target_words",
            "scene_count",
            "word_count",
            "total_llm_calls",
            "had_rubric_feedback",
        ):
            assert key in result.debug, f"missing debug key: {key}"

    def test_debug_total_llm_calls_is_three(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert result.debug["total_llm_calls"] == 3

    def test_debug_had_rubric_feedback_false_when_none(self) -> None:
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input(rubric_feedback=None))

        assert result.debug["had_rubric_feedback"] is False


# ---------------------------------------------------------------------------
# Temperature / token forwarding
# ---------------------------------------------------------------------------


class TestTemperatureForwarding:
    def test_outline_temperature_forwarded(self) -> None:
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
            outline_temperature=0.3,
        )
        agent.run(_make_input())

        assert client.captured_calls[0]["temperature"] == 0.3

    def test_draft_temperature_forwarded(self) -> None:
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
            draft_temperature=0.9,
        )
        agent.run(_make_input())

        assert client.captured_calls[1]["temperature"] == 0.9

    def test_summary_temperature_forwarded(self) -> None:
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
            summary_temperature=0.2,
        )
        agent.run(_make_input())

        assert client.captured_calls[2]["temperature"] == 0.2

    def test_outline_max_tokens_forwarded(self) -> None:
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
            outline_max_tokens=2048,
        )
        agent.run(_make_input())

        assert client.captured_calls[0]["max_tokens"] == 2048

    def test_draft_max_tokens_forwarded(self) -> None:
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
            draft_max_tokens=5000,
        )
        agent.run(_make_input())

        assert client.captured_calls[1]["max_tokens"] == 5000


# ---------------------------------------------------------------------------
# Outline pass failure modes
# ---------------------------------------------------------------------------


class TestOutlinePassFailures:
    def test_outline_llm_exception_raises_runtime_error(self) -> None:
        client = FakeLLMClient(responses=[RuntimeError("provider error")])
        agent = StoryWriterAgent(llm_client=client)

        with pytest.raises(RuntimeError, match="outline pass failed"):
            agent.run(_make_input())

    def test_outline_empty_scenes_list_raises_runtime_error(self) -> None:
        agent = _agent_with_responses([
            json.dumps({"scenes": []}),
        ])

        with pytest.raises(RuntimeError, match="no scenes"):
            agent.run(_make_input())

    def test_all_invalid_scenes_raises_runtime_error(self) -> None:
        bad_outline = json.dumps({
            "scenes": [
                {"scene_id": "scene_01"},  # missing required fields
                {"title": "broken"},        # still missing required fields
            ]
        })
        agent = _agent_with_responses([bad_outline])

        with pytest.raises(RuntimeError, match="all scene outlines failed validation"):
            agent.run(_make_input())

    def test_partial_invalid_scenes_skipped(self) -> None:
        """Three good scenes + one broken scene → broken one skipped, three valid remain."""
        partial_outline = json.dumps({
            "scenes": [
                {
                    "scene_id": "scene_01",
                    "title": "Opening",
                    "summary": "She finds the body and photographs it carefully.",
                    "narrative_pressure": "Establishes the central tension between order and collapse.",
                    "observational_anchor": "The cracked phone screen.",
                    "opens_with": "The body had been arranged with care.",
                },
                {
                    "scene_id": "scene_02",
                    "title": "The Canvass",
                    "summary": "She interviews three witnesses in the flooded block.",
                    "narrative_pressure": "Pressurises the tension between investigation and futility.",
                    "observational_anchor": "The salvaged tax forms.",
                    "opens_with": "She had run out of notebook paper two weeks ago.",
                },
                {
                    "scene_id": "scene_03",
                    "title": "The Verdict",
                    "summary": "Mara convenes a community tribunal with no legal standing.",
                    "narrative_pressure": "Lands on the tension between authority and legitimacy.",
                    "observational_anchor": "The oil-stained concrete floor.",
                    "opens_with": "The tribunal met in what had been a parking garage.",
                },
                {"bad": "data"},  # will be skipped
            ]
        })
        agent = _agent_with_responses([partial_outline, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        result = agent.run(_make_input())

        assert len(result.scene_list) == 3


# ---------------------------------------------------------------------------
# Draft pass failure modes
# ---------------------------------------------------------------------------


class TestDraftPassFailures:
    def test_draft_llm_exception_raises_runtime_error(self) -> None:
        client = FakeLLMClient(responses=[
            _OUTLINE_RESPONSE,
            RuntimeError("provider error"),
        ])
        agent = StoryWriterAgent(llm_client=client)

        with pytest.raises(RuntimeError, match="draft pass failed"):
            agent.run(_make_input())

    def test_empty_full_draft_raises_runtime_error(self) -> None:
        agent = _agent_with_responses([
            _OUTLINE_RESPONSE,
            json.dumps({"full_draft": ""}),
        ])

        with pytest.raises(RuntimeError, match="empty draft"):
            agent.run(_make_input())

    def test_whitespace_only_draft_raises_runtime_error(self) -> None:
        agent = _agent_with_responses([
            _OUTLINE_RESPONSE,
            json.dumps({"full_draft": "   \n  "}),
        ])

        with pytest.raises(RuntimeError, match="empty draft"):
            agent.run(_make_input())


# ---------------------------------------------------------------------------
# Summary pass failure modes
# ---------------------------------------------------------------------------


class TestSummaryPassFailures:
    def test_summary_llm_exception_raises_runtime_error(self) -> None:
        client = FakeLLMClient(responses=[
            _OUTLINE_RESPONSE,
            _DRAFT_RESPONSE,
            RuntimeError("provider error"),
        ])
        agent = StoryWriterAgent(llm_client=client)

        with pytest.raises(RuntimeError, match="summary pass failed"):
            agent.run(_make_input())

    def test_empty_summary_raises_runtime_error(self) -> None:
        agent = _agent_with_responses([
            _OUTLINE_RESPONSE,
            _DRAFT_RESPONSE,
            json.dumps({"back_cover_summary": ""}),
        ])

        with pytest.raises(RuntimeError, match="empty summary"):
            agent.run(_make_input())


# ---------------------------------------------------------------------------
# Craft notes formatting
# ---------------------------------------------------------------------------


class TestFormatCraftNotes:
    def _make_rubric_output(
        self,
        scores: dict[str, int],
        overall_feedback: str = "",
        composite_score: int | None = None,
    ) -> MagicMock:
        """Build a minimal mock rubric output with dimension tier scores."""
        mock = MagicMock()
        mock.overall_feedback = overall_feedback
        mock.composite_score = composite_score

        dims: dict[str, Any] = {}
        for dim_name, score in scores.items():
            dim_mock = MagicMock()
            dim_mock.score = score
            dim_mock.feedback = f"Feedback for {dim_name}"
            dims[dim_name] = dim_mock
        mock.dimensions = dims
        return mock

    def test_fail_score_dimension_included(self) -> None:
        rubric = self._make_rubric_output({"restraint": 0})
        notes = _format_craft_notes(rubric)

        assert "restraint" in notes
        assert "fail" in notes

    def test_acceptable_score_dimension_included(self) -> None:
        rubric = self._make_rubric_output({"specificity": 1})
        notes = _format_craft_notes(rubric)

        assert "specificity" in notes
        assert "acceptable" in notes

    def test_strong_score_dimension_excluded(self) -> None:
        rubric = self._make_rubric_output({"specificity": 2})
        notes = _format_craft_notes(rubric)

        assert notes == ""

    def test_story_serving_choices_included_when_fail(self) -> None:
        rubric = self._make_rubric_output({"story_serving_choices": 0})
        notes = _format_craft_notes(rubric)

        assert "story_serving_choices" in notes

    def test_story_serving_choices_excluded_when_strong(self) -> None:
        rubric = self._make_rubric_output({"story_serving_choices": 2})
        notes = _format_craft_notes(rubric)

        assert notes == ""

    def test_composite_score_included_in_notes(self) -> None:
        rubric = self._make_rubric_output({"restraint": 1}, composite_score=6)
        notes = _format_craft_notes(rubric)

        assert "6/10" in notes

    def test_overall_feedback_appended(self) -> None:
        rubric = self._make_rubric_output(
            {"restraint": 0},
            overall_feedback="The proposal leans too heavily on genre convention.",
        )
        notes = _format_craft_notes(rubric)

        assert "leans too heavily" in notes

    def test_all_dimensions_strong_returns_empty_string(self) -> None:
        rubric = self._make_rubric_output({"restraint": 2, "specificity": 2})
        notes = _format_craft_notes(rubric)

        assert notes == ""

    def test_none_rubric_input_handled_by_caller(self) -> None:
        """Agent wraps None check before calling _format_craft_notes."""
        agent = _agent_with_responses([_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE])
        # No rubric_feedback → craft_notes should be empty, no crash.
        result = agent.run(_make_input(rubric_feedback=None))
        assert result.debug["had_rubric_feedback"] is False


# ---------------------------------------------------------------------------
# _format_scene_list_for_prompt
# ---------------------------------------------------------------------------


class TestFormatSceneListForPrompt:
    def _scene(self, n: int) -> SceneOutline:
        return SceneOutline(
            scene_id=f"scene_0{n}",
            title=f"Scene {n} Title",
            summary=f"What happens in scene {n}.",
            narrative_pressure=f"Tension work for scene {n}.",
            observational_anchor=f"A concrete detail for scene {n}.",
            opens_with=f"The first sentence of scene {n} is specific.",
        )

    def test_each_scene_numbered(self) -> None:
        scenes = [self._scene(1), self._scene(2)]
        text = _format_scene_list_for_prompt(scenes)

        assert "SCENE 1:" in text
        assert "SCENE 2:" in text

    def test_opens_with_verbatim_label_present(self) -> None:
        scenes = [self._scene(1)]
        text = _format_scene_list_for_prompt(scenes)

        assert "Opens with (use verbatim):" in text

    def test_opens_with_content_present(self) -> None:
        scenes = [self._scene(1)]
        text = _format_scene_list_for_prompt(scenes)

        assert "The first sentence of scene 1 is specific." in text

    def test_scenes_separated(self) -> None:
        scenes = [self._scene(1), self._scene(2), self._scene(3)]
        text = _format_scene_list_for_prompt(scenes)

        # Each scene block should be a distinct chunk
        assert text.count("SCENE") == 3


# ---------------------------------------------------------------------------
# Agent name per pass (WI-4 logging hygiene)
# ---------------------------------------------------------------------------


class TestAgentNamePerPass:
    def test_each_pass_uses_distinct_agent_name(self) -> None:
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input())

        names = [call["agent_name"] for call in client.captured_calls]
        assert names == ["story_writer_outline", "story_writer_draft", "story_writer_summary"]


# ---------------------------------------------------------------------------
# make_story_writer_node — node wrapper
# ---------------------------------------------------------------------------


def _make_genre_output(genres: list[str] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.normalized_genres = genres or ["mystery", "post_apocalyptic"]
    return mock


def _make_theme_output(tensions: list[ThematicTension] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.tensions = tensions or [_make_tension()]
    mock.user_tones_carried = ["dark"]
    return mock


def _make_proposal_draft_output(proposal: StoryProposal | None = None) -> MagicMock:
    mock = MagicMock(spec=ProposalDraftAgentOutput)
    mock.proposal = proposal or _make_proposal()
    return mock


def _base_state(**overrides: object) -> dict[str, object]:
    return {
        "run_id": "testrунid001",
        "user_prompt": "dark mystery",
        "proposal_draft_output": _make_proposal_draft_output(),
        "theme_extractor_output": _make_theme_output(),
        "genre_normalizer_output": _make_genre_output(),
        "rubric_judge_output": None,
        "proposal_history": None,
        "best_proposal_index": 0,
        **overrides,
    }


class TestStoryWriterNode:
    def _mock_agent(self, output: StoryWriterAgentOutput | None = None) -> StoryWriterAgent:
        agent = MagicMock(spec=StoryWriterAgent)
        if output is None:
            draft = json.loads(_DRAFT_RESPONSE)["full_draft"].strip()
            summary = json.loads(_SUMMARY_RESPONSE)["back_cover_summary"].strip()
            output = StoryWriterAgentOutput(
                back_cover_summary=summary,
                scene_list=[
                    SceneOutline(
                        scene_id="scene_01",
                        title="The Third Complaint",
                        summary="Mara finds the body arranged in the old courthouse.",
                        narrative_pressure="Establishes tension between order and absence.",
                        observational_anchor="The cracked phone screen.",
                        opens_with=(
                            "The body had been arranged with the careful deliberation "
                            "of someone who expected it to be found."
                        ),
                    ),
                    SceneOutline(
                        scene_id="scene_02",
                        title="The Canvass",
                        summary="Mara interviews three witnesses in the flooded block.",
                        narrative_pressure="Pressurises the tension between investigation and futility.",
                        observational_anchor="The salvaged tax forms.",
                        opens_with=(
                            "She had run out of notebook paper two weeks ago and was "
                            "using the backs of salvaged tax forms."
                        ),
                    ),
                    SceneOutline(
                        scene_id="scene_03",
                        title="The Verdict",
                        summary="Mara convenes a community tribunal with no legal standing.",
                        narrative_pressure="Lands on the tension between authority and legitimacy.",
                        observational_anchor="The oil-stained concrete floor.",
                        opens_with=(
                            "The tribunal met in what had been a parking garage "
                            "because it was the only space large enough."
                        ),
                    ),
                ],
                full_draft=draft,
                word_count=len(draft.split()),
                debug={},
                schema_version=STORY_WRITER_SCHEMA_VERSION,
            )
        agent.run.return_value = output
        return agent

    def test_returns_story_writer_output_key(self) -> None:
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        result = node(_base_state())  # type: ignore[arg-type]

        assert "story_writer_output" in result

    def test_agent_run_called_once(self) -> None:
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        node(_base_state())  # type: ignore[arg-type]

        agent.run.assert_called_once()

    def test_input_uses_proposal_from_draft_output(self) -> None:
        proposal = _make_proposal()
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        node(_base_state(proposal_draft_output=_make_proposal_draft_output(proposal)))  # type: ignore[arg-type]

        call_input: StoryWriterAgentInput = agent.run.call_args[0][0]
        assert call_input.proposal.title == proposal.title

    def test_input_uses_best_proposal_from_history(self) -> None:
        """When proposal_history exists, use the entry at best_proposal_index."""
        best_proposal = _make_proposal()
        best_proposal_output = _make_proposal_draft_output(best_proposal)

        other_proposal = StoryProposal(
            seed_id="S2",
            title="Other Story",
            protagonist="Someone else entirely, a quiet observer of events.",
            setting="A small village at the edge of a receding glacier.",
            plot_arc=(
                "Act 1: The observer notices something wrong with the patterns. "
                "Act 2: She investigates. Act 3: She must decide what to do."
            ),
            thematic_thesis="Observation without action is its own form of complicity.",
            key_scenes=["She notices the first sign.", "She tells no one.", "She acts alone."],
            tensions_addressed=["T1"],
            tone=["quiet"],
            genre_blend=["literary"],
            image_prompt=(
                "A woman standing at the edge of a glacier at dusk, her silhouette "
                "small against the pale blue ice. Muted watercolour style."
            ),
        )
        other_output = _make_proposal_draft_output(other_proposal)

        history = [other_output, best_proposal_output]  # best is index 1

        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        node(_base_state(  # type: ignore[arg-type]
            proposal_history=history,
            best_proposal_index=1,
        ))

        call_input: StoryWriterAgentInput = agent.run.call_args[0][0]
        assert call_input.proposal.title == best_proposal.title

    def test_missing_proposal_draft_raises_runtime_error(self) -> None:
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        with pytest.raises(RuntimeError, match="proposal_draft_output"):
            node(_base_state(proposal_draft_output=None))  # type: ignore[arg-type]

    def test_missing_theme_extractor_raises_runtime_error(self) -> None:
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        with pytest.raises(RuntimeError, match="theme_extractor_output"):
            node(_base_state(theme_extractor_output=None))  # type: ignore[arg-type]

    def test_missing_genre_normalizer_raises_runtime_error(self) -> None:
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        with pytest.raises(RuntimeError, match="genre_normalizer_output"):
            node(_base_state(genre_normalizer_output=None))  # type: ignore[arg-type]

    def test_artifact_store_persist_called(self) -> None:
        agent = self._mock_agent()
        mock_store = MagicMock()
        node = make_story_writer_node(agent, artifact_store=mock_store)

        with patch(
            "storymesh.core.artifacts.persist_node_output"
        ) as mock_persist:
            node(_base_state())  # type: ignore[arg-type]
            mock_persist.assert_called_once()

    def test_artifact_store_none_no_persist_called(self) -> None:
        agent = self._mock_agent()
        node = make_story_writer_node(agent, artifact_store=None)

        with patch(
            "storymesh.core.artifacts.persist_node_output"
        ) as mock_persist:
            node(_base_state())  # type: ignore[arg-type]
            mock_persist.assert_not_called()

    def test_run_id_forwarded_to_current_run_id(self) -> None:
        """current_run_id context var is set and reset around agent.run."""
        from storymesh.llm.base import current_run_id

        captured: list[str] = []

        def capturing_run(input_data: StoryWriterAgentInput) -> StoryWriterAgentOutput:
            captured.append(current_run_id.get(""))
            return self._mock_agent().run.return_value  # type: ignore[return-value]

        agent = MagicMock(spec=StoryWriterAgent)
        agent.run.side_effect = capturing_run

        capturing_node = make_story_writer_node(agent)
        capturing_node(_base_state(run_id="myrunid999"))  # type: ignore[arg-type]

        assert captured == ["myrunid999"]


# ---------------------------------------------------------------------------
# Voice profile overlay injection (WI-3)
# ---------------------------------------------------------------------------


class TestVoiceProfileOverlays:
    """Verify that voice profile overlays reach the LLM system prompts."""

    def test_no_voice_profile_uses_literary_restraint_fallback(self) -> None:
        """When voice_profile is None, literary_restraint is loaded — no crash, overlays empty."""
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=None))

        # literary_restraint has empty overlays — system prompts should not contain overlay markers
        draft_system = client.captured_calls[1]["system_prompt"] or ""
        assert "Direct emotion-naming" not in draft_system
        assert "Kinetic over interior" not in draft_system

    def test_cozy_warmth_craft_overlay_in_draft_system(self) -> None:
        """cozy_warmth craft_overlay appears in the draft pass system prompt."""
        profile = load_voice_profile("cozy_warmth")
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=profile))

        draft_system = client.captured_calls[1]["system_prompt"] or ""
        assert "Direct emotion-naming" in draft_system

    def test_cozy_warmth_avoid_overlay_in_draft_system(self) -> None:
        """cozy_warmth avoid_overlay appears in the draft pass system prompt."""
        profile = load_voice_profile("cozy_warmth")
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=profile))

        draft_system = client.captured_calls[1]["system_prompt"] or ""
        assert "The way X is when Y" in draft_system

    def test_genre_active_craft_overlay_in_draft_system(self) -> None:
        """genre_active craft_overlay appears in the draft pass system prompt."""
        profile = load_voice_profile("genre_active")
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=profile))

        draft_system = client.captured_calls[1]["system_prompt"] or ""
        assert "Kinetic over interior" in draft_system

    def test_cozy_warmth_exemplars_in_outline_system(self) -> None:
        """cozy_warmth exemplars replace the default opens_with examples in the outline prompt."""
        profile = load_voice_profile("cozy_warmth")
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=profile))

        outline_system = client.captured_calls[0]["system_prompt"] or ""
        # cozy_warmth first exemplar contains "lantern"
        assert "lantern" in outline_system

    def test_literary_restraint_exemplars_in_outline_system(self) -> None:
        """literary_restraint exemplars (default fallback) appear in the outline prompt."""
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=None))

        outline_system = client.captured_calls[0]["system_prompt"] or ""
        # literary_restraint first exemplar references arranged body / tax forms
        assert "salvaged tax forms" in outline_system or "arranged" in outline_system

    def test_cozy_warmth_summary_overlay_in_summary_system(self) -> None:
        """cozy_warmth summary_overlay appears in the summary pass system prompt."""
        profile = load_voice_profile("cozy_warmth")
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=profile))

        summary_system = client.captured_calls[2]["system_prompt"] or ""
        assert summary_system  # not empty
        # literary_restraint has no summary_overlay; cozy_warmth should differ
        agent2, client2 = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent2.run(_make_input(voice_profile=None))
        default_system = client2.captured_calls[2]["system_prompt"] or ""
        assert summary_system != default_system

    def test_overlays_do_not_appear_in_wrong_pass(self) -> None:
        """craft_overlay is draft-only; it must not leak into the outline or summary system prompts."""
        profile = load_voice_profile("cozy_warmth")
        agent, client = _capturing_agent(
            [_OUTLINE_RESPONSE, _DRAFT_RESPONSE, _SUMMARY_RESPONSE],
        )
        agent.run(_make_input(voice_profile=profile))

        outline_system = client.captured_calls[0]["system_prompt"] or ""
        summary_system = client.captured_calls[2]["system_prompt"] or ""
        assert "Direct emotion-naming" not in outline_system
        assert "Direct emotion-naming" not in summary_system


# ---------------------------------------------------------------------------
# make_story_writer_node — voice_profile_selector_output integration (WI-3)
# ---------------------------------------------------------------------------


class TestStoryWriterNodeVoiceProfile:
    """Verify the node wrapper reads voice_profile from voice_profile_selector_output."""

    def _mock_agent(self) -> MagicMock:
        agent = MagicMock(spec=StoryWriterAgent)
        draft = json.loads(_DRAFT_RESPONSE)["full_draft"].strip()
        summary = json.loads(_SUMMARY_RESPONSE)["back_cover_summary"].strip()
        output = StoryWriterAgentOutput(
            back_cover_summary=summary,
            scene_list=[
                SceneOutline(
                    scene_id="scene_01",
                    title="Opening",
                    summary="Mara finds the body.",
                    narrative_pressure="Tension between order and collapse.",
                    observational_anchor="The cracked phone screen.",
                    opens_with="The body had been arranged with care.",
                ),
                SceneOutline(
                    scene_id="scene_02",
                    title="The Canvass",
                    summary="Mara interviews witnesses.",
                    narrative_pressure="Investigation without infrastructure.",
                    observational_anchor="The salvaged tax forms.",
                    opens_with="She had run out of notebook paper.",
                ),
                SceneOutline(
                    scene_id="scene_03",
                    title="The Verdict",
                    summary="The tribunal meets.",
                    narrative_pressure="Authority without legitimacy.",
                    observational_anchor="The oil-stained floor.",
                    opens_with="The tribunal met in the parking garage.",
                ),
            ],
            full_draft=draft,
            word_count=len(draft.split()),
            debug={},
            schema_version=STORY_WRITER_SCHEMA_VERSION,
        )
        agent.run.return_value = output
        return agent

    def test_voice_profile_none_when_no_vps_output(self) -> None:
        """When voice_profile_selector_output is absent, voice_profile passed as None."""
        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        node(_base_state(voice_profile_selector_output=None))  # type: ignore[arg-type]

        call_input: StoryWriterAgentInput = agent.run.call_args[0][0]
        assert call_input.voice_profile is None

    def test_voice_profile_passed_from_vps_output(self) -> None:
        """When voice_profile_selector_output is present, its voice_profile is forwarded."""
        profile = load_voice_profile("cozy_warmth")
        vps_mock = MagicMock()
        vps_mock.voice_profile = profile

        agent = self._mock_agent()
        node = make_story_writer_node(agent)

        node(_base_state(voice_profile_selector_output=vps_mock))  # type: ignore[arg-type]

        call_input: StoryWriterAgentInput = agent.run.call_args[0][0]
        assert call_input.voice_profile is profile
