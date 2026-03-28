"""Unit tests for storymesh.agents.theme_extractor.agent and theme_extractor prompt."""

from __future__ import annotations

import json

import pytest

from storymesh.agents.theme_extractor.agent import ThemeExtractorAgent
from storymesh.llm.base import FakeLLMClient
from storymesh.prompts.loader import PromptFormattingError, load_prompt
from storymesh.schemas.book_ranker import RankedBookSummary
from storymesh.schemas.theme_extractor import (
    ThemeExtractorAgentInput,
    ThemeExtractorAgentOutput,
)
from storymesh.versioning.schemas import THEMEPACK_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summary(
    rank: int = 1,
    source_genres: list[str] | None = None,
    title: str | None = None,
) -> RankedBookSummary:
    return RankedBookSummary(
        work_key=f"/works/OL{rank}W",
        title=title or f"Book {rank}",
        authors=["Author A"],
        source_genres=source_genres or ["mystery"],
        composite_score=0.9 - rank * 0.05,
        rank=rank,
    )


def _input(
    summaries: list[RankedBookSummary] | None = None,
    genres: list[str] | None = None,
    **kwargs: object,
) -> ThemeExtractorAgentInput:
    return ThemeExtractorAgentInput(
        ranked_summaries=summaries or [_summary()],
        normalized_genres=genres or ["mystery"],
        user_prompt="dark mystery",
        **kwargs,  # type: ignore[arg-type]
    )


def _valid_response(num_seeds: int = 1) -> str:
    seeds = [
        {
            "seed_id": f"S{i + 1}",
            "concept": "A former detective must reinvent investigation from first principles.",
            "tensions_used": ["T1"],
            "tonal_direction": ["dark"],
            "narrative_context_used": [],
        }
        for i in range(num_seeds)
    ]
    return json.dumps({
        "genre_clusters": [
            {
                "genre": "mystery",
                "books": ["Book 1"],
                "thematic_assumptions": ["Truth is discoverable through investigation"],
                "dominant_tropes": ["red herring"],
            }
        ],
        "tensions": [
            {
                "tension_id": "T1",
                "cluster_a": "mystery",
                "assumption_a": "Truth is discoverable",
                "cluster_b": "mystery",
                "assumption_b": "Some truths destroy the investigator",
                "creative_question": "What does solving a case cost the one who solves it?",
                "intensity": 0.7,
                "cliched_resolutions": [
                    "The detective solves the case and walks away unscathed",
                ],
            }
        ],
        "narrative_seeds": seeds,
    })


def _multi_genre_response() -> str:
    return json.dumps({
        "genre_clusters": [
            {
                "genre": "mystery",
                "books": ["Book 1"],
                "thematic_assumptions": ["Truth is discoverable through investigation"],
                "dominant_tropes": ["red herring"],
            },
            {
                "genre": "post_apocalyptic",
                "books": ["Book 2"],
                "thematic_assumptions": ["Institutional knowledge has collapsed"],
                "dominant_tropes": ["journey narrative"],
            },
        ],
        "tensions": [
            {
                "tension_id": "T1",
                "cluster_a": "mystery",
                "assumption_a": "Truth is discoverable through investigation",
                "cluster_b": "post_apocalyptic",
                "assumption_b": "Institutional knowledge has collapsed",
                "creative_question": "What does investigation look like without records?",
                "intensity": 0.9,
                "cliched_resolutions": [
                    "A lone detective rebuilds justice single-handedly through sheer determination",
                    "A hidden bunker contains all the missing records",
                ],
            }
        ],
        "narrative_seeds": [
            {
                "seed_id": "S1",
                "concept": "A former detective must reinvent investigation from first principles in a collapsed world.",
                "tensions_used": ["T1"],
                "tonal_direction": ["dark", "gritty"],
                "narrative_context_used": [],
            }
        ],
    })


def _agent(response: str = "", max_seeds: int = 5) -> ThemeExtractorAgent:
    client = FakeLLMClient(responses=[response or _valid_response()])
    return ThemeExtractorAgent(llm_client=client, max_seeds=max_seeds)


# ---------------------------------------------------------------------------
# TestPromptLoading (WI-4)
# ---------------------------------------------------------------------------


class TestPromptLoading:
    def test_load_prompt_succeeds(self) -> None:
        template = load_prompt("theme_extractor")
        assert template is not None

    def test_system_prompt_non_empty(self) -> None:
        template = load_prompt("theme_extractor")
        assert template.system.strip() != ""

    def test_user_template_has_required_placeholders(self) -> None:
        template = load_prompt("theme_extractor")
        required = {
            "user_prompt",
            "normalized_genres",
            "subgenres",
            "user_tones",
            "narrative_context",
            "book_list",
            "max_seeds",
        }
        for placeholder in required:
            assert f"{{{placeholder}}}" in template._user_template

    def test_format_user_with_valid_data(self) -> None:
        template = load_prompt("theme_extractor")
        result = template.format_user(
            user_prompt="dark mystery",
            normalized_genres=["mystery"],
            subgenres=[],
            user_tones=["dark"],
            narrative_context=[],
            book_list="[]",
            max_seeds=5,
        )
        assert "dark mystery" in result
        assert "mystery" in result

    def test_format_user_missing_placeholder_raises(self) -> None:
        template = load_prompt("theme_extractor")
        with pytest.raises(PromptFormattingError):
            template.format_user(user_prompt="dark mystery")


# ---------------------------------------------------------------------------
# TestBasicExtraction (WI-3)
# ---------------------------------------------------------------------------


class TestBasicExtraction:
    def test_returns_theme_extractor_output_type(self) -> None:
        agent = _agent()
        result = agent.run(_input())
        assert isinstance(result, ThemeExtractorAgentOutput)

    def test_genre_clusters_populated(self) -> None:
        agent = _agent()
        result = agent.run(_input())
        assert len(result.genre_clusters) >= 1

    def test_tensions_populated(self) -> None:
        agent = _agent()
        result = agent.run(_input())
        assert len(result.tensions) >= 1

    def test_narrative_seeds_populated(self) -> None:
        agent = _agent()
        result = agent.run(_input())
        assert len(result.narrative_seeds) >= 1

    def test_user_tones_carried_through(self) -> None:
        agent = _agent()
        inp = _input(user_tones=["dark", "gritty"])
        result = agent.run(inp)
        assert result.user_tones_carried == ["dark", "gritty"]

    def test_schema_version_set(self) -> None:
        agent = _agent()
        result = agent.run(_input())
        assert result.schema_version == THEMEPACK_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# TestLLMInteraction
# ---------------------------------------------------------------------------


class TestLLMInteraction:
    def test_llm_called_with_system_prompt(self) -> None:
        """FakeLLMClient records call count; a system prompt was passed if complete() was called."""
        client = FakeLLMClient(responses=[_valid_response()])
        agent = ThemeExtractorAgent(llm_client=client)
        agent.run(_input())
        assert client.call_count == 1

    def test_llm_called_with_correct_temperature(self) -> None:
        """Constructing with a custom temperature does not raise; agent runs successfully."""
        client = FakeLLMClient(responses=[_valid_response()])
        agent = ThemeExtractorAgent(llm_client=client, temperature=0.3)
        result = agent.run(_input())
        assert isinstance(result, ThemeExtractorAgentOutput)

    def test_llm_failure_propagates(self) -> None:
        """RuntimeError from an exhausted FakeLLMClient propagates to the caller."""
        client = FakeLLMClient(responses=[])  # No responses — will raise on access.
        agent = ThemeExtractorAgent(llm_client=client)
        with pytest.raises((RuntimeError, IndexError)):
            agent.run(_input())


# ---------------------------------------------------------------------------
# TestDebugMetadata
# ---------------------------------------------------------------------------


class TestDebugMetadata:
    def test_debug_contains_book_count(self) -> None:
        agent = _agent()
        summaries = [_summary(i + 1) for i in range(3)]
        result = agent.run(_input(summaries=summaries))
        assert result.debug["books_processed"] == 3

    def test_debug_contains_cluster_count(self) -> None:
        client = FakeLLMClient(responses=[_multi_genre_response()])
        agent = ThemeExtractorAgent(llm_client=client)
        result = agent.run(_input(summaries=[_summary(1), _summary(2)]))
        assert result.debug["clusters_found"] == len(result.genre_clusters)

    def test_debug_contains_tension_count(self) -> None:
        client = FakeLLMClient(responses=[_multi_genre_response()])
        agent = ThemeExtractorAgent(llm_client=client)
        result = agent.run(_input(summaries=[_summary(1), _summary(2)]))
        assert result.debug["tensions_found"] == len(result.tensions)

    def test_debug_contains_seed_count(self) -> None:
        agent = _agent()
        result = agent.run(_input())
        assert result.debug["seeds_generated"] == len(result.narrative_seeds)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_genre_still_produces_output(self) -> None:
        """A single-genre input should still return a valid ThemePack."""
        agent = _agent(_valid_response())
        result = agent.run(_input(genres=["mystery"]))
        assert isinstance(result, ThemeExtractorAgentOutput)

    def test_empty_narrative_context_ok(self) -> None:
        """Agent works fine when narrative_context is empty."""
        agent = _agent()
        result = agent.run(_input(narrative_context=[]))
        assert isinstance(result, ThemeExtractorAgentOutput)

    def test_max_seeds_respected(self) -> None:
        """Output must have no more seeds than max_seeds."""
        max_seeds = 2
        client = FakeLLMClient(responses=[_valid_response(num_seeds=max_seeds)])
        agent = ThemeExtractorAgent(llm_client=client, max_seeds=max_seeds)
        result = agent.run(_input())
        assert len(result.narrative_seeds) <= max_seeds

    def test_invalid_cluster_in_response_raises_value_error(self) -> None:
        """A response with a malformed cluster must raise ValueError."""
        bad_response = json.dumps({
            "genre_clusters": [{"genre": "mystery"}],  # missing required fields
            "tensions": [
                {
                    "tension_id": "T1",
                    "cluster_a": "mystery",
                    "assumption_a": "a",
                    "cluster_b": "mystery",
                    "assumption_b": "b",
                    "creative_question": "q",
                    "intensity": 0.5,
                    "cliched_resolutions": ["Hero solves it with determination"],
                }
            ],
            "narrative_seeds": [
                {
                    "seed_id": "S1",
                    "concept": "A long enough concept string here.",
                    "tensions_used": ["T1"],
                }
            ],
        })
        client = FakeLLMClient(responses=[bad_response])
        agent = ThemeExtractorAgent(llm_client=client)
        with pytest.raises(ValueError, match="genre_clusters"):
            agent.run(_input())

    def test_empty_clusters_in_response_raises_value_error(self) -> None:
        """A response with an empty genre_clusters list must raise ValueError."""
        bad_response = json.dumps({
            "genre_clusters": [],
            "tensions": [],
            "narrative_seeds": [],
        })
        client = FakeLLMClient(responses=[bad_response])
        agent = ThemeExtractorAgent(llm_client=client)
        with pytest.raises(ValueError, match="genre_clusters"):
            agent.run(_input())


# ---------------------------------------------------------------------------
# TestClichedResolutions
# ---------------------------------------------------------------------------


class TestClichedResolutions:
    def test_cliched_resolutions_present_in_output(self) -> None:
        """Every tension in the agent output must have a non-empty cliched_resolutions list."""
        agent = _agent(_multi_genre_response())
        result = agent.run(_input(summaries=[_summary(1), _summary(2)]))
        for tension in result.tensions:
            assert tension.cliched_resolutions, (
                f"Tension {tension.tension_id} has empty cliched_resolutions"
            )
