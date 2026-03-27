"""
Unit tests for storymesh.agents.genre_normalizer.agent.
"""

from pathlib import Path

import orjson
import pytest

from storymesh.agents.genre_normalizer.agent import GenreNormalizerAgent
from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.exceptions import GenreResolutionError
from storymesh.schemas.genre_normalizer import (
    GenreNormalizerAgentInput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> Path:
    path.write_bytes(orjson.dumps(data))
    return path


def _test_genre_map() -> dict:
    return {
        "fantasy": {
            "alternates": [],
            "genres": ["fantasy"],
            "subgenres": [],
            "default_tones": ["wondrous", "adventurous", "epic"],
        },
        "dark fantasy": {
            "alternates": [],
            "genres": ["fantasy"],
            "subgenres": ["dark_fantasy"],
            "default_tones": ["dark", "ominous", "brooding"],
        },
        "science fiction": {
            "alternates": ["sci-fi", "sci fi", "scifi"],
            "genres": ["science_fiction"],
            "subgenres": [],
            "default_tones": ["speculative", "cerebral", "adventurous"],
        },
        "post-apocalyptic": {
            "alternates": ["post apocalyptic"],
            "genres": ["science_fiction"],
            "subgenres": ["post_apocalyptic"],
            "default_tones": ["bleak", "tense", "survivalist"],
        },
        "enemies to lovers": {
            "alternates": ["enemies-to-lovers"],
            "genres": ["romance"],
            "subgenres": ["enemies_to_lovers"],
            "default_tones": ["passionate", "conflicted", "slow-burn"],
        },
        "mystery": {
            "alternates": [],
            "genres": ["mystery"],
            "subgenres": [],
            "default_tones": ["suspenseful", "cerebral", "atmospheric"],
        },
        "romance": {
            "alternates": [],
            "genres": ["romance"],
            "subgenres": [],
            "default_tones": ["passionate", "warm", "emotional"],
        },
    }


def _test_tone_map() -> dict:
    return {
        "optimistic": {
            "alternates": [],
            "normalized_tones": ["optimistic", "hopeful"],
        },
        "gritty": {
            "alternates": [],
            "normalized_tones": ["gritty", "raw"],
        },
        "dark": {
            "alternates": [],
            "normalized_tones": ["dark", "grim"],
        },
    }


@pytest.fixture()
def store(tmp_path: Path) -> MappingStore:
    genre_path = _write_json(tmp_path / "genre_map.json", _test_genre_map())
    tone_path = _write_json(tmp_path / "tone_map.json", _test_tone_map())
    return MappingStore(genre_map_path=genre_path, tone_map_path=tone_path)


@pytest.fixture()
def agent(store: MappingStore) -> GenreNormalizerAgent:
    return GenreNormalizerAgent(store=store)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAgentConstruction:
    def test_construct_with_store(self, store: MappingStore) -> None:
        agent = GenreNormalizerAgent(store=store)
        assert agent is not None

    def test_construct_with_paths(self, tmp_path: Path) -> None:
        genre_path = _write_json(tmp_path / "genre_map.json", _test_genre_map())
        tone_path = _write_json(tmp_path / "tone_map.json", _test_tone_map())
        agent = GenreNormalizerAgent(
            genre_map_path=genre_path,
            tone_map_path=tone_path,
        )
        assert agent is not None

    def test_construct_with_custom_threshold(self, store: MappingStore) -> None:
        agent = GenreNormalizerAgent(store=store, fuzzy_threshold=0.90)
        assert agent is not None

    def test_construct_with_llm_client(self, store: MappingStore) -> None:
        from storymesh.llm.base import FakeLLMClient

        client = FakeLLMClient(responses=["{}"])
        agent = GenreNormalizerAgent(store=store, llm_client=client)
        assert agent is not None

    def test_construct_without_llm_client(self, store: MappingStore) -> None:
        agent = GenreNormalizerAgent(store=store)
        assert agent is not None


# ---------------------------------------------------------------------------
# Narrative Context tests
# ---------------------------------------------------------------------------

class TestNarrativeContext:
    def test_empty_when_all_tokens_resolved(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="fantasy"))
        assert result.debug["narrative_context"] == []

    def test_empty_when_llm_stub_active(self, agent: GenreNormalizerAgent) -> None:
        """Without a real LLM client, narrative context tokens remain in
        unresolved_tokens. Once the LLM is wired, tokens like '2085' and
        'rebellion' would move to narrative_context."""
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="fantasy about a rebellion in 2085",
        ))
        assert result.debug["narrative_context"] == []


# ---------------------------------------------------------------------------
# Genre Resolution
# ---------------------------------------------------------------------------

class TestGenreResolution:
    def test_single_genre(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="fantasy"))
        assert result.normalized_genres == ["fantasy"]

    def test_subgenre_entry(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="dark fantasy"))
        assert "fantasy" in result.normalized_genres
        assert "dark_fantasy" in result.subgenres

    def test_multiple_genres(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="fantasy mystery"))
        assert "fantasy" in result.normalized_genres
        assert "mystery" in result.normalized_genres
        assert len(result.debug["genre_resolutions"]) == 2

    def test_alias_resolution(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="sci-fi"))
        assert "science_fiction" in result.normalized_genres

    def test_multi_word_genre(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="enemies to lovers"))
        assert "romance" in result.normalized_genres
        assert "enemies_to_lovers" in result.subgenres


# ---------------------------------------------------------------------------
# Genre Deduplication
# ---------------------------------------------------------------------------

class TestGenreDeduplication:
    def test_duplicate_genres_deduplicated(self, agent: GenreNormalizerAgent) -> None:
        """'dark fantasy' and 'fantasy' both produce genre 'fantasy'."""
        result = agent.run(GenreNormalizerAgentInput(raw_genre="dark fantasy fantasy"))
        # Should not have 'fantasy' twice
        assert result.normalized_genres.count("fantasy") == 1


# ---------------------------------------------------------------------------
# Tone Integration
# ---------------------------------------------------------------------------

class TestToneIntegration:
    def test_genre_default_tones(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="fantasy"))
        assert result.debug["default_tones"] == ["wondrous", "adventurous", "epic"]
        assert result.user_tones == []
        assert result.tone_override is False

    def test_explicit_tone_override(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="optimistic post-apocalyptic",
        ))
        assert "optimistic" in result.user_tones
        assert result.tone_override is True
        assert result.override_note is not None

    def test_tone_conflict_detected(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="optimistic post-apocalyptic",
        ))
        assert result.tone_override is True
        assert result.debug["tone_conflicts"] is not None
        assert len(result.debug["tone_conflicts"]) > 0

    def test_tone_profile_ordering(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="gritty optimistic post-apocalyptic",
        ))
        # Explicit tones should come before defaults in profile
        profile = result.debug["tone_profile"]
        gritty_pos = profile.index("gritty")
        bleak_pos = profile.index("bleak")
        assert gritty_pos < bleak_pos


# ---------------------------------------------------------------------------
# Unresolved Tokens
# ---------------------------------------------------------------------------

class TestUnresolvedTokens:
    def test_unresolved_tokens_captured(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="fantasy xyzzyfrob",
        ))
        assert "xyzzyfrob" in result.debug["unresolved_tokens"]

    def test_narrative_context_preserved(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="mystery set in 2075 chicago",
        ))
        assert "mystery" in result.normalized_genres
        # Narrative tokens should be in unresolved
        assert "2075" in result.debug["unresolved_tokens"]
        assert "chicago" in result.debug["unresolved_tokens"]

    def test_no_unresolved_when_all_matched(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="fantasy"))
        assert result.debug["unresolved_tokens"] == []


# ---------------------------------------------------------------------------
# LLM Fallback Control
# ---------------------------------------------------------------------------

class TestLlmFallbackControl:
    def test_llm_fallback_disabled(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="fantasy xyzzy",
            allow_llm_fallback=False,
        ))
        assert "xyzzy" in result.debug["unresolved_tokens"]

    def test_llm_fallback_enabled_stub(self, agent: GenreNormalizerAgent) -> None:
        """With LLM stubbed, results should be identical to disabled."""
        enabled = agent.run(GenreNormalizerAgentInput(
            raw_genre="fantasy xyzzy",
            allow_llm_fallback=True,
        ))
        disabled = agent.run(GenreNormalizerAgentInput(
            raw_genre="fantasy xyzzy",
            allow_llm_fallback=False,
        ))
        assert enabled.debug["unresolved_tokens"] == disabled.debug["unresolved_tokens"]


# ---------------------------------------------------------------------------
# Full End-to-End Scenarios
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_optimistic_post_apocalyptic_enemies_to_lovers_mystery(
        self, agent: GenreNormalizerAgent,
    ) -> None:
        """The canonical test scenario from our design discussions."""
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="optimistic post-apocalyptic enemies to lovers mystery",
        ))

        # Genres
        assert "science_fiction" in result.normalized_genres
        assert "romance" in result.normalized_genres
        assert "mystery" in result.normalized_genres

        # Subgenres
        assert "post_apocalyptic" in result.subgenres
        assert "enemies_to_lovers" in result.subgenres

        # Tones
        assert "optimistic" in result.user_tones
        assert result.tone_override is True
        assert result.override_note is not None

        # Debug: resolutions
        assert len(result.debug["genre_resolutions"]) == 3
        assert len(result.debug["tone_resolutions"]) == 1
        assert result.debug["unresolved_tokens"] == []

        # Debug: audit trail
        genre_tokens = [r["input_token"] for r in result.debug["genre_resolutions"]]
        assert "post apocalyptic" in genre_tokens
        assert "enemies to lovers" in genre_tokens
        assert "mystery" in genre_tokens

    def test_simple_single_genre(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(raw_genre="romance"))
        assert result.normalized_genres == ["romance"]
        assert result.subgenres == []
        assert result.user_tones == []
        assert result.tone_override is False
        assert result.debug["effective_tone"] == "passionate"
        assert result.debug["unresolved_tokens"] == []

    def test_genre_with_narrative_context(self, agent: GenreNormalizerAgent) -> None:
        result = agent.run(GenreNormalizerAgentInput(
            raw_genre="gritty science fiction about a rebellion in 2085",
        ))
        assert "science_fiction" in result.normalized_genres
        assert "gritty" in result.user_tones
        assert "2085" in result.debug["unresolved_tokens"]
        assert "rebellion" in result.debug["unresolved_tokens"]


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unresolvable_input_raises_genre_resolution_error(
        self, agent: GenreNormalizerAgent,
    ) -> None:
        """Prompts with no recognizable genre keywords must raise GenreResolutionError,
        not a Pydantic ValidationError."""
        with pytest.raises(GenreResolutionError):
            agent.run(GenreNormalizerAgentInput(
                raw_genre="a world of warcraft paladin who marries his hammer",
            ))

    def test_genre_resolution_error_is_storymesh_error(
        self, agent: GenreNormalizerAgent,
    ) -> None:
        """GenreResolutionError must be catchable as a StoryMeshError."""
        from storymesh.exceptions import StoryMeshError

        with pytest.raises(StoryMeshError):
            agent.run(GenreNormalizerAgentInput(
                raw_genre="a world of warcraft paladin who marries his hammer",
            ))

    def test_error_no_llm_fallback_disabled(
        self, agent: GenreNormalizerAgent,
    ) -> None:
        """No LLM client + fallback disabled: message guides user to add genre keywords."""
        with pytest.raises(GenreResolutionError, match="genre keyword"):
            agent.run(GenreNormalizerAgentInput(
                raw_genre="a world of warcraft paladin who marries his hammer",
                allow_llm_fallback=False,
            ))

    def test_error_no_llm_fallback_enabled(
        self, agent: GenreNormalizerAgent,
    ) -> None:
        """No LLM client + fallback enabled: message says no LLM client is configured."""
        with pytest.raises(GenreResolutionError, match="No LLM client is configured"):
            agent.run(GenreNormalizerAgentInput(
                raw_genre="a world of warcraft paladin who marries his hammer",
                allow_llm_fallback=True,
            ))

    def test_error_llm_configured_fallback_disabled(
        self, store: MappingStore,
    ) -> None:
        """LLM configured + fallback disabled: message says to enable fallback."""
        import json
        from storymesh.llm.base import FakeLLMClient

        agent_with_llm = GenreNormalizerAgent(
            store=store, llm_client=FakeLLMClient(responses=[])
        )
        with pytest.raises(GenreResolutionError, match="LLM fallback is disabled"):
            agent_with_llm.run(GenreNormalizerAgentInput(
                raw_genre="a world of warcraft paladin who marries his hammer",
                allow_llm_fallback=False,
            ))

    def test_error_llm_configured_fallback_attempted(
        self, store: MappingStore,
    ) -> None:
        """LLM configured + fallback enabled but no genres returned: message says LLM was attempted."""
        import json
        from storymesh.llm.base import FakeLLMClient

        response = json.dumps({"classifications": [
            {"token": "world of warcraft paladin", "type": "narrative_context", "is_stopword": False},
        ]})
        agent_with_llm = GenreNormalizerAgent(
            store=store, llm_client=FakeLLMClient(responses=[response])
        )
        with pytest.raises(GenreResolutionError, match="LLM fallback was attempted"):
            agent_with_llm.run(GenreNormalizerAgentInput(
                raw_genre="a world of warcraft paladin who marries his hammer",
                allow_llm_fallback=True,
            ))