"""
Unit tests for storymesh.agents.genre_normalizer.resolver
"""

import json
from pathlib import Path

import orjson
import pytest

from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.genre_normalizer.resolver import (
    ResolverResult,
    resolve_all,
    resolve_genres,
    resolve_holistic,
    resolve_llm,
    resolve_tones,
)
from storymesh.llm.base import FakeLLMClient
from storymesh.schemas.genre_normalizer import InferredGenre, ResolutionMethod


# Helpers
def _write_json(path: Path, data: object) -> Path:
    path.write_bytes(orjson.dumps(data))
    return path

def _minimal_genre_map() -> dict:
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
            "alternates": ["post apocalyptic", "postapocalyptic"],
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
        "cozy mystery": {
            "alternates": ["cosy mystery"],
            "genres": ["mystery"],
            "subgenres": ["cozy_mystery"],
            "default_tones": ["warm", "lighthearted", "witty"],
        },
    }

def _minimal_tone_map() -> dict:
    return {
        "optimistic": {
            "alternates": [],
            "normalized_tones": ["optimistic", "hopeful"],
        },
        "dark": {
            "alternates": [],
            "normalized_tones": ["dark", "grim"],
        },
        "lighthearted": {
            "alternates": ["light", "lite"],
            "normalized_tones": ["lighthearted", "breezy"],
        },
        "gritty": {
            "alternates": [],
            "normalized_tones": ["gritty", "raw"],
        },
        "darkly comic": {
            "alternates": ["darkly comedic"],
            "normalized_tones": ["darkly comic", "gallows humor"],
        },
    }

def _make_store(
        tmp_path: Path,
        genre_data: dict | None = None,
        tone_data: dict | None = None,
    ) -> MappingStore:
    """Build a MappingStore from minimal test data."""
    genre_path = _write_json(
        tmp_path / "genre_map.json",
        genre_data or _minimal_genre_map()
    )

    tone_path = _write_json(
        tmp_path / "tone_map.json",
        tone_data or _minimal_tone_map()
    )

    return MappingStore(genre_map_path=genre_path, tone_map_path=tone_path)

# Pass 1: Genre resolution
class TestResolveGenres:
    def test_single_genre_exact(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres(["fantasy"], store)
        assert(len(resolutions)) == 1
        assert resolutions[0].canonical_genres == ["fantasy"]
        assert resolutions[0].method == ResolutionMethod.STATIC_EXACT
        assert resolutions[0].confidence == 1.0
        assert leftovers == []

    def test_multi_word_genre(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres(
            ["dark", "fantasy"], store
        )
        assert len(resolutions) == 1
        assert "dark_fantasy" in resolutions[0].canonical_genres or \
            "dark_fantasy" in [sg for r in resolutions for sg in r.default_tones] or \
            resolutions[0].input_token == "dark fantasy"
        assert leftovers == []

    def test_greedy_longest_match_prefers_longer(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # 'dark fantasy' should match as a single n-gram of two words not 'dark' + 'fantasy'.
        resolutions, leftovers = resolve_genres(
            ["dark", "fantasy"], store
        )
        assert len(resolutions) == 1
        assert resolutions[0].input_token == "dark fantasy"

    def test_multiple_genres(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres(
            ["fantasy", "mystery"], store
        )
        tokens = [r.input_token for r in resolutions]
        assert len(resolutions) == 2
        assert "fantasy" in tokens
        assert "mystery" in tokens
        assert leftovers == []

    def test_three_word_genre(self, tmp_path: Path) -> None:
        # 'enemies to lovers' should match as a n-gram of 3 words
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres(
            ["enemies", "to", "lovers"], store
        )
        assert len(resolutions) == 1
        assert resolutions[0].input_token == "enemies to lovers"
        assert "romance" in resolutions[0].canonical_genres
        assert leftovers == []
        assert "enemies_to_lovers" in resolutions[0].subgenres

    def test_unmatched_genre_words_become_leftovers(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres(
            ["optimistic", "fantasy"], store
        )
        assert len(resolutions) == 1
        assert leftovers == ["optimistic"]

    def test_complex_input(self, tmp_path: Path) -> None:
        # Simulate a real user input
        store = _make_store(tmp_path)
        words = ["optimistic", "post", "apocalyptic", "enemies", "to", "lovers", "mystery"]
        resolutions, leftovers = resolve_genres(words, store)

        tokens = [r.input_token for r in resolutions]
        assert "post apocalyptic" in tokens
        assert "enemies to lovers" in tokens
        assert "mystery" in tokens
        assert len(resolutions) == 3
        assert leftovers == ["optimistic"]

    def test_empty_input(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres([], store)
        assert resolutions == []
        assert leftovers == []

    def test_all_unmatched(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_genres(
            ["abcd", "efgh", "ijkl"], store
        )
        assert resolutions == []
        assert leftovers == ["abcd", "efgh", "ijkl"]

# Fuzzy matching tests
class TestFuzzyGenreMatching:
    def test_fuzzy_match_on_close_misspelling(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # 'fantsy' should be a close enough match to 'fantasy'
        resolutions, leftovers = resolve_genres(
            ["fantsy"], store, fuzzy_threshold=0.80
        )
        assert len(resolutions) == 1
        assert resolutions[0].method == ResolutionMethod.STATIC_FUZZY
        assert resolutions[0].confidence > 0.0
        assert resolutions[0].confidence < 1.0

    def test_fuzzy_match_below_threshold_becomes_leftover(self, tmp_path: Path) -> None:
        # Matches below a threshold should become leftovers
        store = _make_store(tmp_path)
        # 'fantsy' should be a close enough match to 'fantasy'
        resolutions, leftovers = resolve_genres(
            ["fantsy"], store, fuzzy_threshold=0.99
        )
        assert len(resolutions) == 0
        assert leftovers == ["fantsy"]

    def test_fuzzy_match_multi_word(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # 'dark fantasty' should match to 'dark fantasy'
        resolutions, leftovers = resolve_genres(
            ["dark", "fantasty"], store, fuzzy_threshold=0.80
        )
        assert len(resolutions) == 1
        assert resolutions[0].method == ResolutionMethod.STATIC_FUZZY
        assert resolutions[0].input_token == "dark fantasty"
        assert resolutions[0].canonical_genres == ["fantasy"]

# Tone resolution tests
class TestResolveTones:
    def test_single_time(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_tones(["optimistic"], store)
        assert len(resolutions) == 1
        assert "optimistic" in resolutions[0].normalized_tones
        assert resolutions[0].is_override is True
        assert resolutions[0].method == ResolutionMethod.STATIC_EXACT
        assert leftovers == []

    def test_multi_word_tone(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_tones(
            ["darkly", "comic"], store
        )
        assert len(resolutions) == 1
        assert resolutions[0].input_token == "darkly comic"
        assert "darkly comic" in resolutions[0].normalized_tones

    def test_multiple_tones(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_tones(
            ["optimistic", "gritty"], store
        )
        assert len(resolutions) == 2
        tokens = [r.input_token for r in resolutions]
        assert "optimistic" in tokens
        assert "gritty" in tokens

    def test_unmatched_tones_become_leftovers(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_tones(
            ["optimistic", "xyzzy"], store
        )
        assert len(resolutions) == 1
        assert leftovers == ["xyzzy"]

    def test_empty_input(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        resolutions, leftovers = resolve_tones([], store)
        assert resolutions == []
        assert leftovers == []

# LLM resolution

class TestResolveLlm:
    def test_no_client_returns_all_resolved(self) -> None:
        prompted_text = "xxyzzy frobb glorp glunk"
        genres, tones, narrative_context, unresolved = resolve_llm(
            raw_input = prompted_text,
            resolved_genres = [],
            resolved_tones = [],
            remaining_text = prompted_text,
            llm_client = None
        )
        assert unresolved == prompted_text.split()

    def test_no_client_empty_input(self) -> None:
        genres, tones, narrative_context, unresolved = resolve_llm(
            raw_input = "fantasy",
            resolved_genres = ["fantasy"],
            resolved_tones = [],
            remaining_text = "",
            llm_client = None
        )
        assert genres == []
        assert tones == []
        assert narrative_context == []
        assert unresolved == []
        
# Full Pipeline — resolve_all

class TestResolveAll:
    def test_simple_genre_only(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="fantasy", store=store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].canonical_genres == ["fantasy"]
        assert result.tone_resolutions == []
        assert result.unresolved_tokens == []

    def test_genre_with_tone_modifier(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="optimistic fantasy", store=store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].canonical_genres == ["fantasy"]
        assert len(result.tone_resolutions) == 1
        assert "optimistic" in result.tone_resolutions[0].normalized_tones

    def test_complex_multi_genre_with_tone(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(
            raw_input="optimistic post-apocalyptic enemies to lovers mystery", 
            store=store
        )
        genre_tokens = [r.input_token for r in result.genre_resolutions]
        assert "post apocalyptic" in genre_tokens
        assert "enemies to lovers" in genre_tokens
        assert "mystery" in genre_tokens
        assert len(result.genre_resolutions) == 3

        tone_tokens = [r.input_token for r in result.tone_resolutions]
        assert "optimistic" in tone_tokens
        assert len(result.tone_resolutions) == 1

        assert result.unresolved_tokens == []

    def test_unresolved_tokens_pass_through(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="fantasy xyzzyfrob", store=store)
        assert len(result.genre_resolutions) == 1
        assert "xyzzyfrob" in result.unresolved_tokens

    def test_empty_input(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="", store=store)
        assert result == ResolverResult()

    def test_whitespace_only_input(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="   ", store=store)
        assert result == ResolverResult()

    def test_normalization_applied(self, tmp_path: Path) -> None:
        """Input with hyphens and mixed case should still resolve."""
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="Dark Fantasy", store=store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].input_token == "dark fantasy"

    def test_llm_fallback_disabled(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(
            raw_input="fantasy xyzzy",
            store=store,
            allow_llm_fallback=False,
        )
        assert len(result.genre_resolutions) == 1
        assert "xyzzy" in result.unresolved_tokens

    def test_all_passes_contribute(self, tmp_path: Path) -> None:
        """Genre from Pass 1, tone from Pass 2, unresolved from Pass 3 stub."""
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="gritty mystery xyzzy", store=store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].input_token == "mystery"
        assert len(result.tone_resolutions) == 1
        assert result.tone_resolutions[0].input_token == "gritty"
        assert result.unresolved_tokens == ["xyzzy"]

    def test_alias_resolution(self, tmp_path: Path) -> None:
        """'sci-fi' should resolve via the alternate on 'science fiction'."""
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="sci-fi", store=store)
        assert len(result.genre_resolutions) == 1
        assert "science_fiction" in result.genre_resolutions[0].canonical_genres




class TestResolveLlmWithClient:
    """Tests for resolve_llm() when an LLM client is provided."""

    def test_genre_classification(self) -> None:
        """LLM classifies a token as a genre."""
        response = json.dumps({"classifications": [
            {
                "token": "solarpunk",
                "type": "genre",
                "genres": ["science_fiction"],
                "subgenres": ["solarpunk"],
                "default_tones": ["optimistic", "hopeful"],
            },
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="gritty solarpunk",
            resolved_genres=[],
            resolved_tones=["gritty"],
            remaining_text="solarpunk",
            llm_client=client,
        )

        assert len(genres) == 1
        assert genres[0].input_token == "solarpunk"
        assert genres[0].canonical_genres == ["science_fiction"]
        assert genres[0].subgenres == ["solarpunk"]
        assert genres[0].default_tones == ["optimistic", "hopeful"]
        assert genres[0].method == ResolutionMethod.LLM_LIVE
        assert genres[0].confidence == 0.8
        assert tones == []
        assert narrative == []
        assert unresolved == []

    def test_tone_classification(self) -> None:
        """LLM classifies a token as a tone."""
        response = json.dumps({"classifications": [
            {
                "token": "moody",
                "type": "tone",
                "normalized_tones": ["moody", "brooding"],
            },
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="moody fantasy",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="moody",
            llm_client=client,
        )

        assert genres == []
        assert len(tones) == 1
        assert tones[0].input_token == "moody"
        assert tones[0].normalized_tones == ["moody", "brooding"]
        assert tones[0].method == ResolutionMethod.LLM_LIVE
        assert tones[0].confidence == 0.8
        assert tones[0].is_override is True

    def test_narrative_context_non_stopword(self) -> None:
        """LLM classifies a token as narrative context (not a stopword)."""
        response = json.dumps({"classifications": [
            {"token": "chicago", "type": "narrative_context", "is_stopword": False},
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="mystery chicago",
            resolved_genres=["mystery"],
            resolved_tones=[],
            remaining_text="chicago",
            llm_client=client,
        )

        assert narrative == ["chicago"]
        assert unresolved == []

    def test_narrative_context_stopword_dropped(self) -> None:
        """Stopword narrative context tokens are silently dropped."""
        response = json.dumps({"classifications": [
            {"token": "in", "type": "narrative_context", "is_stopword": True},
            {"token": "chicago", "type": "narrative_context", "is_stopword": False},
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="mystery in chicago",
            resolved_genres=["mystery"],
            resolved_tones=[],
            remaining_text="in chicago",
            llm_client=client,
        )

        assert narrative == ["chicago"]
        assert unresolved == []

    def test_unknown_classification(self) -> None:
        """LLM classifies a token as unknown."""
        response = json.dumps({"classifications": [
            {"token": "xyzzyfrob", "type": "unknown"},
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="fantasy xyzzyfrob",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="xyzzyfrob",
            llm_client=client,
        )

        assert unresolved == ["xyzzyfrob"]

    def test_mixed_classification_types(self) -> None:
        """LLM returns a mix of all four classification types."""
        response = json.dumps({"classifications": [
            {
                "token": "solarpunk",
                "type": "genre",
                "genres": ["science_fiction"],
                "subgenres": ["solarpunk"],
                "default_tones": ["optimistic"],
            },
            {"token": "moody", "type": "tone", "normalized_tones": ["moody"]},
            {"token": "in", "type": "narrative_context", "is_stopword": True},
            {"token": "2085", "type": "narrative_context", "is_stopword": False},
            {"token": "xyzzy", "type": "unknown"},
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="solarpunk moody in 2085 xyzzy",
            resolved_genres=[],
            resolved_tones=[],
            remaining_text="solarpunk moody in 2085 xyzzy",
            llm_client=client,
        )

        assert len(genres) == 1
        assert genres[0].input_token == "solarpunk"
        assert len(tones) == 1
        assert tones[0].input_token == "moody"
        assert narrative == ["2085"]
        assert unresolved == ["xyzzy"]

    def test_llm_call_exception_falls_back(self) -> None:
        """If the LLM call raises, all tokens become unresolved."""
        client = FakeLLMClient(responses=["this is not valid json at all {{{{"])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="fantasy solarpunk xyzzy",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="solarpunk xyzzy",
            llm_client=client,
        )

        assert genres == []
        assert tones == []
        assert narrative == []
        assert unresolved == ["solarpunk", "xyzzy"]

    def test_invalid_schema_falls_back(self) -> None:
        """Valid JSON that fails _ClassificationResponse validation."""
        response = json.dumps({"wrong_key": "no classifications here"})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="fantasy solarpunk",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="solarpunk",
            llm_client=client,
        )

        assert genres == []
        assert unresolved == ["solarpunk"]

    def test_genre_missing_genres_list_skipped(self) -> None:
        """A genre classification with empty genres list is skipped."""
        response = json.dumps({"classifications": [
            {"token": "solarpunk", "type": "genre", "genres": []},
            {"token": "2085", "type": "narrative_context", "is_stopword": False},
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="solarpunk 2085",
            resolved_genres=[],
            resolved_tones=[],
            remaining_text="solarpunk 2085",
            llm_client=client,
        )

        assert genres == []
        assert narrative == ["2085"]
        assert unresolved == ["solarpunk"]

    def test_tone_missing_tones_list_skipped(self) -> None:
        """A tone classification with empty normalized_tones is skipped."""
        response = json.dumps({"classifications": [
            {"token": "moody", "type": "tone", "normalized_tones": []},
        ]})
        client = FakeLLMClient(responses=[response])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="moody fantasy",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="moody",
            llm_client=client,
        )

        assert tones == []
        assert unresolved == ["moody"]

    def test_empty_remaining_text_skips_llm_call(self) -> None:
        """Empty remaining_text returns immediately without calling the LLM."""
        client = FakeLLMClient(responses=[])  # No responses — would error if called

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="fantasy",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="",
            llm_client=client,
        )

        assert genres == []
        assert tones == []
        assert narrative == []
        assert unresolved == []
        assert client.call_count == 0

    def test_whitespace_remaining_text_skips_llm_call(self) -> None:
        """Whitespace-only remaining_text returns immediately without calling the LLM."""
        client = FakeLLMClient(responses=[])

        genres, tones, narrative, unresolved = resolve_llm(
            raw_input="fantasy",
            resolved_genres=["fantasy"],
            resolved_tones=[],
            remaining_text="   ",
            llm_client=client,
        )

        assert genres == []
        assert tones == []
        assert narrative == []
        assert unresolved == []
        assert client.call_count == 0

    def test_resolved_tones_value_appears_in_prompt(self) -> None:
        """Regression: resolved_tones must inject the list value into the prompt,
        not the resolve_tones function object (which would appear as '<function ...>')."""
        captured_prompts: list[str] = []

        class CapturingClient(FakeLLMClient):
            def complete(
                self,
                prompt: str,
                *,
                system_prompt: str | None = None,
                temperature: float,
                max_tokens: int,
            ) -> str:
                captured_prompts.append(prompt)
                return super().complete(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        response = json.dumps({"classifications": [
            {"token": "chicago", "type": "narrative_context", "is_stopword": False},
        ]})
        client = CapturingClient(responses=[response])

        resolve_llm(
            raw_input="mystery chicago",
            resolved_genres=["mystery"],
            resolved_tones=["suspenseful"],
            remaining_text="chicago",
            llm_client=client,
        )

        assert len(captured_prompts) == 1
        assert "suspenseful" in captured_prompts[0]
        assert "<function" not in captured_prompts[0]

# ---------------------------------------------------------------------------
# TestResolveHolistic (Pass 4)
# ---------------------------------------------------------------------------


def _holistic_response(genres: list[dict] | None = None) -> str:
    return json.dumps({"inferred_genres": genres or []})


def _one_inference(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "canonical_genre": "science_fiction",
        "subgenres": ["techno_thriller"],
        "default_tones": ["cerebral", "tense"],
        "rationale": "Programmer optimizing code implies a technology-forward sci-fi sensibility.",
        "confidence": 0.75,
    }
    return {**defaults, **overrides}


class TestResolveHolistic:
    def test_no_client_returns_empty(self) -> None:
        result = resolve_holistic(
            raw_input="a programmer saves one millisecond",
            resolved_genres=["thriller"],
            resolved_subgenres=[],
            resolved_tones=["optimistic"],
            narrative_context=["programmer", "millisecond"],
            llm_client=None,
        )
        assert result == []

    def test_valid_response_parsed_correctly(self) -> None:
        client = FakeLLMClient(responses=[_holistic_response([_one_inference()])])
        result = resolve_holistic(
            raw_input="techno optimistic thriller about a c++ programmer",
            resolved_genres=["thriller"],
            resolved_subgenres=[],
            resolved_tones=["optimistic"],
            narrative_context=["programmer"],
            llm_client=client,
        )
        assert len(result) == 1
        assert isinstance(result[0], InferredGenre)
        assert result[0].canonical_genre == "science_fiction"
        assert result[0].subgenres == ["techno_thriller"]
        assert result[0].confidence == pytest.approx(0.75)

    def test_deduplication_removes_already_resolved_genre(self) -> None:
        # The LLM incorrectly infers "thriller" which is already resolved.
        bad_inference = _one_inference(canonical_genre="thriller")
        client = FakeLLMClient(responses=[_holistic_response([bad_inference])])
        result = resolve_holistic(
            raw_input="thriller about a programmer",
            resolved_genres=["thriller"],
            resolved_subgenres=[],
            resolved_tones=[],
            narrative_context=["programmer"],
            llm_client=client,
        )
        assert result == []

    def test_llm_failure_returns_empty(self) -> None:
        client = FakeLLMClient(responses=["not valid json {{{{"])
        result = resolve_holistic(
            raw_input="mystery in 1920s harlem",
            resolved_genres=["mystery"],
            resolved_subgenres=[],
            resolved_tones=[],
            narrative_context=["1920s", "harlem"],
            llm_client=client,
        )
        assert result == []

    def test_malformed_schema_returns_empty(self) -> None:
        # Valid JSON but wrong shape — missing "inferred_genres" key.
        client = FakeLLMClient(responses=[json.dumps({"classifications": []})])
        result = resolve_holistic(
            raw_input="mystery",
            resolved_genres=["mystery"],
            resolved_subgenres=[],
            resolved_tones=[],
            narrative_context=[],
            llm_client=client,
        )
        # Missing required field defaults via default_factory — valid but empty.
        assert result == []

    def test_empty_inference_list_is_valid(self) -> None:
        client = FakeLLMClient(responses=[_holistic_response([])])
        result = resolve_holistic(
            raw_input="dark fantasy adventure in a medieval kingdom",
            resolved_genres=["fantasy", "adventure"],
            resolved_subgenres=[],
            resolved_tones=["dark"],
            narrative_context=["medieval", "kingdom"],
            llm_client=client,
        )
        assert result == []


# ---------------------------------------------------------------------------
# TestResolveAllPass4 (integration with resolve_all)
# ---------------------------------------------------------------------------


def _pass4_response(genres: list[dict] | None = None) -> str:
    return json.dumps({"inferred_genres": genres or []})


class TestResolveAllPass4:
    def test_pass4_populates_inferred_genres(self, tmp_path: Path) -> None:
        """resolve_all() result includes inferred_genres from Pass 4."""
        store = _make_store(tmp_path)
        # "mystery" resolves in Pass 1; no leftovers so Pass 3 is skipped.
        # Pass 4 should run and infer historical_fiction.
        client = FakeLLMClient(responses=[_pass4_response([
            {
                "canonical_genre": "historical_fiction",
                "subgenres": ["historical_mystery"],
                "default_tones": ["atmospheric"],
                "rationale": "1920s Harlem setting implies historical fiction.",
                "confidence": 0.85,
            }
        ])])
        result = resolve_all(
            raw_input="mystery",
            store=store,
            llm_client=client,
        )
        assert len(result.inferred_genres) == 1
        assert result.inferred_genres[0].canonical_genre == "historical_fiction"

    def test_pass4_skipped_when_llm_fallback_disabled(self, tmp_path: Path) -> None:
        """allow_llm_fallback=False must skip Pass 4 entirely."""
        store = _make_store(tmp_path)
        # Client would raise if called — ensures it is never called.
        client = FakeLLMClient(responses=[])
        result = resolve_all(
            raw_input="mystery",
            store=store,
            allow_llm_fallback=False,
            llm_client=client,
        )
        assert result.inferred_genres == []
        assert client.call_count == 0

    def test_pass4_skipped_when_no_llm_client(self, tmp_path: Path) -> None:
        """No LLM client — inferred_genres is empty, no crash."""
        store = _make_store(tmp_path)
        result = resolve_all(raw_input="mystery", store=store, llm_client=None)
        assert result.inferred_genres == []
