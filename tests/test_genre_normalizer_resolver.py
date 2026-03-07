"""
Unit tests for storymesh.agents.genre_normalizer.resolver
"""

from pathlib import Path

import orjson

from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.genre_normalizer.resolver import (
    ResolverResult,
    resolve_all,
    resolve_genres,
    resolve_llm,
    resolve_tones,
)
from storymesh.schemas.genre_normalizer import ResolutionMethod


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
    def test_stub_returns_all_unresolved(self) -> None:
        genres, tones, unresolved = resolve_llm(["xyzzy", "frob"])
        assert genres == []
        assert tones == []
        assert unresolved == ["xyzzy", "frob"]

    def test_stub_empty_input(self) -> None:
        genres, tones, unresolved = resolve_llm([])
        assert genres == []
        assert tones == []
        assert unresolved == []

# Full Pipeline — resolve_all

class TestResolveAll:
    def test_simple_genre_only(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all("fantasy", store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].canonical_genres == ["fantasy"]
        assert result.tone_resolutions == []
        assert result.unresolved_tokens == []

    def test_genre_with_tone_modifier(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all("optimistic fantasy", store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].canonical_genres == ["fantasy"]
        assert len(result.tone_resolutions) == 1
        assert "optimistic" in result.tone_resolutions[0].normalized_tones

    def test_complex_multi_genre_with_tone(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(
            "optimistic post-apocalyptic enemies to lovers mystery", store
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
        result = resolve_all("fantasy xyzzyfrob", store)
        assert len(result.genre_resolutions) == 1
        assert "xyzzyfrob" in result.unresolved_tokens

    def test_empty_input(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all("", store)
        assert result == ResolverResult()

    def test_whitespace_only_input(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all("   ", store)
        assert result == ResolverResult()

    def test_normalization_applied(self, tmp_path: Path) -> None:
        """Input with hyphens and mixed case should still resolve."""
        store = _make_store(tmp_path)
        result = resolve_all("Dark Fantasy", store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].input_token == "dark fantasy"

    def test_llm_fallback_disabled(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = resolve_all(
            "fantasy xyzzy",
            store,
            allow_llm_fallback=False,
        )
        assert len(result.genre_resolutions) == 1
        assert "xyzzy" in result.unresolved_tokens

    def test_all_passes_contribute(self, tmp_path: Path) -> None:
        """Genre from Pass 1, tone from Pass 2, unresolved from Pass 3 stub."""
        store = _make_store(tmp_path)
        result = resolve_all("gritty mystery xyzzy", store)
        assert len(result.genre_resolutions) == 1
        assert result.genre_resolutions[0].input_token == "mystery"
        assert len(result.tone_resolutions) == 1
        assert result.tone_resolutions[0].input_token == "gritty"
        assert result.unresolved_tokens == ["xyzzy"]

    def test_alias_resolution(self, tmp_path: Path) -> None:
        """'sci-fi' should resolve via the alternate on 'science fiction'."""
        store = _make_store(tmp_path)
        result = resolve_all("sci-fi", store)
        assert len(result.genre_resolutions) == 1
        assert "science_fiction" in result.genre_resolutions[0].canonical_genres
