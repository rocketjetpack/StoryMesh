"""
Genre and tone resolution logic for GenreNormalizerAgent.

Implements a three-pass resolution pipeline:
1. Greedy longest-match against the genre index
2. Greedy longest-match against the tone index
3. Fallback to LLM-based resolution for any remaining unmatched text
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.genre_normalizer.normalize import normalize_text
from storymesh.llm.base import LLMClient
from storymesh.schemas.genre_normalizer import (
    GenreMapEntry,
    GenreResolution,
    ResolutionMethod,
    ToneMapEntry,
    ToneResolution,
)


# Resolver result class
@dataclass(frozen=True)
class ResolverResult:
    """
    Bundle all outputs from the three resolution passes.
    This is an intermediate class and does not require a Pydantic model.
    """
    genre_resolutions: list[GenreResolution] = field(default_factory=list)
    tone_resolutions: list[ToneResolution] = field(default_factory=list)
    narrative_context: list[str] = field(default_factory=list)
    unresolved_tokens: list[str] = field(default_factory=list)

# Greedy, longest-match logic
def _greedy_longest_match(
    words: list[str],
    index: Mapping[str, GenreMapEntry | ToneMapEntry],
    max_ngram: int,
    fuzzy_threshold: float = 0.85,
    ) -> tuple[list[tuple[str, GenreMapEntry | ToneMapEntry, ResolutionMethod, float]], list[str]]:
    
    """
    Starting from each position in the input word list, try n-grams from longest (capped at max_ngram)
    down to one n-gram. For each n-gram, try an exact match first, then a fuzzy match. On a hit,
    remove those words from the word list and advance.

    Args:
        words: Normalized input list split into words
        index: Flat index lookup (genre or tone)
        max_ngram: Maximum n-gram length to try
        fuzzy_threshold: Minimum rapidfuzz confidence to accept as a match

    Returns:
        A tuple of (matches, leftover_words)
        Each match is (input_token, entry, method, confidence)
        leftover_words are tokens that failed to match via any method
    """

    matches: list[tuple[str, GenreMapEntry | ToneMapEntry, ResolutionMethod, float]] = []
    leftovers: list[str] = []
    consumed: set[int] = set()

    # Pre-group the index keys by word count for more efficient matching
    keys_by_length: dict[int, list[str]] = {}
    for key in index:
        word_count = len(key.split())
        keys_by_length.setdefault(word_count, []).append(key)

    pos = 0
    while pos < len(words):
        if pos in consumed:
            pos += 1
            continue

        matched = False
        window_max = min(max_ngram, len(words) - pos)

        for width in range(window_max, 0, -1):
            ngram = " ".join(words[pos : pos + width])

            # Exact match attempt
            if ngram in index:
                matches.append(
                    (ngram, index[ngram], ResolutionMethod.STATIC_EXACT, 1.0)
                )
                consumed.update(range(pos, pos + width))
                matched = True
                break
            
            # Fuzzy match
            candidates = keys_by_length.get(width)
            if candidates:
                result = process.extractOne(
                    ngram,
                    candidates,
                    scorer = fuzz.ratio,
                    score_cutoff = int(fuzzy_threshold*100.0),
                )

                if result is not None:
                    matched_key, score, _ = result
                    matches.append(
                        (ngram, index[matched_key], ResolutionMethod.STATIC_FUZZY, round(score / 100.0, 4))
                    )
                    consumed.update(range(pos, pos + width))
                    matched = True
                    break

            # FUTURE: LLM Pass

        # Leftovers
        if not matched:
            leftovers.append(words[pos])
            consumed.add(pos)

        pos += 1

    return matches, leftovers

# Pass 1 - Genre resolution
def resolve_genres(
        words: list[str],
        store: MappingStore,
        fuzzy_threshold: float = 0.85
    ) -> tuple[list[GenreResolution], list[str]]:
    """ Pass 1: Resolve genres.

    Args:
        words: Normalized input split into words
        store: Loaded mapping store with a genre index
        fuzzy_threshold: Minimum rapidfuzz score to qualify as a match

    Returns:
        A tuple of (genre_resolutions, leftover_words)
    """

    matches, leftovers = _greedy_longest_match(
        words = words,
        index = store.genre_index,
        max_ngram = store.genre_max_words,
        fuzzy_threshold = fuzzy_threshold
    )

    resolutions = [
        GenreResolution(
            input_token = token,
            canonical_genres = entry.genres,
            default_tones = entry.default_tones,
            subgenres = entry.subgenres,
            method = method,
            confidence = confidence
        ) for token, entry, method, confidence in matches if isinstance(entry, GenreMapEntry)
    ]

    return resolutions, leftovers

# Pass 2 - Tone resolution
def resolve_tones(
        words: list[str],
        store: MappingStore,
        fuzzy_threshold: float = 0.85
    ) -> tuple[list[ToneResolution], list[str]]:
    """ Pass 2: Resolve tones.

    Args:
        words: Leftover words from Pass 1
        store: Loaded mapping store with tone index
        fuzzy_threshold: Minimum rapidfuzz score to qualify as a match

    Returns:
        A tuple of (tone_resolutions, leftover_words)
    """

    matches, leftovers = _greedy_longest_match(
        words = words,
        index = store.tone_index,
        max_ngram = store.tone_max_words,
        fuzzy_threshold = fuzzy_threshold
    )

    resolutions = [
        ToneResolution(
            input_token = token,
            normalized_tones = entry.normalized_tones,
            method = method,
            confidence = confidence
        ) for token, entry, method, confidence in matches if isinstance(entry, ToneMapEntry)
    ]

    return resolutions, leftovers

# Pass 3: LLM fallback (not implemented)
def resolve_llm(
        *,
        raw_input: str,
        resolved_genres: list[str],
        resolved_tones: list[str],
        remaining_text: str,
        llm_client: LLMClient | None = None
    ) -> tuple[list[GenreResolution], list[ToneResolution], list[str], list[str]]:
    """Pass 3: LLM fallback for leftovers.

    Classifies unresolved tokens by asking an LLM to categorize each as
    a genre, tone, narrative context, or unknown.

    Args:
        raw_input: The original user-provided string (for context).
        resolved_genres: Genre names already resolved in Passes 1 and 2.
        resolved_tones: Tone names already resolved in Passes 1 and 2.
        remaining_text: Leftover tokens rejoined as a string.
        llm_client: An LLMClient instance. If None, stub behavior is used.

    Returns:
        A tuple of (genre_resolutions, tone_resolutions, narrative_context, unresolved_tokens)
    """

    if llm_client is None:
        return [], [], [], remaining_text.split()

    return [], [], [], remaining_text.split()

# Full resolution pipeline
def resolve_all(
        *,
        raw_input: str,
        store: MappingStore,
        fuzzy_threshold: float = 0.85,
        allow_llm_fallback: bool = True,
        llm_client: LLMClient | None = None
    ) -> ResolverResult:
    """ Run the full pipeline with all three passes.

    Args:
        raw_input: The user provided raw string
        store: A mapping store
        fuzzy_threshold:  Minimum rapidfuzz score to qualify as a match
        allow_llm_fallback: Controls if LLM fallback (pass 3) is enabled or skipped
    """

    normalized = normalize_text(raw_input)
    words = normalized.split()

    if not words:
        return ResolverResult()

    # Genre resolution
    genre_resolutions, leftover_after_genres = resolve_genres(
        words = words,
        store = store,
        fuzzy_threshold = fuzzy_threshold
    )

    # Tone resolution
    tone_resolutions, leftover_after_tones = resolve_tones(
        words = leftover_after_genres,
        store = store,
        fuzzy_threshold = fuzzy_threshold
    )

    # LLM fallback
    if allow_llm_fallback and leftover_after_tones:
        resolved_genre_names = [
            genre for g in genre_resolutions
                  for genre in g.canonical_genres
        ]
        
        resolved_tone_names = [
            tone for t in tone_resolutions
                for tone in t.normalized_tones
        ]

        remaining_text = " ".join(leftover_after_tones)

        llm_genres, llm_tones, narrative_context, unresolved = resolve_llm(
            raw_input = raw_input,
            resolved_genres = resolved_genre_names,
            resolved_tones = resolved_tone_names,
            remaining_text = remaining_text,
            llm_client = llm_client
        )

        genre_resolutions = genre_resolutions + llm_genres
        tone_resolutions = tone_resolutions + llm_tones
    else:
        narrative_context = []
        unresolved = leftover_after_tones

    return ResolverResult(
        genre_resolutions = genre_resolutions,
        tone_resolutions = tone_resolutions,
        narrative_context = narrative_context,
        unresolved_tokens = unresolved
    )