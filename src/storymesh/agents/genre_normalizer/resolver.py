"""
Genre and tone resolution logic for GenreNormalizerAgent.

Implements a three-pass resolution pipeline:
1. Greedy longest-match against the genre index
2. Greedy longest-match against the tone index
3. Fallback to LLM-based resolution for any remaining unmatched text
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from storymesh.agents.genre_normalizer.loader import MappingStore
from storymesh.agents.genre_normalizer.normalize import normalize_text
from storymesh.llm.base import LLMClient
from storymesh.schemas.genre_normalizer import (
    GenreMapEntry,
    GenreResolution,
    InferredGenre,
    ResolutionMethod,
    ToneMapEntry,
    ToneResolution,
)

logger= logging.getLogger(__name__)

# Resolver result class
@dataclass(frozen=True)
class ResolverResult:
    """Bundle all outputs from the four resolution passes.

    This is an intermediate class and does not require a Pydantic model.
    """

    genre_resolutions: list[GenreResolution] = field(default_factory=list)
    tone_resolutions: list[ToneResolution] = field(default_factory=list)
    narrative_context: list[str] = field(default_factory=list)
    unresolved_tokens: list[str] = field(default_factory=list)
    inferred_genres: list[InferredGenre] = field(default_factory=list)

# Private model for parsing LLM classification responses
class _Classification(BaseModel):
    """
    A single classification entry returned by the LLM during resolution.

    All type-specific fields default to empty/false so the model parses regardless of which
    fields the LLM response actually includes. Validation that the correct fields are populated
    for a given type happens during the conversion to GenreResolution or ToneResolution.
    """
    model_config = { "frozen": True }
    token: str = Field(min_length=1, description="The original token or merged phrase from the input.")
    type: Literal["genre", "tone", "narrative_context", "unknown"]
    genres: list[str] = Field(default_factory=list)
    subgenres: list[str] = Field(default_factory=list)
    default_tones: list[str] = Field(default_factory=list)
    normalized_tones: list[str] = Field(default_factory=list)
    is_stopword: bool = False

class _ClassificationResponse(BaseModel):
    """
    Top level wrapper for the LLM classiciation response.

    Validates that the response contains a 'classifications' list. Individual entries that fail
    validation against the _Classification model are rejected automatically.
    """
    model_config = { "frozen": True }
    classifications: list[_Classification] = Field(default_factory=list)


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
_LLM_LIVE_CONFIDENCE = 0.8

def resolve_llm(
        *,
        raw_input: str,
        resolved_genres: list[str],
        resolved_tones: list[str],
        remaining_text: str,
        llm_client: LLMClient | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
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

    fallback_unresolved = remaining_text.split() if remaining_text else []

    if llm_client is None:
        return [], [], [], remaining_text.split()

    if not remaining_text.strip():
        return [], [], [], remaining_text.split()

    # Load and format the prompt
    from storymesh.prompts.loader import load_prompt  # noqa: PLC0415

    prompt_template = load_prompt("genre_normalizer")
    user_prompt = prompt_template.format_user(
        raw_input = raw_input,
        resolved_genres = resolved_genres,
        resolved_tones = resolved_tones,
        remaining_text = remaining_text
    )

    # Call the actual LLM
    try:
        raw_response = llm_client.complete_json(
            user_prompt,
            system_prompt = prompt_template.system,
            temperature = temperature,
            max_tokens = max_tokens
        )
    except Exception:
        logger.warning(
            "LLM classification call failed: treating all tokens as unresolved.",
            exc_info = True
        )
        return [], [], [], fallback_unresolved
    
    # Validate the response
    try:
        parsed = _ClassificationResponse.model_validate(raw_response)
    except Exception:
        logger.warning(
            "LLM response failed schema validation. Parsed data: %s",
            raw_response,
            exc_info=True,
        )
        return [], [], [], fallback_unresolved
    
    # Convert validated classifications
    genre_resolutions: list[GenreResolution] = []
    tone_resolutions: list[ToneResolution] = []
    narrative_context: list[str] = []
    unresolved: list[str] = []

    for item in parsed.classifications:
        if item.type == "genre":
            if not item.genres:
                logger.warning("Genre classification for '%s' has no genres.", item.token)
                unresolved.append(item.token)
                continue
            genre_resolutions.append(GenreResolution(
                input_token = item.token,
                canonical_genres = item.genres,
                subgenres = item.subgenres,
                default_tones = item.default_tones,
                method = ResolutionMethod.LLM_LIVE,
                confidence = _LLM_LIVE_CONFIDENCE
            ))
        elif item.type == "tone":
            if not item.normalized_tones:
                logger.warning("Tone classification for '%s' has no tones.", item.token)
                unresolved.append(item.token)
                continue
            tone_resolutions.append(ToneResolution(
                input_token = item.token,
                normalized_tones = item.normalized_tones,
                method=ResolutionMethod.LLM_LIVE,
                confidence = _LLM_LIVE_CONFIDENCE,
                is_override = True
            ))
        elif item.type == "narrative_context":
            if not item.is_stopword:
                narrative_context.append(item.token)
        elif item.type == "unknown":
            unresolved.append(item.token)

    # Detect situations where the LLM returns no usable classifications and treat all
    # tokens as unresolved.

    if not genre_resolutions and not tone_resolutions and not narrative_context and not unresolved:
        return [], [], [], fallback_unresolved

    return genre_resolutions, tone_resolutions, narrative_context, unresolved

# Pass 4: Holistic genre inference
_LLM_INFERRED_CONFIDENCE = 0.7


class _InferredGenreItem(BaseModel):
    """A single inferred genre entry from the Pass 4 LLM response."""

    model_config = {"frozen": True}

    canonical_genre: str = Field(min_length=1)
    subgenres: list[str] = Field(default_factory=list)
    default_tones: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=_LLM_INFERRED_CONFIDENCE)


class _HolisticInferenceResponse(BaseModel):
    """Top-level wrapper for the Pass 4 LLM response."""

    model_config = {"frozen": True}

    inferred_genres: list[_InferredGenreItem] = Field(default_factory=list)


def resolve_holistic(
    *,
    raw_input: str,
    resolved_genres: list[str],
    resolved_subgenres: list[str],
    resolved_tones: list[str],
    narrative_context: list[str],
    llm_client: LLMClient | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> list[InferredGenre]:
    """Pass 4: Holistic genre inference from the full user prompt.

    Asks the LLM to identify genres that the prompt *implies* through its
    overall themes, setting, and narrative direction — genres that no single
    token names explicitly.  Each inference includes a rationale string that
    records the textual evidence used.

    This pass is purely additive and best-effort.  A failed LLM call or a
    response that fails validation returns an empty list without crashing the
    pipeline.  No retries are made.

    Args:
        raw_input: The original user-provided string.
        resolved_genres: Canonical genre names resolved in Passes 1–3.
        resolved_subgenres: Subgenre names resolved in Passes 1–3.
        resolved_tones: Tone names resolved in Passes 1–3.
        narrative_context: Narrative context tokens from Pass 3.
        llm_client: LLM client to use. Returns ``[]`` immediately when ``None``.
        temperature: Sampling temperature for the LLM call.
        max_tokens: Token budget for the LLM response.

    Returns:
        List of ``InferredGenre`` objects, deduplicated against
        ``resolved_genres``.  May be empty.
    """
    if llm_client is None:
        return []

    from storymesh.prompts.loader import load_prompt  # noqa: PLC0415

    prompt_template = load_prompt("genre_inference")
    user_prompt = prompt_template.format_user(
        raw_input=raw_input,
        resolved_genres=resolved_genres,
        resolved_subgenres=resolved_subgenres,
        resolved_tones=resolved_tones,
        narrative_context=narrative_context,
    )

    try:
        raw_response = llm_client.complete_json(
            user_prompt,
            system_prompt=prompt_template.system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        logger.warning(
            "Pass 4 holistic inference LLM call failed: returning empty inferred_genres.",
            exc_info=True,
        )
        return []

    try:
        parsed = _HolisticInferenceResponse.model_validate(raw_response)
    except Exception:
        logger.warning(
            "Pass 4 LLM response failed schema validation. Parsed data: %s",
            raw_response,
            exc_info=True,
        )
        return []

    resolved_set = set(resolved_genres)
    results: list[InferredGenre] = []
    for item in parsed.inferred_genres:
        if item.canonical_genre in resolved_set:
            logger.debug(
                "Pass 4: discarding '%s' — already resolved in Passes 1–3.",
                item.canonical_genre,
            )
            continue
        results.append(InferredGenre(
            canonical_genre=item.canonical_genre,
            subgenres=item.subgenres,
            default_tones=item.default_tones,
            rationale=item.rationale,
            confidence=item.confidence,
        ))

    return results


# Full resolution pipeline
def resolve_all(
        *,
        raw_input: str,
        store: MappingStore,
        fuzzy_threshold: float = 0.85,
        allow_llm_fallback: bool = True,
        llm_client: LLMClient | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
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

    # Pass 3: LLM fallback for unresolved tokens
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
            raw_input=raw_input,
            resolved_genres=resolved_genre_names,
            resolved_tones=resolved_tone_names,
            remaining_text=remaining_text,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        genre_resolutions = genre_resolutions + llm_genres
        tone_resolutions = tone_resolutions + llm_tones
    else:
        narrative_context = []
        unresolved = leftover_after_tones

    # Pass 4: Holistic genre inference from full prompt context.
    # Runs only when an LLM client is configured and fallback is enabled.
    # Returns [] gracefully on any failure — never crashes the pipeline.
    if allow_llm_fallback and llm_client is not None:
        resolved_genre_names_p4 = [
            genre for g in genre_resolutions
                  for genre in g.canonical_genres
        ]
        resolved_subgenre_names_p4 = [
            sg for g in genre_resolutions
               for sg in g.subgenres
        ]
        resolved_tone_names_p4 = [
            tone for t in tone_resolutions
                 for tone in t.normalized_tones
        ]
        inferred = resolve_holistic(
            raw_input=raw_input,
            resolved_genres=resolved_genre_names_p4,
            resolved_subgenres=resolved_subgenre_names_p4,
            resolved_tones=resolved_tone_names_p4,
            narrative_context=narrative_context,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        inferred = []

    return ResolverResult(
        genre_resolutions=genre_resolutions,
        tone_resolutions=tone_resolutions,
        narrative_context=narrative_context,
        unresolved_tokens=unresolved,
        inferred_genres=inferred,
    )