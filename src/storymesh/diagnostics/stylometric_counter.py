"""Stylometric counter for StoryMesh prose drafts.

Counts a small fixed set of voice tics in a prose draft. Output is
informational only — no pass/fail thresholds, no pipeline integration.
All detection is pattern-matching; false positives and negatives are expected.

Tic inventory:
  - cascading_which_was     : sentences with 2+ ", which (was|were|had|is|are)" clauses
  - proxy_feeling_simile    : "the way X... when/that/something" simile structures
  - negation_triplet        : paragraphs with 3+ sentences starting "Not " or
                              containing "not X, not Y"
  - sentence_fragment_rate  : fraction of paragraphs whose word count < 6
  - as_if_as_though         : occurrences of "as if" or "as though"
  - numerical_precision     : standalone declarative sentences containing a bare
                              numeral (digit or small integer word)
  - which_was_chain_max     : max depth of ", which (was|were|…)" in any sentence
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_WHICH_WAS = re.compile(r",\s*which\s+(?:was|were|had|is|are)\b", re.IGNORECASE)

_PROXY_SIMILE = re.compile(
    r"\bthe\s+way\s+(?:\w+\s+){0,4}(?:when|that|something)\b",
    re.IGNORECASE,
)

_NOT_SENTENCE_START = re.compile(r"(?:^|\.\s+)Not\b")
_NOT_X_NOT_Y = re.compile(r"\bnot\s+\w+,\s+not\s+\b", re.IGNORECASE)

_AS_IF_AS_THOUGH = re.compile(r"\bas\s+(?:if|though)\b", re.IGNORECASE)

_SMALL_INTEGERS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
}

# Sentence tokeniser: split on sentence-ending punctuation followed by whitespace or end.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


# ---------------------------------------------------------------------------
# Individual tic counters
# ---------------------------------------------------------------------------


def count_cascading_which_was(text: str) -> int:
    """Count sentences that contain two or more ', which (was|were|had|is|are)' clauses."""
    count = 0
    for sentence in _split_sentences(text):
        if len(_WHICH_WAS.findall(sentence)) >= 2:
            count += 1
    return count


def max_which_was_chain_depth(text: str) -> int:
    """Return the maximum number of ', which ...' clauses found in any single sentence."""
    max_depth = 0
    for sentence in _split_sentences(text):
        depth = len(_WHICH_WAS.findall(sentence))
        if depth > max_depth:
            max_depth = depth
    return max_depth


def count_proxy_feeling_simile(text: str) -> int:
    """Count 'the way X when/that/something' simile structures."""
    return len(_PROXY_SIMILE.findall(text))


def count_negation_triplet(text: str) -> int:
    """Count paragraphs containing 3+ 'not X, not Y' or 'Not …' sentence starts."""
    count = 0
    for para in _split_paragraphs(text):
        hits = len(_NOT_SENTENCE_START.findall(para)) + len(_NOT_X_NOT_Y.findall(para))
        if hits >= 3:
            count += 1
    return count


def sentence_fragment_paragraph_rate(text: str) -> float:
    """Return fraction of paragraphs whose word count is below 6."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return 0.0
    short = sum(1 for p in paragraphs if len(p.split()) < 6)
    return round(short / len(paragraphs), 4)


def count_as_if_as_though(text: str) -> int:
    """Count occurrences of 'as if' or 'as though'."""
    return len(_AS_IF_AS_THOUGH.findall(text))


def count_numerical_precision(text: str) -> int:
    """Count standalone declarative sentences containing a bare numeral or small integer word.

    Targets atmospheric numerals like "Three days." or "She had been waiting four hours."
    """
    count = 0
    for sentence in _split_sentences(text):
        if not sentence.endswith("."):
            continue
        words = sentence.lower().split()
        has_digit_word = any(re.search(r"\b\d+\b", w) for w in words)
        has_int_word = any(w in _SMALL_INTEGERS for w in words)
        if has_digit_word or has_int_word:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Aggregated counter
# ---------------------------------------------------------------------------


def count_tics(text: str) -> dict[str, Any]:
    """Count all tics in *text* and return a structured result dict.

    The result is suitable for serialisation as JSON. Per-1000-word rates
    are included for count-based tics; value-only entries are used for
    ratio/max metrics.

    Args:
        text: Raw prose draft string.

    Returns:
        Dict with ``word_count`` and ``tics`` mapping each tic name to its
        count and optional per_1000_words rate.
    """
    word_count = len(text.split()) if text.strip() else 0

    def _rate(n: int) -> float:
        if word_count == 0:
            return 0.0
        return round(n / word_count * 1000, 2)

    cascading = count_cascading_which_was(text)
    proxy = count_proxy_feeling_simile(text)
    negation = count_negation_triplet(text)
    as_if = count_as_if_as_though(text)
    numerical = count_numerical_precision(text)
    chain_max = max_which_was_chain_depth(text)
    frag_rate = sentence_fragment_paragraph_rate(text)

    return {
        "word_count": word_count,
        "tics": {
            "cascading_which_was": {"count": cascading, "per_1000_words": _rate(cascading)},
            "proxy_feeling_simile": {"count": proxy, "per_1000_words": _rate(proxy)},
            "negation_triplet": {"count": negation, "per_1000_words": _rate(negation)},
            "sentence_fragment_rate": {"value": frag_rate},
            "as_if_as_though": {"count": as_if, "per_1000_words": _rate(as_if)},
            "numerical_precision": {"count": numerical, "per_1000_words": _rate(numerical)},
            "which_was_chain_depth_max": {"value": chain_max},
        },
    }
