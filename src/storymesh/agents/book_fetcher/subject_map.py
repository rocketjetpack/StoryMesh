"""Genre-to-subject mapping for the Open Library Search API.

Canonical genre names from the GenreNormalizerAgent (snake_case, e.g.
``science_fiction``) do not always match the subject strings that Open Library
indexes (e.g. ``"science fiction"``).  This module provides a single
translation step between the two.

The mapping is loaded from ``src/storymesh/data/genre_subject_map.json`` once
at import time.  Repeated calls to ``resolve_subjects()`` incur no I/O cost.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level map cache
# ---------------------------------------------------------------------------

def _load_map() -> dict[str, list[str]]:
    """Load genre_subject_map.json from the package data directory."""
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "genre_subject_map.json"
    with open(data_path) as f:
        raw: dict[str, list[str]] = json.load(f)
    # Strip the documentation comment key if present.
    raw.pop("_comment", None)
    return raw


_GENRE_SUBJECT_MAP: dict[str, list[str]] = _load_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_subjects(genres: list[str]) -> list[str]:
    """Map canonical genre names to Open Library subject query strings.

    For each genre:

    - **In the map with entries**: uses the first subject string (ordered
      best-first; extras are documented for future use).
    - **In the map with an empty list**: silently dropped.  These genres have
      no reliable Open Library subject equivalent.
    - **Not in the map**: passed through with underscores replaced by spaces
      as a best-effort fallback (e.g. ``dark_fantasy`` → ``"dark fantasy"``).

    The result is deduplicated preserving first-occurrence order.

    Args:
        genres: Canonical genre names (snake_case) from the
            ``GenreNormalizerAgent`` output or ``InferredGenre`` objects.

    Returns:
        Deduplicated list of Open Library subject query strings ready to pass
        to ``BookFetcherAgentInput.normalized_genres``.
    """
    seen: set[str] = set()
    result: list[str] = []

    for genre in genres:
        if genre in _GENRE_SUBJECT_MAP:
            subjects = _GENRE_SUBJECT_MAP[genre]
            if not subjects:
                logger.debug(
                    "resolve_subjects: '%s' maps to an empty subject list — skipping.",
                    genre,
                )
                continue
            subject = subjects[0]
        else:
            subject = genre.replace("_", " ")
            logger.debug(
                "resolve_subjects: '%s' not in genre_subject_map — using fallback '%s'.",
                genre,
                subject,
            )

        if subject not in seen:
            seen.add(subject)
            result.append(subject)

    return result
