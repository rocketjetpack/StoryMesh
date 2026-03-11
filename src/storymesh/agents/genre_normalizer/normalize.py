"""
Input normalization utilities for the GenreNormalizerAgent.

This module provides text preprocessing that is applied to both the genre and tone map
JSON files in the src/data/ directory. This preprocessing is applied to keys and values
in the JSON files as well as user input during genre and tone mapping steps.

Consistency is critical here. If the same normalization is not applied to both the JSON
files and user input then lookups will fail and the agent will be unable to correctly
map user input to appropriate genres and tones.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """
    Normalize text by:
      - Converting to lowercase
      - Replacing hyphens and underscores with spaces
      - Collapsing multiple whitespace characters into a single space
      - Stripping leading and trailing whitespace

    Args:
        text: The raw string to be normalized.

    Returns:
        The normalized string.
    """

    text = text.lower()
    matches_list = str.maketrans({
        "-": " ",
        "_": " ",
        ",": " ",
        "+": " ",
        "&": " and ",
        "/": " ",
        "\"": " ",
        "(": " ",
        ")": " ",
        ":": " ",
        ";": " ",
        ".": " ",
        "!": " ",
        "?": " ",
    })
    text = text.translate(matches_list)
    text = re.sub(r"\s+", " ", text) # Collapse multiple whitespace characters into a single space
    text = text.strip()
    return text