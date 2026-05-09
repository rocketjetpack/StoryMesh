"""Prompt style metadata for the kiosk picker.

The pipeline supports several prompt styles (subdirectories of
``src/storymesh/prompts/styles/``). Most are user-facing experiments worth
offering at the booth; ``test`` and ``slim`` are internal/dev-only and are
deliberately excluded from the picker.

Each curated entry carries a friendly display name and a one-line description
written for a non-engineer reading them on a touch screen.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from storymesh.kiosk.models import PromptStyleOption

# Style id pre-selected when the picker first renders. Matches the pipeline's
# default style so the recommended option mirrors "do nothing special."
_RECOMMENDED_ID = "default"

_STYLES_DIR = Path(__file__).resolve().parent.parent / "prompts" / "styles"

# Curated metadata. Anything not listed here is hidden from the kiosk picker.
_CURATED: dict[str, tuple[str, str]] = {
    "default": (
        "Classical",
        "The house style. Restrained, literary, balanced — what we'd give to a serious editor.",
    ),
    "bare_minimum": (
        "Open Hand",
        "Stripped-back instructions. The model improvises more freely; results are looser and more surprising.",
    ),
    "context_priming": (
        "Side Door",
        "Slips an unrelated fact into the prompt to nudge the model toward less obvious openings.",
    ),
    "verbalized_sampling": (
        "Hunt the Strange",
        "Asks the model to draft several candidates and pick the most unexpected. Best for original premises.",
    ),
}


@lru_cache(maxsize=1)
def load_prompt_style_options() -> list[PromptStyleOption]:
    """Return the curated styles that exist on disk, recommended first."""
    on_disk = {p.name for p in _STYLES_DIR.iterdir() if p.is_dir()}
    options: list[PromptStyleOption] = []
    for style_id, (name, description) in _CURATED.items():
        if style_id not in on_disk:
            continue
        options.append(
            PromptStyleOption(
                id=style_id,
                name=name,
                description=description,
                is_recommended=(style_id == _RECOMMENDED_ID),
            )
        )
    options.sort(key=lambda opt: (not opt.is_recommended, opt.name))
    return options


def valid_prompt_style_ids() -> set[str]:
    """Set of style ids accepted by the API."""
    return {opt.id for opt in load_prompt_style_options()}
