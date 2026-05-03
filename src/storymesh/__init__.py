from storymesh.orchestration.pipeline import (
    StoryMeshPipeline,
    regenerate_book_assembler,
    regenerate_cover_art,
)
from storymesh.schemas.result import GenerationResult
from storymesh.versioning.package import __version__

__all__ = [
    "generate_synopsis",
    "regenerate_book_assembler",
    "regenerate_cover_art",
    "GenerationResult",
    "__version__",
]

def generate_synopsis(
    user_prompt: str,
    *,
    pass_threshold: int | None = None,
    max_retries: int | None = None,
    min_retries: int = 0,
    skip_resonance_review: bool = True,
) -> GenerationResult:
    """High-level API function to generate a fiction synopsis from the given prompt.

    :param user_prompt: Free-text description of the desired fiction (genres, tones,
        setting, time period, narrative concepts, etc.).
    :param pass_threshold: Override rubric pass threshold. ``None`` uses the config default.
    :param max_retries: Override rubric retry budget. ``None`` uses the default (2).
    :param min_retries: Minimum editorial revision cycles before a passing proposal
        can proceed. Default 0.
    :param skip_resonance_review: When True (default), skip the resonance reviewer
        stage. Set to False by ``high`` and ``very_high`` quality presets.
    :return: A GenerationResult containing the generated synopsis and related metadata.
    """
    pipeline = StoryMeshPipeline(
        pass_threshold=pass_threshold,
        max_retries=max_retries,
        min_retries=min_retries,
        skip_resonance_review=skip_resonance_review,
    )
    return pipeline.generate(user_prompt)