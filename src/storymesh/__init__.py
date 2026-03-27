from storymesh.orchestration.pipeline import StoryMeshPipeline
from storymesh.schemas.result import GenerationResult
from storymesh.versioning.package import __version__

__all__ = ["generate_synopsis", "GenerationResult", "__version__"]

def generate_synopsis(user_prompt: str) -> GenerationResult:
    """
    High-level API function to generate a fiction synopsis from the given prompt.

    :param user_prompt: Free-text description of the desired fiction (genres, tones,
        setting, time period, narrative concepts, etc.).
    :type user_prompt: str
    :return: A GenerationResult containing the generated synopsis and related metadata.
    :rtype: GenerationResult
    """
    pipeline = StoryMeshPipeline()
    return pipeline.generate(user_prompt)