from storymesh.orchestration.pipeline import StoryMeshPipeline
from storymesh.schemas.result import GenerationResult
from storymesh.versioning.package import __version__

__all__ = ["generate_synopsis", "GenerationResult", "__version__"]

def generate_synopsis(genre: str) -> GenerationResult:
    """
    High-level API function to generate a synopsis based on the provided genre.
    
    :param genre: The genre for which to generate the synopsis.
    :type genre: str
    :return: A GenerationResult containing the generated synopsis and related metadata.
    :rtype: GenerationResult
    """
    pipeline = StoryMeshPipeline()
    return pipeline.generate(genre)