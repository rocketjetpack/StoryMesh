"""
Domain exceptions for the StoryMesh pipeline.

All pipeline errors inherit from StoryMeshError so callers can catch the
entire family with a single except clause when needed.
"""


class StoryMeshError(Exception):
    """Base class for all StoryMesh pipeline errors."""


class GenreResolutionError(StoryMeshError):
    """Raised when no genres can be resolved from the user's input.

    This is a terminal condition: without at least one resolved genre the
    downstream pipeline has no meaningful work to perform.
    """
