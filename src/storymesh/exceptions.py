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


class RunNotFoundError(StoryMeshError):
    """Raised when a requested run ID does not exist in the runs directory."""


class LLMOutputTruncatedError(StoryMeshError):
    """Raised when the LLM stops because the max_tokens budget was reached.

    Carries the partial response text and the budget that was exhausted so
    that ``complete_json()`` can escalate the budget and retry automatically.

    Attributes:
        partial_response: The incomplete text returned before truncation.
        token_budget: The ``max_tokens`` value that proved insufficient.
    """

    def __init__(self, *, partial_response: str, token_budget: int) -> None:
        super().__init__(
            f"LLM output truncated at {token_budget} tokens. "
            "Retry with a larger max_tokens budget."
        )
        self.partial_response = partial_response
        self.token_budget = token_budget
