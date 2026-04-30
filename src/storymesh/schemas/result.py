"""Public result type returned by the StoryMesh pipeline."""

from typing import Any

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """The public return type of ``StoryMeshPipeline.generate()``.

    ``final_synopsis`` is populated from ``StoryWriterAgentOutput.back_cover_summary``
    once Stage 6 is implemented; until then it carries a placeholder string.

    ``metadata`` includes ``user_prompt``, ``pipeline_version``, ``run_id``,
    ``stage_timings``, ``run_dir``, and — once book assembly is wired —
    ``story_pdf`` and ``story_epub`` paths.
    """

    final_synopsis: str = Field(
        ..., min_length=1, description="Back-cover summary of the generated story."
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Pipeline errors encountered during generation, if any.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
