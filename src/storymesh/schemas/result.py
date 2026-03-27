from typing import Any

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    final_synopsis: str = Field(
        ..., min_length=1, description="The final generated synopsis."
        )
    scores: dict[str, float] = Field(
        default_factory=dict, description="A dictionary of scores for each item."
        )
    similarity_risk: dict[str, Any] = Field(
        default_factory=dict, description="A dictionary of similarity risk values."
        )
    errors: list[str] = Field(
        default_factory=list,
        description="Pipeline errors encountered during generation, if any.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "frozen": True
    }
