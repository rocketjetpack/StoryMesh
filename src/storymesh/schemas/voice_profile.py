"""Pydantic schema for VoiceProfile and the load_voice_profile loader.

A VoiceProfile is a named, immutable record that conditions prose generation
across the StoryMesh pipeline. Three built-in profiles are shipped with the
package; additional profiles can be added by placing YAML files in
src/storymesh/prompts/voice_profiles/.

Profiles are loaded once per pipeline run by VoiceProfileSelectorAgent, stored
in StoryMeshState, and consumed by StoryWriterAgent (all three passes) and
ResonanceReviewerAgent (revision pass).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from storymesh.versioning.schemas import VOICE_PROFILE_SCHEMA_VERSION

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "prompts" / "voice_profiles"

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

BUILT_IN_PROFILE_IDS: tuple[str, ...] = (
    "literary_restraint",
    "cozy_warmth",
    "genre_active",
)


class VoiceProfile(BaseModel):
    """A named voice profile that conditions prose generation.

    Loaded from src/storymesh/prompts/voice_profiles/<id>.yaml.
    Carried through StoryMeshState and consumed by prose-stage agents.

    The overlay fields (craft_overlay, avoid_overlay, summary_overlay) are
    injected into prompt placeholders at runtime. Empty strings produce a
    prompt byte-identical to the default literary_restraint behavior, which
    is intentional for backward compatibility.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1, description="Lowercase snake_case profile identifier.")
    description: str = Field(min_length=10, description="Human-readable profile description.")
    tone_keywords: list[str] = Field(min_length=1, description="Tone signals that route to this profile.")
    genre_keywords: list[str] = Field(
        default_factory=list,
        description="Genre signals that route to this profile.",
    )

    craft_overlay: str = Field(
        default="",
        description=(
            "Additional craft principles injected into story_writer_draft.yaml. "
            "Empty string preserves default literary_restraint behavior."
        ),
    )
    avoid_overlay: str = Field(
        default="",
        description=(
            "Additional avoid items injected into story_writer_draft.yaml. "
            "Empty string preserves default literary_restraint behavior."
        ),
    )
    exemplars: list[str] = Field(
        min_length=2,
        description="2–4 verbatim opens_with sentences in this voice, used as few-shot examples.",
    )
    summary_overlay: str = Field(
        default="",
        description=(
            "Additional register instructions injected into story_writer_summary.yaml. "
            "Empty string preserves default behavior."
        ),
    )

    schema_version: str = VOICE_PROFILE_SCHEMA_VERSION

    @field_validator("id")
    @classmethod
    def id_must_be_snake_case(cls, v: str) -> str:
        """Enforce lowercase snake_case for profile IDs."""
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"Voice profile id must be lowercase snake_case, got: {v!r}"
            )
        return v


def load_voice_profile(profile_id: str) -> VoiceProfile:
    """Load a VoiceProfile from its YAML data file.

    Reads from src/storymesh/prompts/voice_profiles/<profile_id>.yaml and
    validates the result against the VoiceProfile schema.

    Args:
        profile_id: The snake_case profile identifier, e.g. "cozy_warmth".

    Returns:
        A validated, frozen VoiceProfile instance.

    Raises:
        FileNotFoundError: If no YAML file exists for the given profile_id.
        ValueError: If the YAML is malformed or fails schema validation.
    """
    if not _ID_PATTERN.match(profile_id):
        raise FileNotFoundError(
            f"Voice profile not found: {profile_id!r}. "
            f"Profile IDs must be lowercase snake_case."
        )

    path = _PROFILES_DIR / f"{profile_id}.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"Voice profile not found: {profile_id!r}. "
            f"Expected file at: {path}"
        )

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in voice profile {path}: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        )

    try:
        return VoiceProfile(**data)
    except Exception as exc:
        raise ValueError(
            f"Voice profile {path} failed schema validation: {exc}"
        ) from exc
