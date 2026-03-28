"""Pydantic schemas for the ThemeExtractorAgent.

Defines the input and output contracts for Stage 3 of the StoryMesh pipeline.
The ThemeExtractorAgent identifies thematic assumptions in each genre tradition,
finds contradictions between traditions, and frames those contradictions as
creative questions for story generation. The structured output (the ThemePack)
enables the downstream ProposalDraftAgent to develop narratives that a single
LLM call cannot produce.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from storymesh.schemas.book_ranker import RankedBookSummary
from storymesh.versioning.schemas import THEMEPACK_SCHEMA_VERSION


class ThemeExtractorAgentInput(BaseModel):
    """Input contract for the ThemeExtractorAgent.

    Assembled by the node wrapper from GenreNormalizerAgentOutput,
    BookRankerAgentOutput, and pipeline state. The agent itself has
    no knowledge of the pipeline.
    """

    ranked_summaries: list[RankedBookSummary] = Field(
        min_length=1,
        description="Slim ranked book summaries from BookRankerAgent.",
    )
    normalized_genres: list[str] = Field(
        min_length=1,
        description="Canonical genre names from GenreNormalizerAgent.",
    )
    subgenres: list[str] = Field(
        default_factory=list,
        description="Subgenre names from GenreNormalizerAgent.",
    )
    user_tones: list[str] = Field(
        default_factory=list,
        description="User-specified tone words from GenreNormalizerAgent.",
    )
    tone_override: bool = Field(
        default=False,
        description="Whether user tones diverge from genre default tones.",
    )
    narrative_context: list[str] = Field(
        default_factory=list,
        description=(
            "Narrative tokens (settings, time periods, character archetypes) "
            "from GenreNormalizerAgent. These anchor the creative output "
            "in the user's specific vision."
        ),
    )
    user_prompt: str = Field(
        min_length=1,
        description="Original raw user input string.",
    )


class GenreCluster(BaseModel):
    """A group of books belonging to a single genre tradition.

    Captures the thematic assumptions and dominant tropes that the
    LLM identifies as characteristic of this genre tradition, based
    on the books in the cluster.
    """

    model_config = {"frozen": True}

    genre: str = Field(
        min_length=1,
        description="Canonical genre name (e.g., 'mystery', 'post_apocalyptic').",
    )
    books: list[str] = Field(
        min_length=1,
        description="Titles of books grouped into this genre cluster.",
    )
    thematic_assumptions: list[str] = Field(
        min_length=1,
        description=(
            "Core assumptions this genre tradition takes for granted "
            "(e.g., 'truth is discoverable', 'justice is achievable')."
        ),
    )
    dominant_tropes: list[str] = Field(
        default_factory=list,
        description="Common narrative devices in this genre tradition.",
    )


class ThematicTension(BaseModel):
    """A creative tension between two genre traditions.

    Defined by two opposing assumptions drawn from different genre
    clusters, plus a creative question that frames the tension as
    something a story could explore.
    """

    model_config = {"frozen": True}

    tension_id: str = Field(
        min_length=1,
        description="Short identifier (e.g., 'T1', 'T2').",
    )
    cluster_a: str = Field(
        description="Genre label of the first cluster.",
    )
    assumption_a: str = Field(
        description="The assumption from cluster A that creates the tension.",
    )
    cluster_b: str = Field(
        description="Genre label of the second cluster.",
    )
    assumption_b: str = Field(
        description="The opposing assumption from cluster B.",
    )
    creative_question: str = Field(
        min_length=1,
        description=(
            "The generative question this tension poses for a story "
            "(e.g., 'What does justice look like when there is no one left to enforce it?')."
        ),
    )
    intensity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How fundamentally the assumptions conflict. High intensity "
            "(near 1.0) means a deep thematic contradiction; low intensity "
            "means a stylistic or surface-level difference."
        ),
    )
    cliched_resolutions: list[str] = Field(
        min_length=1,
        description=(
            "Predictable, tropey narrative resolutions to this tension that "
            "the LLM identifies as clichéd. Downstream agents use these as "
            "exclusions (ProposalDraft) and evaluation criteria (RubricJudge)."
        ),
    )


class NarrativeSeed(BaseModel):
    """A concrete story kernel that emerges from one or more thematic tensions.

    Bridges theme extraction and proposal drafting: ProposalDraftAgent
    receives these seeds as starting points rather than re-interpreting
    raw tensions. This keeps single-responsibility intact — ThemeExtractor
    identifies creative potential, ProposalDraft develops it.
    """

    model_config = {"frozen": True}

    seed_id: str = Field(
        min_length=1,
        description="Short identifier (e.g., 'S1', 'S2').",
    )
    concept: str = Field(
        min_length=10,
        description="A 2–3 sentence story kernel describing the core premise.",
    )
    tensions_used: list[str] = Field(
        min_length=1,
        description="Which tension_ids feed this seed.",
    )
    tonal_direction: list[str] = Field(
        default_factory=list,
        description="Tones this seed leans into (from user tones or genre defaults).",
    )
    narrative_context_used: list[str] = Field(
        default_factory=list,
        description=(
            "Which user narrative context tokens (settings, time periods, etc.) "
            "this seed incorporates."
        ),
    )


class ThemeExtractorAgentOutput(BaseModel):
    """Output contract for the ThemeExtractorAgent (Stage 3).

    The ThemePack captures the dialectical structure of genre collisions:
    genre clusters with their thematic assumptions, tensions between those
    assumptions, and narrative seeds that emerge from the tensions. This
    structured intermediate representation is what enables the pipeline to
    produce creative output that a single LLM call cannot.
    """

    model_config = {"frozen": True}

    genre_clusters: list[GenreCluster] = Field(
        min_length=1,
        description="Books grouped by genre tradition with identified thematic assumptions.",
    )
    tensions: list[ThematicTension] = Field(
        min_length=1,
        description="Creative tensions between genre traditions.",
    )
    narrative_seeds: list[NarrativeSeed] = Field(
        min_length=1,
        description=(
            "Concrete story kernels (3–5) that emerge from the tensions. "
            "ProposalDraftAgent selects and develops the best seed."
        ),
    )
    user_tones_carried: list[str] = Field(
        default_factory=list,
        description="User tones passed through for downstream agents.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extraction metadata: number of books processed, genre distribution, "
            "LLM call details, prompt token counts."
        ),
    )
    schema_version: str = THEMEPACK_SCHEMA_VERSION
