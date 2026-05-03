GENRE_CONSTRAINT_SCHEMA_VERSION = "3.1"
BOOK_FETCHER_SCHEMA_VERSION = "1.2"
BOOK_RANKER_SCHEMA_VERSION = "1.1"
PROPOSAL_SCHEMA_VERSION = "1.3"
PROPOSAL_READER_SCHEMA_VERSION = "1.0"
THEMEPACK_SCHEMA_VERSION = "1.1"
RUBRIC_SCHEMA_VERSION = "2.0"
COVER_ART_SCHEMA_VERSION = "1.1"
STORY_WRITER_SCHEMA_VERSION = "1.1"
BOOK_ASSEMBLER_SCHEMA_VERSION = "1.0"
RESONANCE_REVIEWER_SCHEMA_VERSION = "1.0"
VOICE_PROFILE_SCHEMA_VERSION = "1.0"
VOICE_PROFILE_SELECTOR_SCHEMA_VERSION = "1.0"

SCHEMA_VERSIONS: dict[str, str] = {
    "Genre Constraint": GENRE_CONSTRAINT_SCHEMA_VERSION,
    "Book Fetcher": BOOK_FETCHER_SCHEMA_VERSION,
    "Book Ranker": BOOK_RANKER_SCHEMA_VERSION,
    "Proposal": PROPOSAL_SCHEMA_VERSION,
    "Proposal Reader": PROPOSAL_READER_SCHEMA_VERSION,
    "Themepack": THEMEPACK_SCHEMA_VERSION,
    "Rubric": RUBRIC_SCHEMA_VERSION,
    "Cover Art": COVER_ART_SCHEMA_VERSION,
    "Story Writer": STORY_WRITER_SCHEMA_VERSION,
    "Book Assembler": BOOK_ASSEMBLER_SCHEMA_VERSION,
    "Resonance Reviewer": RESONANCE_REVIEWER_SCHEMA_VERSION,
    "Voice Profile": VOICE_PROFILE_SCHEMA_VERSION,
    "Voice Profile Selector": VOICE_PROFILE_SELECTOR_SCHEMA_VERSION,
}

# Version History
# 2026-03-16: Increment Genre Constraint Schema version to 1.2 due to addition
#             of 'narrative_context' component to track tokens/phrases the LLM
#             classification process marks as narrative context.
# 2026-03-21: Increment Genre Constraint schema to 2.0 to account for signifcant
#             changes to the schema to better separate debug information from
#             actual fields that are consumable by downstream agents.
# 2026-03-26: Increment Book Fetcher schema to 1.1. Renamed BookRecord.source_query
#             (str) to source_genres (list[str]) to support deduplication within the
#             agent. Added BookFetcherAgentOutput.debug dict for per-run audit data.
# 2026-03-27: Increment Book Fetcher schema to 1.2. Added readinglog_count,
#             want_to_read_count, already_read_count, currently_reading_count,
#             and number_of_pages_median fields to BookRecord for richer
#             downstream ranking signals.
# 2026-03-27: Add Book Ranker schema 1.0. Introduces ScoreBreakdown, RankedBook,
#             RankedBookSummary, BookRankerAgentInput, and BookRankerAgentOutput.
# 2026-03-28: Increment Genre Constraint schema to 3.0. Promoted
#             narrative_context from debug dict to a top-level field
#             so downstream agents (ThemeExtractorAgent) can consume it
#             as part of the typed contract.
# 2026-03-28: Increment ThemePack schema to 1.1. Added cliched_resolutions
#             (required, min_length=1) to ThematicTension. Downstream agents
#             ProposalDraft and RubricJudge use this list as exclusions and
#             evaluation criteria respectively.
# 2026-03-31: Increment Genre Constraint schema to 3.1. Added LLM_INFERRED
#             ResolutionMethod, InferredGenre model, and inferred_genres field
#             to GenreNormalizerAgentOutput for Pass 4 holistic genre inference.
#             Additive change (default_factory=list); existing consumers unaffected.
# 2026-03-31: Increment Book Ranker schema to 1.1. Replaced select_with_diversity()
#             (source_genres Jaccard, diversity_weight param) with select_diverse()
#             (Open Library subject-tag Jaccard, mmr_lambda + mmr_candidates params).
#             Debug dict now nests MMR metadata under a "mmr" key.
# 2026-04-14: Increment Proposal schema to 1.1. Introduced full schema:
#             ProposalDraftAgentInput, StoryProposal, SelectionRationale,
#             ProposalDraftAgentOutput. Multi-sample architecture with
#             seed-steering and self-selection.
# 2026-04-24: Add Rubric schema 1.0. Introduces RubricJudgeAgentInput,
#             DimensionResult, and RubricJudgeAgentOutput. Six-dimension
#             craft quality rubric with cliché violation tracking and
#             configurable pass threshold.
# 2026-04-29: Increment Proposal schema to 1.2. Added image_prompt (str, min_length=30)
#             to StoryProposal. ProposalDraftAgent generates this field alongside the
#             narrative fields; CoverArtAgent consumes it directly. Breaking change —
#             existing artifacts lack the field and will not deserialize.
# 2026-04-29: Add Cover Art schema 1.0. Introduces CoverArtAgentInput and
#             CoverArtAgentOutput. CoverArtAgent wraps DALL-E 3 image generation;
#             image_path points to the PNG saved in the run artifact directory.
# 2026-04-29: Increment Cover Art schema to 1.1. Switched image provider from
#             dall-e-3 to gpt-image-2. Removed image_style field (gpt-image-2 does
#             not support a style parameter). Quality values updated from
#             'standard'/'hd' to 'low'/'medium'/'high'/'auto'.
# 2026-04-30: Add Story Writer schema 1.0. Introduces StoryWriterAgentInput,
#             SceneOutline, and StoryWriterAgentOutput. Replaces the scaffolded
#             SynopsisWriterAgent placeholder. Produces back_cover_summary,
#             scene_list (6-10 SceneOutline objects), and full_draft prose
#             separated by SCENE_BREAK delimiters for book assembly.
# 2026-05-01: Increment Story Writer schema to 1.1. Renamed thematic_function
#             to narrative_pressure in SceneOutline (describes what pressure the
#             scene is under, not what it means). Added observational_anchor
#             (str, min_length=5) — a concrete physical/sensory detail the scene
#             can return to. Breaking change — existing artifacts will not
#             deserialize.
# 2026-05-01: Increment Rubric schema to 2.0. Breaking change — switched from
#             float (0.0-1.0) Likert scoring to int (0/1/2) three-tier scoring:
#             0=fail, 1=acceptable, 2=strong. Composite is now sum of tier scores
#             (max 10) instead of weighted average. Renamed convention_departure
#             dimension to story_serving_choices. Removed convention_departures
#             list field (analysis folded into D-2 feedback). pass_threshold and
#             composite_score changed from float to int.
# 2026-05-01: Increment Proposal schema to 1.3. Added unknowns (list[str],
#             default_factory=list) to StoryProposal. Optional field — holds
#             unresolved questions the story keeps open. Empty when the story
#             does not benefit from explicit unknowns. Non-breaking additive change.
# 2026-04-30: Add Book Assembler schema 1.0. Introduces BookAssemblerAgentInput
#             and BookAssemblerAgentOutput. Renders story_writer_output and
#             cover_art into PDF (WeasyPrint) and EPUB (ebooklib) deliverables.
#             pdf_path and epub_path are absolute paths in the run directory;
#             either is an empty string when the format was not generated.
# 2026-05-01: Add Proposal Reader schema 1.0. Introduces ProposalReaderFeedback,
#             ProposalReaderAgentInput, and ProposalReaderAgentOutput. Runs on
#             the retry path between RubricJudgeAgent and ProposalDraftAgent.
#             Cross-provider (GPT-4o) reader-perspective evaluation with five
#             non-technical fields: what_engaged_me, what_fell_flat,
#             protagonist_gap, premise_question, reader_direction.
# 2026-05-01: Add Resonance Reviewer schema 1.0. Introduces NearMissMoment,
#             ResonanceReviewerAgentInput, and ResonanceReviewerAgentOutput.
#             Reviews completed prose drafts for near-miss moments (places
#             where the story implies depth but retreats before engaging) and
#             produces targeted expansions. Cross-provider review pass
#             identifies 0-3 moments; revision pass expands avoidance moments
#             in-place. Quality-gated: only runs at high/very_high presets.
