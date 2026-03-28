GENRE_CONSTRAINT_SCHEMA_VERSION = "3.0"
BOOK_FETCHER_SCHEMA_VERSION = "1.2"
BOOK_RANKER_SCHEMA_VERSION = "1.0"
PROPOSAL_SCHEMA_VERSION = "1.0"
THEMEPACK_SCHEMA_VERSION = "1.1"

SCHEMA_VERSIONS: dict[str, str] = {
    "Genre Constraint": GENRE_CONSTRAINT_SCHEMA_VERSION,
    "Book Fetcher": BOOK_FETCHER_SCHEMA_VERSION,
    "Book Ranker": BOOK_RANKER_SCHEMA_VERSION,
    "Proposal": PROPOSAL_SCHEMA_VERSION,
    "Themepack": THEMEPACK_SCHEMA_VERSION,
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
