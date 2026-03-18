GENRE_CONSTRAINT_SCHEMA_VERSION = "1.2"
PROPOSAL_SCHEMA_VERSION = "1.0"
THEMEPACK_SCHEMA_VERSION = "1.0"

SCHEMA_VERSIONS: dict[str, str] = {
    "Genre Constraint": GENRE_CONSTRAINT_SCHEMA_VERSION,
    "Proposal": PROPOSAL_SCHEMA_VERSION,
    "Themepack": THEMEPACK_SCHEMA_VERSION,
}

# Version History
# 2026-03-16: Increment Genre Constraint Schema version to 1.2 due to addition
#             of 'narrative_context' component to track tokens/phrases the LLM
#             classification process marks as narrative context.
