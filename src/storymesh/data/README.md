# StoryMesh Static Mapping Files

These files exist to provide local normalization for a pre-selected, standard
set of genres that are widely used. This list is certainly not exhaustive, and
the GenreNormalizerAgent will (when flagged) pass unrecognized genres to a LLM
for mapping.

## Generation
These files were generated with Claude.ai Opus 4.6 after providing the Pydantic
structures for the appropriate fields and a prompt to populate the files for a
list of genres commonly recognized by organizations like the New York Times.