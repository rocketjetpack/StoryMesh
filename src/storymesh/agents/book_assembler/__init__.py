"""BookAssemblerAgent — Stage 8 of the StoryMesh pipeline.

Assembles the StoryWriterAgent output and CoverArtAgent output into
formatted PDF and EPUB deliverables. No LLM is required — this stage
is purely deterministic rendering.

Optional dependencies:
    PDF:  weasyprint>=62.0  (pip install storymesh[pdf])
    EPUB: ebooklib>=0.18    (pip install storymesh[pdf])
"""
