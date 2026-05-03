"""BookAssemblerAgent — Stage 8 of the StoryMesh pipeline.

Produces a formatted PDF and EPUB from the completed story draft. Two
optional dependencies drive the rendering:

    PDF:  WeasyPrint (HTML/CSS → PDF via libcairo/libpango)
    EPUB: ebooklib  (programmatic EPUB3 construction)

Both are lazy-imported so that a missing library degrades gracefully:
the corresponding format is skipped with a warning rather than crashing
the pipeline. Install both with ``pip install storymesh[pdf]``.

Rendering pipeline:
1. Split ``full_draft`` on ``SCENE_BREAK`` to recover per-scene prose.
2. Zip scene prose with ``scene_list`` titles from the outline pass.
3. Build an HTML document (cover page, title page, synopsis, scenes).
4. Render to PDF via WeasyPrint, or produce a typographic cover if no
   cover image is available.
5. Build an EPUB3 document with the same structure via ebooklib.
6. Return raw bytes for both formats to the node wrapper for persistence.
"""

from __future__ import annotations

import logging
import os
import tempfile
from base64 import b64encode
from dataclasses import dataclass, field
from html import escape as _escape
from pathlib import Path
from typing import Any

from storymesh.schemas.book_assembler import BookAssemblerAgentInput
from storymesh.schemas.story_writer import SCENE_BREAK

logger = logging.getLogger(__name__)


# ── Internal raw result ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BookRawOutput:
    """Internal result type returned by BookAssemblerAgent.run().

    The node wrapper reads pdf_bytes and epub_bytes, saves them to the
    run directory via ArtifactStore, and constructs BookAssemblerAgentOutput
    with the final file paths.
    """

    pdf_bytes: bytes | None
    epub_bytes: bytes | None
    title: str
    word_count: int
    debug: dict[str, Any] = field(default_factory=dict)


# ── Shared CSS constants ───────────────────────────────────────────────────────

# WeasyPrint renders A5 (148×210 mm) with a classic book layout.
_PDF_CSS = """\
@page {
    size: 148mm 210mm;
    margin: 22mm 18mm 25mm 22mm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #888;
        font-family: Georgia, "Times New Roman", serif;
    }
}
@page cover-page {
    margin: 0;
    padding: 0;
    @bottom-center { content: none; }
}
@page front-matter {
    @bottom-center { content: none; }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #111;
    text-align: justify;
    hyphens: auto;
    orphans: 3;
    widows: 3;
}

/* Cover page — full-bleed image or typographic fallback */
.cover-image-page {
    page: cover-page;
    break-after: page;
    width: 148mm;
    height: 210mm;
    overflow: hidden;
}
.cover-image-page img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.cover-text-page {
    page: cover-page;
    break-after: page;
    width: 148mm;
    height: 210mm;
    background: #1a1a2e;
    color: #e8e0d0;
    text-align: center;
    padding-top: 72mm;
    padding-left: 18mm;
    padding-right: 18mm;
}
.cover-text-page h1 {
    font-size: 26pt;
    font-weight: 400;
    letter-spacing: 0.04em;
    margin-bottom: 8mm;
}
.cover-text-page .byline {
    font-size: 10pt;
    font-style: italic;
    letter-spacing: 0.08em;
    color: #b0a898;
}

/* Title page */
.title-page {
    page: front-matter;
    break-after: page;
    text-align: center;
    padding-top: 55mm;
    padding-bottom: 20mm;
}
.title-page h1 {
    font-size: 20pt;
    font-weight: 400;
    letter-spacing: 0.04em;
    margin-bottom: 5mm;
}
.title-page .byline {
    font-size: 10pt;
    font-style: italic;
    color: #555;
    margin-bottom: 6mm;
}
.title-page .genre-tags {
    font-size: 8.5pt;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
}

/* Synopsis page */
.synopsis-page {
    page: front-matter;
    break-after: page;
    padding-top: 6mm;
}
.synopsis-page h2 {
    font-size: 9pt;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    text-align: center;
    color: #666;
    margin-bottom: 7mm;
}
.synopsis-page p {
    font-style: italic;
    font-size: 10.5pt;
    line-height: 1.7;
    text-indent: 0;
}

/* Scene sections */
.scene {
    break-before: page;
}
.scene-title {
    font-size: 11pt;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    text-align: center;
    color: #444;
    margin-bottom: 7mm;
}
.scene p {
    text-indent: 1.5em;
    margin-bottom: 0;
}
.scene p:first-of-type {
    text-indent: 0;
}
.scene p:first-of-type::first-letter {
    font-size: 2.5em;
    font-weight: 400;
    line-height: 1;
}
"""

# Clean, readable EPUB stylesheet.
_EPUB_CSS = """\
body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1em;
    line-height: 1.6;
    margin: 5% 8%;
    text-align: justify;
}
h1.book-title {
    font-size: 1.8em;
    font-weight: normal;
    text-align: center;
    margin: 3em 0 0.4em;
}
p.byline {
    font-size: 0.9em;
    font-style: italic;
    text-align: center;
    color: #555;
    margin-bottom: 0.4em;
}
p.genre-tags {
    font-size: 0.8em;
    text-align: center;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
h2.scene-title {
    font-size: 1.05em;
    font-weight: normal;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
    color: #555;
    margin: 3em 0 1.5em;
}
h2.synopsis-heading {
    font-size: 0.85em;
    font-weight: normal;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    text-align: center;
    color: #777;
    margin: 2.5em 0 1.2em;
}
p {
    text-indent: 1.5em;
    margin: 0;
}
p:first-of-type, p.no-indent {
    text-indent: 0;
}
p.synopsis-text {
    font-style: italic;
    line-height: 1.7;
    text-indent: 0;
}
"""

# EPUB chapter template — placeholders replaced via str.format().
# Rules:
#   - No <!DOCTYPE html>: lxml.etree (used by ebooklib internally) cannot
#     resolve the undeclared DOCTYPE and raises XMLSyntaxError, causing
#     get_body_content() to return b'' and epub.write_epub() to crash.
#   - No <meta charset>: encoding is already declared in the XML prolog.
#   - CSS wrapped in CDATA: prevents any CSS characters that are XML-special
#     (e.g. > in attribute selectors) from breaking the XML parser.
_EPUB_CHAPTER_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head>
  <title>{chapter_title}</title>
  <style type="text/css">
/*<![CDATA[*/
{css}
/*]]>*/
  </style>
</head>
<body>
{body}
</body>
</html>"""


# ── Helper functions ───────────────────────────────────────────────────────────


def _prose_to_html(prose: str) -> str:
    """Convert plain prose text to HTML paragraph elements.

    Double newlines delimit paragraphs. Single newlines within a paragraph
    are treated as soft wraps and joined with a space.

    Args:
        prose: Raw prose string from the story draft.

    Returns:
        HTML string of ``<p>`` elements with content escaped.
    """
    paragraphs = [
        " ".join(chunk.split("\n")).strip()
        for chunk in prose.split("\n\n")
        if chunk.strip()
    ]
    return "\n".join(f"<p>{_escape(p)}</p>" for p in paragraphs)


def _tag_line(proposal_genre_blend: list[str], proposal_tone: list[str]) -> str:
    """Build a short genre/tone tag string for title page display.

    Args:
        proposal_genre_blend: Genre blend list from StoryProposal.
        proposal_tone: Tone list from StoryProposal.

    Returns:
        A human-readable tag string, e.g. ``"Post-Apocalyptic · Mystery · Dark"``.
    """
    tags = proposal_genre_blend + proposal_tone
    return " · ".join(t.title() for t in tags if t.strip())


# ── Agent ──────────────────────────────────────────────────────────────────────


class BookAssemblerAgent:
    """Assembles the final PDF and EPUB deliverables (Stage 8).

    No LLM is required. Both formats are rendered from the story draft
    produced by StoryWriterAgent, with the cover art from CoverArtAgent
    embedded when available.

    PDF generation requires WeasyPrint (system libraries libcairo, libpango).
    EPUB generation requires ebooklib. Either missing dependency degrades
    to a logged warning and an empty path for that format.
    """

    def __init__(
        self,
        *,
        output_formats: list[str] | None = None,
    ) -> None:
        """Construct the agent.

        Args:
            output_formats: Which formats to generate. Valid values are
                ``"pdf"`` and ``"epub"``. Defaults to both.
        """
        self._output_formats: list[str] = (
            list(output_formats) if output_formats else ["pdf", "epub"]
        )

    def run(self, input_data: BookAssemblerAgentInput) -> BookRawOutput:
        """Assemble PDF and EPUB from the pipeline outputs.

        Args:
            input_data: Assembled input from the node wrapper.

        Returns:
            A ``BookRawOutput`` with ``pdf_bytes`` and/or ``epub_bytes`` set.
            Either field is ``None`` when the format was not requested or the
            required rendering library is not installed.
        """
        story = input_data.story_writer_output
        proposal = input_data.proposal

        # Load cover image bytes from the path written by CoverArtAgent.
        cover_bytes: bytes | None = None
        if input_data.cover_art_output and input_data.cover_art_output.image_path:
            cover_path = Path(input_data.cover_art_output.image_path)
            if cover_path.exists():
                cover_bytes = cover_path.read_bytes()
            else:
                logger.warning(
                    "BookAssemblerAgent: cover image not found at %r — "
                    "typographic cover will be used.",
                    str(cover_path),
                )

        # Split the draft into per-scene texts on the SCENE_BREAK delimiter.
        scene_texts = [s.strip() for s in story.full_draft.split(SCENE_BREAK) if s.strip()]

        # Pair each prose block with its scene title from the outline pass.
        # Truncate to the shorter list; use generic fallback titles when the
        # draft produced more sections than the outline has entries.
        outline_titles = [s.title for s in story.scene_list]
        if len(scene_texts) > len(outline_titles):
            extra = [f"Chapter {i + 1}" for i in range(len(outline_titles), len(scene_texts))]
            outline_titles += extra
        scene_pairs: list[tuple[str, str]] = list(zip(outline_titles, scene_texts, strict=False))

        if not scene_pairs:
            # Treat the entire draft as a single untitled section.
            scene_pairs = [("", story.full_draft.strip())]
            logger.warning(
                "BookAssemblerAgent: could not split draft into scenes — "
                "entire draft rendered as a single section."
            )

        genre_tags = _tag_line(proposal.genre_blend, proposal.tone)
        cover_b64 = b64encode(cover_bytes).decode() if cover_bytes else None

        logger.info(
            "BookAssemblerAgent | title=%r scenes=%d cover=%s formats=%s",
            proposal.title,
            len(scene_pairs),
            "yes" if cover_bytes else "no",
            self._output_formats,
        )

        debug: dict[str, Any] = {
            "scene_count": len(scene_pairs),
            "has_cover_image": cover_bytes is not None,
            "word_count": story.word_count,
        }

        pdf_bytes: bytes | None = None
        if "pdf" in self._output_formats:
            pdf_bytes = self._build_pdf(
                title=proposal.title,
                genre_tags=genre_tags,
                back_cover_summary=story.back_cover_summary,
                scene_pairs=scene_pairs,
                cover_b64=cover_b64,
            )
            debug["pdf_generated"] = pdf_bytes is not None

        epub_bytes: bytes | None = None
        if "epub" in self._output_formats:
            epub_bytes = self._build_epub(
                title=proposal.title,
                run_id=input_data.run_id,
                genre_tags=genre_tags,
                back_cover_summary=story.back_cover_summary,
                scene_pairs=scene_pairs,
                cover_bytes=cover_bytes,
            )
            debug["epub_generated"] = epub_bytes is not None

        return BookRawOutput(
            pdf_bytes=pdf_bytes,
            epub_bytes=epub_bytes,
            title=proposal.title,
            word_count=story.word_count,
            debug=debug,
        )

    # ── PDF ────────────────────────────────────────────────────────────────────

    def _build_pdf(
        self,
        *,
        title: str,
        genre_tags: str,
        back_cover_summary: str,
        scene_pairs: list[tuple[str, str]],
        cover_b64: str | None,
    ) -> bytes | None:
        """Render the book as PDF bytes via WeasyPrint.

        Args:
            title: Story title.
            genre_tags: Comma/dot-separated genre and tone tags.
            back_cover_summary: Back-cover synopsis text.
            scene_pairs: List of (scene_title, prose_text) tuples.
            cover_b64: Base64-encoded cover PNG, or None for a typographic cover.

        Returns:
            PDF as raw bytes, or None if WeasyPrint is not installed.
        """
        try:
            from weasyprint import HTML as _WeasyHTML  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "WeasyPrint is not installed — PDF generation skipped. "
                "Install with: pip install storymesh[pdf]"
            )
            return None

        html_str = self._build_html(
            title=title,
            genre_tags=genre_tags,
            back_cover_summary=back_cover_summary,
            scene_pairs=scene_pairs,
            cover_b64=cover_b64,
        )
        return _WeasyHTML(string=html_str).write_pdf()  # type: ignore[no-any-return]

    def _build_html(
        self,
        *,
        title: str,
        genre_tags: str,
        back_cover_summary: str,
        scene_pairs: list[tuple[str, str]],
        cover_b64: str | None,
    ) -> str:
        """Build the full HTML document used as WeasyPrint input.

        Args:
            title: Story title.
            genre_tags: Tag line string for the title page.
            back_cover_summary: Synopsis text.
            scene_pairs: Ordered (title, prose) tuples.
            cover_b64: Base64-encoded cover image, or None.

        Returns:
            Complete HTML document string.
        """
        # Cover page
        if cover_b64:
            cover_html = (
                '<div class="cover-image-page">'
                f'<img src="data:image/png;base64,{cover_b64}" alt="Cover"/>'
                "</div>"
            )
        else:
            cover_html = (
                '<div class="cover-text-page">'
                f"<h1>{_escape(title)}</h1>"
                '<p class="byline">A StoryMesh Production</p>'
                "</div>"
            )

        # Title page
        title_page_html = (
            '<div class="title-page">'
            f"<h1>{_escape(title)}</h1>"
            '<p class="byline">A StoryMesh Production</p>'
            f'<p class="genre-tags">{_escape(genre_tags)}</p>'
            "</div>"
        )

        # Synopsis page
        synopsis_paragraphs = _prose_to_html(back_cover_summary)
        synopsis_html = (
            '<div class="synopsis-page">'
            "<h2>Synopsis</h2>"
            f"{synopsis_paragraphs}"
            "</div>"
        )

        # Scene sections
        scene_parts: list[str] = []
        for scene_title, prose in scene_pairs:
            heading = f'<h2 class="scene-title">{_escape(scene_title)}</h2>' if scene_title else ""
            scene_parts.append(
                '<div class="scene">'
                f"{heading}"
                f"{_prose_to_html(prose)}"
                "</div>"
            )
        scenes_html = "\n".join(scene_parts)

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="utf-8"/>\n'
            f"<title>{_escape(title)}</title>\n"
            f"<style>{_PDF_CSS}</style>\n"
            "</head>\n"
            "<body>\n"
            f"{cover_html}\n"
            f"{title_page_html}\n"
            f"{synopsis_html}\n"
            f"{scenes_html}\n"
            "</body>\n"
            "</html>"
        )

    # ── EPUB ───────────────────────────────────────────────────────────────────

    def _build_epub(
        self,
        *,
        title: str,
        run_id: str,
        genre_tags: str,
        back_cover_summary: str,
        scene_pairs: list[tuple[str, str]],
        cover_bytes: bytes | None,
    ) -> bytes | None:
        """Build an EPUB3 document and return it as raw bytes.

        Uses a temporary file because ``epub.write_epub`` only accepts a
        file path, not a stream.

        Args:
            title: Story title.
            run_id: Pipeline run ID used as the EPUB unique identifier.
            genre_tags: Tag line string.
            back_cover_summary: Synopsis text.
            scene_pairs: Ordered (title, prose) tuples.
            cover_bytes: Raw PNG bytes for the cover image, or None.

        Returns:
            EPUB3 as raw bytes, or None if ebooklib is not installed.
        """
        try:
            from ebooklib import epub  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "ebooklib is not installed — EPUB generation skipped. "
                "Install with: pip install storymesh[pdf]"
            )
            return None

        book = epub.EpubBook()
        book.set_identifier(run_id or "storymesh-unknown")
        book.set_title(title)
        book.set_language("en")
        book.add_author("StoryMesh")

        if cover_bytes:
            book.set_cover("cover.png", cover_bytes)

        # Title page chapter
        title_body = (
            f'<h1 class="book-title">{_escape(title)}</h1>\n'
            '<p class="byline">A StoryMesh Production</p>\n'
            f'<p class="genre-tags">{_escape(genre_tags)}</p>'
        )
        title_ch = epub.EpubHtml(title="Title Page", file_name="title.xhtml", lang="en")
        title_ch.content = _EPUB_CHAPTER_TEMPLATE.format(
            chapter_title="Title Page", css=_EPUB_CSS, body=title_body
        ).encode("utf-8")
        book.add_item(title_ch)

        # Synopsis chapter
        synopsis_body = (
            '<h2 class="synopsis-heading">Synopsis</h2>\n'
            + "\n".join(
                f'<p class="synopsis-text">{_escape(p.strip())}</p>'
                for p in back_cover_summary.split("\n\n")
                if p.strip()
            )
        )
        synopsis_ch = epub.EpubHtml(title="Synopsis", file_name="synopsis.xhtml", lang="en")
        synopsis_ch.content = _EPUB_CHAPTER_TEMPLATE.format(
            chapter_title="Synopsis", css=_EPUB_CSS, body=synopsis_body
        ).encode("utf-8")
        book.add_item(synopsis_ch)

        # Scene chapters
        scene_chapters: list[Any] = []
        for i, (scene_title, prose) in enumerate(scene_pairs):
            heading = (
                f'<h2 class="scene-title">{_escape(scene_title)}</h2>\n'
                if scene_title
                else ""
            )
            chapter_body = heading + _prose_to_html(prose)
            ch_title = scene_title if scene_title else f"Chapter {i + 1}"
            ch = epub.EpubHtml(
                title=ch_title,
                file_name=f"scene_{i + 1:02d}.xhtml",
                lang="en",
            )
            ch.content = _EPUB_CHAPTER_TEMPLATE.format(
                chapter_title=_escape(ch_title), css=_EPUB_CSS, body=chapter_body
            ).encode("utf-8")
            book.add_item(ch)
            scene_chapters.append(ch)

        # Navigation
        all_chapters: list[Any] = [title_ch, synopsis_ch] + scene_chapters
        book.toc = [
            epub.Link(ch.file_name, ch.title, f"nav_{i}")
            for i, ch in enumerate(all_chapters)
        ]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav"] + all_chapters

        # Write to temp file, read bytes, clean up.
        fd, tmp_path = tempfile.mkstemp(suffix=".epub")
        os.close(fd)
        try:
            # epub3_pages=True is the default in this ebooklib version but triggers
            # a page-list scan that calls get_body_content() on every chapter.
            # That body extraction fails on our XHTML content, producing an empty
            # bytes object that causes parse_html_string() to raise ParserError.
            # Page-list navigation is not required for our output so we disable it.
            epub.write_epub(tmp_path, book, options={"epub3_pages": False})
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
