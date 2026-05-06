"""CoverArtAgent — Stage 7 of the StoryMesh pipeline.

Generates a book cover image from the image_prompt field on the selected
StoryProposal. The prompt was written by ProposalDraftAgent at the moment
of story synthesis, when the model had the fullest creative context.

Workflow:
1. Assemble the final image prompt, enforcing flat-canvas framing.
2. Call the image generation provider (gpt-image-2).
3. Composite title and byline text onto the raw PNG bytes using Pillow,
   giving reliable typography independent of the model's text rendering.

Filesystem persistence is handled by the node wrapper.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import time
from typing import Any

from storymesh.llm.image_base import ImageClient
from storymesh.schemas.cover_art import CoverArtAgentInput

logger = logging.getLogger(__name__)

_BYLINE = "A StoryMesh Production"

# Common bold sans-serif font paths across Linux distributions, macOS, Windows.
# The list is tried in order; the first existing path wins.
_COVER_FONT_PATHS = [
    "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",                  # Fedora / RHEL
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",            # Debian / Ubuntu
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",    # many Linux
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",             # Fedora alt
    "/Library/Fonts/Arial Bold.ttf",                                    # macOS
    "C:\\Windows\\Fonts\\arialbd.ttf",                                  # Windows
]


def _build_assembled_prompt(image_prompt: str) -> str:
    """Append flat-canvas enforcement directives to the visual description.

    Title and byline are composited in post-processing via Pillow, so the
    model is explicitly told not to render text. The flat-canvas instruction
    prevents DALL-E from generating a 3D book-object photograph.

    Args:
        image_prompt: Visual design description from StoryProposal.

    Returns:
        Assembled prompt ready for submission to the image generation API.
    """
    return (
        f"{image_prompt}  "
        "Flat 2D artwork filling the entire canvas edge to edge. "
        "No book object, no 3D perspective, no mockup frame, no drop shadow. "
        "No text, no lettering, no title, no words anywhere in the image."
    )


def _load_font(size: int) -> Any:  # noqa: ANN401 — PIL ImageFont types are optional
    """Load the best available bold font at the requested point size.

    Tries each path in ``_COVER_FONT_PATHS`` in order. Falls back to PIL's
    built-in bitmap font if no TrueType font is found (font size is ignored
    for the fallback).

    Args:
        size: Desired font size in points.

    Returns:
        A PIL ``FreeTypeFont`` or ``ImageFont`` instance.
    """
    from PIL import ImageFont  # deferred — optional dependency

    for path in _COVER_FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    logger.warning("No TrueType font found; falling back to PIL default bitmap font.")
    return ImageFont.load_default()


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:  # noqa: ANN401 — PIL types are optional
    """Wrap ``text`` into lines that each fit within ``max_width`` pixels.

    Args:
        draw: PIL ``ImageDraw`` instance used for text measurement.
        text: Text to wrap.
        font: PIL font used for measurement.
        max_width: Maximum line width in pixels.

    Returns:
        List of wrapped lines.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _compose_cover_text(image_bytes: bytes, title: str, byline: str) -> bytes:
    """Composite title and byline text onto the cover image using Pillow.

    Renders semi-transparent dark bands at the top (title) and bottom
    (byline) of the image to ensure legibility over any background colour.
    White text with a soft drop shadow is centered in each band.

    Args:
        image_bytes: Raw PNG bytes from the image generation API.
        title: Story title to render at the top of the cover.
        byline: Byline string to render at the bottom of the cover.

    Returns:
        PNG bytes with title and byline composited onto the image.

    Raises:
        ImportError: If Pillow is not installed.
    """
    from PIL import Image, ImageDraw  # deferred — optional dependency

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    padding = max(16, w // 24)

    # --- Title band (top) ---
    title_font_size = max(40, w // 13)
    title_font = _load_font(title_font_size)
    max_text_w = w - padding * 2
    title_lines = _wrap_text(draw, title, title_font, max_text_w)
    line_h = int(title_font_size * 1.25)
    band_h = len(title_lines) * line_h + padding * 2

    draw.rectangle(((0, 0), (w, band_h)), fill=(0, 0, 0, 170))

    y = padding
    for line in title_lines:
        line_w = int(draw.textlength(line, font=title_font))
        x = (w - line_w) // 2
        draw.text((x + 2, y + 2), line, font=title_font, fill=(0, 0, 0, 180))  # shadow
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255))
        y += line_h

    # --- Byline band (bottom) ---
    byline_font_size = max(22, w // 26)
    byline_font = _load_font(byline_font_size)
    byline_band_h = byline_font_size + padding * 2

    draw.rectangle(((0, h - byline_band_h), (w, h)), fill=(0, 0, 0, 170))

    byline_w = int(draw.textlength(byline, font=byline_font))
    bx = (w - byline_w) // 2
    by = h - byline_band_h + padding
    draw.text((bx + 1, by + 1), byline, font=byline_font, fill=(0, 0, 0, 180))  # shadow
    draw.text((bx, by), byline, font=byline_font, fill=(255, 255, 255, 255))

    composited = Image.alpha_composite(img, overlay)
    out = io.BytesIO()
    composited.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _safe_compose_cover_text(image_bytes: bytes, title: str, byline: str) -> bytes:
    """Attempt PIL text composition; return raw bytes on any failure.

    This wrapper means the pipeline degrades gracefully if Pillow is not
    installed or if the image bytes cannot be decoded (e.g. in unit tests
    that use synthetic placeholder bytes).

    Args:
        image_bytes: Raw PNG bytes from the image generation API.
        title: Story title.
        byline: Byline string.

    Returns:
        Composited PNG bytes, or the original bytes if composition fails.
    """
    try:
        return _compose_cover_text(image_bytes, title, byline)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cover text composition failed — returning raw image bytes. Reason: %s", exc
        )
        return image_bytes


@dataclasses.dataclass(frozen=True)
class GeneratedCoverImage:
    """Raw result from image generation, before filesystem persistence.

    Returned by CoverArtAgent.run(). The node wrapper saves image_bytes
    to disk and constructs the final CoverArtAgentOutput with image_path.
    """

    image_bytes: bytes
    image_prompt: str
    revised_prompt: str | None
    model: str
    image_size: str
    image_quality: str
    latency_ms: int


class CoverArtAgent:
    """Generates a book cover image from the story proposal (Stage 7).

    Makes a single call to the configured image generation provider using
    the image_prompt field from the selected StoryProposal.
    """

    def __init__(
        self,
        *,
        image_client: ImageClient,
        image_size: str = "1024x1792",
        image_quality: str = "auto",
    ) -> None:
        """Construct the agent.

        Args:
            image_client: Image generation client instance. Required.
            image_size: Image dimensions. Default '1024x1792'.
            image_quality: Quality tier. 'auto' (default), 'low', 'medium', or 'high'.
        """
        self._image_client = image_client
        self._image_size = image_size
        self._image_quality = image_quality

    def run(self, input_data: CoverArtAgentInput) -> GeneratedCoverImage:
        """Generate a cover image from the proposal's image prompt.

        Steps:
        1. Build the final image prompt with flat-canvas enforcement.
        2. Call the image generation provider.
        3. Composite title and byline onto the returned PNG bytes via Pillow.

        Args:
            input_data: Input assembled by the node wrapper from the selected proposal.

        Returns:
            GeneratedCoverImage with composited PNG bytes and generation metadata.

        Raises:
            openai.OpenAIError: On API-level failures from the image provider.
            ValueError: If the image provider returns an unexpected response.
        """
        logger.debug(
            "CoverArtAgent starting | title=%r model=%s size=%s quality=%s",
            input_data.title,
            self._image_client.model,
            self._image_size,
            self._image_quality,
        )

        assembled_prompt = _build_assembled_prompt(input_data.image_prompt)

        t0 = time.perf_counter()
        result = self._image_client.generate(
            assembled_prompt,
            size=self._image_size,
            quality=self._image_quality,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        composed_bytes = _safe_compose_cover_text(result.image_bytes, input_data.title, _BYLINE)

        logger.debug(
            "CoverArtAgent complete | latency_ms=%d revised=%s",
            latency_ms,
            result.revised_prompt is not None,
        )

        return GeneratedCoverImage(
            image_bytes=composed_bytes,
            image_prompt=assembled_prompt,
            revised_prompt=result.revised_prompt,
            model=self._image_client.model,
            image_size=self._image_size,
            image_quality=self._image_quality,
            latency_ms=latency_ms,
        )
