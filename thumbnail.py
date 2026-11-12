"""Custom YouTube thumbnail: the story's key frame with its title burned in.

YouTube accepts a custom thumbnail through `thumbnails.set` (see publish.upload_thumbnail).
The image has to be readable at 120 px wide in a feed, so the title is set large, centred, and
given a heavy stroke — the same treatment the captions use, so the thumbnail and the video look
like one product.

Frame choice: the most story-bearing shot, not simply the first. A wide establishing shot is
usually the emptiest frame in the short; `deliver_payload` / `emotional_beat` / `evidence` are
the beats the story is actually about. Falls back to the first frame when none of those exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Thumbnail canvas. YouTube's documented minimum is 1280x720; Shorts are shown vertically, so a
# 9:16 canvas at 1080x1920 is both above that pixel count and the right shape for the feed.
THUMB_WIDTH = 1080
THUMB_HEIGHT = 1920

# Roles that carry the story. Ordered by how much they say on their own.
KEY_ROLES = ("deliver_payload", "emotional_beat", "evidence", "resolution", "build_tension")

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
)


def pick_key_frame(shots, frames: list[Path]) -> Path | None:
    """The frame that best represents the story, or the first frame as a fallback."""
    if not frames:
        return None
    for role in KEY_ROLES:
        for shot, frame in zip(shots, frames):
            if getattr(shot, "narrative_role", "") == role:
                return frame
    return frames[0]


def _resolve_font(font_path: str | None) -> str:
    if font_path and Path(font_path).exists():
        return font_path
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    return ""


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy word wrap against the measured pixel width."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_thumbnail(
    frame_path: Path,
    title: str,
    out_path: Path,
    *,
    font_path: str | None = None,
    accent: str = "#FFD60A",
) -> Path | None:
    """Cover-crop `frame_path` to 9:16, burn `title` into the centre, save as JPEG.

    Returns the written path, or None when Pillow or a usable font is missing — a thumbnail is
    a nice-to-have and must never fail the upload.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        logger.info("Pillow not installed; skipping thumbnail")
        return None

    chosen = _resolve_font(font_path)
    if not chosen:
        logger.warning("no thumbnail font found; skipping thumbnail")
        return None

    try:
        src = Image.open(frame_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not open thumbnail source %s (%s)", frame_path, exc)
        return None

    # Cover-crop to the 9:16 canvas: scale up until both dimensions are covered, then centre.
    scale = max(THUMB_WIDTH / src.width, THUMB_HEIGHT / src.height)
    resized = src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))),
                         Image.LANCZOS)
    left = (resized.width - THUMB_WIDTH) // 2
    top = (resized.height - THUMB_HEIGHT) // 2
    canvas = resized.crop((left, top, left + THUMB_WIDTH, top + THUMB_HEIGHT))

    # Darken the middle band so the title reads over any frame, without dimming the whole image.
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    band_top = int(THUMB_HEIGHT * 0.30)
    band_bottom = int(THUMB_HEIGHT * 0.70)
    for y in range(band_top, band_bottom):
        # Soft edges: strongest in the middle of the band.
        edge = min(y - band_top, band_bottom - y) / ((band_bottom - band_top) / 2)
        odraw.line([(0, y), (THUMB_WIDTH, y)], fill=(0, 0, 0, int(150 * min(1.0, edge * 1.6))))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    max_text_width = int(THUMB_WIDTH * 0.86)

    # Shrink the font until the wrapped title fits in at most three lines.
    size = 150
    lines: list[str] = []
    font = ImageFont.truetype(chosen, size)
    while size >= 60:
        font = ImageFont.truetype(chosen, size)
        lines = _wrap(draw, title, font, max_text_width)
        if len(lines) <= 3:
            break
        size -= 10

    if not lines:
        canvas.save(out_path, format="JPEG", quality=90)
        return out_path

    line_height = int(size * 1.18)
    block_height = line_height * len(lines)
    y = (THUMB_HEIGHT - block_height) // 2

    for line in lines:
        width = draw.textlength(line, font=font)
        x = (THUMB_WIDTH - width) / 2
        # Shadow, then a heavy black stroke, then the fill — same recipe as the captions.
        for off in ((0, 8), (0, 5), (2, 2)):
            draw.text((x + off[0], y + off[1]), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=accent,
                  stroke_width=max(6, size // 18), stroke_fill=(0, 0, 0))
        y += line_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=90)
    logger.info("thumbnail written: %s (%d bytes, %d line(s))",
                out_path, out_path.stat().st_size, len(lines))
    return out_path


__all__ = ["KEY_ROLES", "THUMB_HEIGHT", "THUMB_WIDTH", "pick_key_frame", "render_thumbnail"]
