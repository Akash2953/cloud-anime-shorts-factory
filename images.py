"""Image production for a shot list.

Prompt composition belongs to `scenes.py` (scene/action/object lead, character signature last).
This module only decides *how* each prompt becomes an image, and degrades safely:

  edit   shot 1 rendered fresh becomes the canonical character; later CHARACTER shots are edits of
         it (FLUX.2-klein, multipart). Insert/environment shots are rendered fresh — they have no
         face, so nothing needs conditioning.
  sheet  one character sheet (2x3 grid) cropped into panels. Identity is identical pixels.
  fresh  every shot rendered independently.

All strategies stop on quota/auth errors instead of hammering the API, and record what ran.
"""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from scenes import CHARACTER, STYLE

logger = logging.getLogger(__name__)

# Verified against GET /accounts/{id}/ai/models/search?task=Text-to-Image.
SCHNELL = "@cf/black-forest-labs/flux-1-schnell"
KLEIN = "@cf/black-forest-labs/flux-2-klein-4b"

VALID_STRATEGIES = ("edit", "sheet", "fresh")

# Model-level failures (switch strategy) vs spend failures (stop the run).
_FATAL = ("auth_", "plan_restricted", "model_missing")
_STOP = _FATAL + ("quota_exhausted",)


@dataclass
class ImageResult:
    ok: bool
    data: bytes | None = None
    error: str | None = None


def image_ext(data: bytes) -> str:
    """Pick a file extension from magic bytes so FFmpeg's image2 demuxer can decode it."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("unrecognised image format")


def decode_image_response(resp: requests.Response) -> bytes:
    """Workers AI returns either raw image bytes or base64 under result.image / result.data[0]."""
    if resp.headers.get("content-type", "").startswith("image/"):
        return resp.content
    body = resp.json().get("result", {})
    if isinstance(body, dict):
        image = body.get("image")
        if isinstance(image, str) and image:
            return base64.b64decode(image)
        data = body.get("data")
        if isinstance(data, list) and data and data[0]:
            return base64.b64decode(data[0])
    raise ValueError("no image in response")


def _classify(status: int, text: str) -> str:
    if status in (401, 403):
        return "plan_restricted" if "free plan" in text.lower() else f"auth_{status}"
    if status in (404, 410) or (status == 400 and "7000" in text):
        return "model_missing"
    if status == 429:
        return "quota_exhausted"
    return f"status_{status}"


def generate_fresh(prompt: str, cfa_token: str, account_id: str) -> ImageResult:
    """FLUX.1-schnell text-to-image. Fixed 1024x1024; rejects width/height/seed."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{SCHNELL}"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {cfa_token}", "Content-Type": "application/json"},
            json={"prompt": prompt}, timeout=180)
    except requests.RequestException as exc:  # noqa: BLE001
        return ImageResult(ok=False, error=f"request_failed:{exc}")
    if resp.status_code != 200:
        return ImageResult(ok=False, error=_classify(resp.status_code, resp.text))
    try:
        return ImageResult(ok=True, data=decode_image_response(resp))
    except ValueError as exc:
        return ImageResult(ok=False, error=f"bad_response:{exc}")


def generate_edit(reference: bytes, prompt: str, cfa_token: str, account_id: str) -> ImageResult:
    """FLUX.2-klein image edit: multipart with the canonical frame as `input_image`."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{KLEIN}"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {cfa_token}"},  # requests sets the multipart boundary
            data={"prompt": prompt},
            files={"input_image": ("canonical.jpg", reference, "image/jpeg")},
            timeout=240)
    except requests.RequestException as exc:  # noqa: BLE001
        return ImageResult(ok=False, error=f"request_failed:{exc}")
    if resp.status_code != 200:
        return ImageResult(ok=False, error=_classify(resp.status_code, resp.text))
    try:
        return ImageResult(ok=True, data=decode_image_response(resp))
    except ValueError as exc:
        return ImageResult(ok=False, error=f"bad_response:{exc}")


def crop_grid(sheet_path: Path, out_dir: Path, count: int, columns: int = 3) -> list[Path]:
    """Slice a rendered character sheet into equal panels."""
    import subprocess

    rows = (count + columns - 1) // columns
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(sheet_path)],
        capture_output=True, text=True, check=True).stdout.strip()
    width, height = (int(x) for x in probe.split(",")[:2])
    panel_w, panel_h = width // columns, height // rows

    panels: list[Path] = []
    for i in range(count):
        row, col = divmod(i, columns)
        out = out_dir / f"panel_{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sheet_path),
             "-vf", f"crop={panel_w}:{panel_h}:{col * panel_w}:{row * panel_h}",
             "-frames:v", "1", str(out)], check=True)
        panels.append(out)
    return panels


def _save(data: bytes, img_dir: Path, index: int) -> Path:
    path = img_dir / f"frame_{index}{image_ext(data)}"
    path.write_bytes(data)
    return path


def produce_images(
    prompts: list[str],
    *,
    cfa_token: str,
    account_id: str,
    img_dir: Path,
    strategy: str = "fresh",
    character_flags: list[bool] | None = None,
    budget: dict | None = None,
) -> tuple[list[Path], str]:
    """Render one image per prompt. Returns (frames, strategy_used).

    `character_flags[i]` marks prompts where the girl is on screen; only those take part in the
    canonical-frame conditioning, so inserts and establishing shots stay independent.
    """
    if strategy not in VALID_STRATEGIES:
        logger.warning("Unknown image strategy %r, using 'fresh'", strategy)
        strategy = "fresh"
    img_dir.mkdir(parents=True, exist_ok=True)
    budget = budget if budget is not None else {"attempts": 0, "max": 10_000}
    flags = character_flags or [True] * len(prompts)

    def affordable() -> bool:
        return budget["attempts"] < budget["max"]

    # --- sheet -----------------------------------------------------------------
    if strategy == "sheet" and affordable():
        budget["attempts"] += 1
        n = len(prompts)
        sheet_prompt = (
            f"{STYLE}. Character reference sheet: {CHARACTER}, shown {n} times in a "
            f"3x{(n + 2) // 3} grid of equal panels with no borders, each panel a different pose, "
            "full body, identical design in every panel.")
        res = generate_fresh(sheet_prompt, cfa_token, account_id)
        if res.ok:
            panels = crop_grid(_save(res.data, img_dir, 99), img_dir, n)
            if len(panels) == n:
                logger.info("Character sheet split into %d panels", len(panels))
                return panels, "sheet"
        logger.warning("Sheet strategy failed (%s); falling back to fresh", res.error)
        strategy = "fresh"

    # --- edit ------------------------------------------------------------------
    if strategy == "edit":
        frames: list[Path] = []
        canonical: bytes | None = None
        edits = 0
        stopped = False

        for i, prompt in enumerate(prompts):
            if not affordable():
                logger.warning("Daily image budget exhausted (%d/%d)", budget["attempts"], budget["max"])
                break
            budget["attempts"] += 1

            wants_character = flags[i] if i < len(flags) else True
            if canonical is None or not wants_character or strategy != "edit":
                res = generate_fresh(prompt, cfa_token, account_id)
                if not res.ok:
                    logger.error("Shot %d failed: %s", i + 1, res.error)
                    if res.error.startswith(_STOP):
                        stopped = True
                        break
                    continue
                if canonical is None and wants_character:
                    canonical = res.data
                    logger.info("shot %d: canonical character rendered (flux-1-schnell)", i + 1)
                frames.append(_save(res.data, img_dir, i))
                continue

            res = generate_edit(
                canonical,
                f"Keep the EXACT same character — identical face, hair, hood, red scarf and coat as "
                f"the reference image. {prompt}",
                cfa_token, account_id)
            if res.ok:
                edits += 1
                frames.append(_save(res.data, img_dir, i))
                logger.info("shot %d: edit of the canonical character (flux-2-klein)", i + 1)
                continue

            if res.error.startswith(("quota_exhausted", "auth_")):
                logger.error("Stopping image generation: %s", res.error)
                stopped = True
                break
            logger.warning("Edit failed for shot %d (%s); rendering it fresh", i + 1, res.error)
            if res.error.startswith(_FATAL):
                strategy = "fresh"
                logger.warning("Edit model unusable (%s); remaining shots render fresh", res.error)
            res = generate_fresh(prompt, cfa_token, account_id)
            if res.ok:
                frames.append(_save(res.data, img_dir, i))
            elif res.error.startswith(_STOP):
                stopped = True
                break

        if frames:
            used = "edit" if edits else "fresh"
            logger.info("Strategy '%s': %d shot(s) conditioned on the canonical frame", used, edits)
            return frames, used
        if stopped:
            return [], strategy
        strategy = "fresh"

    # --- fresh -----------------------------------------------------------------
    frames = []
    for i, prompt in enumerate(prompts):
        if not affordable():
            logger.warning("Daily image budget exhausted (%d/%d)", budget["attempts"], budget["max"])
            break
        budget["attempts"] += 1
        res = generate_fresh(prompt, cfa_token, account_id)
        if not res.ok:
            logger.error("Shot %d failed: %s", i + 1, res.error)
            if res.error.startswith(_STOP):
                break
            continue
        frames.append(_save(res.data, img_dir, i))
    return frames, "fresh"
