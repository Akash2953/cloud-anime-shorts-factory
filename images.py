"""Image production for a shot list.

Prompt composition belongs to `scenes.py` (scene/action/object lead, character signature last).
This module only decides *how* each prompt becomes an image, and degrades safely:

  edit   shot 1 rendered fresh becomes the canonical character; later CHARACTER shots are edits of
         it (FLUX.2-klein, multipart). Insert/environment shots are rendered fresh — they have no
         face, so nothing needs conditioning.
  sheet  one character sheet (2x3 grid) cropped into panels. Identity is identical pixels.
  fresh  every shot rendered independently.

Every render goes to Agnes first when an Agnes key is configured, then to Antigravity, then to
Workers AI. All strategies stop on Workers AI quota/auth errors instead of hammering the API,
and record what ran.
"""

import base64
import io
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import agnes
import gemini
from scenes import STYLE, api_error_detail, is_quota_error

logger = logging.getLogger(__name__)

# Verified against GET /accounts/{id}/ai/models/search?task=Text-to-Image.
SCHNELL = "@cf/black-forest-labs/flux-1-schnell"
KLEIN = "@cf/black-forest-labs/flux-2-klein-4b"

VALID_STRATEGIES = ("edit", "sheet", "fresh")

# A frame whose mean |Δluma| against the canonical frame is under this is a near-copy. Measured on
# a collapsed render: five shots planned as close-up, medium, wide, close-up and medium all landed
# at 1.5-7.1, while the shots that actually changed sat above 12. The score is 64x64 greyscale, so
# it ignores resolution and hue and only asks "is this the same picture again".
NEAR_COPY_DIFF = 8.0
_THUMB_SIZE = (64, 64)

# Model-level failures (switch strategy) vs spend failures (stop the run).
_FATAL = ("auth_", "plan_restricted", "model_missing")
_STOP = _FATAL + ("quota_exhausted",)

# A shot that fails transiently is re-rendered in-run. Losing one frame loses its narration line
# too — the orchestrator pairs frames to shots and drops anything unpaired — so a single 500 used
# to truncate the story. Attempts are still spent against the daily budget, so the retries are
# bounded by the same limit that bounds the run.
SHOT_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.5


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
    """Map a failed Workers AI image call to the pipeline's failure vocabulary.

    Quota is detected from the body as well as the status: 4006 ('used up your daily free
    allocation of 10,000 neurons') arrives as a 429, but the allocation is shared with the
    planner, which is why the run has to stop rather than retry.
    """
    if status == 429 or is_quota_error(text):
        return "quota_exhausted"
    if status in (401, 403):
        return "plan_restricted" if "free plan" in text.lower() else f"auth_{status}"
    if status in (404, 410) or (status == 400 and "7000" in text):
        return "model_missing"
    return f"status_{status}"


def _failed(label: str, resp: requests.Response) -> ImageResult:
    """Log why an image call failed, then map it to the error vocabulary."""
    logger.error("Workers AI image %s -> %s (%s)", label, resp.status_code,
                 api_error_detail(resp))
    return ImageResult(ok=False, error=_classify(resp.status_code, resp.text))


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
        return _failed(SCHNELL, resp)
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
        return _failed(KLEIN, resp)
    try:
        return ImageResult(ok=True, data=decode_image_response(resp))
    except ValueError as exc:
        return ImageResult(ok=False, error=f"bad_response:{exc}")


def fresh_image(prompt: str, cfa_token: str, account_id: str, *,
                antigravity_key: str = "", agnes_key: str = "") -> ImageResult:
    """One text-to-image render: Antigravity, then Agnes, then Workers AI.

    The order is the *edit* path's order, not a quality ranking. A short has to be drawn by one
    model family or the art style changes at the cut: an Agnes canonical frame next to Antigravity
    edits (`gemini-3.1-flash-image`) read as two different shows — darker and painterly against
    flatter cel shading — so the provider that can render the edits also renders the frames they
    are edited from. With no Antigravity key there is nothing to match and Agnes goes first: it
    answers 9:16 at 2K (1472x2624) for free, where Gemini's default is 1K.

    Any refusal (rate limit, quota, safety block, timeout) falls through to the next provider
    rather than failing the shot.
    """
    if antigravity_key:
        res = gemini.image(prompt, key=antigravity_key)
        if res.ok and res.data:
            return ImageResult(ok=True, data=res.data)
        logger.warning("Antigravity image failed (%s); falling back to Agnes",
                       res.error or "unknown")
    if agnes_key:
        res = agnes.image(prompt, key=agnes_key)
        if res.ok and res.data:
            return ImageResult(ok=True, data=res.data)
        logger.warning("Agnes image failed (%s); falling back to Workers AI",
                       res.error or "unknown")
    return generate_fresh(prompt, cfa_token, account_id)


def edit_image(reference: bytes, prompt: str, cfa_token: str, account_id: str, *,
               antigravity_key: str = "") -> ImageResult:
    """One identity-conditioned edit: Antigravity, then the Workers AI edit model.

    Antigravity first is measured, not a preference. Handed the same prompt and the same canonical
    frame, `gemini-3.1-flash-image` re-staged the character as the shot asked — back against the
    wall, one hand on the stone, the other on his chest, braced legs — while
    `agnes-image-2.5-flash` returned the canonical frame again (three-quarter turn, hand out at the
    wall). Agnes is therefore not a fallback here: an edit that cannot leave the reference's pose
    is the defect this path exists to avoid, so its refusal sends the shot to `fresh_image`, where
    nothing is copied and the planned action survives.
    """
    if antigravity_key:
        res = gemini.image(prompt, key=antigravity_key, reference=reference)
        if res.ok and res.data:
            return ImageResult(ok=True, data=res.data)
        logger.warning("Antigravity image edit failed (%s); not falling back to Agnes, which "
                       "would repeat the reference's pose", res.error or "unknown")
    return generate_edit(reference, prompt, cfa_token, account_id)


def frame_index(path: Path) -> int:
    """Numeric shot index of a frame file."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10 ** 6


def load_frames(img_dir: Path) -> list[Path]:
    """Frames in shot order.

    Sorted numerically, not lexicographically: `sorted()` puts `frame_10` before `frame_2`, which
    silently reorders every short long enough to reach double digits.
    """
    extension_rank = {".jpg": 0, ".jpeg": 0, ".png": 1, ".webp": 2}
    best: dict[int, Path] = {}
    for path in sorted(img_dir.glob("frame_*.*"),
                       key=lambda p: (frame_index(p), extension_rank.get(p.suffix.lower(), 3))):
        best.setdefault(frame_index(path), path)
    return [best[index] for index in sorted(best)]


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


def _fill_holds(slots: list[Path | None]) -> list[Path]:
    """Hold a neighbouring frame for any beat whose image never rendered.

    Returned frames stay aligned with the prompts that produced them, so the beat still plays:
    the picture holds while its narration is spoken, instead of the shot and its line both
    disappearing from the story.
    """
    out: list[Path | None] = []
    previous: Path | None = None
    held = 0
    for path in slots:
        if path is None:
            out.append(previous)
            held += 1
        else:
            previous = path
            out.append(path)
    first = next((p for p in out if p is not None), None)
    if first is None:
        return []
    if held:
        logger.warning("Held a neighbouring frame for %d beat(s) that never rendered", held)
    return [p if p is not None else first for p in out]


def _fresh_with_retry(
    prompt: str,
    cfa_token: str,
    account_id: str,
    budget: dict,
    shot_no: int,
    *,
    attempts: int = SHOT_ATTEMPTS,
    backoff_s: float | None = None,
    reserve: int = 0,
    antigravity_key: str = "",
    agnes_key: str = "",
) -> tuple[ImageResult | None, str]:
    """Render one fresh image, retrying transient failures within the daily budget.

    Returns `(result, reason)` where reason is one of:
      ok        — rendered; `result.data` holds the bytes
      failed    — out of attempts; the caller holds the neighbouring frame
      fatal     — stop the run (quota/auth/model-missing): retrying only spends more
      no_budget — the day's allowance is gone; stop asking for more images

    `reserve` is the number of shots still to come. A retry is skipped when spending it would
    leave those shots without their first attempt — starving the tail of the story is worse than
    a held frame, because the arc then ends early.
    """
    last = "failed"
    backoff_s = RETRY_BACKOFF_S if backoff_s is None else backoff_s
    for attempt in range(1, attempts + 1):
        if budget["attempts"] >= budget["max"]:
            return None, "no_budget"
        if attempt > 1 and budget["attempts"] + reserve >= budget["max"]:
            logger.warning("shot %d: not retrying, %d shot(s) still need the image budget",
                           shot_no, reserve)
            return None, "failed"
        budget["attempts"] += 1
        res = fresh_image(prompt, cfa_token, account_id, antigravity_key=antigravity_key,
                          agnes_key=agnes_key)
        if res.ok:
            if attempt > 1:
                logger.info("shot %d: rendered on attempt %d/%d", shot_no, attempt, attempts)
            return res, "ok"
        last = res.error or "failed"
        if last.startswith(_STOP):
            return None, "fatal"
        if attempt < attempts:
            logger.warning("shot %d: attempt %d/%d failed (%s); retrying",
                           shot_no, attempt, attempts, last)
            if backoff_s:
                time.sleep(backoff_s * attempt)
    logger.error("shot %d: giving up after %d attempts (%s); holding the previous frame",
                 shot_no, attempts, last)
    return None, "failed"


def luma_thumb(data: bytes) -> np.ndarray | None:
    """64x64 greyscale thumbnail of an encoded image, or None when the bytes are not decodable."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("L").resize(_THUMB_SIZE), dtype=np.float32)
    except Exception:  # noqa: BLE001 — an undecodable payload must not abort a render
        return None


def luma_diff(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Mean absolute luma difference between two thumbnails, or None if either is missing."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.abs(a - b).mean())


def edit_instruction(prompt: str, character: str = "", *, strict: bool = False) -> str:
    """Identity-conditioned edit instruction: shot first, identity last, reframing explicit.

    This used to open with `Keep the EXACT same character — identical face, hair and outfit as the
    reference image`, which made the reference the first and loudest thing in the request: Agnes
    then reproduced the reference composition for five of eight shots planned as close-up, medium,
    wide, close-up and medium (mean luma diff 1.5-7.1 between frames the plan had made different).
    The lesson the plan side already learned — a leading identity paragraph collapses every frame
    into the same portrait — holds for the edit call too. So the shot leads, identity is one
    trailing clause, and the instruction states what has to change. `strict` is the second attempt
    after a render came back as a near-copy of the reference.

    The pose is named separately and on purpose: an edit that only says "same character" keeps the
    reference's stance, head angle, hands and expression, which is how a short ends up with four
    backgrounds and one body. The prompt carries the planned action as `Pose and body: …`; this
    states that the reference supplies the design and nothing else.
    """
    body = " ".join((prompt or "").split()).strip()
    ident = (f"the same character as the reference ({character})" if character
             else "the same character as the reference")
    redraw = ("Treat the reference as a face and costume reference only: ignore its pose, crop, "
              "camera angle and background entirely and draw a completely different frame."
              if strict else
              "A new camera angle, a new framing and a different background from the reference.")
    return (f"{body} {redraw} Carry over only {ident} — the same face, hair and outfit — and "
            "redraw that character performing the action described above. The reference's "
            "stance, head angle, hand position and expression are not this frame's pose and "
            "must not be copied.")


def produce_images(
    prompts: list[str],
    *,
    cfa_token: str,
    account_id: str,
    img_dir: Path,
    strategy: str = "fresh",
    character_flags: list[bool] | None = None,
    character: str = "",
    budget: dict | None = None,
    antigravity_key: str = "",
    agnes_key: str = "",
) -> tuple[list[Path], str]:
    """Render one image per prompt. Returns (frames, strategy_used).

    `character_flags[i]` marks prompts where the arc's protagonist is on screen; only those take
    part in the canonical-frame conditioning, so inserts and establishing shots stay
    independent. `character` is the arc's own identity signature, so each arc conditions on its
    own face instead of a shared one. `agnes_key` puts Agnes first (2K renders), `antigravity_key`
    puts Antigravity next; with both empty the renders go straight to Workers AI.
    """
    if strategy not in VALID_STRATEGIES:
        logger.warning("Unknown image strategy %r, using 'fresh'", strategy)
        strategy = "fresh"
    if strategy == "edit" and not antigravity_key:
        # The edit path is only as good as the model that renders it. Antigravity follows the
        # action; Agnes reproduces the reference's pose from the identical prompt and reference
        # (measured on the same shot: Agnes returned the canonical three-quarter turn, Gemini
        # returned the planned back-against-the-wall), and Workers AI's editor is untested there.
        # Without the provider that can re-stage, an edit freezes the story's body language — so
        # render fresh and let the character signature carry the design.
        logger.info("No Antigravity image key: rendering fresh, since an edit through the "
                    "remaining providers repeats the reference's pose")
        strategy = "fresh"
    img_dir.mkdir(parents=True, exist_ok=True)
    budget = budget if budget is not None else {"attempts": 0, "max": 10_000}
    flags = character_flags or [True] * len(prompts)

    def affordable() -> bool:
        return budget["attempts"] < budget["max"]

    # Every frame already rendered, so a new one can be checked against all of them and not only
    # against the canonical character frame. The canonical-only check let a shot repeat a
    # *neighbouring* shot: measured, shot 6 came back as a near-copy of shot 2 (side view, hand
    # out, same wall) while sitting 6.6 away from the canonical, under the guard's nose.
    prior_thumbs: list[np.ndarray] = []

    def nearest_copy(data: bytes) -> tuple[float | None, int | None]:
        """Smallest luma difference to a frame already rendered, and that frame's ordinal."""
        thumb = luma_thumb(data)
        best: float | None = None
        where: int | None = None
        for ordinal, other in enumerate(prior_thumbs, 1):
            diff = luma_diff(thumb, other)
            if diff is not None and (best is None or diff < best):
                best, where = diff, ordinal
        return best, where

    def remember(data: bytes) -> None:
        thumb = luma_thumb(data)
        if thumb is not None:
            prior_thumbs.append(thumb)

    # --- sheet -----------------------------------------------------------------
    if strategy == "sheet" and affordable():
        n = len(prompts)
        sig = f"{character}, " if character else ""
        sheet_prompt = (
            f"{STYLE}. Character reference sheet: {sig}shown {n} times in a "
            f"3x{(n + 2) // 3} grid of equal panels with no borders, each panel a different pose, "
            "full body, identical design in every panel.")
        budget["attempts"] += 1
        res = fresh_image(sheet_prompt, cfa_token, account_id, antigravity_key=antigravity_key,
                          agnes_key=agnes_key)
        if res.ok:
            panels = crop_grid(_save(res.data, img_dir, 99), img_dir, n)
            if len(panels) == n:
                logger.info("Character sheet split into %d panels", len(panels))
                return panels, "sheet"
        logger.warning("Sheet strategy failed (%s); falling back to fresh", res.error)
        strategy = "fresh"

    # --- edit ------------------------------------------------------------------
    if strategy == "edit":
        frames: list[Path | None] = []
        canonical: bytes | None = None
        edits = 0
        stopped = False

        for i, prompt in enumerate(prompts):
            wants_character = flags[i] if i < len(flags) else True
            if canonical is None or not wants_character or strategy != "edit":
                res, reason = _fresh_with_retry(prompt, cfa_token, account_id, budget, i + 1,
                                                reserve=len(prompts) - i - 1,
                                                antigravity_key=antigravity_key,
                                                agnes_key=agnes_key)
                if res is None:
                    if reason == "fatal":
                        stopped = True
                        break
                    if reason == "no_budget":
                        logger.warning("Daily image budget exhausted (%d/%d)",
                                       budget["attempts"], budget["max"])
                        break
                    # Out of attempts: hold the neighbouring frame rather than lose the beat.
                    frames.append(None)
                    continue
                if canonical is None and wants_character:
                    canonical = res.data
                    logger.info("shot %d: canonical character rendered", i + 1)
                repeat, twin = nearest_copy(res.data)
                if repeat is not None and repeat < NEAR_COPY_DIFF:
                    logger.warning("shot %d: fresh render repeats frame %d (luma diff %.1f); "
                                   "rendering it rather than holding it twice", i + 1, twin,
                                   repeat)
                remember(res.data)
                frames.append(_save(res.data, img_dir, i))
                continue

            if not affordable():
                logger.warning("Daily image budget exhausted (%d/%d)", budget["attempts"], budget["max"])
                break
            budget["attempts"] += 1
            res = edit_image(canonical, edit_instruction(prompt, character),
                             cfa_token, account_id,
                             antigravity_key=antigravity_key)
            if res.ok:
                diff, twin = nearest_copy(res.data)
                if diff is not None and diff < NEAR_COPY_DIFF:
                    logger.warning("shot %d: edit repeats frame %d (luma diff %.1f of %.0f); "
                                   "asking again with the reference limited to identity",
                                   i + 1, twin, diff, NEAR_COPY_DIFF)
                    if affordable():
                        budget["attempts"] += 1
                        retry = edit_image(
                            canonical, edit_instruction(prompt, character, strict=True),
                            cfa_token, account_id, antigravity_key=antigravity_key)
                        if retry.ok:
                            res = retry
                            diff, twin = nearest_copy(res.data)
                    else:
                        logger.warning("shot %d: no image budget left to retry the near-copy",
                                       i + 1)
                    if diff is not None and diff < NEAR_COPY_DIFF:
                        # Retrying the edit rarely helps: an edit is anchored to the reference, so
                        # it reproduces the reference's staging again. A fresh render has no
                        # reference to copy and can actually change the body, so it is the only
                        # remaining move that can separate two beats that look alike.
                        logger.warning("shot %d: still a near-copy of frame %d (luma diff "
                                       "%.1f); rendering it fresh instead", i + 1, twin, diff)
                        if affordable():
                            budget["attempts"] += 1
                            fresher = fresh_image(prompt, cfa_token, account_id,
                                                  antigravity_key=antigravity_key,
                                                  agnes_key=agnes_key)
                            if fresher.ok:
                                fresh_diff, fresh_twin = nearest_copy(fresher.data)
                                if fresh_diff is not None and fresh_diff > diff:
                                    logger.info("shot %d: fresh render is %.1f away from its "
                                                "closest frame", i + 1, fresh_diff)
                                    res, diff, twin = fresher, fresh_diff, fresh_twin
                                else:
                                    logger.warning("shot %d: fresh render is no further from "
                                                   "the earlier frames; keeping the edit", i + 1)
                        else:
                            logger.warning("shot %d: no image budget left for a fresh render",
                                           i + 1)
                        if diff is not None and diff < NEAR_COPY_DIFF:
                            logger.warning("shot %d: still a near-copy of frame %d (luma diff "
                                           "%.1f); keeping it", i + 1, twin, diff)
                elif diff is not None:
                    logger.info("shot %d: edit of the canonical character (closest earlier frame "
                                "%.1f away)", i + 1, diff)
                else:
                    logger.info("shot %d: edit of the canonical character", i + 1)
                remember(res.data)
                edits += 1
                frames.append(_save(res.data, img_dir, i))
                continue

            if res.error.startswith(("quota_exhausted", "auth_")):
                # An exhausted edit model must not end the short: the remaining beats can still be
                # drawn fresh, which also frees them from the reference's pose.
                logger.warning("Edit model unavailable for shot %d (%s); rendering the rest "
                               "fresh", i + 1, res.error)
                strategy = "fresh"
            else:
                logger.warning("Edit failed for shot %d (%s); rendering it fresh", i + 1,
                               res.error)
                if res.error.startswith(_FATAL):
                    strategy = "fresh"
                    logger.warning("Edit model unusable (%s); remaining shots render fresh",
                                   res.error)
            res, reason = _fresh_with_retry(prompt, cfa_token, account_id, budget, i + 1,
                                            reserve=len(prompts) - i - 1,
                                            antigravity_key=antigravity_key, agnes_key=agnes_key)
            if res is not None:
                frames.append(_save(res.data, img_dir, i))
            elif reason == "fatal":
                stopped = True
                break
            elif reason == "no_budget":
                logger.warning("Daily image budget exhausted (%d/%d)",
                               budget["attempts"], budget["max"])
                break
            else:
                frames.append(None)

        filled = _fill_holds(frames)
        if filled:
            used = "edit" if edits else "fresh"
            logger.info("Strategy '%s': %d shot(s) conditioned on the canonical frame", used, edits)
            return filled, used
        if stopped:
            return [], strategy
        strategy = "fresh"

    # --- fresh -----------------------------------------------------------------
    slots: list[Path | None] = []
    for i, prompt in enumerate(prompts):
        res, reason = _fresh_with_retry(prompt, cfa_token, account_id, budget, i + 1,
                                        reserve=len(prompts) - i - 1,
                                        antigravity_key=antigravity_key, agnes_key=agnes_key)
        if res is not None:
            slots.append(_save(res.data, img_dir, i))
            continue
        if reason == "no_budget":
            logger.warning("Daily image budget exhausted (%d/%d)",
                           budget["attempts"], budget["max"])
            break
        if reason == "fatal":
            break
        slots.append(None)
    return _fill_holds(slots), "fresh"
