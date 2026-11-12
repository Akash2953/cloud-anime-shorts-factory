"""Daily production pipeline: narrated vertical short, one upload per day.

    1. Pick the next narration part (a new multi-part series is created when none is pending)
    2. Render one image per beat  (strategy: edit -> canonical frame + conditioned edits)
    3. Kokoro narration, paced into the accepted duration window
    4. Render: each image held for its own line, Ken Burns, AAC, loudness-normalised
    5. Upload to YouTube as private

Every stage degrades instead of crashing, so an unattended run either publishes or leaves
the queue untouched for tomorrow.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from audio import Narration, join_with_gaps, write_wav, SAMPLE_RATE
from config import load_config, Limits
from images import load_frames, produce_images
from publish import exchange_refresh_token, upload_video
from render import render_narration
from scenes import (MAX_PARTS, Shot, make_character, plan_arc, plan_shorts_detailed,
                    reuse_image_refs, shots_for_duration, unique_prompts,
                    validate_story_position)
from state import compute_fingerprint, parse_state, save_state, today_key

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BEAT_GAP = 0.35      # breath between narration lines
TAIL_HOLD = 0.6      # last frame held after the voice stops
MIN_CUT = 0.8        # shortest a still may be held, so a silent beat cannot flash past



def load_state(path: Path) -> dict:
    return parse_state(path.read_text(encoding="utf-8") if path.exists() else None)


def advance_series(state: dict, uploaded_part: int) -> None:
    """The queue only moves after a successful upload, so a failed day retries the same part."""
    series = state.get("series") or {}
    if series.get("parts"):
        series["next_part"] = max(int(series.get("next_part", 0)), uploaded_part)


def resolve_part(cfg: dict, state: dict, *, part: int | None = None,
                 plan_only: bool = False) -> tuple[list[Shot], int, int, dict]:
    """Return (shots, part_number, total_parts, state). Creates or extends the series as needed.

    `part` asks for one specific part without consulting (or advancing) the publish queue — that
    is how a later part gets planned and previewed while an earlier one is still unpublished.
    """
    # plan-only mode ignores the existing series so every call spins up a fresh story.
    if plan_only:
        parts, index, pinned, total_parts, auto = [], 0, 0, 1, False
    else:
        series = state.get("series") or {}
        parts = series.get("parts") or []
        index = (int(part) - 1) if part else int(series.get("next_part", 0))
        requested = cfg.get("SERIES_PARTS") or 1
        auto = isinstance(requested, str) and requested.lower() == "auto"
        pinned = int(series.get("total_parts") or 0)  # an arc in flight keeps the length it began with
        total_parts = pinned or (1 if auto else int(requested))
        if part and index > len(parts):
            raise RuntimeError(f"part {part} needs parts 1..{index} planned first "
                               f"(only {len(parts)} stored)")
        if part and index < total_parts:
            total_parts = max(total_parts, index + 1)
        elif parts and index >= total_parts:
            logger.info("Series %s complete (%d/%d parts uploaded); starting a new story",
                        series.get("id"), len(parts), total_parts)
            parts, index = [], 0
            pinned = 0
            total_parts = 1 if auto else int(requested)

        if parts and index < len(parts):
            shots = [Shot.from_dict(d) for d in parts[index]]
            if series.get("total_parts") != total_parts:
                # Pin the arc on the first run that uses it: changing SERIES_PARTS mid-arc must not
                # make a part 2 that follows an uploaded "#1/3" call itself "#2 of 5".
                series["total_parts"] = total_parts
            # An in-flight arc keeps the protagonist it opened with, even for state written before
            # a character was pinned: a signature regenerated from the id is stable either way.
            series.setdefault("character", make_character(series.get("id", "arc")))
            state["series"] = series
            logger.info("Continuing series %s — part %d/%d (%d shots)",
                        series.get("id"), index + 1, total_parts, len(shots))
            return shots, index + 1, total_parts, state

    target_seconds = float(cfg.get("PREFERRED_SECONDS")
                           or (cfg["MIN_SECONDS"] + cfg["MAX_SECONDS"]) // 2)
    want = int(cfg.get("SHOTS_PER_PART") or shots_for_duration(target_seconds))
    part_number = index + 1
    # Each arc draws its own protagonist, fixed for the whole arc. A new story (index 0) gets a
    # signature seeded from today's id; an in-flight continuation keeps the one it opened with.
    if index == 0:
        character = make_character(today_key())
    else:
        character = series.setdefault("character", make_character(series.get("id", "arc")))
    # `SERIES_PARTS=auto`: part 1 of a new story also decides how long the arc is. A series with
    # no pinned length that is already past part 1 (hand-made state) closes here instead.
    decide_arc = bool(auto and not pinned and part_number == 1 and not part)
    if auto and not pinned and not decide_arc:
        total_parts = part_number
    story_so_far = [d.get("narration", "") for earlier in parts for d in earlier]
    if total_parts > 1:
        logger.info("Planning part %d of %d (%d line(s) of story so far)",
                    part_number, total_parts, len(story_so_far))

    shots, image_prompts, fallback, declared_parts = plan_shorts_detailed(
        theme="", character=character, api_key=cfg.get("OPENROUTER_API_KEY", ""),
        want=want, target_seconds=target_seconds,
        part=part_number, total_parts=None if decide_arc else total_parts,
        story_so_far=story_so_far,
        parts_range=(int(cfg.get("PARTS_MIN") or 1), int(cfg.get("PARTS_MAX") or MAX_PARTS)),
    )
    degraded = fallback
    if decide_arc and not fallback:
        total_parts = declared_parts or 1
        logger.info("Arc length for this story: %d part(s)", total_parts)
    if not shots:
        arc, fallback = plan_arc(
            theme="", parts=1, per_part=want, api_key=cfg.get("OPENROUTER_API_KEY", ""),
        )
        shots = arc[0] if arc else []
        image_prompts = {}
        degraded = True  # the arc planner never sees the story so far, so it cannot continue one
    if not shots:
        raise RuntimeError("No usable shot plan from any provider")

    for problem in validate_story_position(shots, part_number, total_parts):
        logger.warning("Story position: %s", problem)

    logger.info("Plan: %d shots, roles=%s", len(shots), [s.narrative_role for s in shots])
    for i, shot in enumerate(shots):
        char = ", character" if shot.character else ", no character"
        logger.info("  shot %2d [%s/%s%s]", i + 1, shot.shot_size, shot.narrative_role, char)

    if degraded:
        if part_number > 1:
            logger.warning("Planner fell back to a generic plan for part %d; rendering it but not "
                           "saving it, so the next run re-plans this part instead of continuing "
                           "a story it never read", part_number)
            return shots, part_number, total_parts, state
        logger.warning("Template plan (no provider answered): rendering it but not saving the "
                       "series, so tomorrow retries the real providers")
        return shots, part_number, total_parts, state

    if parts:  # next part of a running series: append, never overwrite the earlier parts
        series["parts"] = parts + [[s.as_dict() for s in shots]]
        series["total_parts"] = total_parts
        state["series"] = series
        logger.info("Series %s: part %d/%d planned (%d parts stored)",
                    series.get("id"), part_number, total_parts, len(series["parts"]))
        return shots, part_number, total_parts, state

    series = {
        "id": today_key(),
        "theme": "",
        "character": character,  # this arc's protagonist, so it is not shared across arcs
        "parts": [[s.as_dict() for s in shots]],
        "total_parts": total_parts,
        "next_part": 0,  # part 1 is pending until an upload actually succeeds
        "strategy": cfg.get("IMAGE_STRATEGY", "edit"),
    }
    state["series"] = series
    logger.info("New series %s: %d part(s), part 1 of %d planned (%d shots)",
                series["id"], total_parts, total_parts, len(shots))
    return shots, 1, total_parts, state


def beat_track(shots: list[Shot], spoken_segments: list[np.ndarray], *,
               gap: float = BEAT_GAP, tail: float = TAIL_HOLD,
               min_cut: float = MIN_CUT) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Lay the spoken lines back onto the full beat list and time the stills to them.

    A beat whose narration is `none` gets its planned `seconds` of silence instead of a spoken
    line, and each still is held for its own line, so the cut cannot truncate the voice.
    Returns (per-beat audio, full track, per-beat hold in seconds).
    """
    spoken = [i for i, shot in enumerate(shots) if shot.speech]
    beats = [np.zeros(int(round(shot.seconds * SAMPLE_RATE)), dtype=np.float32) for shot in shots]
    for i, segment in zip(spoken, spoken_segments):
        beats[i] = segment
    track, _starts = join_with_gaps(beats, gap_seconds=gap)
    track = np.concatenate([track, np.zeros(int(tail * SAMPLE_RATE), dtype=np.float32)])
    holds = [max(len(beat) / SAMPLE_RATE + (tail if i == len(beats) - 1 else gap), min_cut)
             for i, beat in enumerate(beats)]
    return beats, track, holds


def run_pipeline(*, render_only: bool = False, offline_images: bool = False,
                 reuse_frames: bool = False, plan_only: bool = False, part: int | None = None) -> None:
    cfg = load_config()
    state = load_state(Path("state.json"))
    fps = 24
    shots, part_num, total_parts, state = resolve_part(cfg, state, part=part, plan_only=plan_only)
    if plan_only:
        Path("state.json").write_text(save_state(state), encoding="utf-8")
        logger.info("Plan-only: wrote %d shot(s), part %d/%d — stopping before images/render/upload",
                    len(shots), part_num, total_parts)
        return
    target_seconds = cfg.get("PREFERRED_SECONDS") or (cfg["MIN_SECONDS"] + cfg["MAX_SECONDS"]) // 2
    strategy = cfg.get("IMAGE_STRATEGY", "edit")
    # The arc's own protagonist. For state written before a character was pinned, regenerate it
    # deterministically from the arc id so an old run still conditions on one consistent face.
    character = ((state.get("series") or {}).get("character")
                 or make_character((state.get("series") or {}).get("id", "arc")))

    if reuse_frames:
        img_dir = Path("build/frames")
        frames = load_frames(img_dir)
        if len(frames) >= len(shots):
            frames = frames[:len(shots)]
            logger.info("Reused %d existing frames; skipping image generation", len(frames))
        else:
            logger.warning("Only %d existing frames but need %d", len(frames), len(shots))
            frames = []
    elif offline_images:
        from images import OFFLINE_IMAGE_DATA
        img_dir = Path("build/frames")
        img_dir.mkdir(parents=True, exist_ok=True)
        n = len(shots)
        frames = [images._save(OFFLINE_IMAGE_DATA, img_dir, i) for i in range(n)]
        strategy = "offline"
    else:
        img_dir = Path("build/frames")
        daily_cfg = state.setdefault("daily", {}).setdefault(today_key(), {})
        budget = {"attempts": daily_cfg.get("image_attempts", 0),
                  "max": int(cfg.get("MAX_IMAGE_ATTEMPTS") or Limits.max_image_attempts_per_day)}
        try:
            # Use detailed image prompts if available, otherwise fall back to prompt() builder.
            # Duplicate prompts collapse to one render: the frame is held for every beat that
            # references it, so the daily image allowance buys more shots.
            image_prompts_raw = (state.get("series") or {}).get("image_prompts", {})
            shots = reuse_image_refs(shots, int(budget["max"]))
            final_prompts, slots = unique_prompts(shots, image_prompts_raw, character)
            logger.info("Rendering %d image(s) for %d shots", len(final_prompts), len(shots))

            chars = [any(shot.character for shot, slot in zip(shots, slots) if slot == k)
                     for k in range(len(final_prompts))]
            rendered, strategy = produce_images(
                final_prompts, cfa_token=cfg["CFAI_TOKEN"],
                account_id=cfg["CF_ACCOUNT_ID"], img_dir=img_dir,
                strategy=strategy, character_flags=chars, character=character, budget=budget,
            )
            paired = [(shot, rendered[slot]) for shot, slot in zip(shots, slots)
                      if slot < len(rendered)]
            shots = [shot for shot, _ in paired]
            frames = [frame for _, frame in paired]
        except Exception as exc:
            logger.error("Image production failed (%s); aborting", exc)
            raise
        state.setdefault("daily", {})[today_key()]["image_attempts"] = budget["attempts"]
        Path("state.json").write_text(save_state(state), encoding="utf-8")
        if budget["attempts"] >= int(cfg.get("MAX_IMAGE_ATTEMPTS") or Limits.max_image_attempts_per_day):
            logger.warning("Daily image budget exhausted (%d/%d)",
                           budget["attempts"], int(cfg.get("MAX_IMAGE_ATTEMPTS") or Limits.max_image_attempts_per_day))

    if not frames:
        logger.error("No images produced — aborting, upload counter untouched.")
        return

    if len(frames) != len(shots):
        logger.warning("Frame count %d doesn't match shot count %d; truncating to %d",
                       len(frames), len(shots), min(len(frames), len(shots)))
        shots = shots[:len(frames)]

    narration_tts = Narration(voice=cfg["KOKORO_VOICE"])
    speech = [shot.speech for shot in shots]
    spoken = [i for i, text in enumerate(speech) if text]
    silent_hold = sum(shot.seconds for shot in shots if not shot.speech)
    if len(spoken) < len(shots):
        logger.info("%d of %d beats are silent: held %.1fs without voice",
                    len(shots) - len(spoken), len(shots), silent_hold)
    if not spoken:
        logger.warning("Every beat is silent; the short will render without narration")

    # Only spoken beats are synthesised; the silent ones become their planned hold.
    budget = target_seconds - BEAT_GAP * (len(shots) - 1) - TAIL_HOLD - silent_hold
    raw = [narration_tts.wav(speech[i]) for i in spoken]
    scaled, speed = (narration_tts.synth_scaled([speech[i] for i in spoken], target_seconds=budget)
                     if spoken else ([], 1.0))
    natural = sum(len(a) for a in scaled) / SAMPLE_RATE

    beats, audio, durations = beat_track(shots, scaled)
    total = sum(durations)
    speech_duration = sum(len(b) for b in beats) / SAMPLE_RATE
    # seconds of audio per second of video: must stay ~1.0, so a >6 reading means real drift.
    cut_rate = (len(audio) / SAMPLE_RATE) / total if total else 999
    logger.info("Natural speech %.1fs; scaled by %.2f -> %.1fs speech, %.1fs total",
                natural, speed, speech_duration, total)

    if speed < 0.6 or cut_rate > 6:
        logger.warning("Speech scaling too extreme (%.2fx); using the natural read", speed)
        beats, audio, durations = beat_track(shots, raw)
        total = sum(durations)
        speed = 1.0

    audio_path = Path("build/narration.wav")
    write_wav(audio_path, audio)

    logger.info("Render: %d beats, %.1fs total, audio=narration.wav",
                len(durations), total)
    video_path = Path("output_short.mp4")
    if not render_narration(frames, durations, audio_path, video_path, fps=fps):
        logger.error("Rendering failed; aborting before upload")
        return

    if render_only:
        logger.info("Render-only: wrote %s (%s); frames kept for --reuse-frames",
                     video_path, strategy)
        return

    access_token = exchange_refresh_token(
        cfg["YOUTUBE_CLIENT_ID"], cfg["YOUTUBE_CLIENT_SECRET"], cfg["YOUTUBE_REFRESH_TOKEN"])
    if not access_token:
        logger.error("YouTube token exchange failed; queue untouched, retrying tomorrow")
        return
    title = f"Anime Short #{part_num}/{total_parts} — {state.get('series', {}).get('id', 'new')}"
    description_lines = [
        f"# Part {part_num} of {total_parts}",
        "",
        "# Narration:",
    ]
    for i, line in enumerate((s.speech for s in shots if s.speech), 1):
        description_lines.append(f"{i}. {line}")
    description = "\n".join(description_lines)
    tags = ["anime", "shorts", "storytime", "mystery", "lantern",
            "rainy city", "vertical video"]
    privacy = cfg.get("PUBLISH_PRIVACY", "private")
    video_id = upload_video(video_path.read_bytes(), title, description, access_token,
                            privacy_status=privacy, tags=tags)
    if not video_id:
        logger.error("Upload failed; queue untouched, retrying the same part tomorrow")
        return

    daily = state.setdefault("daily", {}).setdefault(today_key(), {})
    daily["uploads"] = daily.get("uploads", 0) + 1
    advance_series(state, part_num)
    Path("state.json").write_text(save_state(state), encoding="utf-8")
    logger.info("Uploaded %s; daily uploads=%d/%d, strategy=%s, next_part=%s",
                video_id, daily["uploads"], Limits.max_daily_uploads, strategy,
                (state.get("series") or {}).get("next_part"))
    if daily["uploads"] >= Limits.max_daily_uploads:
        logger.info("Upload limit reached today — skipping further publishing this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Anime Shorts Factory pipeline")
    parser.add_argument("--render-only", action="store_true",
                        help="stop after rendering and skip upload")
    parser.add_argument("--plan-only", action="store_true",
                        help="generate a fresh story only; write state and exit before images/render/upload")
    parser.add_argument("--offline-images", action="store_true",
                        help="generate placeholder frames (no API key required)")
    parser.add_argument("--reuse-frames", action="store_true",
                        help="skip regeneration; reuse build/frames/frame_*.ext")
    parser.add_argument("--part", type=int, default=None, metavar="N",
                        help="plan/preview part N of the arc without touching the publish queue")
    args = parser.parse_args()
    run_pipeline(render_only=args.render_only, offline_images=args.offline_images,
                 reuse_frames=args.reuse_frames, plan_only=args.plan_only, part=args.part)
