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
from images import produce_images
from publish import exchange_refresh_token, upload_video
from render import render_narration
from scenes import Shot, plan_arc, plan_shorts_detailed, shots_for_duration
from state import compute_fingerprint, parse_state, save_state, today_key

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BEAT_GAP = 0.35      # breath between narration lines
TAIL_HOLD = 0.6      # last frame held after the voice stops
THEME = (
    "cyberpunk anime, floating paper lanterns over a rain-soaked city, "
    "a hooded girl searching for her missing brother"
)


def load_state(path: Path) -> dict:
    return parse_state(path.read_text(encoding="utf-8") if path.exists() else None)


def resolve_part(cfg: dict, state: dict) -> tuple[list[Shot], int, int, dict]:
    """Return (shots, part_number, total_parts, series). Creates the series if needed."""
    series = state.get("series") or {}
    parts = series.get("parts") or []
    index = int(series.get("next_part", 0))

    if parts and index < len(parts):
        shots = [Shot.from_dict(d) for d in parts[index]]
        return shots, index + 1, len(parts), state

    shots, image_prompts, fallback = plan_shorts_detailed(
        theme=THEME, cfa_token=cfg["CFAI_TOKEN"], account_id=cfg["CF_ACCOUNT_ID"],
        api_key=cfg.get("OPENROUTER_API_KEY", ""),
    )
    if not shots:
        shots, _ = plan_arc(
            theme=THEME, parts=1, per_part=shots_for_duration(cfg["TARGET_SECONDS"]),
            cfa_token=cfg["CFAI_TOKEN"], account_id=cfg["CF_ACCOUNT_ID"],
            api_key=cfg.get("OPENROUTER_API_KEY", ""),
        )

    if cfg.get("PREFERRED_SECONDS"):
        shots = shots_for_duration(int(cfg["PREFERRED_SECONDS"]))
    else:
        shots = shots_for_duration(cfg["TARGET_SECONDS"])
    logger.info("Plan: %d shots, roles=%s", len(shots),
                [s.narrative_role for s in shots])

    series = {
        "id": today_key(),
        "theme": THEME,
        "parts": [[s.as_dict() for s in shots]],
        "next_part": 1,
        "strategy": cfg.get("IMAGE_STRATEGY", "edit"),
        "image_prompts": {str(k): v for k, v in image_prompts.items()},
    }
    state["series"] = series
    logger.info("New series %s: %d part(s), [%d] shots",
                today_key(), len(series["parts"]), len(shots))
    for i, shot in enumerate(shots):
        char = ", character" if shot.character else ", no character"
        logger.info("  shot %2d [%s/%s%s]", i + 1, shot.shot_size, shot.narrative_role, char)
    return shots, 1, 1, state


def run_pipeline(*, render_only: bool = False, offline_images: bool = False,
                 reuse_frames: bool = False) -> None:
    cfg = load_config()
    state = load_state(Path("state.json"))
    fps = 24
    shots, part_num, total_parts, state = resolve_part(cfg, state)
    target_seconds = cfg.get("PREFERRED_SECONDS") or (cfg["MIN_SECONDS"] + cfg["MAX_SECONDS"]) // 2
    strategy = cfg.get("IMAGE_STRATEGY", "edit")

    if reuse_frames:
        img_dir = Path("build/frames")
        frames = sorted(img_dir.glob("frame_*.jpg")) + sorted(img_dir.glob("frame_*.png"))
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
            # Use detailed image prompts if available, otherwise fall back to prompt() builder
            image_prompts_raw = (state.get("series") or {}).get("image_prompts", {})
            detailed_prompts = [image_prompts_raw.get(str(i), "") for i in range(len(shots))]
            if any(p for p in detailed_prompts):
                final_prompts = [p or shot.prompt() for p, shot in zip(detailed_prompts, shots)]
                logger.info("Using %d detailed image prompts from planner output",
                            sum(1 for p in final_prompts if p))
            else:
                final_prompts = [shot.prompt() for shot in shots]
                logger.info("Falling back to generated prompts for %d shots", len(final_prompts))

            chars = [s.character for s in shots]
            frames, strategy = produce_images(
                final_prompts, cfa_token=cfg["CFAI_TOKEN"],
                account_id=cfg["CF_ACCOUNT_ID"], img_dir=img_dir,
                strategy=strategy, character_flags=chars, budget=budget,
            )
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

    durations = [shot.seconds for shot in shots]
    narration = Narration(voice=cfg["KOKORO_VOICE"])
    # Prefer audio_notes if present, otherwise use narration text
    beats = [s.audio_notes or s.narration for s in shots]
    beats_text = [s.narration for s in shots]

    raw = [narration.wav(b) for b in beats_text]
    segments, speed = narration.synth_scaled(beats_text, target_seconds=target_seconds - BEAT_GAP * (len(beats_text) - 1) - TAIL_HOLD)
    natural = sum(len(s) for s in segments) / SAMPLE_RATE

    audio, starts = join_with_gaps(segments, gap_seconds=BEAT_GAP)
    tail_padding = int(TAIL_HOLD * SAMPLE_RATE)
    silence = np.zeros(tail_padding, dtype=np.int16)
    audio = np.concatenate([audio, silence])

    cut_rate = len(audio) / (sum(durations) + BEAT_GAP * (len(durations) - 1) + TAIL_HOLD)
    duration_cutoff = speed >= 0.6 and cut_rate <= 6
    speech_duration = sum(len(s) for s in segments) / SAMPLE_RATE
    total = speech_duration + BEAT_GAP * (len(segments) - 1) + TAIL_HOLD
    logger.info("Natural speech %.1fs; scaled by %.2f -> %.1fs speech, %.1fs total",
                natural, speed, speech_duration, total)

    if not duration_cutoff:
        logger.warning("Speech scaling %s too extreme (%.2f); using original",
                       "< 0.6x" if speed < 0.6 else "> 6x", speed)
        segments, speed = raw, 1.0
        audio, starts = join_with_gaps(raw, gap_seconds=BEAT_GAP)
        total = sum(len(r) for r in raw) / SAMPLE_RATE + BEAT_GAP * (len(raw) - 1) + TAIL_HOLD
        cut_rate = len(audio) / total if total > 0 else 999
        duration_cutoff = True

    audio_path = Path("build/narration.wav")
    write_wav(audio_path, audio)

    logger.info("Render: %d beats, %.1fs total, audio=narration.wav",
                len(segments), total)
    video_path = render_narration(
        frames, durations, audio_path, Path("output_short.mp4"), fps=fps,
    )

    if render_only:
        logger.info("Render-only: wrote %s (%s); frames kept for --reuse-frames",
                     video_path, strategy)
        return

    token = exchange_refresh_token(cfg)
    title = f"Anime Short #{part_num}/{total_parts} — {state.get('series', {}).get('id', 'new')}"
    description_lines = [
        f"# Part {part_num} of {total_parts}",
        "",
        "# Narration:",
    ]
    for i, shot in enumerate(shots):
        description_lines.append(f"{i+1}. {shot.narration}")
    description = "\n".join(description_lines)
    tags = ["anime", "shorts", "storytime", "mystery", "lantern",
            "rainy city", "vertical video"]
    privacy = cfg.get("PUBLISH_PRIVACY", "private")
    upload_video(video_path, title=title, description=description, tags=tags,
                 privacy_status=privacy, client_id=token.client_id,
                 client_secret=token.client_secret, refresh_token=token.refresh_token)
    uploads_today = state.setdefault("daily", {}).setdefault(today_key(), {}).get("uploads", 0) + 1
    state.setdefault("daily", {})[today_key()]["uploads"] = uploads_today
    if uploads_today >= Limits.max_daily_uploads:
        logger.info("Upload limit reached today — skipping further publishing this run.")
        return
    Path("state.json").write_text(save_state(state), encoding="utf-8")
    logger.info("Uploaded; daily uploads=%d/%d, strategy=%s",
                uploads_today, Limits.max_daily_uploads, strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Anime Shorts Factory pipeline")
    parser.add_argument("--render-only", action="store_true",
                        help="stop after rendering and skip upload")
    parser.add_argument("--offline-images", action="store_true",
                        help="generate placeholder frames (no API key required)")
    parser.add_argument("--reuse-frames", action="store_true",
                        help="skip regeneration; reuse build/frames/frame_*.ext")
    args = parser.parse_args()
    run_pipeline(render_only=args.render_only, offline_images=args.offline_images,
                 reuse_frames=args.reuse_frames)
