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

from audio import Narration, join_with_gaps, write_wav
from config import load_config
from images import produce_images
from publish import exchange_refresh_token, upload_video
from render import render_narration
from state import compute_fingerprint, parse_state, save_state, today_key
from scenes import Shot, plan_arc, shots_for_duration

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
        logger.info("Continuing series %s — part %d/%d (%d shots)",
                    series.get("id"), index + 1, len(parts), len(shots))
        return shots, index + 1, len(parts), series

    per_part = cfg["SHOTS_PER_PART"] or shots_for_duration(cfg["PREFERRED_SECONDS"])
    arc, fallback = plan_arc(
        theme=THEME,
        parts=cfg["SERIES_PARTS"],
        per_part=per_part,
        cfa_token=cfg["CFAI_TOKEN"],
        account_id=cfg["CF_ACCOUNT_ID"],
        api_key=cfg["OPENROUTER_API_KEY"],
    )
    series = {
        "id": today_key(),
        "theme": THEME,
        "parts": [[sh.as_dict() for sh in part] for part in arc],
        "next_part": 0,
        "strategy": cfg["IMAGE_STRATEGY"],
    }
    if fallback:
        logger.warning("Scene planner unavailable — template plan for this run only")
    else:
        state["series"] = series
        logger.info("New series %s: %d part(s), %s shots",
                    series["id"], len(arc), [len(p) for p in arc])
    return arc[0], 1, len(arc), series


def run_pipeline(*, render_only: bool = False, offline_images: bool = False,
                 no_audio: bool = False, reuse_frames: bool = False,
                 strategy: str | None = None) -> Path | None:
    cfg = load_config()
    state_path = Path("state.json")
    state = load_state(state_path)
    today = today_key()
    daily = state.get("daily", {}).get(today, {"uploads": 0, "image_attempts": 0})

    logger.info("Run start — target %.0f-%.0fs, strategy=%s",
                cfg["MIN_SECONDS"], cfg["MAX_SECONDS"], strategy or cfg["IMAGE_STRATEGY"])

    if daily.get("uploads", 0) >= 1 and not render_only:
        logger.info("Daily upload limit reached for %s. Exiting.", today)
        return None

    # 1. Which part of which story.
    shots, part_no, part_total, series = resolve_part(cfg, state)
    for i, shot in enumerate(shots, 1):
        logger.info("  shot %2d [%s/%s%s] %s", i, shot.shot_size, shot.narrative_role,
                    "" if shot.character else ", no character", shot.place)
    logger.info("Plan: %d shots, %d without the character, roles=%s",
                len(shots), sum(1 for s in shots if not s.character),
                sorted({s.narrative_role for s in shots}))

    # 2. Images.
    work = Path("build")
    img_dir = work / "frames"
    if reuse_frames and img_dir.exists() and any(img_dir.glob("frame_*.*")):
        frames = sorted(img_dir.glob("frame_*.*"))
        used_strategy = "reused"
        logger.info("Reusing %d cached frames (no quota)", len(frames))
    elif offline_images:
        img_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(len(shots)):
            path = img_dir / f"frame_{i}.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "testsrc2=s=1024x1024", "-frames:v", "1",
                            "-vf", f"hue=h={i * 55}", str(path)], check=True)
            frames.append(path)
        used_strategy = "offline"
        logger.info("Offline mode: %d stub frames", len(frames))
    else:
        budget = {
            "attempts": daily.get("image_attempts", 0),
            "max": int(cfg["MAX_IMAGE_ATTEMPTS"] or 8),
        }
        frames, used_strategy = produce_images(
            [sh.prompt() for sh in shots],
            cfa_token=cfg["CFAI_TOKEN"],
            account_id=cfg["CF_ACCOUNT_ID"],
            img_dir=img_dir,
            strategy=strategy or series.get("strategy") or cfg["IMAGE_STRATEGY"],
            character_flags=[sh.character for sh in shots],
            budget=budget,
        )
        daily["image_attempts"] = budget["attempts"]

    state.setdefault("daily", {})[today] = {
        "uploads": daily.get("uploads", 0),
        "image_attempts": daily.get("image_attempts", 0),
    }
    state_path.write_text(save_state(state), encoding="utf-8")

    if not frames:
        logger.error("No images produced — aborting, upload counter untouched.")
        return None
    if len(frames) < len(shots):
        logger.warning("Only %d of %d shots have images; narrating just those", len(frames), len(shots))
        shots = shots[:len(frames)]
    beats = [sh.narration for sh in shots]

    video_path = Path("output_short.mp4")
    published = False   # only a real upload allows the temp tree to be deleted
    try:
        # 3. Narration: natural pace unless the result falls outside the accepted window.
        gaps = BEAT_GAP * (len(beats) - 1)
        if no_audio:
            audio_path, durations, speed = None, [cfg["MIN_SECONDS"] / len(beats)] * len(beats), 1.0
            logger.warning("--no-audio: silent video")
        else:
            narration = Narration(voice=cfg["KOKORO_VOICE"])
            raw = [narration.wav(b) for b in beats]
            natural_seconds = sum(len(a) for a in raw) / 24000
            # Only ever speed UP when a cut runs long. Slowing narration down to reach a
            # minimum length just makes the read drawl, and a shorter short is fine.
            ceiling = cfg["MAX_SECONDS"] - gaps - TAIL_HOLD
            if natural_seconds <= ceiling:
                segments, speed = raw, 1.0
                logger.info("Natural speech %.1fs — kept at speed 1.00 (limit %.0fs)",
                            natural_seconds, cfg["MAX_SECONDS"])
                if natural_seconds < cfg["MIN_SECONDS"]:
                    logger.info("Below the %.0fs soft floor — accepting the shorter cut",
                                cfg["MIN_SECONDS"])
            else:
                logger.info("Natural speech %.1fs exceeds the %.0fs limit — speeding up",
                            natural_seconds, cfg["MAX_SECONDS"])
                segments, speed = narration.synth_scaled(beats, target_seconds=ceiling)
            audio, starts = join_with_gaps(segments, gap_seconds=BEAT_GAP)
            audio_path = write_wav(work / "narration.wav", audio)
            speech = sum(len(s) for s in segments) / 24000
            tail = min(max(cfg["MIN_SECONDS"] - (speech + gaps), 0.4), 4.0)
            durations = [len(seg) / 24000 + BEAT_GAP for seg in segments]
            durations[-1] = durations[-1] - BEAT_GAP + tail
            logger.info("Narration %.2fs at speed %.2f, cues %s, total %.2fs",
                        speech, speed, [round(s, 1) for s in starts], sum(durations))

        # 4. Render.
        if audio_path is None:
            silent = work / "silence.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "anullsrc=r=48000:cl=stereo", "-t", str(sum(durations)),
                            str(silent)], check=True)
            ok = render_narration(frames, durations, silent, video_path)
        else:
            ok = render_narration(frames, durations, audio_path, video_path)
        if not ok or not video_path.exists():
            logger.error("Render failed.")
            return None

        if render_only:
            logger.info("Render-only: wrote %s (%s); frames kept for --reuse-frames",
                        video_path.resolve(), used_strategy)
            return video_path

        # 5. Publish.
        access_token = exchange_refresh_token(
            cfg["YOUTUBE_CLIENT_ID"], cfg["YOUTUBE_CLIENT_SECRET"], cfg["YOUTUBE_REFRESH_TOKEN"])
        if not access_token:
            logger.error("YouTube token exchange failed — nothing uploaded.")
            return None

        title = beats[0][:80]
        if part_total > 1:
            title += f" — Part {part_no}/{part_total}"
        video_id = upload_video(
            video_path.read_bytes(),
            title=title[:100] + " #shorts",
            description="\n\n".join(beats)[:4000],
            access_token=access_token,
            privacy_status=cfg["PUBLISH_PRIVACY"],
        )
        if not video_id:
            logger.error("Upload failed; series queue left untouched for tomorrow.")
            return None

        published = True
        daily["uploads"] = daily.get("uploads", 0) + 1
        daily["video_id"] = video_id
        daily["strategy"] = used_strategy
        daily["seconds"] = round(sum(durations), 2)
        state["daily"][today] = daily
        state["processed"] = state.get("processed", []) + [
            compute_fingerprint(" ".join(beats), [f.name for f in frames])
        ]
        # Only a published part advances the series.
        series["next_part"] = part_no
        if series["next_part"] >= len(series.get("parts", [])):
            logger.info("Series %s complete (%d parts)", series.get("id"), part_total)
            state.pop("series", None)
        else:
            state["series"] = series
            logger.info("Part %d published; %d part(s) queued for the next run",
                        part_no, len(series["parts"]) - series["next_part"])
        state_path.write_text(save_state(state), encoding="utf-8")
        logger.info("Uploaded private: https://youtu.be/%s (%s, %.1fs)",
                    video_id, used_strategy, sum(durations))
        return video_path

    finally:
        if published:
            for p in frames:
                p.unlink(missing_ok=True)
            for d in (img_dir, work):
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
        else:
            logger.warning("Keeping %s for inspection/retry.", work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Anime Shorts Factory pipeline")
    parser.add_argument("--render-only", action="store_true", help="stop before upload")
    parser.add_argument("--offline-images", action="store_true",
                        help="test-pattern frames instead of the API (no quota)")
    parser.add_argument("--reuse-frames", action="store_true", help="reuse build/frames/ (no quota)")
    parser.add_argument("--no-audio", action="store_true", help="skip Kokoro narration")
    parser.add_argument("--strategy", choices=("edit", "sheet", "fresh"),
                        help="override the image strategy for this run")
    args = parser.parse_args()
    run_pipeline(
        render_only=args.render_only,
        offline_images=args.offline_images,
        no_audio=args.no_audio,
        reuse_frames=args.reuse_frames,
        strategy=args.strategy,
    )
