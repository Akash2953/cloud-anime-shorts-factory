"""Live end-to-end check of the scene-plan pipeline.

    python verify.py                  # plan -> images -> Kokoro -> timed render
    python verify.py --skip-images    # reuse build/frames/ (no image quota spent)

Asserts each stage instead of trusting it, and prints credential presence only, never values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from audio import SAMPLE_RATE, Narration, join_with_gaps, write_wav
from config import load_config
from images import KLEIN, SCHNELL, image_ext, load_frames, produce_images
from render import render_narration
from scenes import Shot, make_character, plan_shots, shots_for_duration

BUILD = Path("build")
FRAMES = BUILD / "frames"
VIDEO = Path("output_short_check.mp4")
MIN_SECONDS, MAX_SECONDS = 30.0, 68.0
GAP = 0.35
RESULTS: list[tuple[str, bool, str]] = []


def record(stage: str, ok: bool, detail: str) -> bool:
    RESULTS.append((stage, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {stage}: {detail}")
    return ok


def img_size(data: bytes) -> tuple[int, int]:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            if data[i + 1] in (0xC0, 0xC1, 0xC2):
                return (int.from_bytes(data[i + 7:i + 9], "big"),
                        int.from_bytes(data[i + 5:i + 7], "big"))
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    raise ValueError("unrecognised image format")


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true", help="reuse build/frames/")
    parser.add_argument("--shots", type=int, default=0, help="override the shot count")
    parser.add_argument("--strategy", default="edit", choices=("edit", "sheet", "fresh"))
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config()
    print("--- credentials (presence only) ---")
    for name in ("CFAI_TOKEN", "CF_ACCOUNT_ID", "OPENROUTER_API_KEY",
                 "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"):
        print(f"  {name:<22} {'set' if cfg.get(name) else 'MISSING'}")
    print(f"  image models            {SCHNELL} + {KLEIN} (edit)\n")

    count = args.shots or shots_for_duration(cfg["PREFERRED_SECONDS"])

    # 1. Scene plan -----------------------------------------------------------
    shots, fallback = plan_shots(
        theme="cyberpunk anime, floating paper lanterns over a rain-soaked city, "
              "a hooded girl searching for her missing brother",
        count=count,
        api_key=cfg["OPENROUTER_API_KEY"])
    roles = {s.narrative_role for s in shots}
    no_char = sum(1 for s in shots if not s.character)
    record("plan", not fallback and len(shots) >= 8 and len(roles) >= 4 and no_char >= 3,
           f"{len(shots)} shots, {len(roles)} roles, {no_char} without character, "
           f"fallback={fallback}")
    for i, s in enumerate(shots, 1):
        print(f"       {i:>2}. [{s.shot_size:<12} {s.narrative_role:<17} "
              f"{'char' if s.character else 'env '}] {s.place[:52]}")

    # 2. Images ---------------------------------------------------------------
    frames: list[Path] = []
    if args.skip_images and FRAMES.exists() and any(FRAMES.glob("frame_*.*")):
        frames = load_frames(FRAMES)
        record("images", len(frames) >= 8, f"reused {len(frames)} cached frames (no quota)")
    else:
        FRAMES.mkdir(parents=True, exist_ok=True)
        for stale in FRAMES.glob("frame_*.*"):
            stale.unlink()
        character = make_character("verify")
        frames, used = produce_images(
            [s.prompt(character) for s in shots],
            cfa_token=cfg["CFAI_TOKEN"], account_id=cfg["CF_ACCOUNT_ID"],
            img_dir=FRAMES, strategy=args.strategy,
            character_flags=[s.character for s in shots], character=character,
            budget={"attempts": 0, "max": args.max_images})
        sizes = []
        for f in frames:
            try:
                sizes.append(img_size(f.read_bytes()))
            except ValueError:
                sizes.append((0, 0))
        record("images", len(frames) >= 8, f"strategy={used}, {len(frames)} frames, "
                                           f"{len(set(sizes))} distinct sizes, first={sizes[:2]}")
    if not frames:
        return 1
    shots = shots[:len(frames)]

    # 3. Narration ------------------------------------------------------------
    beats = [s.narration for s in shots]
    gaps = GAP * (len(beats) - 1)
    narration = Narration(voice=cfg["KOKORO_VOICE"])
    raw = [narration.wav(b) for b in beats]
    natural = sum(len(a) for a in raw) / SAMPLE_RATE
    ceiling = MAX_SECONDS - gaps - 0.6
    if natural <= ceiling:
        segments, speed = raw, 1.0
    else:
        segments, speed = narration.synth_scaled(beats, target_seconds=ceiling)
    speech = sum(len(s) for s in segments) / SAMPLE_RATE
    audio, starts = join_with_gaps(segments, gap_seconds=GAP)
    audio_path = write_wav(BUILD / "narration_check.wav", audio)
    cut_rate = (speech + gaps) / len(beats)
    record("narration", speed >= 0.6 and 1.5 <= cut_rate <= 6.0,
           f"{speech:.2f}s speech, speed {speed:.2f}, ~{cut_rate:.1f}s per cut over {len(beats)} cuts")

    # 4. Render ---------------------------------------------------------------
    durations = [len(s) / SAMPLE_RATE + GAP for s in segments]
    durations[-1] = durations[-1] - GAP + 0.6
    ok = render_narration(frames, durations, audio_path, VIDEO)
    if not ok or not VIDEO.exists():
        record("render", False, "ffmpeg produced no output")
    else:
        meta = ffprobe(VIDEO)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
        duration = float(meta["format"]["duration"])
        crop = subprocess.run(["ffmpeg", "-i", str(VIDEO), "-vf", "cropdetect=24:2:0",
                               "-frames:v", "40", "-f", "null", "-"],
                              capture_output=True, text=True).stderr
        lines = [ln for ln in crop.splitlines() if "crop=" in ln]
        crop_box = lines[-1].split("crop=")[-1].strip() if lines else "?"
        checks = {
            "1080x1920": (v["width"], v["height"]) == (1080, 1920),
            f"<= {MAX_SECONDS:.0f}s": duration <= MAX_SECONDS + 0.5,
            "h264 yuv420p": v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p",
            "aac stereo 48k": bool(a) and a["codec_name"] == "aac"
                              and a["channels"] == 2 and int(a["sample_rate"]) == 48000,
            "no bars": crop_box.startswith("1080:1920:0:0"),
            "cut every 1.5-6s": 1.5 <= duration / len(frames) <= 6.0,
        }
        record("render", all(checks.values()),
               f"{v['width']}x{v['height']} {duration:.2f}s {VIDEO.stat().st_size/1e6:.1f} MB, "
               f"crop={crop_box}")
        for name, passed in checks.items():
            print(f"       {'ok  ' if passed else 'no  '} {name}")

    print("\n--- summary ---")
    for stage, ok_, _ in RESULTS:
        print(f"  {stage:<10} {'PASS' if ok_ else 'FAIL'}")
    print(f"\nartifact: {VIDEO.resolve()}")
    return 1 if any(not ok for _, ok, _ in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
