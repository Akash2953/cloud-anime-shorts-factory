"""FFmpeg video rendering for vertical 9:16 shorts."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _fit_filter(width: int, height: int, fit: str) -> str:
    """Build the scale/pad chain for the chosen fill strategy.

    cover   — scale up until the frame is full, then centre-crop the overflow.
              Use when sources are square (Workers AI returns 1024x1024).
    contain — scale to fit and letterbox with black bars.

    Both force limited-range yuv420p: JPEG sources are full-range, and letting that
    through produces yuvj420p files that render washed out or crushed on phones.
    """
    if fit == "cover":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:out_range=limited,"
            f"crop={width}:{height},setsar=1,format=yuv420p"
        )
    if fit == "contain":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:out_range=limited,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
        )
    raise ValueError(f"unknown fit mode: {fit}")


def _fit_chain(width: int, height: int, fit: str) -> str:
    """scale+pad/crop chain only (no format), reusable inside bigger graphs."""
    if fit == "cover":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:out_range=limited,"
            f"crop={width}:{height}"
        )
    if fit == "contain":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:out_range=limited,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    raise ValueError(f"unknown fit mode: {fit}")


def _ken_burns(index: int, frames: int, zoom_max: float = 1.12) -> str:
    """Slow push-in on even beats, pull-out on odd ones, always centred."""
    step = (zoom_max - 1.0) / max(frames, 1)
    centred = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    if index % 2 == 0:
        z = f"min(zoom+{step:.6f},{zoom_max})"
    else:
        z = f"if(lte(on,1),{zoom_max},max(zoom-{step:.6f},1.0))"
    return f"zoompan=z='{z}':{centred}:d={frames}:s=1080x1920:fps=30"


def render_narration(
    image_paths: list[Path],
    durations: list[float],
    audio_path: Path,
    output_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    fit: str = "cover",
    ken_burns: bool = True,
) -> bool:
    """One image per narration beat: each still is held for its own line length, with
    Ken Burns motion, over the concatenated voice track.

    durations[i] must cover the audio for beat i (caller adds the inter-line gap).
    """
    if len(image_paths) != len(durations) or not image_paths:
        logger.error("render_narration needs one duration per image (%d vs %d)",
                     len(image_paths), len(durations))
        return False

    cmd: list[str] = ["ffmpeg", "-y"]
    for img in image_paths:
        cmd += ["-i", str(img.resolve())]
    cmd += ["-i", str(audio_path.resolve())]

    graph: list[str] = []
    for i, duration in enumerate(durations):
        frames = max(int(round(duration * fps)), 1)
        chain = [_fit_chain(width, height, fit)]
        if ken_burns:
            chain.append(_ken_burns(i, frames))
        chain.append(f"trim=duration={duration:.3f}")
        chain.append("setpts=PTS-STARTPTS")
        chain.append("format=yuv420p")
        graph.append(f"[{i}:v]{','.join(chain)}[v{i}]")

    concat_inputs = "".join(f"[v{i}]" for i in range(len(durations)))
    audio_index = len(image_paths)
    graph.append(f"{concat_inputs}concat=n={len(durations)}:v=1:a=0[vout]")
    # Normalise to -14 LUFS / -1.5 dBTP so every short arrives at YouTube at the same level.
    graph.append(f"[{audio_index}:a]apad,asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")

    total = sum(durations)
    cmd += [
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", f"{total:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Rendering %d beats, %.1fs total, audio=%s", len(durations), total, audio_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("FFmpeg failed: %s", result.stderr[-1500:])
        return False
    return output_path.exists()


def render_short(
    image_paths: list[Path],
    output_path: Path,
    *,
    duration_per_image: float = 3.0,
    width: int = 1080,
    height: int = 1920,
    fit: str = "cover",
) -> bool:
    if not image_paths:
        logger.error("No images provided for rendering")
        return False

    # Create temporary concat list for FFmpeg
    concat_file = output_path.parent / "concat.txt"
    try:
        lines = []
        for p in image_paths:
            lines.append(f"file '{p.resolve()}'")
            lines.append(f"duration {duration_per_image}")
        # repeat last frame to hold duration
        lines.append(f"file '{image_paths[-1].resolve()}'")
        concat_file.write_text("\n".join(lines), encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            _fit_filter(width, height, fit),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            # Tag as standard limited-range BT.709 so YouTube/phones decode colours correctly.
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-t",
            str(len(image_paths) * duration_per_image),
            str(output_path),
        ]

        logger.info("Running FFmpeg: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("FFmpeg failed: %s", result.stderr[-800:])
            return False

        return output_path.exists()
    finally:
        if concat_file.exists():
            concat_file.unlink()
