"""FFmpeg video rendering for vertical 9:16 shorts."""

import logging
import math
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


def _ken_burns(index: int, frames: int, fps: int, zoom_max: float = 1.12) -> str:
    """Slow push-in on even beats, pull-out on odd ones, always centred.

    `fps` must be the render's frame rate: zoompan's own default is 30, and emitting 30 fps
    frames that are then trimmed to a duration computed at 24 fps makes every beat 1.25x short,
    which desyncs the whole short from its voice track.
    """
    step = (zoom_max - 1.0) / max(frames, 1)
    centred = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    if index % 2 == 0:
        z = f"min(zoom+{step:.6f},{zoom_max})"
    else:
        z = f"if(lte(on,1),{zoom_max},max(zoom-{step:.6f},1.0))"
    return f"zoompan=z='{z}':{centred}:d={frames}:s=1080x1920:fps={fps}"


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
    overlays: list[tuple] | None = None,
    caption_animation: str = "slide",
    music_path: Path | None = None,
) -> bool:
    """One image per narration beat: each still is held for its own line length, with
    Ken Burns motion, over the concatenated voice track.

    durations[i] must cover the audio for beat i (caller adds the inter-line gap).

    Optional `overlays`: list of (png_path, start_s, end_s) triples, each glyph-rendered
    strip is composited over the base video between start_s and end_s. Used by captions.py
    to layer word-synced subtitles on top of the Ken Burns frames; PNG height is read by
    ffmpeg at filter time so every strip lands in the bottom safe area. Default None keeps
    callers that don't pass overlays off the new code path.

    A 4th element may be added to a triple to choose the vertical anchor:
      "bottom" (default) — strip centre at 75% of the canvas height (centre of the bottom half)
      "center"           — strip centre at 50% of the canvas height (centre of the frame)
    The CTA block uses "center" so the ask sits in the middle of the picture rather than in the
    caption lane.

    `music_path` adds a background bed: it is looped to the full length, faded in and out, and
    sidechain-ducked under the voice by `music.mix_graph`, then the whole mix is normalised to
    -14 LUFS. The music is mixed at render time (not at upload) so `--render-only` produces the
    same audio that ships.

    `caption_animation` picks how each strip enters and leaves:
      none  — hard on/off at the cue boundaries (cheapest, one still frame per overlay)
      fade  — 100 ms alpha fade in and out
      slide — fade plus a 14 px upward slide over the first 120 ms (default)
    A still PNG cannot be animated by time-based filters, so animated modes feed each overlay
    as a short looped video stream (`-loop 1 -t dur -itsoffset start`). Fade timings are then
    absolute, because `-itsoffset` shifts the stream's PTS to the cue start.
    """
    if len(image_paths) != len(durations) or not image_paths:
        logger.error("render_narration needs one duration per image (%d vs %d)",
                     len(image_paths), len(durations))
        return False

    cmd: list[str] = ["ffmpeg", "-y"]
    for img in image_paths:
        cmd += ["-i", str(img.resolve())]
    cmd += ["-i", str(audio_path.resolve())]
    audio_index = len(image_paths)
    # Overlay inputs come after the audio (and the music) input; index = overlay_base + k.
    # Animated modes need a real video stream per overlay, so the PNG is looped for exactly its
    # cue length and offset to the cue start. `none` keeps the cheap single-still input.
    animated = caption_animation in ("fade", "slide")
    # Music sits right after the voice so the overlay indices stay contiguous.
    has_music = music_path is not None and Path(music_path).exists()
    if has_music:
        cmd += ["-i", str(Path(music_path).resolve())]
    overlay_base = audio_index + (2 if has_music else 1)

    if overlays:
        for ov in overlays:
            png, start_s, end_s = ov[0], ov[1], ov[2]
            if animated:
                dur = max(end_s - start_s, 0.1)
                cmd += ["-itsoffset", f"{start_s:.3f}", "-loop", "1",
                        "-t", f"{dur:.3f}", "-i", str(Path(png).resolve())]
            else:
                cmd += ["-i", str(Path(png).resolve())]  

    graph: list[str] = []
    for i, duration in enumerate(durations):
        # ceil, not round: the trimmed beat can never be longer than the frames Ken Burns actually
        # generated, and 13 beats of half-frame shortfall ended the video 1 frame before the audio
        # (44.833s picture vs 44.885s sound) so the last word lost its tail.
        frames = max(int(math.ceil(duration * fps)), 1)
        chain = [_fit_chain(width, height, fit)]
        if ken_burns:
            chain.append(_ken_burns(i, frames, fps))
        chain.append(f"trim=duration={duration:.3f}")
        chain.append("setpts=PTS-STARTPTS")
        chain.append("format=yuv420p")
        graph.append(f"[{i}:v]{','.join(chain)}[v{i}]")

    concat_inputs = "".join(f"[v{i}]" for i in range(len(durations)))
    # `-t` can only trim, never extend: per-beat frame quantisation left the concatenated picture
    # a frame or two short, so the tail of the last word was cut off. Hold the final frame long
    # enough for the clamp to do the cutting instead.
    graph.append(f"{concat_inputs}concat=n={len(durations)}:v=1:a=0[vcat]")

    # Burn caption overlays (if any) between the concat and the final tpad. Each overlay is a
    # transparent RGBA strip; ffmpeg's overlay filter positions it so its vertical centre lands
    # at y = main_h * 3/4, the centre of the lower half of a 9:16 canvas. PNGs of varying
    # density therefore always sit the same way; `y` references the strip's own height (`h`),
    # so a 2-word strip and a 3-word strip land centred on the same line.
    if overlays:
        last = "[vcat]"
        for k, ov in enumerate(overlays):
            _png, start_s, end_s = ov[0], ov[1], ov[2]
            anchor = ov[3] if len(ov) > 3 else "bottom"
            cap_index = overlay_base + k
            out = f"[vcap{k}]"
            # "bottom" puts the strip centre at 75% of the canvas height (centre of the bottom
            # half); "center" puts it at 50% (centre of the frame).
            base_y = ("main_h/2-overlay_h/2" if anchor == "center"
                      else "main_h*3/4-overlay_h/2")
            y = base_y
            if animated:
                dur = max(end_s - start_s, 0.1)
                fade_d = min(0.10, dur / 3)
                chain = ["format=rgba"]
                if caption_animation in ("fade", "slide"):
                    # Absolute timings: -itsoffset already moved this stream to start_s.
                    chain.append(f"fade=t=in:st={start_s:.3f}:d={fade_d:.3f}:alpha=1")
                    chain.append(f"fade=t=out:st={end_s - fade_d:.3f}:d={fade_d:.3f}:alpha=1")
                graph.append(f"[{cap_index}:v]{','.join(chain)}[capA{k}]")
                if caption_animation == "slide":
                    y = f"{base_y}+14*max(0,1-(t-{start_s:.3f})/0.12)"
                graph.append(f"{last}[capA{k}]overlay=x=0:y='{y}':eof_action=pass{out}")
            else:
                graph.append(
                    f"{last}[{cap_index}:v]overlay=enable='between(t,"
                    f"{start_s:.3f},{end_s:.3f})':x=0:y={y}{out}"
                )
            last = out
        graph.append(f"{last}tpad=stop_mode=clone:stop_duration=1[vout]")
    else:
        graph.append("[vcat]tpad=stop_mode=clone:stop_duration=1[vout]")
    # Normalise to -14 LUFS / -1.5 dBTP so every short arrives at YouTube at the same level.
    total = sum(durations)
    if has_music:
        from music import mix_graph
        # Sidechain-ducked bed under the narration, normalised as one mix to -14 LUFS.
        graph.append(mix_graph(total, voice_index=audio_index, music_index=audio_index + 1))
        logger.info("Music bed: %s (ducked under voice, %.1fs)",
                    Path(music_path).name, total)
    else:
        graph.append(f"[{audio_index}:a]apad,asetpts=PTS-STARTPTS,"
                     f"loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
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

    logger.info("Rendering %d beats, %.1fs total, audio=%s%s", len(durations), total,
                audio_path.name, " + music" if has_music else "")
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
