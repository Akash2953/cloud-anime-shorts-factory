"""Background music for narrated shorts.

Mode (option 2): one track per part, rotated deterministically by part number.
Files live in `build/music/`. The same track is picked on every retry, so the
finished arc can be reproduced; the rotation is `(part - 1) % len(tracks)`,
which keeps part 1 on the first file forever while every part you add lands on
its own.

The mixer graph here is the same one I will drop into `render.render_narration`
once the renderer is rebuilt around it, so the demo script and the production
pipeline pass through one filter graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio import SAMPLE_RATE, write_wav

MUSIC_DIR = Path(__import__("os").environ.get("MUSIC_DIR", "build/music"))
MUSIC_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

# Quiet subtitles for a YouTube description credit. The orchestrator will append
# these when an upload includes an attribution-required track.
MUSIC_NOTE = (
    'Music: "{title}" by {artist} ({license}) — {url}'
)


def enumerate_music(music_dir: Path | None = None) -> list[Path]:
    """Sorted list of usable audio files. Empty when the folder is missing."""
    music_dir = music_dir or MUSIC_DIR
    if not music_dir.exists():
        return []
    return sorted(p for p in music_dir.iterdir()
                  if p.suffix.lower() in MUSIC_EXTS and p.is_file())


def pick_music(part_num: int, music_dir: Path | None = None) -> Path | None:
    """One track per part, deterministically rotated across the arc."""
    tracks = enumerate_music(music_dir)
    if not tracks:
        return None
    return tracks[(part_num - 1) % len(tracks)]


# ffmpeg filter graph fragment. Inputs: [voice, music]. Output: `[aout]`.
# Placeholders substituted by `mix_graph()`. Drops to mono before sidechain
# because Kokoro's mono key links to the same channel it ducks.
_MIX_GRAPH = (
    "[1:a]aloop=loop=-1:size=2e9,atrim=0:duration={dur:.3f},"
    "afade=t=in:st=0:d={fade_in:.2f},"
    "afade=t=out:st={fade_out_start:.3f}:d={fade_out:.2f},"
    "volume={music_db}dB[mix];"
    "[0:a]asplit=2[voice][key];"
    "[mix][key]sidechaincompress=threshold={thf}:ratio={ratio}:"
    "attack={atk}:release={rel}[ducked];"
    "[voice][ducked]amix=inputs=2:dropout_transition=0,"
    "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
)


def mix_graph(
    duration_seconds: float,
    *,
    fade_in: float = 1.5,
    fade_out: float = 2.5,
    music_db: float = -18.0,
    threshold_pct: float = 0.04,
    ratio: float = 8.0,
    attack_ms: int = 300,
    release_ms: int = 500,
) -> str:
    """Build the per-run graph string. The `[aout]` label is what render maps."""
    fade_start = max(0.0, duration_seconds - fade_out)
    return _MIX_GRAPH.format(
        dur=duration_seconds, fade_in=fade_in, fade_out=fade_out,
        fade_out_start=fade_start, music_db=music_db, thf=threshold_pct,
        ratio=ratio, atk=attack_ms, rel=release_ms,
    )


def synthesize_ambient_bed(
    duration_seconds: float,
    *,
    sample_rate: int = SAMPLE_RATE,
    note_hz: float = 220.0,         # A3 root — kept low so it sits under voice
    fith_hz: float = 330.0,         # perfect fifth
    sub_hz: float = 55.0,           # sub
    bpm: float = 70.0,
    seed: int = 0xC001,
) -> np.ndarray:
    """Local fallback bed: a slow, soft sub + drone + pulsing fifth pad.

    Used when `build/music/` is empty so the mixer code and the ducking graph
    can be heard end-to-end before any real audio is licensed. This is not
    meant to ship: replace it by dropping CC0 / CC-BY files into MUSIC_DIR.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    beat = 60.0 / bpm

    # Two interleaved voices: a sub drone and a slightly detuned fifth pad.
    # Detune by ±2 cents so the fifth breathes between chorus and detuned, never
    # stiffens into a metronome under the voice-over.
    detune = 0.002
    pad = (
        0.6 * np.sin(2 * np.pi * note_hz * t) +
        0.6 * np.sin(2 * np.pi * note_hz * (1 + detune) * t) +
        0.5 * np.sin(2 * np.pi * fith_hz * t) +
        0.5 * np.sin(2 * np.pi * fith_hz * (1 - detune) * t) +
        0.4 * np.sin(2 * np.pi * sub_hz * t)
    )

    # Slow LFO on the pad's amplitude to keep the texture breathing.
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * (bpm / 240.0) * t * 4.0)

    # A gentle chord progression every `chord_period` beats. Sections: A3/e
    # fifth -> G3/d fifth -> F3/a fifth -> A3/e fifth. The fifth is held; only
    # the root changes, which keeps ANY listener's ear from spotting a melody
    # under the narration.
    chord_period = 8 * beat
    sections = [220.0, 196.0, 174.6, 220.0]
    note_step = (t // chord_period).astype(int) % len(sections)
    chord_root = np.array(sections)[note_step]
    chord_fifth = chord_root * 1.5  # keeps the ratio honest
    ring = (
        0.35 * np.sin(2 * np.pi * chord_root * t) +
        0.35 * np.sin(2 * np.pi * chord_fifth * t)
    )

    # 8th-note pulses: very soft, on the low mix of the pad, plausible as
    # either rain or finger taps. No real kick to compete with the voice.
    eighth = (t * (bpm / 60.0 * 2.0) % 1.0) < 0.04
    pulse = rng.standard_normal(n) * eighth.astype(np.float64) * 0.06

    bed = (pad * lfo * 0.45 + ring * 0.30 + pulse).astype(np.float32)
    # Soft-clamp to keep peaks inside ±0.85 so the sidechain doesn't duck too hard.
    bed = np.tanh(bed * 0.85) * 0.85
    return bed


def write_fallback_bed(path: Path, duration_seconds: float) -> Path:
    """Synthesise the demo bed to a wav file ffmpeg can ingest.

    Returns the path written. Render never calls this directly: it expects a
    real file from MUSIC_DIR. This helper exists only so option 2 can be
    exercised end-to-end before any music is licensed.
    """
    bed = synthesize_ambient_bed(duration_seconds)
    return write_wav(path, bed)
