"""Narration audio via Kokoro-82M (local, ONNX, no API key).

Model files live in models/kokoro/ (gitignored):
  kokoro-v1.0.onnx     311 MB  — GitHub release thewh1teagle/kokoro-onnx model-files-v1.0
  voices-v1.0.bin      ~4 MB   — npz of style matrices, packed from the HF per-voice bins

`python audio.py --download` provisions both, which is what CI runs.
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
MODEL_DIR = Path("models/kokoro")
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
VOICE_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/{name}.bin"

# English voices that suit narration. Style matrices are 510x1x256 float32.
DEFAULT_VOICES: tuple[str, ...] = ("af_heart", "am_michael", "af_bella", "am_fenrir")

# A line counts as voiced only if it has some length and some signal. Kokoro occasionally hands
# back an empty array, or a whisper of DC offset, for a line it failed to voice — and a beat with
# no narration is a hole in the story rather than a cosmetic defect, so it is retried and then
# refused instead of rendered.
MIN_LINE_SECONDS = 0.15
MIN_LINE_PEAK = 1e-3
LINE_ATTEMPTS = 3
LINE_BACKOFF_S = 0.5


class AudioLineError(RuntimeError):
    """A narration line could not be synthesised, or came back empty or silent."""


def _line_problem(audio: np.ndarray | None) -> str | None:
    """Why `audio` is not a usable narration line, or None when it is."""
    if audio is None:
        return "no audio returned"
    if audio.size == 0:
        return "empty audio"
    seconds = audio.size / SAMPLE_RATE
    if seconds < MIN_LINE_SECONDS:
        return f"only {seconds:.2f}s of audio"
    peak = float(np.abs(audio).max())
    if not np.isfinite(peak) or peak < MIN_LINE_PEAK:
        return "audio is silent"
    return None


def _download(url: str, dest: Path, *, retries: int = 4) -> None:
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", dest.name, attempt, retries)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            if tmp.stat().st_size < 100_000:
                raise ValueError(f"{dest.name} suspiciously small: {tmp.stat().st_size} bytes")
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download failed (%s): %s", dest.name, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def ensure_model(model_dir: Path = MODEL_DIR, voices: tuple[str, ...] = DEFAULT_VOICES) -> tuple[Path, Path]:
    """Make sure the ONNX model and the packed voice bundle exist."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILE
    voices_path = model_dir / VOICES_FILE

    if not model_path.exists():
        _download(MODEL_URL, model_path)

    if not voices_path.exists():
        packed: dict[str, np.ndarray] = {}
        for name in voices:
            single = model_dir / f"{name}.bin"
            if not single.exists():
                _download(VOICE_URL.format(name=name), single)
            packed[name] = np.fromfile(single, dtype=np.float32).reshape(510, 1, 256)
        # np.savez appends .npz unless given a file object — that would break kokoro_onnx.
        with open(voices_path, "wb") as fh:
            np.savez(fh, **packed)
        logger.info("Packed %d voices into %s", len(packed), voices_path)

    return model_path, voices_path


class Narration:
    """Lazy Kokoro wrapper: text -> float32 waveform at 24 kHz."""

    def __init__(self, voice: str = "af_heart", model_dir: Path = MODEL_DIR, speed: float = 1.0):
        from kokoro_onnx import Kokoro  # imported here so the rest of the pipeline works without it

        model_path, voices_path = ensure_model(model_dir)
        self._k = Kokoro(str(model_path), str(voices_path))
        self.voice = voice
        self.speed = speed
        logger.info("Kokoro loaded (voice=%s, speed=%.2f)", voice, speed)

    def wav(self, text: str, *, speed: float | None = None,
            voice: str | None = None) -> np.ndarray:
        """Synthesise one line. `voice` overrides the instance voice for this call only.

        The CTA uses a different voice from the narration (brighter, faster) without paying to
        load a second Kokoro model — the voice is a style matrix, not a separate network.
        """
        audio, sr = self._k.create(
            text,
            voice=voice or self.voice,
            speed=self.speed if speed is None else speed,
            lang="en-us",
        )
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate {sr}")
        return np.asarray(audio, dtype=np.float32)

    def line(
        self,
        text: str,
        *,
        label: str = "",
        voice: str | None = None,
        speed: float | None = None,
        attempts: int = LINE_ATTEMPTS,
        backoff_s: float | None = None,
    ) -> np.ndarray:
        """Synthesise one line, retrying, and refuse to return silence.

        Raises `AudioLineError` when the line still has no usable audio after `attempts` tries.
        Callers abort the run instead of rendering a beat whose narration is missing: a hole in
        the voice track cannot be repaired later, so the part stays queued and is retried.
        """
        backoff_s = LINE_BACKOFF_S if backoff_s is None else backoff_s
        what = label or f"{text[:48]!r}"
        last = ""
        for attempt in range(1, attempts + 1):
            try:
                audio: np.ndarray | None = self.wav(text, speed=speed, voice=voice)
            except Exception as exc:  # noqa: BLE001 — kokoro raises a variety of errors
                audio, last = None, str(exc) or exc.__class__.__name__
            if audio is not None:
                problem = _line_problem(audio)
                if problem is None:
                    if attempt > 1:
                        logger.info("%s: synthesised on attempt %d/%d", what, attempt, attempts)
                    return audio
                last = problem
            if attempt < attempts:
                logger.warning("%s: attempt %d/%d failed (%s); retrying",
                               what, attempt, attempts, last)
                if backoff_s:
                    time.sleep(backoff_s * attempt)
        raise AudioLineError(f"{what}: no usable audio after {attempts} attempts ({last})")


def write_wav(path: Path, audio: np.ndarray) -> Path:
    import soundfile as sf

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.89 / peak)  # leave headroom, never clip
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def join_with_gaps(segments: list[np.ndarray], gap_seconds: float = 0.35) -> tuple[np.ndarray, list[float]]:
    """Concatenate lines with a breath between them; return audio and each cue's start time."""
    gap = np.zeros(int(SAMPLE_RATE * gap_seconds), dtype=np.float32)
    parts: list[np.ndarray] = []
    starts: list[float] = []
    cursor = 0.0
    for i, seg in enumerate(segments):
        if i:
            parts.append(gap)
            cursor += gap_seconds
        starts.append(cursor)
        parts.append(seg)
        cursor += len(seg) / SAMPLE_RATE
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32), starts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision / smoke-test Kokoro TTS")
    parser.add_argument("--download", action="store_true", help="download the ONNX model and voices")
    parser.add_argument("--say", help="synthesise this line to test.wav")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.download:
        m, v = ensure_model()
        print(f"model:  {m} ({m.stat().st_size / 1e6:.0f} MB)\nvoices: {v} ({v.stat().st_size / 1e6:.1f} MB)")
    if args.say:
        t = time.time()
        audio = Narration().wav(args.say)
        out = write_wav(Path("test.wav"), audio)
        print(f"{out}: {len(audio) / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz in {time.time() - t:.1f}s")
