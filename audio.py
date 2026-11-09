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

    def wav(self, text: str, *, speed: float | None = None) -> np.ndarray:
        audio, sr = self._k.create(
            text,
            voice=self.voice,
            speed=self.speed if speed is None else speed,
            lang="en-us",
        )
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate {sr}")
        return np.asarray(audio, dtype=np.float32)

    def synth_scaled(
        self, lines: list[str], target_seconds: float, *, iterations: int = 4, tol: float = 0.02
    ) -> tuple[list[np.ndarray], float]:
        """Fit the speaking rate to `target_seconds` by bisection.

        A single ratio (measured/target) overshoots, because Kokoro's fixed pauses make
        duration non-monotone in `speed`. Bisection on a monotone-decreasing assumption
        converges in a few passes and keeps the best candidate seen.
        """
        segments = [self.wav(line) for line in lines]
        speed = self.speed
        if target_seconds <= 0 or not segments:
            return segments, speed

        measured = sum(len(a) for a in segments) / SAMPLE_RATE
        logger.info("Pass 0: %.2fs at speed %.2f (target %.2fs)", measured, speed, target_seconds)
        if abs(measured - target_seconds) / target_seconds <= tol:
            return segments, speed

        best = (abs(measured - target_seconds), speed, segments)
        lo, hi = 0.7, 1.5
        # Faster speech is shorter, so measured > target means search higher.
        if measured > target_seconds:
            lo = self.speed
        else:
            hi = self.speed

        for attempt in range(1, iterations + 1):
            speed = round((lo + hi) / 2, 3)
            candidate = [self.wav(line, speed=speed) for line in lines]
            measured = sum(len(a) for a in candidate) / SAMPLE_RATE
            error = abs(measured - target_seconds)
            logger.info("Pass %d: %.2fs at speed %.3f (error %.1f%%)",
                        attempt, measured, speed, error / target_seconds * 100)
            if error < best[0]:
                best = (error, speed, candidate)
            if error / target_seconds <= tol:
                break
            if measured > target_seconds:
                lo = speed
            else:
                hi = speed

        return best[2], best[1]


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
