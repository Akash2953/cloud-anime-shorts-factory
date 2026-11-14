"""Narration line guards: retry, and refuse to ship a beat whose voice is missing.

The failure these lock down: Kokoro could hand back an empty array (or a line it never voiced),
and the run rendered anyway — a hole in the voice track that no later stage can repair. A line
is now retried and then refused, which aborts the run *before* the render, leaving the part
queued so tomorrow retries the same one.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import audio
from audio import AudioLineError, Narration, SAMPLE_RATE
from scenes import _template_shots

CFG = {
    "KOKORO_VOICE": "af_heart", "PREFERRED_SECONDS": 55.0, "MIN_SECONDS": 45.0,
    "MAX_SECONDS": 75.0, "IMAGE_STRATEGY": "edit", "SERIES_PARTS": 1,
    "CTA_SECONDS": 0, "MUSIC_ENABLED": False, "WORD_CAPTIONS": False,
}


class ScriptedKokoro:
    """Kokoro stand-in: a tone by default, silence for chosen voices, failures on demand."""

    def __init__(self, *, fail_texts=(), raise_texts=(), silent_voices=()):
        self.fail_texts = set(fail_texts)
        self.raise_texts = set(raise_texts)
        self.silent_voices = set(silent_voices)
        self.requests: list[tuple[str, str | None]] = []

    def create(self, text, voice=None, speed=1.0, lang="en-us"):
        self.requests.append((text, voice))
        if voice in self.silent_voices:
            return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE
        if text in self.raise_texts:
            raise RuntimeError("kokoro crashed")
        if text in self.fail_texts:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        return np.full(int(1.2 * SAMPLE_RATE), 0.1, dtype=np.float32), SAMPLE_RATE


def make_narration(kokoro: ScriptedKokoro, voice: str = "af_heart") -> Narration:
    """A real Narration whose model backend is scripted — no ONNX load, no weights."""
    narration = Narration.__new__(Narration)
    narration.voice = voice
    narration.speed = 1.0
    narration._k = kokoro
    return narration


class TestNarrationLine(unittest.TestCase):
    def setUp(self):
        self._backoff = audio.LINE_BACKOFF_S
        audio.LINE_BACKOFF_S = 0

    def tearDown(self):
        audio.LINE_BACKOFF_S = self._backoff

    def test_clean_line_is_returned_in_one_attempt(self):
        kokoro = ScriptedKokoro()
        line = make_narration(kokoro).line("The lantern answered him.")
        self.assertAlmostEqual(len(line) / SAMPLE_RATE, 1.2, places=3)
        self.assertEqual(len(kokoro.requests), 1)

    def test_crash_is_retried_then_recovers(self):
        calls: dict[str, int] = {"n": 0}

        class Flaky:
            def create(self, text, voice=None, speed=1.0, lang="en-us"):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("kokoro crashed once")
                return np.full(int(1.2 * SAMPLE_RATE), 0.1, dtype=np.float32), SAMPLE_RATE

        narration = make_narration(Flaky())
        line = narration.line("first try dies", label="beat 1")
        self.assertGreater(len(line), 0)
        self.assertEqual(calls["n"], 2)

    def test_empty_audio_is_a_failed_line(self):
        kokoro = ScriptedKokoro(fail_texts={"a line with no voice"})
        with self.assertRaises(AudioLineError) as caught:
            make_narration(kokoro).line("a line with no voice", label="beat 2")
        self.assertIn("empty audio", str(caught.exception))
        self.assertEqual(len(kokoro.requests), audio.LINE_ATTEMPTS)

    def test_crash_is_retried_up_to_the_attempt_limit(self):
        kokoro = ScriptedKokoro(raise_texts={"always dies"})
        with self.assertRaises(AudioLineError):
            make_narration(kokoro).line("always dies", label="beat 3")
        self.assertEqual(len(kokoro.requests), audio.LINE_ATTEMPTS)

    def test_silent_audio_is_refused(self):
        kokoro = ScriptedKokoro(silent_voices={"af_heart"})
        with self.assertRaises(AudioLineError) as caught:
            make_narration(kokoro).line("nobody home")
        self.assertIn("silent", str(caught.exception))

    def test_trimmed_audio_is_refused(self):
        kokoro = ScriptedKokoro(fail_texts={"blip"})
        kokoro.create = lambda text, voice=None, speed=1.0, lang="en-us": (
            np.full(int(0.05 * SAMPLE_RATE), 0.1, dtype=np.float32), SAMPLE_RATE)
        with self.assertRaises(AudioLineError) as caught:
            make_narration(kokoro).line("blip")
        self.assertIn("only 0.05s", str(caught.exception))

    def test_voice_and_speed_are_passed_through(self):
        kokoro = ScriptedKokoro()
        make_narration(kokoro).line("hello", voice="af_bella", speed=1.15)
        self.assertEqual(kokoro.requests[0][1], "af_bella")

    def test_there_is_no_time_scaling_path(self):
        # Pacing is a product rule, not a tuning knob: the pipeline extends the video instead, so
        # a helper that compresses the read must not exist to be called again by accident.
        self.assertFalse(hasattr(make_narration(ScriptedKokoro()), "synth_scaled"))


class TestPipelineAbortsOnAMissingLine(unittest.TestCase):
    """run_pipeline must stop before the render, leaving the publish queue untouched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "build" / "frames").mkdir(parents=True)
        for i in range(3):
            (self.dir / "build" / "frames" / f"frame_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        self._backoff = audio.LINE_BACKOFF_S
        audio.LINE_BACKOFF_S = 0

    def tearDown(self):
        audio.LINE_BACKOFF_S = self._backoff
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def _state(self) -> dict:
        return {"processed": [], "daily": {}, "series": {
            "id": "2026-01-01", "theme": "t", "parts": [], "next_part": 0,
            "total_parts": 1, "image_prompts": {}, "character": "a girl in a red coat"}}

    def _run(self, narration, *, cfg_extra=None, render=None):
        from orchestrator import run_pipeline
        shots = _template_shots(3)
        cfg = dict(CFG, **(cfg_extra or {}))
        state = self._state()
        with patch("orchestrator.load_config", return_value=cfg), \
             patch("orchestrator.load_state", return_value=state), \
             patch("orchestrator.resolve_part", return_value=(shots, 1, 1, state)), \
             patch("orchestrator.Narration", return_value=narration), \
             patch("orchestrator.render_narration", return_value=True) as mocked:
            run_pipeline(render_only=True, reuse_frames=True)
        return mocked

    def test_a_line_that_will_not_voice_aborts_before_render(self):
        shots = _template_shots(3)
        kokoro = ScriptedKokoro(fail_texts={shots[1].speech})
        render = self._run(make_narration(kokoro))
        self.assertFalse(render.called)                    # nothing is rendered
        self.assertFalse((self.dir / "output_short.mp4").exists())

    def test_a_crashed_line_also_aborts_before_render(self):
        shots = _template_shots(3)
        kokoro = ScriptedKokoro(raise_texts={shots[0].speech})
        render = self._run(make_narration(kokoro))
        self.assertFalse(render.called)

    def test_unvoiceable_cta_is_dropped_and_the_short_still_renders(self):
        from cta import CTA_VOICE
        kokoro = ScriptedKokoro(silent_voices={CTA_VOICE})
        render = self._run(make_narration(kokoro), cfg_extra={"CTA_SECONDS": 4})
        self.assertTrue(render.called)
        frames = render.call_args.args[0]
        self.assertEqual(len(frames), 3)                   # the CTA beat is gone, the story is not


if __name__ == "__main__":
    unittest.main()
