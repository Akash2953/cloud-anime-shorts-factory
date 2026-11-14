"""Render resilience: a failed or truncated encode is retried, never left on disk.

The failure these lock down: ffmpeg died mid-write and left a half-written `output_short.mp4`.
Nothing re-ran the encode, and a later `--render-only` or thumbnail step would read that file as
if it were a short.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import render

FPS = 24


def _clip(path: Path, seconds: float) -> Path:
    """A real, short MP4 of the given length, so ffprobe can measure it."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=s=360x640:r={FPS}",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
         "-t", f"{seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


class TestEncodeRetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.good = _clip(self.dir / "good.mp4", 3.0)
        self.short = _clip(self.dir / "short.mp4", 0.5)
        self.out = self.dir / "out.mp4"
        self.calls: list[int] = []
        self.real_run = subprocess.run

    def tearDown(self):
        self.tmp.cleanup()

    def fake_ffmpeg(self, attempts):
        """`attempts` is a list of ('fail'|'empty'|'partial'|'ok') per ffmpeg invocation."""
        def run(cmd, **kwargs):
            if Path(cmd[0]).name != "ffmpeg":
                return self.real_run(cmd, **kwargs)      # let ffprobe through
            n = len(self.calls)
            self.calls.append(n)
            step = attempts[min(n, len(attempts) - 1)]
            if step == "fail":
                self.out.write_bytes(b"half-written")
                return subprocess.CompletedProcess(cmd, 1, "", "encoder died")
            if step == "empty":
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if step == "partial":
                shutil.copy(self.short, self.out)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            shutil.copy(self.good, self.out)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return run

    def encode(self, attempts, expected=3.0):
        with patch("render.subprocess.run", side_effect=self.fake_ffmpeg(attempts)):
            return render._encode(["ffmpeg", "-x", str(self.out)], self.out,
                                  expected_seconds=expected)

    def test_a_failed_encode_is_retried_and_recovered(self):
        self.assertTrue(self.encode(["fail", "ok"]))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(render._probe_seconds(self.out), 3.0)

    def test_a_truncated_encode_is_detected_and_retried(self):
        # Exit code 0 but a 0.5s file for a 3s timeline: the encode lied, so it is redone.
        self.assertTrue(self.encode(["partial", "ok"]))
        self.assertEqual(len(self.calls), 2)

    def test_an_empty_output_is_a_failure(self):
        self.assertTrue(self.encode(["empty", "ok"]))
        self.assertEqual(len(self.calls), 2)

    def test_every_attempt_failing_leaves_no_video_behind(self):
        self.out.write_bytes(b"stale video from an earlier run")
        self.assertFalse(self.encode(["fail", "fail"]))
        self.assertFalse(self.out.exists())      # a later stage cannot mistake it for a short

    def test_a_good_encode_is_not_repeated(self):
        self.assertTrue(self.encode(["ok"]))
        self.assertEqual(len(self.calls), 1)


class TestRenderNarrationEndToEnd(unittest.TestCase):
    """The real filter graph, so the duration check cannot reject a legitimate render."""

    def test_a_real_render_produces_the_timeline_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            frame_a = d / "a.png"
            frame_b = d / "b.png"
            for frame, size in ((frame_a, "540x960"), (frame_b, "600x900")):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                                "-i", f"testsrc2=s={size}", "-frames:v", "1", str(frame)],
                               check=True)
            audio = d / "sil.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "anullsrc=r=24000:cl=mono", "-t", "4.5", str(audio)],
                           check=True)
            out = d / "out.mp4"
            ok = render.render_narration([frame_a, frame_a, frame_b], [1.5, 1.5, 1.5],
                                         audio, out, fps=FPS)
            self.assertTrue(ok)
            # A held frame (the same path twice) must render like any other beat.
            self.assertAlmostEqual(render._probe_seconds(out) or 0, 4.5, delta=0.5)

    def test_the_duration_check_is_tolerant_but_not_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            clip = _clip(d / "clip.mp4", 3.0)
            out = d / "out.mp4"
            real = subprocess.run

            def run(cmd, **kwargs):
                if Path(cmd[0]).name != "ffmpeg":
                    return real(cmd, **kwargs)          # ffprobe
                shutil.copy(clip, out)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with patch("render.subprocess.run", side_effect=run):
                self.assertTrue(render._encode(["ffmpeg"], out, expected_seconds=3.2))
            with patch("render.subprocess.run", side_effect=run):
                self.assertFalse(render._encode(["ffmpeg"], out, expected_seconds=9.0))


if __name__ == "__main__":
    unittest.main()
