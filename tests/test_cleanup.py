"""Cleanup: artefacts go after a successful upload, and stay after a failed one.

Two things must hold. Nothing outside the pipeline's own artefact list may be touched — `build/`
can hold unrelated work — and a run that failed before publishing must keep its media so it can be
re-rendered or inspected.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from audio import SAMPLE_RATE
from cleanup import ARTIFACTS, remove_artifacts
from scenes import _template_shots

CFG = {
    "KOKORO_VOICE": "af_heart", "PREFERRED_SECONDS": 55.0, "MIN_SECONDS": 45.0,
    "MAX_SECONDS": 75.0, "IMAGE_STRATEGY": "edit", "SERIES_PARTS": 1,
    "CTA_SECONDS": 0, "MUSIC_ENABLED": False, "WORD_CAPTIONS": False, "THUMBNAIL": False,
    "PUBLISH_PRIVACY": "public", "CLEANUP_AFTER_UPLOAD": True,
    "YOUTUBE_CLIENT_ID": "id", "YOUTUBE_CLIENT_SECRET": "secret",
    "YOUTUBE_REFRESH_TOKEN": "refresh",
}


class TestRemoveArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_run_artifacts(self) -> None:
        (self.root / "build" / "frames").mkdir(parents=True)
        (self.root / "build" / "frames" / "frame_0.jpg").write_bytes(b"x" * 100)
        (self.root / "build" / "captions").mkdir()
        (self.root / "build" / "captions" / "cap_00.png").write_bytes(b"y" * 50)
        (self.root / "build" / "narration.wav").write_bytes(b"z" * 200)
        (self.root / "output_short.mp4").write_bytes(b"v" * 500)

    def test_removes_the_runs_own_artifacts(self):
        self.make_run_artifacts()
        removed, freed = remove_artifacts(self.root)
        for name in ("build/frames", "build/captions", "build/narration.wav", "output_short.mp4"):
            self.assertFalse((self.root / name).exists(), name)
            self.assertIn(self.root / name, removed)
        self.assertEqual(freed, 850)
        self.assertTrue((self.root / "build").exists())   # the parent stays

    def test_unrelated_build_content_is_untouched(self):
        self.make_run_artifacts()
        other = self.root / "build" / "someone-elses-project"
        other.mkdir(parents=True)
        (other / "notes.md").write_text("do not delete me")
        remove_artifacts(self.root)
        self.assertTrue((other / "notes.md").exists())

    def test_missing_artifacts_are_not_an_error(self):
        removed, freed = remove_artifacts(self.root)
        self.assertEqual((removed, freed), ([], 0))

    def test_keep_list_protects_a_named_artifact(self):
        self.make_run_artifacts()
        removed, _freed = remove_artifacts(self.root, keep=("output_short.mp4",))
        self.assertTrue((self.root / "output_short.mp4").exists())
        self.assertNotIn(self.root / "output_short.mp4", removed)

    def test_the_artifact_list_covers_the_pipeline_outputs(self):
        for name in ("output_short.mp4", "build/frames", "build/captions",
                     "build/narration.wav", "build/music_bed.wav", "build/thumbnail.jpg"):
            self.assertIn(name, ARTIFACTS)


class TestPipelineCleanup(unittest.TestCase):
    """run_pipeline deletes only after a successful upload."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        frames = self.dir / "build" / "frames"
        frames.mkdir(parents=True)
        for i in range(3):
            (frames / f"frame_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
        (self.dir / "build" / "captions").mkdir()
        (self.dir / "build" / "captions" / "cap_00.png").write_bytes(b"png")

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def run_pipeline(self, *, video_id: str | None, cfg_extra: dict | None = None) -> None:
        from orchestrator import run_pipeline
        shots = _template_shots(3)
        state = {"processed": [], "daily": {}, "series": {
            "id": "2026-01-01", "theme": "t", "parts": [], "next_part": 0, "total_parts": 1,
            "image_prompts": {}, "character": "a girl in a red coat", "title": "The Lantern"}}

        tone = np.full(int(1.0 * SAMPLE_RATE), 0.1, dtype=np.float32)
        narration = MagicMock()
        narration.line.return_value = tone
        narration.synth_scaled.return_value = ([tone] * len(shots), 1.0)

        def fake_render(frames, durations, audio, out, **kwargs):
            Path(out).write_bytes(b"v" * 500)      # the render step really writes the MP4
            return True

        with patch("orchestrator.load_config", return_value=dict(CFG, **(cfg_extra or {}))), \
             patch("orchestrator.load_state", return_value=state), \
             patch("orchestrator.resolve_part", return_value=(shots, 1, 1, state)), \
             patch("orchestrator.Narration", return_value=narration), \
             patch("orchestrator.render_narration", side_effect=fake_render), \
             patch("orchestrator.exchange_refresh_token", return_value="token"), \
             patch("orchestrator.upload_video", return_value=video_id):
            run_pipeline(reuse_frames=True)

    def test_a_successful_upload_clears_the_artifacts(self):
        self.run_pipeline(video_id="abc123")
        self.assertFalse((self.dir / "output_short.mp4").exists())
        self.assertFalse((self.dir / "build" / "frames").exists())
        self.assertFalse((self.dir / "build" / "captions").exists())
        self.assertFalse((self.dir / "build" / "narration.wav").exists())

    def test_a_failed_upload_keeps_everything(self):
        self.run_pipeline(video_id=None)
        self.assertTrue((self.dir / "output_short.mp4").exists())
        self.assertTrue((self.dir / "build" / "frames").exists())
        self.assertTrue((self.dir / "build" / "narration.wav").exists())

    def test_the_knob_turns_cleanup_off(self):
        self.run_pipeline(video_id="abc123", cfg_extra={"CLEANUP_AFTER_UPLOAD": False})
        self.assertTrue((self.dir / "output_short.mp4").exists())
        self.assertTrue((self.dir / "build" / "frames").exists())


if __name__ == "__main__":
    unittest.main()
