"""Strategy-engine tests — no network, no quota.

The Workers AI calls are monkeypatched, so these assert routing and degradation logic only:
which strategy ran, whether beats were conditioned on the canonical frame, and whether a
fatally-unavailable edit model drops to fresh renders without losing the run.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

import images

JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-payload"


def _jpeg(tag: bytes) -> bytes:
    return JPEG + tag


class TestProduceImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.calls: list[tuple[str, str]] = []
        self.edits: list[bytes] = []
        self.beats = [f"beat number {i} is a complete sentence." for i in range(1, 5)]

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_fresh(self, ok=True, error=None):
        def fn(prompt, token, account):
            self.calls.append(("fresh", prompt))
            return images.ImageResult(ok=ok, data=_jpeg(b"fresh") if ok else None, error=error)
        return fn

    def _fake_edit(self, ok=True, error=None):
        def fn(reference, prompt, token, account):
            self.calls.append(("edit", prompt))
            self.edits.append(reference)
            return images.ImageResult(ok=ok, data=_jpeg(b"edit") if ok else None, error=error)
        return fn

    def test_edit_strategy_conditions_every_beat_on_canonical(self):
        images.generate_fresh = self._fake_fresh()
        images.generate_edit = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit")
        self.assertEqual(used, "edit")
        self.assertEqual(len(frames), 4)
        self.assertEqual([c[0] for c in self.calls], ["fresh", "edit", "edit", "edit"])
        # every edit is conditioned on beat 1's frame, never on the previous edit (no drift chain)
        self.assertEqual(len(self.edits), 3)
        self.assertTrue(all(ref == _jpeg(b"fresh") for ref in self.edits))

    def test_identity_anchor_is_in_the_edit_prompt(self):
        images.generate_fresh = self._fake_fresh()
        images.generate_edit = self._fake_edit()
        images.produce_images(self.beats, cfa_token="t", account_id="a",
                              img_dir=self.dir, strategy="edit")
        edit_prompts = [p for kind, p in self.calls if kind == "edit"]
        self.assertTrue(all("EXACT same character" in p for p in edit_prompts))

    def test_plan_restricted_edit_model_falls_back_to_fresh(self):
        images.generate_fresh = self._fake_fresh()
        images.generate_edit = self._fake_edit(ok=False, error="plan_restricted")
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit")
        self.assertEqual(used, "fresh")
        self.assertEqual(len(frames), 4)
        # one attempt at an edit, then straight to fresh for the rest
        self.assertEqual([c[0] for c in self.calls], ["fresh", "edit", "fresh", "fresh", "fresh"])

    def test_transient_edit_failure_only_affects_that_beat(self):
        state = {"n": 0}

        def flaky(reference, prompt, token, account):
            state["n"] += 1
            if state["n"] == 1:  # first edit fails, rest succeed
                return images.ImageResult(ok=False, error="status_500")
            return images.ImageResult(ok=True, data=_jpeg(b"edit"))

        images.generate_fresh = self._fake_fresh()
        images.generate_edit = flaky
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit")
        self.assertEqual(used, "edit")
        self.assertEqual(len(frames), 4)

    def test_fresh_strategy_never_calls_edit(self):
        images.generate_fresh = self._fake_fresh()
        images.generate_edit = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="fresh")
        self.assertEqual(used, "fresh")
        self.assertTrue(all(kind == "fresh" for kind, _ in self.calls))
        self.assertEqual(len(frames), 4)

    def test_budget_stops_generation(self):
        images.generate_fresh = self._fake_fresh()
        images.generate_edit = self._fake_edit()
        budget = {"attempts": 6, "max": 8}
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir,
            strategy="fresh", budget=budget)
        self.assertEqual(len(frames), 2)   # only what the remaining budget allowed
        self.assertEqual(budget["attempts"], 8)

    def test_quota_exhausted_from_canonical_frame_aborts_cleanly(self):
        images.generate_fresh = self._fake_fresh(ok=False, error="quota_exhausted")
        images.generate_edit = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit")
        self.assertEqual(frames, [])
        # quota exhaustion must stop the run instead of hammering the API
        self.assertEqual([c[0] for c in self.calls], ["fresh"])


class TestCropGrid(unittest.TestCase):
    def test_sheet_splits_into_equal_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sheet = d / "sheet.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "testsrc2=s=900x600", "-frames:v", "1", str(sheet)], check=True)
            panels = images.crop_grid(sheet, d, 6, columns=3)
            self.assertEqual(len(panels), 6)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0", str(panels[0])],
                capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(probe, "300,300")


if __name__ == "__main__":
    unittest.main()
