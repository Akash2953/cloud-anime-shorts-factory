"""Strategy-engine tests — no network, no quota.

The Workers AI calls are monkeypatched, so these assert routing and degradation logic only:
which strategy ran, whether beats were conditioned on the canonical frame, and whether a
fatally-unavailable edit model drops to fresh renders without losing the run.
"""

import io
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
        def fn(prompt, token, account, **kw):
            self.calls.append(("fresh", prompt))
            return images.ImageResult(ok=ok, data=_jpeg(b"fresh") if ok else None, error=error)
        return fn

    def _fake_edit(self, ok=True, error=None):
        def fn(reference, prompt, token, account, **kw):
            self.calls.append(("edit", prompt))
            self.edits.append(reference)
            return images.ImageResult(ok=ok, data=_jpeg(b"edit") if ok else None, error=error)
        return fn

    def test_edit_strategy_conditions_every_beat_on_canonical(self):
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit",
            antigravity_key="ag")
        self.assertEqual(used, "edit")
        self.assertEqual(len(frames), 4)
        self.assertEqual([c[0] for c in self.calls], ["fresh", "edit", "edit", "edit"])
        # every edit is conditioned on beat 1's frame, never on the previous edit (no drift chain)
        self.assertEqual(len(self.edits), 3)
        self.assertTrue(all(ref == _jpeg(b"fresh") for ref in self.edits))

    def test_identity_anchor_is_last_in_the_edit_prompt(self):
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit()
        images.produce_images(self.beats, cfa_token="t", account_id="a",
                              img_dir=self.dir, strategy="edit",
                              antigravity_key="ag", character="a man in a grey poncho")
        edit_prompts = [p for kind, p in self.calls if kind == "edit"]
        self.assertTrue(edit_prompts)
        for prompt, beat in zip(edit_prompts, self.beats[1:]):
            # The shot leads: leading with "keep the EXACT same character" made Agnes reproduce
            # the reference composition for most of the short.
            self.assertTrue(prompt.startswith(beat), prompt[:80])
            self.assertLess(prompt.index(beat), prompt.index("same character as the reference"))
            self.assertIn("grey poncho", prompt)
            # The reference supplies the design only — its pose is explicitly not this frame's.
            self.assertIn("performing the action described above", prompt)
            self.assertIn("must not be copied", prompt)
            self.assertNotIn("EXACT same character", prompt)

    def test_plan_restricted_edit_model_falls_back_to_fresh(self):
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit(ok=False, error="plan_restricted")
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit",
            antigravity_key="ag")
        self.assertEqual(used, "fresh")
        self.assertEqual(len(frames), 4)
        # one attempt at an edit, then straight to fresh for the rest
        self.assertEqual([c[0] for c in self.calls], ["fresh", "edit", "fresh", "fresh", "fresh"])

    def test_transient_edit_failure_only_affects_that_beat(self):
        state = {"n": 0}

        def flaky(reference, prompt, token, account, **kw):
            state["n"] += 1
            if state["n"] == 1:  # first edit fails, rest succeed
                return images.ImageResult(ok=False, error="status_500")
            return images.ImageResult(ok=True, data=_jpeg(b"edit"))

        images.fresh_image = self._fake_fresh()
        images.edit_image = flaky
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit",
            antigravity_key="ag")
        self.assertEqual(used, "edit")
        self.assertEqual(len(frames), 4)

    def test_fresh_strategy_never_calls_edit(self):
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="fresh")
        self.assertEqual(used, "fresh")
        self.assertTrue(all(kind == "fresh" for kind, _ in self.calls))
        self.assertEqual(len(frames), 4)

    def test_budget_stops_generation(self):
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit()
        budget = {"attempts": 6, "max": 8}
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir,
            strategy="fresh", budget=budget)
        self.assertEqual(len(frames), 2)   # only what the remaining budget allowed
        self.assertEqual(budget["attempts"], 8)

    def test_quota_exhausted_from_canonical_frame_aborts_cleanly(self):
        images.fresh_image = self._fake_fresh(ok=False, error="quota_exhausted")
        images.edit_image = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit",
            antigravity_key="ag")
        self.assertEqual(frames, [])
        # quota exhaustion must stop the run instead of hammering the API
        self.assertEqual([c[0] for c in self.calls], ["fresh"])

    def test_no_antigravity_key_renders_fresh_instead_of_editing(self):
        """The remaining providers repeat the reference's pose, so an edit is not offered."""
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit()
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit")
        self.assertEqual(used, "fresh")
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(kind == "fresh" for kind, _ in self.calls))

    def test_exhausted_edit_model_falls_back_to_fresh_for_the_rest(self):
        """An exhausted edit model must not end the short; the remaining beats still render."""
        images.fresh_image = self._fake_fresh()
        images.edit_image = self._fake_edit(ok=False, error="quota_exhausted")
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="edit",
            antigravity_key="ag")
        self.assertEqual(used, "fresh")
        self.assertEqual(len(frames), 4)
        self.assertEqual([c[0] for c in self.calls], ["fresh", "edit", "fresh", "fresh", "fresh"])


class TestEditCollapseGuard(unittest.TestCase):
    """A frame that repeats one already rendered is asked again, once.

    Measured failure this guards: Agnes returned the reference portrait for five of eight shots
    the plan had made different (close-up, medium, wide, close-up, medium), so the short played as
    one held portrait — and a later run returned shot 6 as a near-copy of shot 2, which the guard
    missed while it only compared against the canonical frame. Frames are now compared with every
    frame already rendered: identity is not the only thing an edit can duplicate.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.calls: list[tuple[str, str]] = []
        self.flat = _png(flat=40)
        self.other = _png(gradient=True)
        self.third = _png(flat=180)
        self.beats = [f"beat number {i} is a complete sentence." for i in range(1, 4)]

    def tearDown(self):
        self.tmp.cleanup()

    def _fresh(self, prompt, token, account, **kw):
        self.calls.append(("fresh", prompt))
        return images.ImageResult(ok=True, data=self.flat)

    def _edit_returning(self, *frames: bytes):
        """One image per edit call, in order (so a retry gets the next one in the list)."""
        queue = list(frames)

        def edit(reference, prompt, token, account, **kw):
            self.calls.append(("edit", prompt))
            return images.ImageResult(ok=True, data=queue.pop(0) if queue else self.third)
        return edit

    def test_a_near_copy_of_the_canonical_is_retried_once(self):
        images.fresh_image = self._fresh
        images.edit_image = self._edit_returning(self.flat, self.other, self.third)
        budget = {"attempts": 0, "max": 10}
        frames, used = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir,
            strategy="edit", budget=budget, antigravity_key="ag")
        self.assertEqual(used, "edit")
        self.assertEqual(len(frames), 3)
        edits = [p for kind, p in self.calls if kind == "edit"]
        self.assertEqual(len(edits), 3)          # beat 2 + one retry, then beat 3
        self.assertIn("ignore its pose, crop, camera angle and background", edits[1])
        self.assertEqual(budget["attempts"], 4)  # 1 fresh + edit + retry + edit

    def test_a_near_copy_of_a_neighbouring_shot_is_retried_too(self):
        images.fresh_image = self._fresh
        images.edit_image = self._edit_returning(self.other, self.other, self.third)
        images.produce_images(self.beats, cfa_token="t", account_id="a",
                              img_dir=self.dir, strategy="edit", antigravity_key="ag")
        edits = [p for kind, p in self.calls if kind == "edit"]
        # beat 2 renders, beat 3 repeats it, the repeat is retried — not left holding a twin.
        self.assertEqual(len(edits), 3)
        self.assertIn("ignore its pose, crop, camera angle and background", edits[2])

    def test_a_second_near_copy_is_replaced_by_a_fresh_render(self):
        def fresh(prompt, token, account, **kw):
            self.calls.append(("fresh", prompt))
            return images.ImageResult(
                ok=True, data=self.flat if len(self.calls) == 1 else self.other)

        images.fresh_image = fresh
        # Both edit attempts return the canonical frame again; only a fresh render can differ.
        images.edit_image = self._edit_returning(self.flat, self.flat)
        frames, _ = images.produce_images(self.beats[:2], cfa_token="t", account_id="a",
                                          img_dir=self.dir, strategy="edit", antigravity_key="ag")
        self.assertEqual([c[0] for c in self.calls], ["fresh", "edit", "edit", "fresh"])
        self.assertEqual(Path(frames[1]).read_bytes(), self.other)

    def test_a_different_edit_is_not_retried(self):
        images.fresh_image = self._fresh
        images.edit_image = self._edit_returning(self.other, self.third)
        images.produce_images(self.beats, cfa_token="t", account_id="a",
                              img_dir=self.dir, strategy="edit", antigravity_key="ag")
        self.assertEqual(len([c for c in self.calls if c[0] == "edit"]), 2)

    def test_an_undecodable_payload_does_not_crash_the_guard(self):
        images.fresh_image = self._fake_jpeg_fresh(self)
        images.edit_image = self._fake_jpeg_edit(self)
        frames, _ = images.produce_images(self.beats, cfa_token="t", account_id="a",
                                          img_dir=self.dir, strategy="edit", antigravity_key="ag")
        self.assertEqual(len(frames), 3)

    @staticmethod
    def _fake_jpeg_fresh(case):
        def fn(prompt, token, account, **kw):
            case.calls.append(("fresh", prompt))
            return images.ImageResult(ok=True, data=_jpeg(b"fresh"))
        return fn

    @staticmethod
    def _fake_jpeg_edit(case):
        def fn(reference, prompt, token, account, **kw):
            case.calls.append(("edit", prompt))
            return images.ImageResult(ok=True, data=_jpeg(b"edit"))
        return fn


def _png(*, flat: int = 0, gradient: bool = False) -> bytes:
    """A real, decodable PNG so the luma guard can compare two frames."""
    from PIL import Image

    img = Image.new("L", (128, 128), flat)
    if gradient:
        img.putdata([(x * 2 + y) % 256 for y in range(128) for x in range(128)])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TestShotRetry(unittest.TestCase):
    """A failed shot must not silently delete its narration line from the story."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.calls: list[str] = []
        self.beats = [f"beat number {i} is a complete sentence." for i in range(1, 5)]
        # Retries are exercised without the real backoff sleep.
        self._backoff = images.RETRY_BACKOFF_S
        images.RETRY_BACKOFF_S = 0

    def tearDown(self):
        images.RETRY_BACKOFF_S = self._backoff
        self.tmp.cleanup()

    def _scripted(self, failures: dict[int, int]):
        """Fail the Nth shot `failures[N]` times with a retryable 500, then succeed."""
        seen: dict[int, int] = {}

        def fn(prompt, token, account, **kw):
            self.calls.append(prompt)
            # Prompts are `beat number N is a complete sentence.` — key attempts per shot, so a
            # retry is attributed to the shot it belongs to rather than to the next one.
            n = int(prompt.split("number ")[1].split()[0]) - 1
            seen[n] = seen.get(n, 0) + 1
            if seen[n] <= failures.get(n, 0):
                return images.ImageResult(ok=False, error="status_500")
            return images.ImageResult(ok=True, data=_jpeg(str(n).encode()))
        return fn

    def test_transient_failure_is_retried_within_the_same_shot(self):
        images.fresh_image = self._scripted({1: 1})   # shot 2 fails once
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="fresh")
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(self.calls), 5)             # 4 shots + 1 retry
        self.assertTrue(all(p.exists() for p in frames))

    def test_shot_that_never_renders_holds_the_previous_frame(self):
        images.fresh_image = self._scripted({2: images.SHOT_ATTEMPTS})  # shot 3 never renders
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="fresh")
        self.assertEqual(len(frames), 4)                 # every beat still has a frame
        self.assertEqual(frames[2], frames[1])           # the hold
        self.assertEqual(len(self.calls), 4 + images.SHOT_ATTEMPTS - 1)

    def test_leading_hold_backfills_from_the_first_frame_that_rendered(self):
        images.fresh_image = self._scripted({0: images.SHOT_ATTEMPTS})  # shot 1 never renders
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="fresh")
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0], frames[1])

    def test_retries_are_charged_to_the_daily_budget(self):
        images.fresh_image = self._scripted({0: 1})
        budget = {"attempts": 0, "max": 4}
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir,
            strategy="fresh", budget=budget)
        self.assertEqual(budget["attempts"], 4)
        self.assertLessEqual(len(frames), 4)

    def test_retry_does_not_starve_the_remaining_shots(self):
        # 4 beats, 5 attempts: shot 1 can retry once, then the reserve protects beats 2-4.
        images.fresh_image = self._scripted({0: 99})
        budget = {"attempts": 0, "max": 5}
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir,
            strategy="fresh", budget=budget)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(len(frames), 4)                 # every beat still plays
        self.assertEqual(frames[0], frames[1])           # shot 1 held

    def test_fatal_error_is_not_retried(self):
        def fn(prompt, token, account, **kw):
            self.calls.append("x")
            return images.ImageResult(ok=False, error="auth_401")

        images.fresh_image = fn
        frames, _ = images.produce_images(
            self.beats, cfa_token="t", account_id="a", img_dir=self.dir, strategy="fresh")
        self.assertEqual(frames, [])
        self.assertEqual(len(self.calls), 1)             # one call, no retry


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


class TestLoadFrames(unittest.TestCase):
    """`--reuse-frames` must hand the beats back in shot order."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, payload: bytes = JPEG) -> None:
        (self.dir / name).write_bytes(payload)

    def test_double_digits_sort_numerically(self):
        for i in (0, 1, 2, 10, 11, 12):
            self.write(f"frame_{i}.jpg")
        self.assertEqual([p.name for p in images.load_frames(self.dir)],
                         ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg", "frame_10.jpg",
                          "frame_11.jpg", "frame_12.jpg"])

    def test_one_file_per_beat_and_jpg_wins(self):
        self.write("frame_0.jpg")
        self.write("frame_0.png", b"\x89PNG")
        self.write("frame_1.png", b"\x89PNG")
        loaded = images.load_frames(self.dir)
        self.assertEqual([p.name for p in loaded], ["frame_0.jpg", "frame_1.png"])

    def test_sheet_panels_are_not_beats(self):
        self.write("frame_0.jpg")
        self.write("panel_0.jpg")
        self.write("panel_1.jpg")
        self.assertEqual(len(images.load_frames(self.dir)), 1)


if __name__ == "__main__":
    unittest.main()
