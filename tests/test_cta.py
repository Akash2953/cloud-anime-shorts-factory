"""Tests for the end-of-short CTA, the thumbnail composer and the story title."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cta import CTA_LAST, CTA_MIDDLE, CTA_SINGLE, cta_line, cta_shot
from orchestrator import story_title
from scenes import Shot, fallback_title, parse_title, title_from_line
from thumbnail import THUMB_HEIGHT, THUMB_WIDTH, pick_key_frame, render_thumbnail


def _shot(role="build_tension", narration="She waits."):
    return Shot(narration=narration, shot_size="wide", place="a street", action="waits",
                story_object="no object", narrative_role=role)


class TestCtaLine(unittest.TestCase):
    def test_middle_part_names_the_next_part(self):
        line = cta_line(1, 3)
        self.assertIn("Part 2", line)
        self.assertIn("Subscribe", line)

    def test_middle_part_of_two(self):
        self.assertIn("Part 2", cta_line(1, 2))

    def test_last_part_asks_for_the_channel(self):
        self.assertEqual(cta_line(3, 3), CTA_LAST)
        self.assertNotIn("Part 4", cta_line(3, 3))

    def test_single_part_asks_for_the_channel(self):
        self.assertEqual(cta_line(1, 1), CTA_SINGLE)

    def test_lines_are_short_enough_to_speak(self):
        # ~14 words lands in about 3.5 s at the narrator's pace.
        for line in (CTA_MIDDLE.format(next=2), CTA_LAST, CTA_SINGLE):
            self.assertLessEqual(len(line.split()), 16, line)

    def test_cta_shot_is_a_spoken_beat(self):
        shot = cta_shot(1, 2, seconds=4.0)
        self.assertTrue(shot.speech)
        self.assertEqual(shot.narrative_role, "call_to_action")
        self.assertFalse(shot.character)
        self.assertEqual(shot.seconds, 4.0)


class TestStoryTitle(unittest.TestCase):
    def test_parses_the_title_line(self):
        text = "Title: The Lantern She Left Burning\nShort 1: 8 shots, 8 images."
        self.assertEqual(parse_title(text), "The Lantern She Left Burning")

    def test_tolerates_markdown_and_quotes(self):
        self.assertEqual(parse_title('**Title:** "Rain on the Last Train"'), "Rain on the Last Train")

    def test_missing_title_returns_empty(self):
        self.assertEqual(parse_title("Short 1: 8 shots, 8 images."), "")

    def test_absurdly_long_title_is_rejected(self):
        self.assertEqual(parse_title("Title: " + "word " * 20), "")

    def test_fallback_title_uses_the_opening_clause(self):
        shots = [_shot(narration="The night train hissed to a halt at Inari Station, spilling him out.")]
        self.assertEqual(fallback_title(shots), "The night train hissed to a halt")

    def test_fallback_title_skips_silent_beats(self):
        shots = [_shot(narration="none"), _shot(narration="Rain fell on the empty platform.")]
        self.assertEqual(fallback_title(shots), "Rain fell on the empty platform")

    def test_fallback_title_with_no_speech(self):
        self.assertEqual(fallback_title([_shot(narration="none")]), "")

    def test_planner_title_ending_on_a_conjunction_is_rejected(self):
        # A cut sentence is not a name: fall through to a title built from the narration.
        self.assertEqual(parse_title("Title: Rain drummed on the tin roof while"), "")
        self.assertEqual(parse_title("Title: of"), "")
        self.assertEqual(parse_title("Title: The Girl Who Waited Because"), "")

    def test_a_real_title_still_parses(self):
        self.assertEqual(parse_title("Title: The Lantern That Remembered"),
                         "The Lantern That Remembered")

    def test_fallback_title_never_ends_mid_clause(self):
        # The regression: this uploaded as "Rain drummed on the tin roof while".
        shots = [_shot(narration="Rain drummed on the tin roof while the city slept, and Mio "
                                 "wiped fresh oil from her fingers.")]
        self.assertEqual(fallback_title(shots), "Rain drummed on the tin roof")

    def test_fallback_title_skips_a_clause_that_cannot_stand_alone(self):
        shots = [_shot(narration="And then, because she had promised, the bell rang.")]
        self.assertEqual(fallback_title(shots), "the bell rang")


class TestStoryTitleFromState(unittest.TestCase):
    """`story_title` is the second fallback — it names the upload from the plan in state.json."""

    def test_series_title_wins(self):
        state = {"series": {"title": "The Lantern That Remembered", "parts": []}}
        self.assertEqual(story_title(state), "The Lantern That Remembered")

    def test_missing_title_falls_back_to_a_complete_phrase(self):
        state = {"series": {"title": "", "parts": [[
            {"narration": "Rain drummed on the tin roof while the city slept."}]]}}
        self.assertEqual(story_title(state), "Rain drummed on the tin roof")

    def test_nothing_usable_is_untitled(self):
        state = {"series": {"title": "", "parts": [[{"narration": "none"}]]}}
        self.assertEqual(story_title(state), "Untitled Story")

    def test_title_from_line_rejects_pure_fragments(self):
        self.assertEqual(title_from_line(""), "")
        self.assertEqual(title_from_line("and"), "")
        self.assertEqual(title_from_line("the of at"), "")


class TestThumbnail(unittest.TestCase):
    def test_picks_the_payload_frame_not_the_first(self):
        shots = [_shot("establish_context"), _shot("build_tension"), _shot("deliver_payload")]
        frames = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        self.assertEqual(pick_key_frame(shots, frames), Path("c.jpg"))

    def test_falls_back_to_the_first_frame(self):
        shots = [_shot("establish_context"), _shot("introduce_subject")]
        frames = [Path("a.jpg"), Path("b.jpg")]
        self.assertEqual(pick_key_frame(shots, frames), Path("a.jpg"))

    def test_no_frames_returns_none(self):
        self.assertIsNone(pick_key_frame([_shot()], []))

    def test_renders_a_9x16_jpeg_with_the_title(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "frame.jpg"
            Image.new("RGB", (1024, 1024), (40, 20, 60)).save(src)
            out = Path(tmp) / "thumb.jpg"
            written = render_thumbnail(src, "The Lantern She Left Burning", out)
            self.assertIsNotNone(written)
            self.assertTrue(out.exists())
            with Image.open(out) as im:
                self.assertEqual(im.size, (THUMB_WIDTH, THUMB_HEIGHT))
            # YouTube rejects thumbnails over 2 MB.
            self.assertLess(out.stat().st_size, 2_000_000)


if __name__ == "__main__":
    unittest.main()
