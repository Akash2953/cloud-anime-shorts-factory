"""Unit tests for the deterministic stages of captions.py.

These cover pack_groups / classify_word / filter_usable_words. The whisper alignment and
Pillow PNG rendering paths are exercised manually because they need faster-whisper / Pillow
installed; the deterministic packing logic is what fails most often during tuning.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from captions import (
    GroupCue,
    WordCue,
    _trim_trailing_hallucination,
    build_caption_overlays,
    build_line_level_cues,
    classify_word,
    filter_usable_words,
    find_gaps,
    is_hallucination_token,
    pack_groups,
)


def _w(start_end_text):
    return WordCue(*start_end_text)


class TestPackGroups(unittest.TestCase):
    def test_two_close_one_far(self):
        # The example the spec asks for. she<-tight->ran<-wide-->towards -> 2 groups.
        words = [_w((0.00, 0.20, "she")), _w((0.22, 0.45, "ran")), _w((1.20, 1.55, "towards"))]
        groups = pack_groups(words)
        self.assertEqual([g.text for g in groups], ["she ran", "towards"])
        # Group window: starts AT/just before first.start, ends after last.end + tail (100 ms).
        self.assertAlmostEqual(groups[0].start, 0.00, delta=0.005)
        self.assertAlmostEqual(groups[0].end, 0.55, delta=0.005)
        # Second group must start after first_group.end - overlap_ms (default 20 ms).
        self.assertAlmostEqual(groups[1].start, 1.16, delta=0.005)
        self.assertAlmostEqual(groups[1].end, 1.65, delta=0.005)

    def test_three_tight_words_pack_into_one_group(self):
        words = [
            _w((0.10, 0.30, "the")),
            _w((0.32, 0.50, "red")),
            _w((0.55, 0.80, "door")),
        ]
        groups = pack_groups(words)
        self.assertEqual([g.text for g in groups], ["the red door"])

    def test_dramatic_long_word_stands_alone(self):
        # Even mid a tight cluster, a 0.8 s word gets its own group.
        words = [
            _w((0.00, 0.20, "she")),
            _w((0.22, 1.00, "never")),
            _w((1.05, 1.20, "left")),
        ]
        groups = pack_groups(words)
        self.assertEqual([g.text for g in groups], ["she", "never", "left"])

    def test_max_group_cap(self):
        # 5 tight words -> 1 group of 3 + 1 group of 2.
        words = [
            _w((0.0, 0.2, "a")), _w((0.25, 0.4, "b")), _w((0.45, 0.6, "c")),
            _w((0.65, 0.8, "d")), _w((0.85, 1.0, "e")),
        ]
        groups = pack_groups(words)
        self.assertEqual([g.text for g in groups], ["a b c", "d e"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(pack_groups([]), [])


class TestClassifyWord(unittest.TestCase):
    palette = ["#FF2D55"]

    def test_pronoun_is_white(self):
        self.assertEqual(classify_word("she", palette=self.palette)[0], "white")
        self.assertEqual(classify_word("him", palette=self.palette)[0], "white")

    def test_preposition_is_white(self):
        # Drives the example's "towards" -> white result.
        self.assertEqual(classify_word("towards", palette=self.palette)[0], "white")

    def test_content_verb_is_accent(self):
        # Drives the example's "ran" -> accent.
        self.assertEqual(classify_word("ran", palette=self.palette)[0], "accent")

    def test_soft_mute_is_white(self):
        # Common verbs that we don't want shouting in neon.
        self.assertEqual(classify_word("said", palette=self.palette)[0], "white")
        self.assertEqual(classify_word("went", palette=self.palette)[0], "white")

    def test_punctuation_trimmed(self):
        # A trailing comma must not flip the classification.
        self.assertEqual(classify_word("city,", palette=self.palette)[0], "accent")


class TestDriftFilter(unittest.TestCase):
    def test_drift_too_high_returns_empty(self):
        words = [_w((0.0, 0.5, "a")), _w((0.6, 1.1, "b"))]
        # span 1.1 s, audio 3.0 s -> ~63% drift -> drop word captions.
        self.assertEqual(filter_usable_words(words, audio_duration=3.0), [])

    def test_drift_within_tolerance_kept(self):
        words = [_w((0.0, 0.5, "a")), _w((0.6, 1.0, "b"))]
        # span 1.0 s, audio 1.1 s -> ~9% drift -> keep.
        result = filter_usable_words(words, audio_duration=1.1)
        self.assertEqual(len(result), 2)

    def test_empty_returns_empty(self):
        self.assertEqual(filter_usable_words([], audio_duration=1.0), [])


class TestGapDetection(unittest.TestCase):
    """The 30 s hole: whisper skipped from 27.44 s to 33.92 s and nothing was captioned."""

    def test_four_second_skip_is_detected(self):
        words = [
            _w((26.88, 27.00, "lock.")),
            _w((27.44, 29.82, "inside,")),
            _w((33.92, 34.44, "a")),
        ]
        self.assertEqual(find_gaps(words), [(29.82, 33.92)])

    def test_normal_beat_gaps_are_not_reported(self):
        # 0.35 s breathing gaps are below the 1.2 s threshold.
        words = [_w((0.0, 1.0, "a")), _w((1.35, 2.0, "b")), _w((2.35, 3.0, "c"))]
        self.assertEqual(find_gaps(words), [])

    def test_single_word_has_no_gaps(self):
        self.assertEqual(find_gaps([_w((0.0, 1.0, "a"))]), [])

    def test_empty_has_no_gaps(self):
        self.assertEqual(find_gaps([]), [])


class TestHallucinationTokens(unittest.TestCase):
    """The tail of the Inari short rendered `8 8 8` over the apad silence."""

    def test_bare_digits_are_hallucinations(self):
        for tok in ("8", "8.", "7", "12", "3,000"):
            self.assertTrue(is_hallucination_token(tok), tok)

    def test_symbols_are_hallucinations(self):
        for tok in ("...", "-", "--", "♪", "?"):
            self.assertTrue(is_hallucination_token(tok), tok)

    def test_real_words_are_kept(self):
        for tok in ("stayed.", "kotatsu", "Inari", "a", "I", "seven"):
            self.assertFalse(is_hallucination_token(tok), tok)

    def test_number_words_are_kept(self):
        # "seven years" is real narration; only bare digits are invented.
        self.assertFalse(is_hallucination_token("seven"))


class TestTrailingHallucinationTrim(unittest.TestCase):
    """The end of the Inari short rendered `8 8 8`, then `immediate aid.`, after `stayed.`"""

    SCRIPT = "At dawn he knelt in the overgrown garden, pressing each seed into dark soil, the house exhaling behind him as he stayed."

    def test_digit_tail_is_trimmed(self):
        words = [
            _w((54.42, 54.62, "as")),
            _w((54.70, 54.92, "stayed.")),
            _w((54.92, 55.04, "8.")),
            _w((55.50, 55.68, "7")),
            _w((55.68, 55.74, "8.")),
        ]
        kept = _trim_trailing_hallucination(words, self.SCRIPT)
        self.assertEqual([w.text for w in kept], ["as", "stayed."])

    def test_word_tail_is_trimmed(self):
        # The shape whisper produced with condition_on_previous_text=False.
        words = [
            _w((54.70, 54.92, "stayed.")),
            _w((54.92, 55.04, "immediate")),
            _w((55.04, 55.12, "aid.")),
        ]
        kept = _trim_trailing_hallucination(words, self.SCRIPT)
        self.assertEqual([w.text for w in kept], ["stayed."])

    def test_clean_transcript_is_untouched(self):
        words = [_w((0.0, 0.4, "at")), _w((0.4, 0.8, "dawn")), _w((0.8, 1.2, "he"))]
        self.assertEqual(_trim_trailing_hallucination(words, self.SCRIPT), words)

    def test_mid_transcript_mismatch_is_kept(self):
        # "Inari" often comes back as "ARRI"; it sits at a real timestamp, so keep it.
        words = [_w((0.0, 0.4, "ARRI")), _w((0.4, 0.8, "station,")), _w((0.8, 1.2, "stayed."))]
        kept = _trim_trailing_hallucination(words, self.SCRIPT)
        self.assertEqual(len(kept), 3)

    def test_absurd_trim_is_refused(self):
        # If the whole transcript mismatches, do not delete the captions.
        words = [_w((i * 0.5, i * 0.5 + 0.4, f"zzz{i}")) for i in range(10)]
        self.assertEqual(len(_trim_trailing_hallucination(words, self.SCRIPT)), 10)

    def test_empty_script_is_a_no_op(self):
        words = [_w((0.0, 0.4, "stayed."))]
        self.assertEqual(_trim_trailing_hallucination(words, ""), words)


class TestGroupCueText(unittest.TestCase):
    def test_text_joins_with_space(self):
        g = GroupCue(start=0.0, end=0.5, words=(_w((0.0, 0.2, "she")), _w((0.2, 0.5, "ran"))))
        self.assertEqual(g.text, "she ran")


class TestLineLevelFallback(unittest.TestCase):
    """A short must never publish without subtitles just because alignment is unavailable.

    The failure that shipped the caption-less video: faster-whisper was missing from the Python
    that rendered it, so `align_words` returned [], `build_caption_overlays` returned [], and the
    orchestrator rendered with `overlays=[]`. Nothing failed, nothing warned at ERROR level.
    """

    SEGMENTS = [
        ("Rain hammers the sleeping city and the sky glows tonight.", 0.0, 6.8),
        ("She pulls her hood up and lifts the lantern against the wind.", 6.8, 13.6),
    ]

    def test_one_cue_per_beat_with_the_beat_timing(self):
        groups = build_line_level_cues(
            " ".join(s[0] for s in self.SEGMENTS), segments=self.SEGMENTS, audio_duration=13.6)
        self.assertEqual(len(groups), 2)
        self.assertEqual((groups[0].start, groups[0].end), (0.0, 6.8))
        self.assertEqual((groups[1].start, groups[1].end), (6.8, 13.6))
        self.assertTrue(groups[0].text.startswith("Rain hammers"))

    def test_cues_are_clamped_to_the_track(self):
        groups = build_line_level_cues(
            "one beat only", segments=[("one beat only", 10.0, 99.0)], audio_duration=12.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].end, 12.0)

    def test_a_beat_starting_past_the_track_is_dropped(self):
        groups = build_line_level_cues(
            "late", segments=[("late", 20.0, 26.0)], audio_duration=12.0)
        self.assertEqual(groups, [])

    def test_without_beat_timing_the_text_is_spread_over_the_track(self):
        groups = build_line_level_cues("First line here. Second line here.", audio_duration=10.0)
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[0].start, 0.0)
        self.assertAlmostEqual(groups[0].end, 5.0)
        self.assertAlmostEqual(groups[1].end, 10.0)

    def test_no_timing_and_no_text_produces_nothing(self):
        self.assertEqual(build_line_level_cues("", audio_duration=10.0), [])
        self.assertEqual(build_line_level_cues("Some narration.", audio_duration=0.0), [])

    def test_alignment_failure_falls_back_instead_of_returning_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("captions.align_words", return_value=[]), \
                 patch("captions.render_group_pngs",
                       return_value=[(out_dir / "cap_00.png", 0.0, 6.8)]) as render:
                overlays = build_caption_overlays(
                    Path("narration.wav"),
                    " ".join(s[0] for s in self.SEGMENTS),
                    audio_duration=13.6,
                    out_dir=out_dir,
                    segments=self.SEGMENTS,
                )
        self.assertEqual(len(overlays), 1)
        render.assert_called_once()
        rendered_groups = render.call_args[0][0]
        self.assertEqual(len(rendered_groups), 2)

    def test_missing_faster_whisper_logs_a_warning_not_info(self):
        """INFO here is what made the broken run look healthy in the log."""
        from captions import align_words
        with patch.dict("sys.modules", {"faster_whisper": None}), \
             self.assertLogs("captions", level="WARNING") as logs:
            self.assertEqual(align_words(Path("nope.wav"), "text"), [])
        self.assertTrue(any("line-level" in line for line in logs.output))

    def test_a_video_with_narration_never_gets_zero_overlays(self):
        """End-to-end guard on the regression itself: empty cues must not mean empty captions."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("captions.align_words", return_value=[]), \
                 patch("captions._resolve_font", return_value=("", "none")), \
                 patch("captions.render_group_pngs", return_value=[]):
                # No font available is a genuine render failure and may stay empty.
                self.assertEqual(build_caption_overlays(
                    Path("n.wav"), "A line of narration.", audio_duration=5.0,
                    out_dir=out_dir), [])
            # With a font, the same missing alignment still yields overlays.
            with patch("captions.align_words", return_value=[]), \
                 patch("captions.render_group_pngs",
                       return_value=[(out_dir / "cap_00.png", 0.0, 5.0)]):
                overlays = build_caption_overlays(
                    Path("n.wav"), "A line of narration.", audio_duration=5.0, out_dir=out_dir)
            self.assertEqual(len(overlays), 1)


if __name__ == "__main__":
    unittest.main()
