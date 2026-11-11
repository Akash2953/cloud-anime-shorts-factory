"""Option 2 rotation: one track per part, deterministic across retries."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music import enumerate_music, mix_graph, pick_music, synthesize_ambient_bed


class TestRotation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="music_tests_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed(self, files: list[str]) -> Path:
        if list(self.tmp.iterdir()):
            shutil.rmtree(self.tmp)
            self.tmp.mkdir()
        for name in files:
            (self.tmp / name).write_bytes(b"")
        return self.tmp

    def test_part_1_picks_the_first_track(self):
        d = self._seed(["a.wav", "b.mp3", "c.m4a"])
        with patch("music.MUSIC_DIR", d):
            self.assertEqual(pick_music(1).name, "a.wav")

    def test_sequence_rotates_randomly_across_the_arc(self):
        d = self._seed(["a.wav", "b.wav", "c.wav"])
        with patch("music.MUSIC_DIR", d):
            seen = [pick_music(n).name for n in range(1, 8)]
        self.assertEqual(seen, ["a.wav", "b.wav", "c.wav", "a.wav", "b.wav", "c.wav", "a.wav"])

    def test_an_empty_folder_returns_none(self):
        d = self._seed([])
        with patch("music.MUSIC_DIR", d):
            self.assertIsNone(pick_music(1))
            self.assertEqual(enumerate_music(), [])

    def test_unknown_extensions_are_skipped(self):
        d = self._seed(["track.wav", "readme.txt", "cover.jpg"])
        with patch("music.MUSIC_DIR", d):
            self.assertEqual([p.name for p in enumerate_music()], ["track.wav"])

    def test_a_stable_arc_picks_the_same_track_each_retry(self):
        d = self._seed(["x.wav", "y.wav"])
        with patch("music.MUSIC_DIR", d):
            a = pick_music(2).name
            b = pick_music(2).name
        self.assertEqual(a, b)


class TestMixGraph(unittest.TestCase):
    def test_the_graph_template_has_key_labels(self):
        g = mix_graph(55.0)
        self.assertIn("[1:a]", g)
        self.assertIn("[0:a]", g)              # voice comes first in the input list
        self.assertIn("[aout]", g)             # render maps this label
        self.assertIn("sidechaincompress", g)  # the duck
        self.assertIn("loudnorm=I=-14", g)     # YouTube target loudness

    def test_fades_align_with_the_duration(self):
        g = mix_graph(50.0, fade_out=2.0)
        self.assertIn("afade=t=out:st=48.000:d=2.00", g)

    def test_ducking_thresholds_round_trip(self):
        g = mix_graph(40.0, threshold_pct=0.05, ratio=4.0,
                     attack_ms=200, release_ms=300)
        self.assertIn("threshold=0.05:ratio=4", g)
        self.assertIn("attack=200:release=300", g)


class TestAmbientBed(unittest.TestCase):
    def test_the_synth_pads_for_the_full_duration(self):
        import numpy as np
        from audio import SAMPLE_RATE
        bed = synthesize_ambient_bed(5.0)
        self.assertEqual(bed.dtype, np.float32)
        self.assertEqual(bed.shape, (5 * SAMPLE_RATE,))
        # peaks should stay well below clipping even with tanh clamp
        self.assertLess(float(np.max(np.abs(bed))), 0.95)

    def test_two_chains_of_the_same_length_are_identical(self):
        a = synthesize_ambient_bed(2.0)
        b = synthesize_ambient_bed(2.0)
        import numpy as np
        self.assertTrue(np.allclose(a, b))   # seed-driven, so two runs match


if __name__ == "__main__":
    unittest.main()
