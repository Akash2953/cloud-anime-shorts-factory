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

    def test_input_indices_are_configurable(self):
        # render_narration puts images first, so the voice is rarely input 0.
        g = mix_graph(55.0, voice_index=8, music_index=9)
        self.assertIn("[9:a]", g)
        self.assertIn("[8:a]", g)
        self.assertNotIn("[0:a]", g)

    def test_amix_does_not_normalise_the_voice_down(self):
        # amix's default divides by the input count, which would halve the narration.
        self.assertIn("normalize=0", mix_graph(55.0))

    def test_the_voice_is_padded_to_the_music_length(self):
        self.assertIn("[0:a]apad", mix_graph(55.0))

    def test_fade_out_never_starts_before_the_track(self):
        # A 1 s bed with a 2.5 s fade would otherwise open already fading out.
        g = mix_graph(1.0)
        self.assertIn("afade=t=out:st=0.667", g)


class TestResolveMusic(unittest.TestCase):
    def test_falls_back_to_a_bed_when_no_tracks_exist(self):
        import tempfile
        from music import resolve_music
        with tempfile.TemporaryDirectory() as tmp:
            with patch("music.MUSIC_DIR", Path(tmp) / "missing"):
                bed = Path(tmp) / "bed.wav"
                path, is_fallback = resolve_music(1, 2.0, bed_path=bed)
            self.assertTrue(is_fallback)
            self.assertEqual(path, bed)
            self.assertTrue(bed.exists())

    def test_uses_the_licensed_track_when_present(self):
        import tempfile
        from music import resolve_music
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "track.wav").write_bytes(b"")
            with patch("music.MUSIC_DIR", d):
                path, is_fallback = resolve_music(1, 2.0, bed_path=d / "bed.wav")
            self.assertFalse(is_fallback)
            self.assertEqual(path.name, "track.wav")
            self.assertFalse((d / "bed.wav").exists())


class TestCredit(unittest.TestCase):
    def test_sidecar_produces_an_attribution_line(self):
        import json, tempfile
        from music import credit_for
        with tempfile.TemporaryDirectory() as tmp:
            track = Path(tmp) / "bed.mp3"
            track.write_bytes(b"")
            (Path(tmp) / "bed.mp3.json").write_text(json.dumps({
                "title": "Rain on Metal", "artist": "Someone",
                "license": "CC-BY 4.0", "url": "https://example.test/x",
            }), encoding="utf-8")
            line = credit_for(track)
        self.assertIn("Rain on Metal", line)
        self.assertIn("CC-BY 4.0", line)

    def test_no_sidecar_means_no_credit(self):
        import tempfile
        from music import credit_for
        with tempfile.TemporaryDirectory() as tmp:
            track = Path(tmp) / "bed.mp3"
            track.write_bytes(b"")
            self.assertIsNone(credit_for(track))


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
