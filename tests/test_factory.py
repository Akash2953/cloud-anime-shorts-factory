"""State fingerprinting, narration joining, and duplicate-prevention contracts."""

import unittest
from state import compute_fingerprint, parse_state, save_state


class TestNarrationJoin(unittest.TestCase):
    def test_gaps_and_cue_times(self):
        import numpy as np
        from audio import SAMPLE_RATE, join_with_gaps

        segs = [np.ones(SAMPLE_RATE, dtype=np.float32),      # 1.0s
                np.ones(SAMPLE_RATE * 2, dtype=np.float32)]  # 2.0s
        audio, starts = join_with_gaps(segs, gap_seconds=0.5)
        self.assertEqual(starts, [0.0, 1.5])
        self.assertAlmostEqual(len(audio) / SAMPLE_RATE, 3.5, places=3)


class TestShortsFactory(unittest.TestCase):
    def test_fingerprint(self):
        fp1 = compute_fingerprint("hello", ["url1", "url2"])
        fp2 = compute_fingerprint("hello", ["url1", "url2"])
        fp3 = compute_fingerprint("world", ["url1", "url2"])
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)
        self.assertTrue(fp1.startswith("hash32:"))

    def test_state_persistence(self):
        st = {"processed": ["hash32:123"], "daily": {"2026-11-07": {"uploads": 1}}}
        raw = save_state(st)
        parsed = parse_state(raw)
        self.assertEqual(parsed, st)


if __name__ == "__main__":
    unittest.main()
