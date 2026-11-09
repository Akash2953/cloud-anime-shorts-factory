"""Unit tests for Cloud Anime Shorts Factory router, fingerprinting, and story fallback."""

import unittest
from state import compute_fingerprint, parse_state, save_state
from story import generate_story, _split_beats, _template_beats


class TestBeats(unittest.TestCase):
    def test_split_strips_numbering_and_bullets(self):
        raw = "1. Rain falls on the neon street tonight.\n2) A girl walks alone.\n- Lanterns drift above her."
        beats = _split_beats(raw, 3)
        self.assertEqual(len(beats), 3)
        self.assertEqual(beats[0], "Rain falls on the neon street tonight.")
        self.assertFalse(beats[1].startswith("2"))

    def test_split_breaks_packed_sentences(self):
        raw = ("1. She runs through the flooded street tonight. She hides behind an old brick wall. "
               "She waits for the morning light.\n2. The city sleeps beneath the endless rain.")
        beats = _split_beats(raw, 6)
        self.assertEqual(len(beats), 4)
        self.assertTrue(all(b.endswith(".") for b in beats))

    def test_split_drops_truncated_final_beat(self):
        raw = "1. One fine complete sentence here now.\n2. Another complete sentence here too.\n3. Third one is fine.\n4. And then the lantern"
        beats = _split_beats(raw, 4)
        self.assertEqual(len(beats), 3)
        self.assertTrue(beats[-1].endswith("."))

    def test_split_respects_count(self):
        raw = "\n".join(f"{i}. Sentence number {i} is here." for i in range(1, 10))
        self.assertEqual(len(_split_beats(raw, 6)), 6)

    def test_template_beats_returns_requested_count(self):
        self.assertEqual(len(_template_beats("neon rain", 6)), 6)
        self.assertEqual(len(_template_beats("neon rain", 3)), 3)

    def test_leaked_chain_of_thought_is_never_narrated(self):
        # Observed verbatim from openrouter/free: reasoning text dressed up as beats.
        leaked = (
            "**Analyze the Request:**\n"
            "* **Topic:** Cyberpunk anime, lanterns.\n"
            "* **Format:** Exactly 6 numbered lines (1. to 6.), nothing else.\n"
            "1. Rain hammers the neon rooftops while a hooded girl waits.\n"
            "2. A lantern flickers once and dies against the storm.\n"
            "3. She follows the dark between the puddles without a sound."
        )
        beats = _split_beats(leaked, 6)
        self.assertEqual(len(beats), 3)
        self.assertFalse(any("**" in b for b in beats))
        self.assertFalse(any("Analyze" in b or "Format" in b or "Topic" in b for b in beats))

    def test_markdown_and_unterminated_lines_rejected(self):
        raw = (
            "## Heading that should not be spoken aloud here.\n"
            "A real sentence of narration lands cleanly in the output.\n"
            "Word count: seventeen words is clearly meta information here.\n"
            "Trailing sentence without any terminal punctuation at all"
        )
        beats = _split_beats(raw, 5)
        self.assertEqual(beats, ["A real sentence of narration lands cleanly in the output."])


class TestNarrationJoin(unittest.TestCase):
    def test_gaps_and_cue_times(self):
        import numpy as np
        from audio import SAMPLE_RATE, join_with_gaps

        segs = [np.ones(SAMPLE_RATE, dtype=np.float32),      # 1.0s
                np.ones(SAMPLE_RATE * 2, dtype=np.float32)]  # 2.0s
        audio, starts = join_with_gaps(segs, gap_seconds=0.5)
        self.assertEqual(starts, [0.0, 1.5])
        self.assertAlmostEqual(len(audio) / SAMPLE_RATE, 3.5, places=3)


class TestOpenRouterFallback(unittest.TestCase):
    def test_skips_empty_and_rate_limited_models(self):
        """openrouter/free can return content=None from a reasoning model; must move on."""
        from unittest.mock import MagicMock, patch

        import story

        responses = [
            {"choices": [{"finish_reason": "length", "message": {"content": None}}]},
            None,
            {"choices": [{"finish_reason": "stop",
                           "message": {"content": "1. Rain falls on the neon street tonight.\n"
                                                   "2. A hooded girl walks alone through the storm."}}]},
        ]

        def fake_post(url, json=None, headers=None, timeout=None):
            resp = MagicMock()
            item = responses.pop(0)
            if item is None:
                resp.status_code, resp.text = 429, "rate limited"
            else:
                resp.status_code, resp.text = 200, "ok"
                resp.json.return_value = item
            return resp

        with patch("story.requests.post", fake_post):
            ok, text = story._call_openrouter_messages([{"role": "user", "content": "x"}], "key")
        self.assertTrue(ok)
        self.assertIn("Rain falls on the neon street", text)

    def test_all_models_failing_returns_false(self):
        from unittest.mock import MagicMock, patch

        import story

        def fake_post(url, json=None, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code, resp.text = 429, "rate limited"
            return resp

        with patch("story.requests.post", fake_post):
            ok, text = story._call_openrouter_messages([{"role": "user", "content": "x"}], "key")
        self.assertFalse(ok)
        self.assertIsNone(text)


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

    def test_story_fallback(self):
        story, used_fallback = generate_story(theme="magic academy", api_key="", attempts=0)
        self.assertTrue(used_fallback)
        self.assertTrue(len(story) > 0)


if __name__ == "__main__":
    unittest.main()
