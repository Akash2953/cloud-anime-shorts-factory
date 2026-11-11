"""Queue tests: which part a run picks, and when the series is allowed to move on.

The failure these lock down: a new plan used to be replaced by the shot *count* before it was
ever used, a template plan was written into the series, and `next_part` never advanced because
the state file was only written after an early `return`.
"""

import unittest
from unittest.mock import patch

import numpy as np

from orchestrator import advance_series, beat_track, resolve_part
from scenes import Shot, _template_shots

CFG = {
    "CFAI_TOKEN": "token", "CF_ACCOUNT_ID": "account", "OPENROUTER_API_KEY": "key",
    "SHOTS_PER_PART": 0, "PREFERRED_SECONDS": 55.0, "MIN_SECONDS": 45.0, "MAX_SECONDS": 75.0,
    "IMAGE_STRATEGY": "edit", "SERIES_PARTS": 1,
}


def plan(n: int = 13) -> list[Shot]:
    return _template_shots(n)


def series_state(parts: int, next_part: int) -> dict:
    return {"processed": [], "daily": {}, "series": {
        "id": "2026-01-01", "theme": "t",
        "parts": [[s.as_dict() for s in plan(3)] for _ in range(parts)],
        "next_part": next_part, "image_prompts": {},
    }}


class TestBeatTrack(unittest.TestCase):
    """Silent beats hold the frame; every still is held long enough for its own line."""

    def beat(self, narration: str, seconds: float = 3.0) -> Shot:
        return Shot(narration=narration, shot_size="wide", place="a rooftop",
                    action="she waits", story_object="no object",
                    narrative_role="build_tension", seconds=seconds)

    def voice(self, seconds: float):
        return np.zeros(int(seconds * 24000), dtype=np.float32)

    def test_silent_beat_keeps_its_planned_hold(self):
        shots = [self.beat("The city glows."), self.beat("none", 4.0)]
        beats, _, holds = beat_track(shots, [self.voice(2.0)])
        self.assertEqual(len(beats[1]) / 24000, 4.0)
        self.assertAlmostEqual(holds[0], 2.0 + 0.35)
        self.assertAlmostEqual(holds[1], 4.0 + 0.6)

    def test_the_cut_is_exactly_as_long_as_the_track(self):
        shots = [self.beat("The city glows."), self.beat("none", 2.0),
                 self.beat("She lifts it.")]
        _, audio, holds = beat_track(shots, [self.voice(2.0), self.voice(2.5)])
        self.assertAlmostEqual(len(audio) / 24000, sum(holds), places=3)

    def test_a_fully_silent_short_still_renders(self):
        shots = [self.beat("silent", 2.0), self.beat("", 3.0)]
        beats, audio, holds = beat_track(shots, [])
        self.assertEqual([round(len(b) / 24000, 2) for b in beats], [2.0, 3.0])
        self.assertAlmostEqual(len(audio) / 24000, 2.0 + 3.0 + 0.35 + 0.6, places=3)
        self.assertEqual(len(holds), 2)

    def test_a_rewritten_voice_track_maps_back_onto_the_right_beats(self):
        shots = [self.beat("none", 1.0), self.beat("She lifts it.")]
        beats, _, _ = beat_track(shots, [self.voice(2.0)])
        self.assertEqual(len(beats[0]) / 24000, 1.0)   # silence, not the voice
        self.assertEqual(len(beats[1]) / 24000, 2.0)   # the voice lands on the spoken beat

    def test_a_zero_length_beat_still_gets_a_frame(self):
        _, _, holds = beat_track([self.beat("none", 0.0)], [])
        self.assertGreaterEqual(holds[0], 0.8)


class TestSeriesParts(unittest.TestCase):
    """`SERIES_PARTS` must reach the planner, and a middle part must not resolve."""

    def test_total_parts_from_config_reaches_the_planner(self):
        cfg = dict(CFG, SERIES_PARTS=3)
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            shots, part, total, out = resolve_part(cfg, state)
        self.assertEqual((part, total), (1, 3))
        self.assertEqual(planner.call_args.kwargs["total_parts"], 3)
        self.assertEqual(planner.call_args.kwargs["part"], 1)
        self.assertEqual(out["series"]["total_parts"], 3)

    def test_the_next_part_is_appended_and_told_the_story_so_far(self):
        state = series_state(parts=1, next_part=1)
        state["series"]["total_parts"] = 3
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            shots, part, total, out = resolve_part(CFG, state)
        self.assertEqual((part, total), (2, 3))
        self.assertEqual(len(out["series"]["parts"]), 2)   # part 1 kept, part 2 appended
        self.assertEqual(planner.call_args.kwargs["part"], 2)
        self.assertEqual(len(planner.call_args.kwargs["story_so_far"]), 3)  # part 1's 3 lines

    def test_a_finished_arc_starts_a_new_story(self):
        state = series_state(parts=3, next_part=3)
        state["series"]["total_parts"] = 3
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            shots, part, total, out = resolve_part(CFG, state)
        self.assertEqual((part, total), (1, 1))
        self.assertEqual(len(out["series"]["parts"]), 1)   # fresh arc, not a 4th part
        self.assertEqual(planner.call_args.kwargs["part"], 1)
        self.assertEqual(planner.call_args.kwargs["story_so_far"], [])

    def test_an_existing_part_pins_the_arc_length(self):
        state = series_state(parts=1, next_part=0)          # hand-built series, no total_parts
        with patch("orchestrator.plan_shorts_detailed") as planner:
            shots, part, total, out = resolve_part(dict(CFG, SERIES_PARTS=3), state)
        planner.assert_not_called()                          # planned shots reused, not redone
        self.assertEqual((part, total), (1, 3))
        self.assertEqual(out["series"]["total_parts"], 3)   # pinned for the rest of the arc

    def test_a_middle_part_must_not_resolve(self):
        from scenes import validate_story_position
        resolved = plan(13)
        resolved[-1].narrative_role = "resolution"
        self.assertTrue(validate_story_position(resolved, part=2, total_parts=3))
        hooked = plan(13)
        hooked[-1].narrative_role = "build_tension"
        self.assertEqual(validate_story_position(hooked, part=2, total_parts=3), [])

    def test_a_final_part_must_resolve(self):
        from scenes import validate_story_position
        open_ended = plan(13)
        open_ended[-1].narrative_role = "build_tension"
        self.assertTrue(validate_story_position(open_ended, part=3, total_parts=3))

    def test_the_planner_is_told_the_position_and_the_hook_rule(self):
        from scenes import plan_shorts_detailed
        with patch("scenes._openrouter", return_value="x " * 60) as planner:
            plan_shorts_detailed(theme="t", api_key="k", want=13, part=2, total_parts=3,
                                 story_so_far=["Earlier line."])
        sent = planner.call_args[0][0]
        self.assertIn("part 2 of 3", sent[0]["content"])
        self.assertIn("never resolution", sent[0]["content"])
        self.assertIn("Story so far", sent[1]["content"])
        self.assertIn("Earlier line.", sent[1]["content"])

    def test_a_degraded_plan_is_not_allowed_to_continue_an_arc(self):
        state = series_state(parts=1, next_part=1)
        state["series"]["total_parts"] = 3
        with patch("orchestrator.plan_shorts_detailed", return_value=([], {}, True, 3)), \
             patch("orchestrator.plan_arc", return_value=([plan(13)], False)):
            shots, part, total, out = resolve_part(CFG, state)
        self.assertEqual((part, total), (2, 3))
        self.assertEqual(len(out["series"]["parts"]), 1)   # generic plan rendered, not stored

    def test_parts_are_clamped_to_seven(self):
        from scenes import plan_shorts_detailed
        with patch("scenes._openrouter", return_value="x " * 60) as planner:
            plan_shorts_detailed(theme="t", api_key="k", part=9, total_parts=9)
        self.assertIn("part 7 of 7", planner.call_args[0][0][0]["content"])


class TestAutoArcLength(unittest.TestCase):
    """`SERIES_PARTS=auto`: the planner picks the arc length once, the arc is then pinned."""

    def test_new_story_lets_the_planner_choose_and_pins_it(self):
        cfg = dict(CFG, SERIES_PARTS="auto", PARTS_MIN=1, PARTS_MAX=7)
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 5)) as planner:
            shots, part, total, out = resolve_part(cfg, state)
        self.assertIsNone(planner.call_args.kwargs["total_parts"])   # remained undecided
        self.assertEqual(planner.call_args.kwargs["parts_range"], (1, 7))
        self.assertEqual((part, total), (1, 5))
        self.assertEqual(out["series"]["total_parts"], 5)            # pinned for later days

    def test_a_pinned_arc_does_not_re_decide(self):
        state = series_state(parts=1, next_part=1)
        state["series"]["total_parts"] = 5
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            shots, part, total, out = resolve_part(dict(CFG, SERIES_PARTS="auto"), state)
        self.assertEqual(planner.call_args.kwargs["total_parts"], 5)  # known, so hook rules apply
        self.assertEqual((part, total), (2, 5))
        self.assertEqual(out["series"]["total_parts"], 5)

    def test_auto_can_still_pick_one_part(self):
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)):
            shots, part, total, out = resolve_part(dict(CFG, SERIES_PARTS="auto"), state)
        self.assertEqual((part, total), (1, 1))
        self.assertEqual(out["series"]["total_parts"], 1)

    def test_a_hand_built_part_is_pinned_as_a_single_part(self):
        state = series_state(parts=1, next_part=0)      # hand-made: no total_parts recorded
        with patch("orchestrator.plan_shorts_detailed") as planner:
            shots, part, total, out = resolve_part(dict(CFG, SERIES_PARTS="auto"), state)
        planner.assert_not_called()                     # its shots are reused as they are
        self.assertEqual((part, total), (1, 1))
        self.assertEqual(out["series"]["total_parts"], 1)

    def test_an_uploaded_part_with_no_recorded_length_starts_a_new_story(self):
        state = series_state(parts=1, next_part=1)      # part 1 already went out
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 4)) as planner:
            shots, part, total, out = resolve_part(dict(CFG, SERIES_PARTS="auto"), state)
        self.assertIsNone(planner.call_args.kwargs["total_parts"])
        self.assertEqual((part, total), (1, 4))
        self.assertEqual(len(out["series"]["parts"]), 1)

    def test_the_declared_arc_line_is_read_and_clamped(self):
        from scenes import _declared_arc
        self.assertEqual(_declared_arc("Arc: 5 parts\nShort 1: 13 shots", 1, 7), 5)
        self.assertEqual(_declared_arc("**Arc:** 3 parts", 1, 7), 3)
        self.assertEqual(_declared_arc("arc = 2 parts", 1, 7), 2)
        self.assertEqual(_declared_arc("Arc: 9 parts", 1, 7), 7)      # clamped to PARTS_MAX
        self.assertEqual(_declared_arc("no arc line here", 2, 7), 2)  # falls back to PARTS_MIN


class TestPartPreview(unittest.TestCase):
    """`--part N` plans a later part without pretending the earlier one was published."""

    def test_previewing_a_later_part_leaves_the_queue_alone(self):
        state = series_state(parts=1, next_part=0)          # part 1 still unpublished
        state["series"]["total_parts"] = 3
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 3)) as planner:
            shots, part, total, out = resolve_part(CFG, state, part=2)
        self.assertEqual(planner.call_args.kwargs["part"], 2)
        self.assertEqual(len(planner.call_args.kwargs["story_so_far"]), 3)  # part 1's lines
        self.assertEqual((part, total), (2, 3))
        self.assertEqual(len(out["series"]["parts"]), 2)   # part 2 appended
        self.assertEqual(out["series"]["next_part"], 0)    # part 1 still the one to publish

    def test_a_stored_part_is_reused_not_replanned(self):
        state = series_state(parts=2, next_part=0)
        state["series"]["total_parts"] = 3
        with patch("orchestrator.plan_shorts_detailed") as planner:
            shots, part, total, out = resolve_part(CFG, state, part=2)
        planner.assert_not_called()
        self.assertEqual((part, total), (2, 3))
        self.assertEqual(out["series"]["next_part"], 0)

    def test_a_gap_in_the_arc_is_an_error(self):
        state = series_state(parts=1, next_part=0)
        with self.assertRaises(RuntimeError):
            resolve_part(CFG, state, part=3)


class TestResolvePart(unittest.TestCase):
    def test_pending_part_is_reused_without_asking_a_provider(self):
        state = series_state(parts=1, next_part=0)
        with patch("orchestrator.plan_shorts_detailed") as planner:
            shots, part, total, _ = resolve_part(CFG, state)
        planner.assert_not_called()
        self.assertEqual((part, total, len(shots)), (1, 1, 3))

    def test_new_plan_is_used_and_part_one_stays_pending(self):
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {1: "frame one"}, False, 1)) as planner:
            shots, part, total, out = resolve_part(CFG, state)
        self.assertEqual(len(shots), 13)          # the plan, not its length
        self.assertEqual((part, total), (1, 1))
        self.assertEqual(planner.call_args.kwargs["want"], 8)   # 55s / 6.8s per cut
        self.assertEqual(out["series"]["next_part"], 0)
        self.assertEqual(len(out["series"]["parts"]), 1)
        self.assertEqual(out["series"]["total_parts"], 1)
        # prompts ride on each shot (image_generation_prompt), never in a part-numbered dict
        self.assertNotIn("image_prompts", out["series"])

    def test_template_plan_is_rendered_but_never_persisted(self):
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed", return_value=([], {}, True, 1)), \
             patch("orchestrator.plan_arc", return_value=([plan(13)], True)):
            shots, part, total, out = resolve_part(CFG, state)
        self.assertEqual(len(shots), 13)
        self.assertEqual((part, total), (1, 1))
        self.assertNotIn("series", out)

    def test_shot_count_follows_the_pacing_window(self):
        state = {"processed": [], "daily": {}}
        cfg = dict(CFG, PREFERRED_SECONDS=0.0)  # falls back to the MIN/MAX midpoint
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(9), {}, False, 1)) as planner:
            resolve_part(cfg, state)
        self.assertEqual(planner.call_args.kwargs["want"], 9)   # (45+75)//2 = 60 / 6.8


class TestAdvanceSeries(unittest.TestCase):
    def test_a_successful_upload_advances_the_queue(self):
        state = series_state(parts=1, next_part=0)
        advance_series(state, 1)
        self.assertEqual(state["series"]["next_part"], 1)

    def test_the_same_part_cannot_go_backwards(self):
        state = series_state(parts=4, next_part=2)
        advance_series(state, 1)
        self.assertEqual(state["series"]["next_part"], 2)

    def test_a_failed_run_leaves_the_queue_where_it_was(self):
        state = series_state(parts=1, next_part=0)
        self.assertEqual(state["series"]["next_part"], 0)

    def test_no_series_means_nothing_to_advance(self):
        state = {"processed": [], "daily": {}}
        advance_series(state, 1)
        self.assertNotIn("series", state)


if __name__ == "__main__":
    unittest.main()
