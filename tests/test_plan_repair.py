"""Any missing detail is re-asked for, not silently filled in by the parser."""

import unittest
from unittest.mock import patch

from scenes import (PLAN_REPAIR_ROUNDS, _plan_gaps, _ranges, parse_detailed_response,
                    plan_shorts_detailed, request_title)

from tests.test_scenes import detailed_reply

NL = chr(10)


class TestIncompletePlanRepair(unittest.TestCase):
    def plan(self, replies, **kwargs):
        with patch("scenes.request_title", return_value=""), \
             patch("scenes._openrouter", side_effect=list(replies)) as planner:
            result = plan_shorts_detailed(theme="t", api_key="k", **kwargs)
        return result, planner

    def test_a_complete_plan_costs_no_repair(self):
        (_shots, _prompts, _fallback, _arc, _title), planner = self.plan(
            [detailed_reply(8)], want=8)
        self.assertEqual(planner.call_count, 1)

    def test_a_missing_field_is_named_in_the_repair_request(self):
        broken = detailed_reply(8).replace(f"- Action: something happens in 3{NL}", "")
        (shots, _prompts, fallback, _arc, _title), planner = self.plan(
            [broken, detailed_reply(8)], want=8)
        self.assertEqual(planner.call_count, 2)
        instruction = planner.call_args[0][0][-1]["content"]
        self.assertIn("`Action` field", instruction)
        self.assertIn("shot(s) 3", instruction)
        self.assertEqual(len(shots), 8)
        self.assertFalse(fallback)

    def test_a_missing_hook_line_is_repaired(self):
        no_hook = detailed_reply(8).replace(
            'Hook: "She crosses the flooded walkway and counts the lanterns above her."', "")
        (_shots, _prompts, _fallback, _arc, _title), planner = self.plan(
            [no_hook, detailed_reply(8)], want=8)
        self.assertEqual(planner.call_count, 2)
        self.assertIn("`Hook:` line is missing", planner.call_args[0][0][-1]["content"])

    def test_a_hook_the_opening_shot_ignores_is_repaired(self):
        # The shipped failure: the planner stated a hook, then opened the short on weather.
        scenery = detailed_reply(8).replace(
            '- Narration: "She crosses', '- Narration: "The mist clung to the roofs and she crosses', 1)
        (_shots, _prompts, fallback, _arc, _title), planner = self.plan(
            [scenery, detailed_reply(8)], want=8)
        self.assertEqual(planner.call_count, 2)
        self.assertIn("does not open with the Hook", planner.call_args[0][0][-1]["content"])
        self.assertFalse(fallback)

    def test_an_over_long_script_is_repaired_against_the_word_ceiling(self):
        (_shots, _prompts, _fallback, _arc, _title), planner = self.plan(
            [detailed_reply(8), detailed_reply(8)], want=8, max_words=120)
        self.assertEqual(planner.call_count, 2)
        self.assertIn("carries at most 120", planner.call_args[0][0][-1]["content"])

    def test_a_missing_image_prompt_is_named_in_the_repair_request(self):
        broken = detailed_reply(8).replace(f"Image 5: scene 5 in the rain.{NL}", "")
        (_shots, _prompts, _fallback, _arc, _title), planner = self.plan(
            [broken, detailed_reply(8)], want=8)
        self.assertIn("Image 5", planner.call_args[0][0][-1]["content"])

    def test_an_undeclared_arc_is_repaired_when_the_length_is_undecided(self):
        broken = detailed_reply(8).replace(f"Arc: 1 part{NL}", "")
        (_shots, _prompts, _fallback, _arc, _title), planner = self.plan(
            [broken, detailed_reply(8)], want=8, part=1, total_parts=None)
        self.assertIn("Arc: N parts", planner.call_args[0][0][-1]["content"])

    def test_repair_rounds_are_bounded_and_a_useless_fix_is_discarded(self):
        broken = detailed_reply(8).replace(f"- Action: something happens in 3{NL}", "")
        (shots, _prompts, _fallback, _arc, _title), planner = self.plan(
            [broken, broken, broken], want=8)
        # the first plan is kept once a repair stops reducing the gap count
        self.assertEqual(planner.call_count, 2)
        self.assertLessEqual(planner.call_count, 1 + PLAN_REPAIR_ROUNDS)
        self.assertEqual(len(shots), 8)

    def test_dropped_shot_blocks_are_reported(self):
        reply = detailed_reply(3).replace('- Narration: "She crosses', '- Note: "She crosses')
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=3)
        self.assertTrue(any("dropped" in gap or "Narration" in gap for gap in gaps), gaps)

    def test_gap_ranges_are_readable(self):
        self.assertEqual(_ranges([1, 2, 3, 5, 9]), "1-3, 5, 9")
        self.assertEqual(_ranges([4]), "4")


class TestPlanGaps(unittest.TestCase):
    def test_gap_detection_names_every_missing_field(self):
        reply = detailed_reply(4, complete=False)
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=4, need_arc=True)
        joined = " | ".join(gaps)
        for field in ("Place", "Story object", "Narrative role"):
            self.assertIn(field, joined)
        self.assertTrue(any("shot(s) 1-4" in gap for gap in gaps), gaps)

    def test_a_complete_plan_has_no_gaps(self):
        reply = "Arc: 2 parts" + NL + detailed_reply(4)
        shots, prompts = parse_detailed_response(reply)
        self.assertEqual(_plan_gaps(reply, shots, prompts, want=4, need_arc=True), [])

    def test_narration_density_is_a_gap(self):
        reply = detailed_reply(4).replace(
            "She crosses the flooded walkway and counts the lanterns above her again, because "
            "the last time she stood here there were twelve of them and now there are only "
            "eleven left.", "Gold eyes. Targeting.")
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=4)
        self.assertTrue(any("captions" in gap for gap in gaps), gaps)

    def test_title_request_is_separate_and_cheap(self):
        """The title is one short follow-up, not a full storyboard re-request."""
        with patch("scenes._plan_request", return_value="The Lantern") as planner:
            self.assertEqual(request_title(_template_shots(3), "k"), "The Lantern")
        self.assertIn("Reply with the title only", planner.call_args[0][0][-1]["content"])

    def test_a_short_title_reply_passes_the_word_floor(self):
        """Regression: the 40-word storyboard floor rejected every title reply as unusable."""
        from unittest.mock import MagicMock

        import scenes

        body = {"choices": [{"message": {"content": "The Iron Lantern"},
                             "finish_reason": "stop"}], "usage": {}}
        messages = [{"role": "user", "content": "name this story"}]
        with patch("scenes.requests.post",
                   return_value=MagicMock(status_code=200, json=lambda: body)):
            self.assertIsNone(scenes._openrouter(messages, "k"))              # storyboard floor
            self.assertEqual(scenes._openrouter(messages, "k", min_words=1),
                             "The Iron Lantern")                             # title floor


class TestTitleFromPlan(unittest.TestCase):
    """When no title arrives at all, the plan's own `Story object` names the story."""

    def shot(self, obj: str, role: str = "deliver_payload"):
        from scenes import Shot
        return Shot(narration="A line of prose long enough to count as narration.",
                    shot_size="wide", place="a street", action="she waits",
                    story_object=obj, narrative_role=role)

    def test_story_object_becomes_a_title(self):
        from scenes import title_from_plan
        self.assertEqual(title_from_plan([self.shot("fresh yellow ribbon among faded ones")]),
                         "The Fresh Yellow Ribbon")
        self.assertEqual(title_from_plan([self.shot("iron lantern hanging from eaves")]),
                         "The Iron Lantern")

    def test_payload_beat_is_preferred_over_an_establishing_shot(self):
        from scenes import title_from_plan
        shots = [self.shot("shrine roof", role="establish_context"),
                 self.shot("iron lantern", role="deliver_payload")]
        self.assertEqual(title_from_plan(shots), "The Iron Lantern")

    def test_no_object_gives_nothing(self):
        from scenes import title_from_plan
        self.assertEqual(title_from_plan([self.shot("no object")]), "")


class TestSplitComposition(unittest.TestCase):
    """A prompt that asks for two scenes in one frame renders a diptych with a seam."""

    SPLIT = ("Anime key visual. Dawn over a city from the shrine steps. "
             "Composition splits horizontal: tree and ribbons upper left, steps and boy lower right. "
             "Dew on the step edges.")

    def test_a_split_request_is_detected(self):
        from scenes import _asks_for_split
        self.assertTrue(_asks_for_split(self.SPLIT))
        self.assertTrue(_asks_for_split("Two panels side by side, left a tree, right a boy."))
        self.assertFalse(_asks_for_split("A girl waits on a rooftop while rain falls."))

    def test_a_corrective_clause_does_not_retrigger_detection(self):
        from scenes import _asks_for_split
        self.assertFalse(_asks_for_split("One continuous single frame, no split panels, no diptych."))

    def test_anchoring_rewrites_the_request_into_one_frame(self):
        from scenes import anchor_prompt
        out = anchor_prompt(self.SPLIT, character=False, character_sig="a boy in a coat")
        self.assertNotIn("splits horizontal", out.lower())
        self.assertIn("One continuous single frame", out)
        # The rest of the prompt survives; only the split sentence is dropped, because keeping
        # "upper left / lower right" would still describe two regions of one frame.
        self.assertIn("Dew on the step edges", out)
        self.assertIn("Dawn over a city from the shrine steps", out)

    def test_the_gap_validator_flags_split_prompts(self):
        from scenes import _plan_gaps, parse_detailed_response
        reply = detailed_reply(3).replace("Image 2: scene 2 in the rain.",
                                          "Image 2: " + self.SPLIT)
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=3)
        self.assertTrue(any("split/panel composition" in gap for gap in gaps), gaps)
        self.assertTrue(any("Image 2" in gap for gap in gaps), gaps)

    def test_a_split_plan_triggers_a_repair_request(self):
        from scenes import _plan_gaps, parse_detailed_response
        reply = NL.join([
            "Arc: 1 part",
            "Short 1: 1 shots, 1 images.",
            "",
            "Shot 1:",
            "- Duration: 6",
            "- Shot type: wide",
            "- Character on screen: no",
            "- Camera movement: static",
            "- Place: shrine steps",
            "- Story object: a yellow ribbon",
            "- Narrative role: resolution",
            "- Visual description: dawn over the steps",
            "- Action: he descends",
            '- Narration: "He blew the flame out and walked down the stone steps into the '
            'morning, the ribbon still moving behind him long after he was gone."',
            "- Audio notes: birds",
            "- Transition: cut",
            "- Image prompt ref: Image 1",
            "",
            "--- IMAGE GENERATION PROMPTS ---",
            "Image 1: " + self.SPLIT,
            "",
        ])
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=1, need_arc=True)
        self.assertTrue(any("Image 1" in gap and "split" in gap for gap in gaps), gaps)


def _template_shots(n: int):
    from scenes import _template_shots as template
    return template(n)


if __name__ == "__main__":
    unittest.main()
