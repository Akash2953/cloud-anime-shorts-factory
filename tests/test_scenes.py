"""Scene-plan tests: parsing, prompt architecture, pacing and fallback.

The point of these is the thing that broke the first 45s short: a shot list must vary place,
object and shot size, keep most faces off screen, and put the scene BEFORE the character in the
image prompt (a leading character paragraph collapsed every frame into a portrait).
"""

import unittest

from scenes import (
    CHARACTER, NARRATIVE_ROLES, SHOT_SIZES, Shot, _template_shots, parse_plan,
    plan_shots, shots_for_duration,
)

SAMPLE = """PART 1
1 | establishing | crane_down | high above a rain-soaked neon skyline | rain hammers the towers | no object | establish_context | none | Rain hammers the sleeping city and the sky glows.
2 | medium_wide | dolly_in | on the rooftop ledge in the wind | she pulls her hood up | hexagonal paper lantern | introduce_subject | character | She pulls the hood up and lifts the lantern.
3 | insert | static | on a gloved hand gripping rusted iron | the flame bends in the wind | lantern flame | build_tension | none | The flame bends sideways and almost dies."""


class TestParsing(unittest.TestCase):
    def test_parses_fields_including_camera(self):
        shots = parse_plan(SAMPLE, 10)
        self.assertEqual(len(shots), 3)
        self.assertEqual(shots[0].shot_size, "establishing")
        self.assertEqual(shots[0].camera, "crane_down")
        self.assertFalse(shots[0].character)
        self.assertTrue(shots[1].character)
        self.assertEqual(shots[1].story_object, "hexagonal paper lantern")

    def test_tolerates_missing_camera_column(self):
        line = "1 | insert | on a wet iron railing | her hand slips | lantern | build_tension | none | Her hand slips and the lantern spins."
        shots = parse_plan(line, 5)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].camera, "static")

    def test_rejects_unusable_lines(self):
        bad = "\n".join([
            "no pipes here at all",
            "1 | medium | place | action | object | build_tension | none | too short",
            "2 | medium | place | action | object | build_tension | none | **Analyze the request** please.",
        ])
        self.assertEqual(parse_plan(bad, 5), [])

    def test_coerces_unknown_enum_values(self):
        line = "1 | huge_imax | a place with walls | something happens now | object | epic_finale | characters | A real sentence of narration lands here."
        shots = parse_plan(line, 5)
        self.assertEqual(shots[0].shot_size, "medium")
        self.assertEqual(shots[0].narrative_role, "build_tension")
        self.assertTrue(shots[0].character)


class TestPromptArchitecture(unittest.TestCase):
    def test_scene_leads_and_character_signature_is_last(self):
        shot = Shot(narration="x.", shot_size="wide", place="in a flooded brick alley",
                    action="she drops into the water", story_object="puddle reflections",
                    narrative_role="build_tension", character=True, camera="tilt_down")
        prompt = shot.prompt()
        self.assertIn("in a flooded brick alley", prompt)
        self.assertLess(prompt.index("flooded brick alley"), prompt.index(CHARACTER))
        self.assertTrue(prompt.startswith("Anime key visual"))
        self.assertIn("Key object: puddle reflections", prompt)

    def test_insert_shots_carry_no_character(self):
        shot = Shot(narration="x.", shot_size="insert", place="on a hand", action="it shakes",
                    story_object="paper lantern", narrative_role="build_tension", character=False)
        self.assertNotIn(CHARACTER, shot.prompt())


class TestPacing(unittest.TestCase):
    def test_shot_count_for_45s_sits_in_short_form_change_rate(self):
        n = shots_for_duration(45)
        self.assertEqual(n, 13)
        self.assertTrue(2.0 <= 45 / n <= 4.5)

    def test_template_plan_mixes_roles_sizes_and_faces(self):
        shots = _template_shots(13)
        self.assertEqual(len(shots), 13)
        self.assertTrue({s.shot_size for s in shots} <= set(SHOT_SIZES))
        self.assertTrue({s.narrative_role for s in shots} <= set(NARRATIVE_ROLES))
        self.assertGreaterEqual(len({s.shot_size for s in shots}), 5)
        self.assertGreaterEqual(len({s.narrative_role for s in shots}), 6)
        # most shots must not need a face — that is what keeps identity from being the bottleneck
        self.assertGreaterEqual(sum(1 for s in shots if not s.character), 6)

    def test_never_three_identical_shot_sizes_in_a_row(self):
        sizes = [s.shot_size for s in _template_shots(13)]
        for i in range(len(sizes) - 2):
            self.assertFalse(sizes[i] == sizes[i + 1] == sizes[i + 2],
                             f"three identical shot sizes at {i}: {sizes}")

    def test_places_change(self):
        places = [s.place for s in _template_shots(13)]
        self.assertEqual(len(places), len(set(places)))


class TestOfflineFallback(unittest.TestCase):
    def test_no_credentials_yields_template_plan(self):
        shots, fallback = plan_shots(theme="lantern city", count=13, cfa_token="", account_id="", api_key="")
        self.assertTrue(fallback)
        self.assertEqual(len(shots), 13)
        self.assertTrue(all(s.narration.endswith(".") for s in shots))


if __name__ == "__main__":
    unittest.main()
