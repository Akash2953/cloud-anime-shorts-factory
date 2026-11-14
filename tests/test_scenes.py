"""Scene-plan tests: parsing, prompt architecture, pacing and fallback.

The point of these is the thing that broke the first 45s short: a shot list must vary place,
object and shot size, keep most faces off screen, and put the scene BEFORE the character in the
image prompt (a leading character paragraph collapsed every frame into a portrait).
"""

import unittest

from scenes import (
    HOOK_MAX_WORDS, NARRATIVE_ROLES, SHOT_SIZES, STYLE, Shot, _hook_problem, _infer_character,
    _opens_with, _plan_gaps, _spoken_words, _template_shots, _total_spoken_words, anchor_prompt,
    fit_shot_count,
    make_character, parse_detailed_response, parse_hook, parse_plan, plan_shots, reuse_image_refs,
    shots_for_duration, unique_prompts, words_for_duration,
)

# The signature is now a per-arc value, not a global. Tests pin one value so the
# "raincoat"/"red scarf" substring assertions keep their meaning.
CHARACTER = "a girl in a black hooded raincoat with a red scarf"

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
        prompt = shot.prompt(CHARACTER)
        self.assertIn("in a flooded brick alley", prompt)
        self.assertLess(prompt.index("flooded brick alley"), prompt.index(CHARACTER))
        self.assertTrue(prompt.startswith("Anime key visual"))
        self.assertIn("Key object: puddle reflections", prompt)

    def test_insert_shots_carry_no_character(self):
        shot = Shot(narration="x.", shot_size="insert", place="on a hand", action="it shakes",
                    story_object="paper lantern", narrative_role="build_tension", character=False)
        self.assertNotIn(CHARACTER, shot.prompt(CHARACTER))


class TestPacing(unittest.TestCase):
    def test_shot_count_gives_each_shot_room_for_a_sentence(self):
        n = shots_for_duration(55)
        self.assertEqual(n, 8)                       # 55 / 6.8s per cut
        self.assertTrue(5.5 <= 55 / n <= 8.0)        # long enough for 2-3 sentences of story

    def test_short_and_long_windows_stay_inside_the_clamp(self):
        self.assertEqual((shots_for_duration(30), shots_for_duration(120)), (6, 10))

    def test_narration_budget_matches_the_narrator_rate(self):
        self.assertEqual(words_for_duration(55), 145)  # 158 wpm measured on rendered shorts
        self.assertTrue(words_for_duration(55) > 3 * words_for_duration(18))

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


class TestSilentBeats(unittest.TestCase):
    """A beat the planner marks silent must be silent, not a robot saying the word 'none'."""

    def make(self, text: str, seconds: float = 3.0) -> Shot:
        return Shot(narration=text, shot_size="wide", place="a rooftop", action="she waits",
                    story_object="no object", narrative_role="build_tension", seconds=seconds)

    def test_silent_markers_yield_no_speech(self):
        for marker in ("none", "None.", " SILENT ", "(silent)", "n/a", "-", "", "   "):
            self.assertEqual(self.make(marker).speech, "", marker)

    def test_a_line_that_merely_mentions_none_is_still_spoken(self):
        self.assertEqual(self.make("None of the lanterns answer.").speech,
                         "None of the lanterns answer.")

    def test_real_narration_is_kept_and_tidied(self):
        self.assertEqual(self.make("  She   lifts the lantern. ").speech,
                         "She lifts the lantern.")

    def test_a_silent_beat_survives_parsing(self):
        block = ('Shot 1:\n- Duration: 2\n- Character on screen: no\n'
                 '- Visual description: a rain-soaked skyline\n- Action: lanterns drift\n'
                 '- Narration: "none"\n- Image prompt ref: Image 1\n')
        shots, _ = parse_detailed_response(block)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].speech, "")
        self.assertEqual(shots[0].seconds, 2.0)


class TestDetailedParsing(unittest.TestCase):
    """The detailed planner's own `character` verdict drives edit-vs-fresh image routing."""

    BLOCK = (
        'Short 1: 2 shots, 2 images.\n\nShot 1:\n- Duration: 3\n- Shot type: wide\n'
        '- Character on screen: no\n- Camera movement: slow pan\n'
        '- Visual description: A rain-soaked skyline with no one in it\n'
        '- Action: Lanterns drift over the city\n- Narration: "The city glows."\n'
        '- Image prompt ref: Image 1\n\nShot 2:\n- Duration: 4\n- Shot type: medium\n'
        '- Character on screen: yes\n- Camera movement: static\n'
        '- Visual description: A rooftop ledge in the wind\n'
        '- Action: She pulls her hood up\n- Narration: "She lifts the lantern."\n'
        '- Image prompt ref: Image 2\n'
    )

    def test_explicit_character_field_is_parsed(self):
        shots, _ = parse_detailed_response(self.BLOCK)
        self.assertEqual([s.character for s in shots], [False, True])

    def test_missing_field_falls_back_to_frame_description(self):
        legacy = self.BLOCK.replace("- Character on screen: no\n", "").replace(
            "- Character on screen: yes\n", "")
        shots, _ = parse_detailed_response(legacy)
        self.assertFalse(shots[0].character)   # empty skyline
        self.assertTrue(shots[1].character)    # "She pulls her hood up"

    def test_narration_alone_does_not_imply_presence(self):
        block = ('Shot 1:\n- Duration: 3\n- Shot type: insert\n'
                 '- Visual description: a graffiti-covered wall\n'
                 '- Action: the mark dries in the rain\n'
                 '- Narration: "Her brother left this mark."\n- Image prompt ref: Image 1\n')
        shots, _ = parse_detailed_response(block)
        self.assertFalse(shots[0].character)

    def test_infer_character_tokens(self):
        self.assertTrue(_infer_character("the girl's hand on wet iron"))
        self.assertTrue(_infer_character("", "she climbs the fire escape"))
        self.assertFalse(_infer_character("hundreds of paper lanterns burning"))

    def test_a_shot_round_trips_through_state(self):
        shot = Shot(narration="She lifts it.", shot_size="wide", place="a rooftop",
                    action="she lifts", story_object="lantern", narrative_role="evidence",
                    character=True, seconds=3.5, image_prompt_ref=7,
                    image_generation_prompt="a prompt")
        back = Shot.from_dict(shot.as_dict())
        self.assertEqual((back.image_prompt_ref, back.character, back.seconds), (7, True, 3.5))
        self.assertEqual(back.image_generation_prompt, "a prompt")

    def test_place_object_and_role_are_read_from_the_planner(self):
        block = ('Shot 1:\n- Duration: 3\n- Shot type: insert\n- Character on screen: no\n'
                 '- Camera movement: static\n'
                 '- Place: the gap between two humming vending machines\n'
                 '- Story object: hexagonal paper lantern\n- Narrative role: evidence\n'
                 '- Visual description: A girl pressing into the shadow between machines\n'
                 '- Action: she holds her breath\n- Narration: "She folds into the gap."\n'
                 '- Image prompt ref: Image 5\n')
        shots, _ = parse_detailed_response(block)
        self.assertEqual(shots[0].place, "the gap between two humming vending machines")
        self.assertEqual(shots[0].story_object, "hexagonal paper lantern")
        self.assertEqual(shots[0].narrative_role, "evidence")  # not the positional guess

    def test_place_is_not_a_sixty_character_slice(self):
        desc = ("a flooded brick alley under a broken neon sign, pink and blue reflections "
                "shivering across the standing water")
        block = ('Shot 1:\n- Duration: 3\n- Visual description: ' + desc + '\n'
                 '- Action: she drops down\n- Narration: "Her boots split the water."\n'
                 '- Image prompt ref: Image 1\n')
        shots, _ = parse_detailed_response(block)
        place = shots[0].place
        self.assertGreater(len(place), 60)
        self.assertTrue(desc.startswith(place))
        self.assertFalse(place.endswith((" ,", " ")))  # cut on a word boundary

    def test_absent_story_object_is_not_invented(self):
        block = ('Shot 1:\n- Duration: 3\n- Visual description: a rain-soaked skyline\n'
                 '- Action: lanterns drift\n- Narration: "The city glows."\n'
                 '- Image prompt ref: Image 1\n')
        shots, _ = parse_detailed_response(block)
        self.assertEqual(shots[0].story_object, "no object")

    def test_unknown_role_falls_back_to_the_beat_position(self):
        block = ('Shot 1:\n- Duration: 3\n- Narrative role: climax\n'
                 '- Visual description: a rain-soaked skyline\n- Action: lanterns drift\n'
                 '- Narration: "The city glows."\n- Image prompt ref: Image 1\n')
        shots, _ = parse_detailed_response(block)
        self.assertIn(shots[0].narrative_role, NARRATIVE_ROLES)
        self.assertEqual(shots[0].narrative_role, "establish_context")


class TestPromptAnchoring(unittest.TestCase):
    """Planner-written prompts are generic; the anchor makes them share art direction and identity."""

    GENERIC_ENV = ("Cyberpunk art style, rainy cityscape at dusk with floating paper lanterns, "
                   "warm lighting, 9:16 portrait composition, emotional beat of melancholy, "
                   "engagement hook of mystery.")
    GENERIC_CHAR = ("Anime style, hooded girl on a rooftop, cool lighting, 9:16 composition, "
                    "engagement hook of intrigue.")

    def test_character_prompt_gets_the_identity_signature(self):
        out = anchor_prompt(self.GENERIC_CHAR, character=True, character_sig=CHARACTER)
        self.assertIn(CHARACTER, out)
        self.assertIn("red scarf", out)

    def test_environment_prompt_gets_no_face(self):
        out = anchor_prompt(self.GENERIC_ENV, character=False, character_sig=CHARACTER)
        self.assertNotIn("raincoat", out)

    def test_environment_prompt_drops_a_signature_the_planner_wrote(self):
        leaked = self.GENERIC_ENV.replace("engagement hook of mystery.",
                                          f"moody rain, {CHARACTER}.")
        out = anchor_prompt(leaked, character=False, character_sig=CHARACTER)
        self.assertNotIn("raincoat", out)
        self.assertIn("moody rain.", out)
        self.assertNotIn(", .", out)

    def test_art_direction_is_shared_across_frames(self):
        for out in (anchor_prompt(self.GENERIC_ENV, False, character_sig=CHARACTER),
                    anchor_prompt(self.GENERIC_CHAR, True, character_sig=CHARACTER)):
            self.assertIn("cel-shaded", out)
            self.assertIn("lineart", out)
            self.assertIn(STYLE.split(",")[0], out)

    def test_identity_never_leads_the_prompt(self):
        out = anchor_prompt(self.GENERIC_CHAR, character=True, character_sig=CHARACTER)
        self.assertLess(out.index("rooftop"), out.index("raincoat"))

    def test_meta_language_is_stripped(self):
        out = anchor_prompt(self.GENERIC_ENV, character=False, character_sig=CHARACTER)
        self.assertNotIn("engagement hook", out)

    def test_anchor_is_idempotent(self):
        once = anchor_prompt(self.GENERIC_CHAR, character=True, character_sig=CHARACTER)
        self.assertEqual(anchor_prompt(once, character=True, character_sig=CHARACTER), once)

    def test_planner_output_is_anchored_end_to_end(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        reply = ("Shot 1:\n- Duration: 3\n- Character on screen: no\n"
                 "- Visual description: a rain-soaked skyline\n- Action: lanterns drift\n"
                 "- Narration: \"The city glows.\"\n- Image prompt ref: Image 1\n"
                 "Shot 2:\n- Duration: 3\n- Character on screen: yes\n"
                 "- Visual description: a rooftop ledge\n- Action: she lifts the lantern\n"
                 "- Narration: \"She lifts it.\"\n- Image prompt ref: Image 2\n"
                 "--- IMAGE GENERATION PROMPTS ---\n"
                 "Image 1: Cyberpunk art style, rainy skyline, emotional beat of awe.\n"
                 "Image 2: Anime style, hooded girl on a rooftop, cool lighting.\n")
        with patch("scenes._openrouter", return_value=reply):
            shots, prompts, fallback, _arc, _title = plan_shorts_detailed(theme="lantern city",
                                                                  character=CHARACTER, api_key="k")
        self.assertFalse(fallback)
        self.assertIn("raincoat", prompts[2])
        self.assertNotIn("raincoat", prompts[1])
        self.assertTrue(shots[1].image_generation_prompt.endswith("No text, no watermark."))
        self.assertIn(CHARACTER, shots[1].image_generation_prompt)


def detailed_reply(n: int, *, complete: bool = True) -> str:
    """A synthetic detailed-planner reply: n shot blocks, one image prompt each.

    Complete by default (every field the prompt contract asks for, plus an `Arc:` line), so the
    repair loop stays out of the way unless a test is specifically about a missing detail.
    """
    header = (f"Arc: 1 part\nHook: \"She crosses the flooded walkway and counts the lanterns "
              f"above her.\"\nShort 1: {n} shots, {n} images.\n")
    blocks = [header]
    for i in range(1, n + 1):
        extra = ("- Place: a flooded walkway\n- Story object: a paper lantern\n"
                 "- Narrative role: build_tension\n- Audio notes: rain\n- Transition: cut\n"
                 if complete else "")
        blocks.append(
            f"\nShot {i}:\n- Duration: 3\n- Shot type: wide\n- Character on screen: no\n"
            f"- Camera movement: static\n{extra}- Visual description: place {i}\n"
            f"- Action: something happens in {i}\n"
            f"- Narration: \"She crosses the flooded walkway and counts the lanterns above her "
            f"again, because the last time she stood here there were twelve of them and now "
            f"there are only eleven left.\"\n"
            f"- Image prompt ref: Image {i}\n")
    blocks.append("\n--- IMAGE GENERATION PROMPTS ---\n")
    blocks += [f"Image {i}: scene {i} in the rain.\n" for i in range(1, n + 1)]
    return "".join(blocks)


def shot(ref: int, place: str = "a rooftop") -> Shot:
    return Shot(narration="A line of narration.", shot_size="wide", place=place,
                action="she waits", story_object="no object", narrative_role="build_tension",
                image_prompt_ref=ref)


class TestShotCount(unittest.TestCase):
    def test_overlong_plan_is_trimmed_keeping_the_resolution(self):
        plan = [shot(i) for i in range(1, 16)]
        plan[-1].narration = "The last line."
        fitted = fit_shot_count(plan, 13)
        self.assertEqual(len(fitted), 13)
        self.assertEqual(fitted[-1].narration, "The last line.")

    def test_short_plan_is_not_padded(self):
        plan = [shot(i) for i in range(1, 8)]
        self.assertEqual(len(fit_shot_count(plan, 13)), 7)

    def test_unset_target_is_a_no_op(self):
        plan = [shot(i) for i in range(1, 6)]
        self.assertEqual(len(fit_shot_count(plan, 0)), 5)

    def test_declared_count_mismatch_is_detected(self):
        from scenes import _declared_count
        self.assertEqual(_declared_count(detailed_reply(9)), 9)
        self.assertEqual(_declared_count("no count line here"), 0)


class TestImageReuse(unittest.TestCase):
    def test_reuse_caps_distinct_images_at_the_budget(self):
        plan = [shot(i) for i in range(1, 9)]
        reuse_image_refs(plan, 4)
        self.assertLessEqual(len({s.image_prompt_ref for s in plan}), 4)
        self.assertTrue(all(s.image_prompt_ref for s in plan))

    def test_reuse_pairs_neighbouring_beats(self):
        plan = [shot(i) for i in range(1, 14)]
        reuse_image_refs(plan, 8)
        self.assertLessEqual(len({s.image_prompt_ref for s in plan}), 8)
        groups: dict[int, list[int]] = {}
        for i, s in enumerate(plan):
            groups.setdefault(s.image_prompt_ref, []).append(i)
        self.assertTrue(all(len(v) <= 2 for v in groups.values()), groups)
        self.assertTrue(all(v == list(range(v[0], v[0] + len(v))) for v in groups.values()),
                        "a shared frame must cover adjacent beats only")

    def test_reuse_keeps_a_plan_that_fits(self):
        plan = [shot(i) for i in range(1, 5)]
        reuse_image_refs(plan, 8)
        self.assertEqual([s.image_prompt_ref for s in plan], [1, 2, 3, 4])

    def test_a_shared_ref_renders_as_one_image(self):
        plan = [shot(1), shot(1)]
        plan[0].image_generation_prompt = "frame A"
        plan[1].image_generation_prompt = "per-shot rewrite of frame A"
        prompts, slots = unique_prompts(plan, {1: "frame A"})
        self.assertEqual(prompts, ["frame A"])
        self.assertEqual(slots, [0, 0])

    def test_unique_prompts_render_a_shared_image_once(self):
        plan = [shot(1), shot(2), shot(1)]
        for i, s in enumerate(plan):
            s.image_generation_prompt = f"prompt {s.image_prompt_ref} (shot {i})"
        plan[0].image_generation_prompt = plan[2].image_generation_prompt = "shared prompt"
        prompts, slots = unique_prompts(plan, {})
        self.assertEqual(len(prompts), 2)
        self.assertEqual(slots, [0, 1, 0])

    def test_unique_prompts_fall_back_to_the_built_prompt(self):
        prompts, slots = unique_prompts([shot(0), shot(0)], {})
        self.assertEqual(len(prompts), 1)
        self.assertEqual(slots, [0, 0])
        self.assertIn("cel-shaded", prompts[0])


class TestNarrationDensity(unittest.TestCase):
    """8-word captions must be caught and repaired, not rendered."""

    def test_captions_trigger_one_repair_and_prose_is_kept(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        captions = detailed_reply(8)
        prose = detailed_reply(8).replace(
            "She crosses the flooded walkway and counts the lanterns above her again, because "
            "the last time she stood here there were twelve of them and now there are only "
            "eleven left.", "Gold eyes. Targeting.")
        with patch("scenes.request_title", return_value=""), \
             patch("scenes._openrouter", side_effect=[prose, captions]) as planner:
            shots, _, fallback, _, _title = plan_shorts_detailed(theme="t", api_key="k", want=8,
                                                        target_seconds=55)
        self.assertEqual(planner.call_count, 2)              # one repair, then it stops
        self.assertGreater(_spoken_words(shots), 15)
        self.assertIn("captions", planner.call_args[0][0][-1]["content"])

    def test_the_request_states_the_word_budget(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes.request_title", return_value=""), \
             patch("scenes._openrouter", return_value=detailed_reply(8)) as planner:
            plan_shorts_detailed(theme="t", api_key="k", want=8, target_seconds=55)
        sent = planner.call_args[0][0]
        self.assertIn("about 145 words", sent[0]["content"])
        self.assertIn("caption", sent[0]["content"])
        self.assertIn("145 words in total", sent[1]["content"])


class TestPlannerCountRequest(unittest.TestCase):
    def test_request_states_the_required_shot_count(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes.request_title", return_value=""), \
             patch("scenes._openrouter", return_value=detailed_reply(8)) as planner:
            shots, _, _, _, _title = plan_shorts_detailed(theme="lantern city", api_key="k", want=8)
        self.assertEqual(len(shots), 8)
        sent = planner.call_args[0][0]
        self.assertIn("exactly 8 shots", sent[0]["content"])
        self.assertIn("Shots required: 8", sent[1]["content"])

    def test_one_repair_attempt_when_the_plan_is_short(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes.request_title", return_value=""), \
             patch("scenes._openrouter",
                   side_effect=[detailed_reply(3), detailed_reply(6)]) as planner:
            shots, _, fallback, _arc, _title = plan_shorts_detailed(theme="lantern city", api_key="k", want=8)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(len(shots), 6)
        self.assertFalse(fallback)

    def test_no_second_attempt_when_the_plan_is_close_enough(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes.request_title", return_value=""), \
             patch("scenes._openrouter", return_value=detailed_reply(7)) as planner:
            plan_shorts_detailed(theme="lantern city", api_key="k", want=8)
        self.assertEqual(planner.call_count, 1)

    def test_planning_never_spends_workers_ai(self):
        """The daily neurons belong to image generation; text planning is OpenRouter's job."""
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes._workers_ai") as workers, \
             patch("scenes._openrouter", return_value=detailed_reply(13)):
            shots, _, fallback, _arc, _title = plan_shorts_detailed(theme="lantern city", api_key="k", want=13)
        workers.assert_not_called()
        self.assertEqual((len(shots), fallback), (13, False))

    def test_no_key_degrades_to_the_template_without_a_request(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes._openrouter") as orouter:
            shots, _, fallback, _arc, _title = plan_shorts_detailed(theme="lantern city", api_key="")
        orouter.assert_not_called()
        self.assertEqual((shots, fallback), ([], True))


class TestTitleRequest(unittest.TestCase):
    """A plan that forgets `Title:` gets asked for one, instead of uploading cut prose."""

    def test_missing_title_triggers_a_follow_up_request(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes._openrouter",
                   side_effect=[detailed_reply(8), "The Lantern That Remembered"]) as planner:
            shots, _, fallback, _arc, title = plan_shorts_detailed(
                theme="t", api_key="k", want=8)
        self.assertFalse(fallback)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(title, "The Lantern That Remembered")
        # the follow-up is a title request, not a second storyboard: it carries the beats only
        self.assertIn("Reply with the title only", planner.call_args[0][0][-1]["content"])

    def test_a_stated_title_costs_no_extra_request(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        reply = detailed_reply(8).replace("Short 1: 8 shots, 8 images.",
                                          "Title: The Lantern That Remembered\n"
                                          "Short 1: 8 shots.")
        with patch("scenes._openrouter", return_value=reply) as planner:
            _, _, _, _, title = plan_shorts_detailed(theme="t", api_key="k", want=8)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(title, "The Lantern That Remembered")

    def test_a_useless_reply_falls_back_to_a_complete_phrase(self):
        from unittest.mock import patch
        from scenes import plan_shorts_detailed
        with patch("scenes._openrouter",
                   side_effect=[detailed_reply(8), "", ""]):
            _, _, _, _, title = plan_shorts_detailed(theme="t", api_key="k", want=8)
        self.assertTrue(title)
        self.assertFalse(title.lower().endswith((" while", " and", " because", " of")))

    def test_title_request_cleans_chatty_replies(self):
        from unittest.mock import patch
        from scenes import request_title
        from scenes import _template_shots
        for reply, expected in [("Title: The Lantern", "The Lantern"),
                                ('"The Lantern"', "The Lantern"),
                                ("Sure! Here is a title: The Lantern.", "The Lantern"),
                                ("The Lantern\nHope that helps!", "The Lantern")]:
            with patch("scenes._plan_request", return_value=reply):
                self.assertEqual(request_title(_template_shots(3), "k"), expected, reply)

    def test_title_request_reports_failure_without_beats(self):
        from unittest.mock import patch
        from scenes import request_title
        with patch("scenes._plan_request") as planner:
            self.assertEqual(request_title([], "k"), "")
        planner.assert_not_called()


class TestProviderErrors(unittest.TestCase):
    """A spent free allocation must be named in the log and must not burn the second model."""

    QUOTA_BODY = {"errors": [{"message": "AiError: you have used up your daily free allocation "
                                          "of 10,000 neurons, please upgrade", "code": 4006}],
                  "success": False}

    def fake(self, status, body):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.status_code = status
        response.json.return_value = body
        response.text = str(body)
        return response

    def test_quota_error_is_named_and_skips_the_second_workers_model(self):
        from unittest.mock import patch
        from scenes import _workers_ai
        with patch("scenes.requests.post", return_value=self.fake(429, self.QUOTA_BODY)) as post:
            with self.assertLogs("scenes", level="WARNING") as logs:
                self.assertIsNone(_workers_ai([{"role": "user", "content": "x"}], "t", "a"))
        self.assertEqual(post.call_count, 1)      # the allocation is per account, not per model
        joined = " ".join(logs.output)
        self.assertIn("4006", joined)
        self.assertIn("daily free allocation", joined)
        self.assertIn("OpenRouter", joined)

    def test_other_failures_still_try_the_second_model(self):
        from unittest.mock import patch
        from scenes import _workers_ai
        body = {"errors": [{"message": "internal error", "code": 5001}]}
        with patch("scenes.requests.post", return_value=self.fake(500, body)) as post:
            self.assertIsNone(_workers_ai([{"role": "user", "content": "x"}], "t", "a"))
        self.assertEqual(post.call_count, 2)

    def test_openrouter_failure_reason_is_logged(self):
        from unittest.mock import patch
        from scenes import _openrouter
        body = {"error": {"code": 402, "message": "Insufficient credits"}}
        with patch("scenes.requests.post", return_value=self.fake(402, body)):
            with self.assertLogs("scenes", level="WARNING") as logs:
                self.assertIsNone(_openrouter([{"role": "user", "content": "x"}], "key"))
        self.assertIn("402", " ".join(logs.output))
        self.assertIn("Insufficient credits", " ".join(logs.output))

    def test_a_429_model_falls_through_to_a_live_one(self):
        from unittest.mock import patch
        from scenes import _openrouter
        good = {"choices": [{"message": {"content": "Shot 1: a real storyboard reply. " * 10}}]}
        with patch("scenes.requests.post",
                   side_effect=[self.fake(429, {"error": {"code": 429, "message": "Provider "
                                                                     "returned error"}}),
                                self.fake(200, good)]) as post:
            text = _openrouter([{"role": "user", "content": "x"}], "key")
        self.assertEqual(post.call_count, 2)
        self.assertIn("real storyboard", text)

    def test_default_model_list_ends_with_the_auto_router(self):
        from scenes import DEFAULT_OPENROUTER_MODELS
        self.assertEqual(DEFAULT_OPENROUTER_MODELS[-1], "openrouter/free")
        self.assertNotIn("google/gemma-4-31b-it:free", DEFAULT_OPENROUTER_MODELS)

    def test_error_detail_reads_both_body_shapes(self):
        from scenes import api_error_detail, is_quota_error
        self.assertEqual(api_error_detail(self.fake(429, self.QUOTA_BODY))[:4], "4006")
        self.assertEqual(api_error_detail(self.fake(402, {"error": {"code": 402,
                                                               "message": "no credits"}})),
                         "402: no credits")
        self.assertTrue(is_quota_error(api_error_detail(self.fake(429, self.QUOTA_BODY))))
        self.assertTrue(is_quota_error(api_error_detail(
            self.fake(402, {"error": {"code": 402, "message": "Insufficient credits"}}))))
        self.assertFalse(is_quota_error("5001: internal error"))

    def test_image_quota_is_classified_from_the_body_too(self):
        from images import _classify
        self.assertEqual(_classify(429, "anything"), "quota_exhausted")
        self.assertEqual(_classify(503, str(self.QUOTA_BODY)), "quota_exhausted")
        self.assertEqual(_classify(401, "nope"), "auth_401")


class TestOpeningHook(unittest.TestCase):
    """The first spoken line decides whether the rest of the short is watched at all.

    These pin the two failure modes that shipped: a hook framed as setup ("once upon a time") and a
    stated hook that the opening shot does not actually speak, leaving a weather description as the
    first thing a viewer hears.
    """

    GOOD = ("In Oakhaven memories are currency, and Kael just found one that wasn't.")

    def test_hook_is_read_from_its_own_line(self):
        reply = f'Hook: "{self.GOOD}"\nTitle: The Memory Thief\n'
        self.assertEqual(parse_hook(reply), self.GOOD)

    def test_markdown_and_single_quotes_are_tolerated(self):
        self.assertEqual(parse_hook(f"**Hook:** '{self.GOOD}'"), self.GOOD)
        self.assertEqual(parse_hook(f"- Hook = {self.GOOD}"), self.GOOD)

    def test_prose_about_hooks_is_not_a_hook_line(self):
        self.assertEqual(parse_hook("The last shot must end on a hook, never a resolution."), "")
        self.assertEqual(parse_hook("Short 1: 8 shots, 8 images."), "")

    def test_setup_openers_are_rejected(self):
        for opener in ("Once upon a time, in a far village, a boy waited.",
                       "In a world where memories are sold, one boy refused.",
                       "Have you ever wondered what a memory is worth to you?",
                       "This is a story about a boy who sold his memories."):
            self.assertTrue(_hook_problem(opener), opener)

    def test_a_story_of_frame_survives_when_it_carries_the_cost(self):
        self.assertEqual(_hook_problem(
            "This is the story of a boy who sold memories and forgot his own sister."), "")

    def test_length_band_is_enforced_both_ways(self):
        self.assertIn("words", _hook_problem("Kael sells memories."))
        long = " ".join(["Kael"] * (HOOK_MAX_WORDS + 1))
        self.assertIn("too long", _hook_problem(long))
        self.assertEqual(_hook_problem(self.GOOD), "")

    def test_shot_one_must_speak_the_stated_hook(self):
        hook = self.GOOD
        self.assertTrue(_opens_with(hook.lower() + " He did not know it yet.", hook))
        self.assertFalse(_opens_with(
            "The mist clung to the rooftops when Kael arrived, vials singing.", hook))

    def test_a_hook_the_opening_shot_ignores_is_a_gap(self):
        reply = detailed_reply(2).replace(
            'Hook: "She crosses the flooded walkway and counts the lanterns above her."',
            f'Hook: "{self.GOOD}"'
        ).replace('- Narration: "She crosses',
                  '- Narration: "The mist clung to the roofs and she crosses', 1)
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=2)
        joined = " | ".join(gaps)
        self.assertIn("does not open with the Hook", joined)
        self.assertIn("The mist clung", joined)

    def test_a_missing_hook_line_is_a_gap(self):
        reply = detailed_reply(4).replace(
            'Hook: "She crosses the flooded walkway and counts the lanterns above her."\n', "")
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=4)
        self.assertTrue(any("`Hook:` line is missing" in gap for gap in gaps), gaps)


class TestWordCeiling(unittest.TestCase):
    """Speech is read at a fixed pace, so the word count is the runtime.

    The renderer extends the video instead of speeding the narrator up, which makes an over-long
    plan a planning problem: it has to be caught here, in words, before any audio is made.
    """

    def long_plan(self, n: int = 8, words: int = 30) -> list[Shot]:
        line = " ".join(["word"] * words) + "."
        return [Shot(narration=line, shot_size="wide", place=f"place {i}", action="she waits",
                     story_object="no object", narrative_role="build_tension",
                     image_prompt_ref=i) for i in range(1, n + 1)]

    def test_a_plan_past_the_ceiling_is_a_gap(self):
        reply = detailed_reply(4)
        shots, prompts = parse_detailed_response(reply)
        gaps = _plan_gaps(reply, shots, prompts, want=4, max_words=40)
        self.assertTrue(any("carries at most 40" in gap for gap in gaps), gaps)

    def test_a_plan_inside_the_ceiling_is_not(self):
        reply = detailed_reply(4)
        shots, prompts = parse_detailed_response(reply)
        self.assertEqual(_plan_gaps(reply, shots, prompts, want=4, max_words=400), [])

    def test_the_total_is_what_is_measured_not_the_average(self):
        self.assertEqual(_total_spoken_words(self.long_plan(3, 10)), 30)


class TestOfflineFallback(unittest.TestCase):
    def test_no_api_key_yields_template_plan(self):
        from unittest.mock import patch
        with patch("scenes.requests.post") as post:
            shots, fallback = plan_shots(theme="lantern city", count=13, api_key="")
        post.assert_not_called()      # no key, no request: straight to the template
        self.assertTrue(fallback)
        self.assertEqual(len(shots), 13)
        self.assertTrue(all(s.narration.endswith(".") for s in shots))


if __name__ == "__main__":
    unittest.main()
