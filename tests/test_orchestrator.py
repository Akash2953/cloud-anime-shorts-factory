"""Queue tests: which part a run picks, and when the series is allowed to move on.

The failure these lock down: a new plan used to be replaced by the shot *count* before it was
ever used, a template plan was written into the series, and `next_part` never advanced because
the state file was only written after an early `return`.
"""

import unittest
from unittest.mock import patch

import numpy as np

from orchestrator import (
    advance_series, beat_track, caption_preflight, published_video, quota_spent_today,
    record_published, resolve_part, resolve_voice, run_pipeline, story_description,
    story_upload_title, upload_privacy, uploads_today,
)
from scenes import GENRES, Shot, genre_by_name, next_genre, _template_shots, words_for_duration
from state import today_key

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


class TestGenreRotation(unittest.TestCase):
    """Twelve stories in a row came back dark fantasy with a missing person in them.

    The cause was an empty `theme`: the planner had no world to write in, so it reused the
    prompt's own examples. A genre is now drawn per story, with a story engine and the light the
    frames are lit by, and nothing repeats until the whole cycle has had a turn.
    """

    def test_every_genre_gets_a_turn_before_any_repeats(self):
        history, seen = [], []
        for _ in range(len(GENRES)):
            genre = next_genre(history)
            history.append(genre.name)
            seen.append(genre.name)
        self.assertEqual(len(set(seen)), len(GENRES))

    def test_the_next_cycle_draws_a_different_order(self):
        history = [g.name for g in GENRES]
        second = []
        for _ in range(len(GENRES)):
            genre = next_genre(history)
            history.append(genre.name)
            second.append(genre.name)
        self.assertEqual(sorted(second), sorted(g.name for g in GENRES))
        self.assertNotEqual(second, [g.name for g in GENRES])

    def test_a_genre_name_maps_back_to_its_world(self):
        self.assertEqual(genre_by_name("sports").name, "sports")
        self.assertIsNone(genre_by_name(""))
        self.assertIsNone(genre_by_name("a genre that does not exist"))

    def test_new_story_draws_a_genre_and_records_it(self):
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            resolve_part(CFG, state)
        drawn = planner.call_args.kwargs["genre"]
        self.assertIsNotNone(drawn)
        self.assertEqual(state["series"]["genre"], drawn.name)
        self.assertEqual(state["genres"], [drawn.name])

    def test_a_second_story_draws_a_different_genre(self):
        first = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)):
            resolve_part(CFG, first)
        second = {"processed": [], "daily": {}, "genres": first["genres"]}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            resolve_part(CFG, second)
        self.assertNotEqual(planner.call_args.kwargs["genre"].name, first["genres"][0])

    def test_a_later_part_stays_in_the_world_the_arc_opened_in(self):
        state = series_state(parts=1, next_part=1)
        state["series"].update(total_parts=3, genre="sports")
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            resolve_part(CFG, state)
        self.assertEqual(planner.call_args.kwargs["genre"].name, "sports")
        self.assertNotIn("genres", state)   # the arc does not spend a second genre slot

    def test_an_arc_planned_before_genres_existed_still_plans(self):
        state = series_state(parts=1, next_part=1)
        state["series"].update(total_parts=3)
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(13), {}, False, 1)) as planner:
            resolve_part(CFG, state)
        self.assertIsNone(planner.call_args.kwargs["genre"])


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

    def test_the_planner_gets_a_word_ceiling_derived_from_max_seconds(self):
        # Speech is never time-scaled, so the ceiling is what keeps a plan from outrunning the
        # short: MAX_SECONDS minus the breaths and the tail hold, converted at the fixed pace.
        state = {"processed": [], "daily": {}}
        with patch("orchestrator.plan_shorts_detailed",
                   return_value=(plan(8), {}, False, 1)) as planner:
            resolve_part(CFG, state)
        ceiling = planner.call_args.kwargs["max_words"]
        self.assertGreaterEqual(ceiling, 180)          # 8 shots, 75 s window
        self.assertLess(ceiling, words_for_duration(75))

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


class TestUploadLedger(unittest.TestCase):
    """A part that is already on YouTube must not be uploaded a second time.

    "The Glass Storm Keeper #1/3" shipped twice because a local run's state was never committed,
    so the next day's cron re-planned the same part. The ledger cannot see across machines, but it
    does stop a run from re-publishing a part whose upload it recorded.
    """

    def published_state(self) -> dict:
        return {"processed": [], "daily": {}, "series": {
            "id": "2026-11-15", "parts": [[{}]], "next_part": 0, "total_parts": 3,
            "published": {"1": {"video_id": "ny9S4RLqRXU",
                                 "title": "The Glass Storm Keeper"}}}}

    def test_the_ledger_round_trips(self):
        state = {"series": {}}
        record_published(state, 2, "vid42", "A Story")
        self.assertEqual(published_video(state, 2), "vid42")

    def test_state_written_before_the_ledger_has_no_entries(self):
        self.assertEqual(published_video({}, 1), "")
        self.assertEqual(published_video({"series": {"parts": [[{}]]}}, 1), "")

    def test_the_pipeline_refuses_the_second_upload_of_a_part(self):
        state = self.published_state()
        with patch("orchestrator.load_config", return_value=CFG), \
             patch("orchestrator.load_state", return_value=state), \
             patch("orchestrator.resolve_part", return_value=(plan(3), 1, 3, state)), \
             patch("orchestrator.produce_images") as images, \
             self.assertLogs("orchestrator", level="ERROR") as logs:
            run_pipeline()
        images.assert_not_called()
        self.assertIn("already on YouTube", "\n".join(logs.output))


class TestNarratorPinning(unittest.TestCase):
    """One arc, one narrator: a new story matches the protagonist, an old arc keeps its voice."""

    def test_a_new_story_matches_a_male_protagonist(self):
        voice = resolve_voice({"KOKORO_VOICE": "auto"}, {"id": "2026-01-01"}, "a boy in a track jacket")
        self.assertTrue(voice.startswith("am_") or voice.startswith("bm_"))

    def test_an_arc_that_shipped_before_pinning_keeps_the_old_default(self):
        series = {"id": "2026-11-22", "parts": [[{}]], "voice": ""}
        self.assertEqual(resolve_voice({"KOKORO_VOICE": "auto"}, series,
                                       "a young woman in a chef's apron"), "af_heart")

    def test_a_stored_voice_is_never_redrawn(self):
        series = {"id": "x", "parts": [[{}]], "voice": "am_fenrir"}
        self.assertEqual(resolve_voice({"KOKORO_VOICE": "auto"}, series, "a girl"), "am_fenrir")

    def test_an_explicit_setting_still_wins(self):
        self.assertEqual(resolve_voice({"KOKORO_VOICE": "af_bella"}, {"id": "x"}, "a boy"),
                         "af_bella")


class TestUploadVisibility(unittest.TestCase):
    """The channel publishes hands-off, so a missing setting must not silently mean private."""

    def test_missing_setting_publishes_public(self):
        self.assertEqual(upload_privacy({}), "public")

    def test_blank_and_misspelled_values_publish_public(self):
        self.assertEqual(upload_privacy({"PUBLISH_PRIVACY": "  "}), "public")
        with self.assertLogs("orchestrator", level="WARNING"):
            self.assertEqual(upload_privacy({"PUBLISH_PRIVACY": "pubic"}), "public")

    def test_an_explicit_test_upload_is_respected(self):
        self.assertEqual(upload_privacy({"PUBLISH_PRIVACY": "UNLISTED"}), "unlisted")
        self.assertEqual(upload_privacy({"PUBLISH_PRIVACY": "private"}), "private")


class TestCaptionPreflight(unittest.TestCase):
    """The run must say out loud when its interpreter cannot produce subtitles."""

    def test_a_healthy_interpreter_is_quiet(self):
        with patch("captions.missing_caption_dependencies", return_value=[]), \
             self.assertNoLogs("orchestrator", level="WARNING"):
            caption_preflight({"WORD_CAPTIONS": True})

    def test_missing_faster_whisper_warns_with_the_interpreter_and_the_fix(self):
        with patch("captions.missing_caption_dependencies", return_value=["faster_whisper"]), \
             self.assertLogs("orchestrator", level="WARNING") as logs:
            caption_preflight({"WORD_CAPTIONS": True})
        text = "\n".join(logs.output)
        self.assertIn("faster_whisper", text)
        self.assertIn("requirements.txt", text)
        self.assertIn("line per beat", text)

    def test_missing_pillow_is_an_error_because_no_subtitles_can_render(self):
        with patch("captions.missing_caption_dependencies", return_value=["PIL"]), \
             self.assertLogs("orchestrator", level="ERROR") as logs:
            caption_preflight({"WORD_CAPTIONS": True})
        self.assertIn("WITHOUT subtitles", "\n".join(logs.output))

    def test_captions_disabled_skips_the_check(self):
        with patch("captions.missing_caption_dependencies") as check:
            caption_preflight({"WORD_CAPTIONS": False})
        check.assert_not_called()


class TestUploadMetadata(unittest.TestCase):
    """What the upload request actually carries: `snippet.title` and `snippet.description`.

    The description used to be assembled inline in the run and never checked, so a change to the
    shot list could silently drop narration or leak the call to action into the script.
    """

    TAGS = ["anime", "shorts", "storytime", "mystery", "lantern", "rainy city"]

    def shots(self):
        def shot(narration, role="build_tension"):
            return Shot(narration=narration, shot_size="medium", place="a rooftop",
                        action="He raises the lantern.", story_object="lantern",
                        narrative_role=role, camera="static", character=True, seconds=7.0)

        return [shot("The lantern answered with a storm."),
                shot("He sold his name for the light."),
                shot("Subscribe so you do not miss part 2.", "call_to_action")]

    def test_narration_is_listed_in_order_and_the_cta_is_not(self):
        desc = story_description(self.shots(), 1, 3)
        self.assertIn("# Part 1 of 3", desc)
        self.assertIn("1. The lantern answered with a storm.", desc)
        self.assertIn("2. He sold his name for the light.", desc)
        self.assertNotIn("Subscribe", desc)

    def test_hashtags_close_the_description_with_shorts_first_and_no_duplicates(self):
        desc = story_description(self.shots(), 1, 3, tags=self.TAGS)
        hashtags = desc.splitlines()[-1]
        self.assertTrue(hashtags.startswith("#Shorts "))
        self.assertEqual(hashtags.lower().count("#shorts"), 1)
        self.assertIn("#rainycity", hashtags)   # a space would split the hashtag in two

    def test_the_music_credit_sits_above_the_hashtags(self):
        desc = story_description(self.shots(), 2, 3, tags=self.TAGS, music_credit="Music: Someone")
        self.assertLess(desc.index("Music: Someone"), desc.index("#Shorts"))

    def test_a_long_story_is_capped_at_youtube_limit(self):
        from scenes import Shot as S
        many = [S(narration="word " * 400, shot_size="medium", place="p", action="a",
                  story_object="o", narrative_role="build_tension", camera="static",
                  character=False, seconds=7.0) for _ in range(10)]
        self.assertLessEqual(len(story_description(many, 1, 1)), 5000)

    def test_part_marker_is_in_the_title(self):
        state = {"series": {"title": "The Glass Storm Keeper", "parts": []}}
        self.assertEqual(story_upload_title(state, 1, 3), "The Glass Storm Keeper #1/3")
        self.assertEqual(story_upload_title(state, 1, 1), "The Glass Storm Keeper")


class TestDailyQuota(unittest.TestCase):
    """A second trigger on the same day must cost nothing: no plan, no images, no upload.

    The workflow now has a fallback cron because GitHub's schedule dropped whole days in Nov 2026.
    That fallback arrives after a late primary has already published, and a run that only checked
    the upload cap *after* uploading still paid for a full render first.
    """

    def state_with(self, uploads: int) -> dict:
        daily = {today_key(): {"image_attempts": 8, "uploads": uploads}} if uploads else {}
        return {"processed": [], "daily": daily}

    def test_no_upload_today_leaves_the_quota_unspent(self):
        self.assertEqual(uploads_today(self.state_with(0)), 0)
        self.assertFalse(quota_spent_today(self.state_with(0)))

    def test_an_upload_today_spends_the_quota(self):
        self.assertEqual(uploads_today(self.state_with(1)), 1)
        self.assertTrue(quota_spent_today(self.state_with(1)))

    def test_yesterdays_upload_does_not_spend_todays_quota(self):
        self.assertFalse(quota_spent_today({"daily": {"2020-01-01": {"uploads": 4}}}))

    def test_manual_and_preview_runs_are_never_guarded(self):
        state = self.state_with(1)
        self.assertFalse(quota_spent_today(state, part=3))
        self.assertFalse(quota_spent_today(state, plan_only=True))
        self.assertFalse(quota_spent_today(state, render_only=True))

    def test_a_spent_quota_exits_before_planning_or_render(self):
        state = self.state_with(1)
        with patch("orchestrator.load_config", return_value=CFG), \
             patch("orchestrator.load_state", return_value=state), \
             patch("orchestrator.resolve_part") as resolve:
            run_pipeline()
        resolve.assert_not_called()
        self.assertEqual(state["daily"][today_key()]["uploads"], 1)   # untouched


if __name__ == "__main__":
    unittest.main()
