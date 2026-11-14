"""Scene planning: story -> shot list.

Adapted (not imported) from the OpenMontage artifact schemas:
  scene_plan  -> per shot: shot_size, camera, narrative_role, place/action/story object
  character   -> a fixed costume signature whose props are named per shot

Why this exists: "one narration line = one image" produced seven medium portraits of the same
face on the same street. A story needs establishing shots, inserts, place changes and a job for
every cut. Each shot therefore carries:
    narration      one voice-over line (short: 8-16 words, so the cut rate stays 2-4s)
    shot_size      establishing | wide | medium_wide | medium | medium_close | close_up |
                   over_shoulder | insert
    place          "rooftop ledge above the neon skyline"
    action         "she lowers herself over the edge"
    story_object   "hexagonal paper lantern"      <- the thing that changes per beat
    narrative_role establish_context | introduce_subject | build_tension | deliver_payload |
                   emotional_beat | evidence | resolution | call_to_action
    camera         static | pan_left | dolly_in | crane_up | handheld | ...
    character      True when the girl is on screen. Insert and environment shots carry no face,
                   which is what makes identity drift a non-issue for most of the video.

A newer detailed prompt can also produce full image-generation prompts and per-shot audio
notes; see PLAN_SYSTEM_PROMPT_DETAILED and plan_shots_detailed().
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

import requests

import gemini

logger = logging.getLogger(__name__)

MAX_PARTS = 7

# Seconds per cut. 3.4 s is the short-form change rate, but at that rhythm each beat gets ~8 words
# of narration — 13 captions rather than a story. Fewer, longer cuts let one continuous voice-over
# carry the scene instead of a string of clipped assertions.
SECONDS_PER_CUT = 6.8

# Kokoro af_heart at speed 1.0 measured on rendered shorts: 104 words in 39.6 s.
NARRATOR_WPM = 158


def words_for_duration(seconds: float) -> int:
    """How much narration a shot list of this length should carry, at the narrator's real rate."""
    return max(20, int(round(seconds * NARRATOR_WPM / 60)))

SHOT_SIZES = {
    "establishing": "Wide establishing shot",
    "wide": "Wide shot",
    "medium_wide": "Medium wide shot",
    "medium": "Medium shot",
    "medium_close": "Medium close shot",
    "close_up": "Close-up shot",
    "over_shoulder": "Over-the-shoulder shot",
    "insert": "Extreme close-up insert shot",
}

NARRATIVE_ROLES = (
    "establish_context", "introduce_subject", "build_tension", "deliver_payload",
    "emotional_beat", "evidence", "resolution", "call_to_action",
)

CAMERA_MOVES = (
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down", "dolly_in", "dolly_out",
    "tracking_left", "tracking_right", "crane_up", "crane_down", "handheld", "zoom_in",
)

STYLE = (
    "Anime key visual, 2D cel-shaded illustration, clean bold lineart, saturated flat colours, "
    "cinematic rain-lit lighting, no text, no watermark"
)

# The character signature is deliberately SHORT and placed last in the prompt: a leading
# paragraph of description overpowers the scene and collapses every frame into a portrait.
# Each arc gets its own signature (make_character): fixed for the whole arc so every part and
# every image edit stays consistent, while different arcs draw different protagonists.

_CHARACTER_POOL = (
    ("a girl", "a boy", "a young woman", "a man", "a child"),
    ("a black hooded raincoat", "a red trench coat", "a grey hooded poncho",
     "a navy hooded cloak", "a green hooded jacket", "a charcoal waxed coat"),
    ("with a red scarf", "with a paper mask", "with glowing goggles",
     "with a knitted beanie", "with a worn satchel", "with silver rings"),
)


def make_character(series_id: str) -> str:
    """A short, stable identity signature: unique per arc, fixed within it.

    Seeded by the series id so every part and image edit of one arc keeps the same face, while
    different arcs come out different. Deterministic, so no LLM call and no spend.
    """
    rng = random.Random(series_id or "arc")
    return f"{rng.choice(_CHARACTER_POOL[0])} in {rng.choice(_CHARACTER_POOL[1])} {rng.choice(_CHARACTER_POOL[2])}"

# Default shot rhythm for a ~45s short: 13 cuts, 2-5s each, roles in narrative order.
# (shot_size, place, action, story_object, narrative_role, camera, character, seconds)
TEMPLATE_PLAN = [
    ("establishing", "high above a rain-soaked neon skyline", "rain hammers the towers and the whole city glows beneath her", "no object", "establish_context", "crane_down", False, 3.0),
    ("medium_wide", "the rooftop ledge, wind tearing at a black coat", "a hooded girl pulls her hood up and lifts a paper lantern", "hexagonal paper lantern", "introduce_subject", "dolly_in", True, 3.5),
    ("insert", "on a gloved hand gripping rusted iron", "the lantern sways and its flame bends in the wind", "hexagonal paper lantern", "build_tension", "static", False, 2.0),
    ("wide", "down in a flooded brick alley full of pink and blue reflections", "she drops down and her boots split the water", "puddle reflections", "build_tension", "tilt_down", True, 3.0),
    ("over_shoulder", "at the alley mouth where a guard stands under an awning", "a torch beam swings toward the sound of splashing", "torch light", "evidence", "pan_right", False, 2.5),
    ("medium", "inside a narrow gap between two humming vending machines", "she presses flat into the shadow and holds her breath", "glowing vending machines", "build_tension", "static", True, 3.0),
    ("wide", "on a rusted fire escape climbing the brick wall into darkness", "she climbs fast, lantern swinging from her wrist", "rusted fire escape", "deliver_payload", "tilt_up", True, 3.5),
    ("insert", "on a wet iron railing far above the street", "her hand slips, then catches, and the lantern spins", "hexagonal paper lantern", "build_tension", "handheld", False, 2.0),
    ("establishing", "across a rooftop garden strung with hundreds of paper lanterns", "every lantern burns at once above the wet concrete", "hundreds of paper lanterns", "deliver_payload", "crane_up", False, 4.0),
    ("wide", "in the middle of that lantern garden", "she stands small among the lights, looking for one that answers", "one answering lantern", "emotional_beat", "dolly_out", True, 3.5),
    ("insert", "on a single paper lantern with a faded name written on it", "the flame inside it leans toward her hand", "named paper lantern", "evidence", "static", False, 2.5),
    ("close_up", "in the lantern garden, warm light across her face", "she whispers a name and the city finally goes quiet", "named paper lantern", "emotional_beat", "dolly_in", True, 3.0),
    ("wide", "above the rooftop garden as the rain turns to mist", "lanterns rise together into the dark and she lets go", "rising lanterns", "resolution", "crane_up", False, 4.0),
]

PLAN_SYSTEM_PROMPT = (
    "You are a shot planner for vertical anime shorts. Output exactly {n} numbered shots and "
    "nothing else. One shot per line, fields separated by ' | ' in this exact order:\n"
    "n | shot_size | place | action | story_object | narrative_role | character | narration\n"
    "Rules:\n"
    "- shot_size: one of {sizes}\n"
    "- narrative_role: one of {roles}\n"
    "- character: the word 'character' if the girl is on screen, otherwise 'none'\n"
    "- narration: ONE spoken sentence of 8-16 words ending with punctuation\n"
    "- aim for a mix: at least 3 shots with character 'none' (places, objects, inserts)\n"
    "- every shot needs a DIFFERENT place or a different object; never repeat a place twice in a row\n"
    "- place and action are concrete and physical (a painter can draw them), no mood words\n"
    "- never repeat a shot_size three times in a row\n"
    "The shots must tell one continuous story in order, ending on a resolution."
)

# Detailed prompt — produces per-shot schema + a separate numbered Image Generation Prompts
# section. Designed to be parsed programmatically.
PLAN_SYSTEM_PROMPT_DETAILED = (
    """## Role

You are an expert anime story writer, storyboard artist, and YouTube Shorts editor. You build
original anime stories and turn them into shot-by-shot schemas and image generation prompts for
vertical slideshow Shorts optimized for viewer engagement.

## Task

Create an original anime story and break it into one YouTube Short (vertical, 9:16, under 60
seconds). The Short is a slideshow of anime still images with narration. The slideshow must read
as a coherent story, not a random image sequence.

**Story requirements:**
- The story must be complete.
- Each Short advances the story and resolves or deepens a clear thread.
- Use as many images as possible while keeping each image clearly relevant.
- **Narration is the story.** The Narration lines, read in order, must sound like one storyteller
telling one story out loud — not a sequence of captions for the pictures. Each line continues
the previous line's thought, sentence or action, and may carry a clause of explanation, memory
or consequence that the frame cannot show. Write full sentences, with the character named or
clearly referred to, and use linking words (and, but, so, until, because, then) where the story
needs them. One word sentences, labels, slogans and noun stacks are forbidden: "Gold eyes.
Targeting." is not narration, it is a caption.
- **Place changes are spoken.** When a shot's `Place` is not the one before it, the first clause of
that shot's Narration carries the move — "By the time he reached the Spire", "Back at the shrine,
the rain had stopped", "Inside, the machine was already awake". Fuse the move into the sentence
that does the story work; never spend a whole line announcing a location ("Later, at the
temple.") — that is padding, and it describes what the frame already shows.

**Per Shot output format (one entry per shot, in order):**

```
Shot N:
- Duration: <seconds>
- Shot type: <wide|medium|close-up|extreme close-up|...>
- Character on screen: <yes|no>
- Camera movement: <static|slow pan|slow zoom|Ken Burns push-in|parallax|...>
- Place: <the concrete location, 3-8 words>
- Story object: <the one object the frame is about, or "no object">
- Narrative role: <establish_context|introduce_subject|build_tension|deliver_payload|emotional_beat|evidence|resolution|call_to_action>
- Visual description: <what appears on screen>
- Action: <action taking place>
- Narration: "<2-3 full sentences of connected prose: what she does, what it costs her, what she has just understood>"     (or "none" if silent; Shot 1's Narration must begin with the Hook sentence, word for word)
- Audio notes: "<music/SFX/ambient or silence>"
- Transition: <cut|fade|dissolve>
- Image prompt ref: Image N
```

Begin with a count line: "Short 1: N shots, M images."

**Before the title and the count line, state the Hook:**
```
Hook: "<one spoken sentence, 10-16 words>"
```
The Hook is the first thing the narrator says, so it is the first sentence of Shot 1's Narration,
word for word. It has one job: a stranger scrolling past, who has never seen this story, has to
need the next ten seconds. Ten to sixteen words. Build it out of the story's own world — who, what
is wrong here, and what it will cost — and leave one of those three unresolved. The hook is not a
summary, a genre label, or a title read aloud.

Its shape is: a person, in this story's place, against one broken rule or one price — stated as a
flat fact, not as a question, a greeting or a scene-setting description. Shot 1 is that sentence
plus at most one short clause of consequence; it still has to fit the per-line word budget this
request sets. Write a new hook for this story: never reuse a line, name or place from an earlier
story, and never copy the shape of these failures as if it were a template.

- Fails: "Once upon a time, in a faraway village, there lived a boy named Kael." (setup, no gap)
- Fails: "The mist clung to the rooftops when Kael arrived." (weather and scenery first)
- Fails: "Have you ever wondered what your memories are worth?" (a question defers the payoff)

A "This is the story of…" frame is allowed, but only when the words after it already carry the
contradiction or the cost — a boy who sold memories and forgot his own sister — never a bare
name, and never as the whole of the hook's content.

**Before the count line, state the story's name:**
```
Title: <3-7 words, the story's own name, no date, no "Anime Short", no part number>
```
The title is what the video is published as, so it must name this story ("The Lantern She
Left Burning"), not describe the format. Do not put "Part 1" in it — the publisher adds the
part number itself. The title is not spoken: it is the upload title, the thumbnail and the
channel grid, so the narrator never spends opening words announcing it.

**Shot 1 is the hook frame.** It carries the Hook sentence, so it must be the most arresting
frame in the plan — the protagonist's face mid-emotion, or the story object doing the impossible
thing — at medium, medium close or close. Never open on an establishing view, a skyline or an
empty street, however atmospheric: a wide view of a place nobody has been introduced to yet is
the sound of a thumb moving to the next video.

**After all shots, add this section header exactly:**
```
--- IMAGE GENERATION PROMPTS ---
```
Then list every image prompt, one per line, numbered sequentially (Image 1:, Image 2:, …). Write
**70-110 words** per prompt, because a viewer holds this frame for six or seven seconds and the
frame has to keep rewarding that attention. Each prompt must be fully self-contained and contain,
in this order: (1) art style, (2) setting, time, lighting and mood, (3) composition as a vertical
9:16 portrait, (4) emotional beat, (5) one or two concrete physical details that sell the place
(weather on a surface, wear on an object, what the light is landing on). The composition is ONE
continuous single frame: never a split, diptych, panel, collage or side-by-side layout, and never
two scenes in one image — the renderer holds this frame for six seconds and a split reads as a
glitch, not a moment.
__CHARACTER_SIGNATURE__

Keep each field concise and concrete so the output can be extracted programmatically.

**Character on screen.** Report whether the girl is physically in frame, not whether she is
mentioned: `no` for establishing shots, empty places and standalone objects; `yes` for any shot
where her face, body or hands are visible (including body-part inserts).

**Place, Story object, Narrative role.** `Place` is a location a painter could stand in, not a
sentence: "a flooded brick alley", "the gap between two vending machines". `Story object` is the
single drawable thing the frame is about, or `no object`. `Narrative role` is the job this cut
does for the story, and must be one of the listed values, spelled exactly."""
)


# The detailed planner is allowed to write a silent beat; these are the words it uses for one.
SILENT_MARKERS = {"none", "n/a", "na", "silent", "no narration", "no voice", "-", "—", "(silent)"}


@dataclass
class Shot:
    narration: str
    shot_size: str
    place: str
    action: str
    story_object: str
    narrative_role: str
    camera: str = "static"
    character: bool = False
    seconds: float = 3.0
    # Detailed-mode extras:
    shot_type: str = ""
    camera_movement: str = ""
    visual_description: str = ""
    audio_notes: str = ""
    transition: str = ""
    image_prompt_ref: int = 0
    image_generation_prompt: str = ""

    @property
    def speech(self) -> str:
        """Narration text for TTS, or "" for a silent beat.

        The schema lets the planner write `none`, `silent` or "" for a beat it wants carried by
        music alone. Feeding that to Kokoro speaks the word "none" over the shot.
        """
        text = " ".join((self.narration or "").split()).strip()
        if not text or text.lower().strip(" .\"'") in SILENT_MARKERS:
            return ""
        return text

    def prompt(self, character: str = "") -> str:
        """Scene and action lead; the character signature is short and comes last."""
        size = SHOT_SIZES.get(self.shot_size, "Medium shot")
        obj = "" if self.story_object in ("", "no object") else f" Key object: {self.story_object}."
        who = f" {character}, seen as part of the scene." if self.character and character else ""
        return (
            f"{STYLE}. {size}: {self.place}. {self.action}.{obj} "
            f"Camera: {self.camera.replace('_', ' ')}.{who}"
        )

    def as_dict(self) -> dict:
        return {
            "narration": self.narration, "shot_size": self.shot_size, "place": self.place,
            "action": self.action, "story_object": self.story_object,
            "narrative_role": self.narrative_role, "camera": self.camera,
            "character": self.character, "seconds": self.seconds,
            "image_prompt_ref": self.image_prompt_ref,
            "image_generation_prompt": self.image_generation_prompt,
            "audio_notes": self.audio_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Shot":
        return cls(
            narration=d["narration"], shot_size=d.get("shot_size", "medium"),
            place=d.get("place", "a rainy city street"), action=d.get("action", "she waits"),
            story_object=d.get("story_object", "no object"),
            narrative_role=d.get("narrative_role", "build_tension"),
            camera=d.get("camera", "static"), character=bool(d.get("character", False)),
            seconds=float(d.get("seconds", 3.0)),
            image_prompt_ref=int(d.get("image_prompt_ref", 0)),
            image_generation_prompt=d.get("image_generation_prompt", ""),
            audio_notes=d.get("audio_notes", ""),
        )


def _coerce(value: str, allowed, default: str) -> str:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in allowed else default


def parse_plan(text: str, want: int) -> list[Shot]:
    """Parse the pipe-delimited shot lines, dropping anything that is not a usable shot."""
    shots: list[Shot] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        line = re.sub(r"^\s*\d+\s*[.)\]]?\s*", "", line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 9:
            _, size, camera, place, action, obj, role, has_char, narration = parts[:9]
        elif len(parts) >= 8:  # model omitted camera
            _, size, place, action, obj, role, has_char, narration = parts[:8]
            camera = "static"
        else:
            continue
        narration = re.sub(r"\s+", " ", narration).strip("\"' ")
        words = len(narration.split())
        if words < 4 or not re.search(r"[.!?]$", narration):
            continue
        if any(tok in narration for tok in ("**", "|", "##")):
            continue
        shots.append(Shot(
            narration=narration,
            shot_size=_coerce(size, SHOT_SIZES, "medium"),
            place=place or "a rainy city street",
            action=action or "the city waits",
            story_object=obj or "no object",
            narrative_role=_coerce(role, NARRATIVE_ROLES, "build_tension"),
            camera=_coerce(camera, CAMERA_MOVES, "static"),
            character=has_char.lower().startswith("char"),
            seconds=3.0,
        ))
        if len(shots) == want:
            break
    return shots


def _template_shots(count: int) -> list[Shot]:
    rows = TEMPLATE_PLAN
    # Keep the last row: slicing the first N rows dropped the resolution, so an 8-beat fallback
    # ended on build_tension and the story stopped without closing.
    picks = rows if count >= len(rows) else rows[:max(count - 1, 1)] + [rows[-1]]
    shots = [
        Shot(narration=_template_narration(r), shot_size=r[0], place=r[1], action=r[2],
             story_object=r[3], narrative_role=r[4], camera=r[5], character=r[6], seconds=r[7])
        for r in picks
    ]
    while len(shots) < count:  # repeat the middle for longer plans
        shots.append(Shot(**{**shots[len(shots) % len(rows)].as_dict(),
                             "narration": _template_narration(rows[len(shots) % len(rows)])}))
    return shots[:count]


_TEMPLATE_LINES = [
    "Rain hammers the sleeping city and the whole skyline glows beneath her.",
    "She pulls the hood up and lifts the lantern into the storm.",
    "The flame bends sideways, and the wind almost takes it.",
    "She drops into the flooded alley and the water splits.",
    "A torch beam swings toward the sound of her boots.",
    "She folds into the gap and holds her breath.",
    "The fire escape rattles as she climbs out of the street.",
    "Her hand slips, catches, and the lantern spins.",
    "Hundreds of lanterns burn above the wet rooftop.",
    "She stands small among them, searching for one that answers.",
    "A faded name is written on the paper, and the flame leans.",
    "She whispers a name and the city finally goes quiet.",
    "The lanterns rise together, and she lets go.",
]


def _template_narration(row) -> str:
    idx = TEMPLATE_PLAN.index(row) if row in TEMPLATE_PLAN else 0
    return _TEMPLATE_LINES[idx % len(_TEMPLATE_LINES)]


def _split_parts(text: str, parts: int, per_part: int) -> list[list[Shot]]:
    chunks = re.split(r"^[\W_]*(?:PART|Part)\s*\d+[\W_]*$", text or "", flags=re.MULTILINE)
    if len(chunks) < 2:
        return [parse_plan(text, per_part)]
    result = [parse_plan(c, per_part) for c in chunks]
    result = [r for r in result if len(r) >= 4]
    return result[:parts] or [parse_plan(text, per_part)]


def shots_for_duration(duration_seconds: float) -> int:
    """8 shots for 55s: ~6.8s per cut, long enough for two or three sentences of story."""
    return max(6, min(10, int(round(duration_seconds / SECONDS_PER_CUT))))


def plan_shots(
    *,
    theme: str,
    count: int,
    api_key: str = "",
    gemini_key: str = "",
) -> tuple[list[Shot], bool]:
    """Return (shots, used_fallback). Text planning takes no Workers AI credentials."""
    system = PLAN_SYSTEM_PROMPT.format(
        n=count, sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERA_MOVES),
        roles=", ".join(NARRATIVE_ROLES))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Write the {count} shots for this story: {theme}"},
    ]

    text = _plan_request(messages, api_key, gemini_key=gemini_key)
    shots = parse_plan(text, count) if text else []
    if len(shots) >= 6:
        logger.info("Scene plan: %d shots (%d without the character)",
                    len(shots), sum(1 for s in shots if not s.character))
        return shots, False

    logger.warning("Scene planner returned %d usable shots; using the template plan", len(shots))
    return _template_shots(count), True


def plan_arc(*, theme: str, parts: int, per_part: int, api_key: str = "",
             gemini_key: str = ""
             ) -> tuple[list[list[Shot]], bool]:
    parts = max(1, min(parts, MAX_PARTS))
    if parts == 1:
        shots, fallback = plan_shots(theme=theme, count=per_part, api_key=api_key,
                                     gemini_key=gemini_key)
        return [shots], fallback

    head = PLAN_SYSTEM_PROMPT.format(
        n=per_part, sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERA_MOVES),
            roles=", ".join(NARRATIVE_ROLES))
    messages = [
        {"role": "system", "content": f"{head}\nWrite {parts} parts. Put a line 'PART <n>' before "
                                      f"each part's shots. Keep the story continuous across parts."},
        {"role": "user", "content": f"Write PART 1 through PART {parts} for: {theme}"},
    ]
    text = _plan_request(messages, api_key, gemini_key=gemini_key)
    if text:
        parsed = _split_parts(text, parts, per_part)
        if parsed and all(len(p) >= 4 for p in parsed):
            return parsed, False
    shots, fallback = plan_shots(theme=theme, count=per_part, api_key=api_key,
                                 gemini_key=gemini_key)
    return [shots], fallback


# Every field the prompt contract asks for in a shot block. Completeness is checked against the
# raw reply, not the parsed Shot: the parser substitutes a default for anything missing (`Place`
# becomes a shortened visual description, `Action` becomes "she watches", a character-less block is
# dropped), so a fabricated field is indistinguishable from a written one after parsing.
_SHOT_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Duration", ("- duration:",)),
    ("Shot type", ("- shot type:",)),
    ("Character on screen", ("- character on screen:", "- character:")),
    ("Camera movement", ("- camera movement:", "- camera:")),
    ("Place", ("- place:", "- location:")),
    ("Story object", ("- story object:",)),
    ("Narrative role", ("- narrative role:", "- narrative beat:")),
    ("Visual description", ("- visual description:", "- visual:")),
    ("Action", ("- action:",)),
    ("Narration", ("- narration:",)),
    ("Image prompt ref", ("- image prompt ref:",)),
)

PLAN_REPAIR_ROUNDS = 2


def _shot_blocks(text: str) -> list[str]:
    """Raw text of each `Shot N:` block, tolerating markdown decoration around the label."""
    parts = re.split(r"(?im)^\s*(?:#+\s*)?\*{0,2}\s*Shot\s+\d+\s*:\*{0,2}", text or "")
    return [part.strip() for part in parts[1:] if part.strip()]


def _ranges(numbers: list[int]) -> str:
    """`[1, 2, 3, 5]` -> `"1-3, 5"` for a readable repair instruction."""
    out: list[str] = []
    start = prev = None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else str(start))
    return ", ".join(out)


def _plan_gaps(text: str, shots: list[Shot], image_prompts: dict[int, str], *,
               want: int = 0, need_arc: bool = False, hook_required: bool = True,
               max_words: int = 0) -> list[str]:
    """What the planner left out of its reply, phrased as repair instructions.

    Anything listed here is detail the pipeline would otherwise silently invent its own answer
    for — a missing image prompt, a missing field per shot, an undeclared arc length, a hook the
    opening shot does not actually speak, a script the narrator cannot finish inside the short, or
    narration written as captions instead of prose.
    """
    gaps: list[str] = []
    blocks = _shot_blocks(text)
    if not shots:
        gaps.append("no `Shot N:` blocks were written")
        return gaps
    if want and len(shots) < want - 2:
        gaps.append(f"only {len(shots)} shots were written; the pacing needs {want}")
    if len(blocks) > len(shots):
        gaps.append(f"{len(blocks) - len(shots)} shot block(s) had neither a Narration line nor an "
                    "image ref, so they were dropped: every shot needs both")
    for name, labels in _SHOT_FIELDS:
        missing = [i for i, block in enumerate(blocks, 1)
                   if not any(label in block.lower() for label in labels)]
        if missing:
            gaps.append(f"shot(s) {_ranges(missing)} are missing the `{name}` field")
    if hook_required:
        gaps.extend(_hook_gaps(text, shots))
    if not image_prompts:
        gaps.append("the `--- IMAGE GENERATION PROMPTS ---` section is missing")
    else:
        refs = sorted({shot.image_prompt_ref for shot in shots if shot.image_prompt_ref})
        missing = [ref for ref in refs if ref not in image_prompts]
        if missing:
            gaps.append(f"no image generation prompt for Image {_ranges(missing)}")
        split = sorted(ref for ref, prompt in image_prompts.items()
                       if _asks_for_split(prompt))
        if split:
            gaps.append(f"Image {_ranges(split)} asks for a split/panel composition: describe ONE "
                        "continuous single frame instead (no diptych, no side-by-side panels, "
                        "no two scenes in one image)")
    if need_arc and not re.search(r"\barc\b\s*[:=]?\s*(\d+)\s*parts?\b", text or "", re.I):
        gaps.append("the `Arc: N parts` line is missing, so the story's length is unknown")
    if _spoken_words(shots) < 11:
        gaps.append(f"the narration reads as captions ({_spoken_words(shots):.0f} words per "
                    "shot): rewrite every Narration line as 2-3 connected sentences of a "
                    "storyteller's voice-over, 16-20 words each, each line continuing the one "
                    "before it, without describing the picture")
    if max_words and _total_spoken_words(shots) > max_words:
        total = _total_spoken_words(shots)
        gaps.append(f"the narration is {total} words, but at the narrator's fixed pace this short "
                    f"carries at most {max_words}: cut about {total - max_words} words by "
                    f"tightening lines and dropping beats — the read is never sped up to fit")
    return gaps


def _hook_gaps(text: str, shots: list[Shot]) -> list[str]:
    """The opening line is the one thing a viewer judges before the story reaches them.

    A stated hook that Shot 1 does not speak is worse than no hook, because the plan then opens on
    scenery while claiming to open on tension — so the stated line and the spoken line are checked
    against each other, and the mismatch is repaired rather than shipped.
    """
    hook = parse_hook(text)
    if not hook:
        missing = ("the `Hook:` line is missing: state the narrator's opening sentence (10-16 words, "
                   "who + what is wrong + what it costs) above the title")
        return [missing]
    problem = _hook_problem(hook)
    if problem:
        return [problem]
    opening = (shots[0].narration or "").strip()
    if not _opens_with(opening, hook):
        first = re.split(r"(?<=[.!?])\s+", opening)[0][:90] if opening else "nothing"
        mismatch = (f"Shot 1 does not open with the Hook line. Its Narration begins {first!r}: the "
                    f"first words the narrator says must be the Hook, word for word, before any "
                    f"scenery or backstory")
        return [mismatch]
    return []


def _request_repair(messages: list[dict], text: str, gaps: list[str], api_key: str, *,
                    gemini_key: str = "") -> str | None:
    """Ask for the whole storyboard again, naming every detail that was missing."""
    instruction = ("Your storyboard is incomplete. Keep the same story and output the WHOLE "
                   "storyboard again, plain text, in exactly the same format, with nothing "
                   "omitted. Fix all of these:\n" + "\n".join(f"- {gap}" for gap in gaps))
    return _plan_request(messages + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": instruction},
    ], api_key, gemini_key=gemini_key)


def parse_detailed_response(text: str) -> tuple[list[Shot], dict[int, str]]:
    """Parse the structured detailed-format output into Shots + image prompts.

    Returns (shots, image_prompts) where image_prompts maps image_number -> prompt_text.
    Falls back gracefully when the LLM output doesn't match the expected structure.
    """
    shots: list[Shot] = []
    image_prompts: dict[int, str] = {}

    if not text:
        return shots, image_prompts

    # Split into shot entries and image-prompt section
    img_section_match = re.search(r"---\s*IMAGE GENERATION PROMPTS\s*---", text, re.IGNORECASE)
    if img_section_match:
        shot_text = text[:img_section_match.start()]
        img_text = text[img_section_match.end():]
    else:
        shot_text = text
        img_text = ""

    # Parse individual shot blocks
    shot_blocks = re.split(r"(?=^Shot\s+\d+\s*:)", shot_text, flags=re.MULTILINE)
    for block in shot_blocks:
        block = block.strip()
        if not block or not re.match(r"Shot\s+\d+", block, re.IGNORECASE):
            continue

        duration = 3.0
        shot_type = "medium"
        has_character: bool | None = None
        camera_movement = "static"
        place = ""
        story_object = ""
        role = ""
        visual_desc = ""
        action = ""
        narration = ""
        audio_notes = ""
        transition = "cut"
        img_ref = 0

        for line in block.splitlines():
            line = line.strip()
            lower = line.lower()
            if lower.startswith("- duration:"):
                try:
                    duration = float(re.search(r"([\d.]+)", line).group(1))
                except Exception:
                    pass
            elif lower.startswith("- shot type:"):
                st = line.split(":", 1)[1].strip()
                shot_type = _map_shot_type(st)
            elif lower.startswith(("- character on screen:", "- character:")):
                value = line.split(":", 1)[1].strip().lower()
                has_character = value.startswith(("y", "t", "character"))
            elif lower.startswith("- camera movement:"):
                camera_movement = line.split(":", 1)[1].strip()
            elif lower.startswith("- place:"):
                place = line.split(":", 1)[1].strip().strip('"')
            elif lower.startswith(("- story object:", "- key object:", "- object:")):
                story_object = line.split(":", 1)[1].strip().strip('"')
            elif lower.startswith(("- narrative role:", "- role:")):
                role = line.split(":", 1)[1].strip()
            elif lower.startswith("- visual description:"):
                visual_desc = line.split(":", 1)[1].strip()
            elif lower.startswith("- action:"):
                action = line.split(":", 1)[1].strip()
            elif lower.startswith("- narration:"):
                narration = line.split(":", 1)[1].strip().strip('"')
            elif lower.startswith("- audio notes:"):
                audio_notes = line.split(":", 1)[1].strip().strip('"')
            elif lower.startswith("- transition:"):
                transition = line.split(":", 1)[1].strip()
            elif lower.startswith("- image prompt ref:"):
                m = re.search(r"(\d+)", line)
                if m:
                    img_ref = int(m.group(1))

        # Skip shots without narration (unless explicitly "none")
        if not narration and img_ref == 0:
            continue

        if has_character is None:  # model omitted the field: read the frame description
            has_character = _infer_character(visual_desc, action)

        shot = Shot(
            narration=narration or "none",
            shot_size=shot_type,
            place=place or _shorten(visual_desc) or "a rainy city street",
            action=action or "she watches",
            story_object=story_object or "no object",
            narrative_role=_coerce(role, NARRATIVE_ROLES, "") or _infer_role(img_ref),
            camera=camera_movement.replace(" ", "_"),
            character=has_character,
            seconds=duration,
            visual_description=visual_desc,
            audio_notes=audio_notes,
            transition=transition,
            image_prompt_ref=img_ref,
        )
        shots.append(shot)

    # Parse image generation prompts section
    for m in re.finditer(r"Image\s+(\d+)\s*:\s*(.+?)(?=\nImage\s+\d+\s*:|$)", img_text,
                         re.IGNORECASE | re.DOTALL):
        num = int(m.group(1))
        prompt = " ".join(m.group(2).split())  # collapse whitespace
        image_prompts[num] = prompt

    return shots, image_prompts


_CHARACTER_TOKENS = re.compile(
    r"\b(she|her|hers|herself|girl|girls|hood|hooded|protagonist)\b", re.IGNORECASE)


def _strip_signature(text: str, character: str) -> str:
    """Remove the identity signature from a prompt for a frame the shot list says is empty."""
    if character:
        text = re.sub(r",?\s*" + re.escape(character) + r"(,\s*seen as part of the scene)?",
                      "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*([.,])", r"\1", text)  # 'wonder, .' -> 'wonder.'
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


# A prompt that asks for two scenes in one frame gets exactly that: FLUX renders a diptych, a
# split-screen or a diagonal composite, and the seam lands in the middle of a six-second hold.
# "Composition splits horizontal: tree upper left, boy lower right" shipped in a published short.
_SPLIT_LAYOUT = re.compile(
    r"\b(diptych|triptych|split[- ]?screen|split composition|splits?|panels?|collage|"
    r"side[- ]by[- ]side|two (?:scenes|images|panels)|divided (?:in|into) two|"
    r"composite of two)\b", re.IGNORECASE)

# A corrective clause contains the very words it forbids ("no split panels, no diptych"), so a
# match only counts when it is not being ruled out.
_NEGATIONS = ("no ", "not ", "never ", "without ", "avoid ", "single ", "one ")


def _asks_for_split(text: str) -> bool:
    """True when the text *requests* a split/panel frame rather than forbidding one."""
    for match in _SPLIT_LAYOUT.finditer(text or ""):
        window = (text or "")[max(0, match.start() - 28):match.start()].lower()
        if not any(negation in window for negation in _NEGATIONS):
            return True
    return False


def _unsplit_layout(text: str) -> tuple[str, bool]:
    """Drop a request for a split/panel composition and ask for one frame instead.

    Sentence-level, because the offending clause is always its own sentence ("Composition splits
    horizontal: ..."). Returns (text, whether anything was rewritten).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [sentence for sentence in sentences if not _asks_for_split(sentence)]
    if len(kept) == len(sentences):
        return text, False
    cleaned = " ".join(kept).strip()
    return (f"{cleaned} One continuous single frame, no split panels, no diptych.").strip(), True


def anchor_prompt(prompt: str, character: bool, character_sig: str = "") -> str:
    """Force the art direction and identity anchor onto a planner-written image prompt.

    The detailed planner writes prompts that share no style words between frames ('Cyberpunk art
    style', 'Vibrant cyberpunk art') and name no costume accent, so identity has nothing to
    hold onto. Order is preserved from `Shot.prompt()`: style, scene, identity last, format last.
    The shot's `character` flag wins over what the prompt happens to mention. `character_sig` is
    the arc's own signature, so each arc anchors to its own protagonist.
    """
    text = " ".join((prompt or "").split()).strip()
    text, unsplit = _unsplit_layout(text)
    if unsplit:
        logger.info("Rewrote a split/panel composition request into one continuous frame")
    text = re.sub(r"[,.]?\s*engagement hook of [^.,]+[.,]?", "", text, flags=re.IGNORECASE)
    if not character:
        text = _strip_signature(text, character_sig)
    text = " ".join(text.split()).strip()
    low = text.lower()

    style = "" if ("cel-shaded" in low or "lineart" in low) else STYLE.rstrip(".") + "."
    body = text.rstrip(" .") + "." if text else ""
    trail = []
    if character and character_sig and character_sig.lower() not in low:
        trail.append(f"{character_sig}, seen as part of the scene.")
    if "9:16" not in text and "portrait" not in low:
        trail.append("Vertical 9:16 portrait composition.")
    if "no text" not in low:
        trail.append("No text, no watermark.")
    return " ".join(part for part in (style, body, *trail) if part)


def _infer_character(*fields: str) -> bool:
    """True when the frame itself contains the girl.

    Only frame-describing fields are passed in (visual description, action) — narration is not a
    reliable signal because "her brother" appears in lines about shots she is not in.
    """
    return bool(_CHARACTER_TOKENS.search(" ".join(f for f in fields if f)))


def _shorten(text: str, limit: int = 120) -> str:
    """Trim at a word boundary.

    The old `visual_desc[:60]` cut mid-word ("partially hidden by her ho") and threw away the
    location whenever the description started with the subject.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.") or text[:limit]


def _infer_role(img_ref: int) -> str:
    """Best-effort role when the planner did not state one: beats open, complicate, resolve."""
    roles = ["establish_context", "introduce_subject", "build_tension", "deliver_payload",
             "emotional_beat", "evidence", "resolution", "call_to_action"]
    return roles[min(img_ref - 1, len(roles) - 1)]


def _map_shot_type(raw: str) -> str:
    """Map freeform shot-type strings to known shot_size keys."""
    low = raw.lower()
    mapping = {
        "extreme close-up": "insert", "close-up": "close_up",
        "medium close": "medium_close", "medium": "medium",
        "medium wide": "medium_wide", "wide": "wide",
        "establishing": "establishing", "over the shoulder": "over_shoulder",
    }
    for key, val in mapping.items():
        if key in low:
            return val
    return "medium"


def compose_image_prompts(shots: list[Shot], image_prompts: dict[int, str],
                          character: str = "") -> list[str]:
    """Build per-shot image prompts, falling back to the standard prompt builder."""
    results = []
    for s in shots:
        if s.image_generation_prompt:
            results.append(s.image_generation_prompt)
        elif s.image_prompt_ref and s.image_prompt_ref in image_prompts:
            results.append(image_prompts[s.image_prompt_ref])
        else:
            results.append(s.prompt(character))
    return results


def _prompt_for_ref(image_prompts: dict, ref: int) -> str:
    """Prompts are keyed by int in memory and by str once they round-trip through state.json."""
    return image_prompts.get(ref) or image_prompts.get(str(ref)) or ""


def unique_prompts(shots: list[Shot], image_prompts: dict | None = None,
                   character: str = "") -> tuple[list[str], list[int]]:
    """Return (prompts to render once, per-shot index into them).

    Shots that share an `image_prompt_ref` are rendered once and that frame is held for each of
    their beats: image quota (10k neurons/day) is the scarce resource, not shots.
    """
    image_prompts = image_prompts or {}
    ordered: list[str] = []
    slots: list[int] = []
    seen: dict[str, int] = {}
    for s in shots:
        # The ref wins over the shot's own prompt: a reused ref must render as one image.
        prompt = ((_prompt_for_ref(image_prompts, s.image_prompt_ref) if s.image_prompt_ref else "")
                  or s.image_generation_prompt or s.prompt(character))
        index = seen.get(prompt)
        if index is None:
            index = len(ordered)
            seen[prompt] = index
            ordered.append(prompt)
        slots.append(index)
    return ordered, slots


MIN_SHOTS = 8  # below this the cut rate drops out of the 1-3s short-form window


def fit_shot_count(shots: list[Shot], want: int) -> list[Shot]:
    """Trim an over-long plan to `want` shots, keeping the final shot as the resolution.

    Short plans are never padded with invented beats — the count is reported instead.
    """
    if not want or want <= 0:
        return shots
    if len(shots) <= want:
        if len(shots) < want:
            logger.info("Planner wrote %d shots, %d wanted; keeping the arc it wrote",
                        len(shots), want)
        return shots
    kept = shots[:want - 1] + [shots[-1]]
    logger.info("Planner wrote %d shots; trimmed to %d (resolution kept)", len(shots), want)
    return kept


def reuse_image_refs(shots: list[Shot], budget: int) -> list[Shot]:
    """Fold the plan onto at most `budget` distinct images.

    Consecutive beats share a frame, split as evenly as possible, so the short keeps its full
    shot list and its resolution inside the daily allowance. Piling every overflow beat onto one
    image would hold the same frame for five cuts; pairing neighbours costs one repeated frame
    per pair instead. A plan that already fits is left alone.
    """
    if budget <= 0 or len(shots) <= budget:
        return shots

    counter = max((s.image_prompt_ref for s in shots), default=0)  # every shot needs a ref
    for s in shots:
        if not s.image_prompt_ref:
            counter += 1
            s.image_prompt_ref = counter

    base, extra = divmod(len(shots), budget)
    folded = 0
    pos = 0
    for group in range(budget):
        size = base + (1 if group < extra else 0)
        chunk = shots[pos:pos + size]
        ref = chunk[0].image_prompt_ref
        for s in chunk[1:]:
            s.image_prompt_ref = ref
            folded += 1
        pos += size
    logger.info("%d shots folded onto %d image(s); %d beat(s) share a frame",
                len(shots), budget, folded)
    return shots


def _spoken_words(shots: list[Shot]) -> float:
    """Average words per spoken line — the difference between narration and captions."""
    spoken = [s.speech for s in shots if s.speech]
    return sum(len(line.split()) for line in spoken) / len(spoken) if spoken else 0.0


def _total_spoken_words(shots: list[Shot]) -> int:
    """How many words the narrator has to read: at a fixed pace, this is the runtime."""
    return sum(len(s.speech.split()) for s in shots if s.speech)


def validate_story_position(shots: list[Shot], part: int, total_parts: int) -> list[str]:
    """Check that a plan ends the way its position in the arc requires.

    The failure this catches is quiet: a single-part short whose last beat is a threshold instead
    of a payoff reads as unfinished, and a middle part that resolves leaves nothing to continue.
    """
    if not shots:
        return []
    warnings: list[str] = []
    last_role = shots[-1].narrative_role
    if part >= total_parts and last_role != "resolution":
        warnings.append(f"final part ends on '{last_role}', not a resolution")
    if part < total_parts and last_role == "resolution":
        warnings.append(f"part {part} of {total_parts} resolves early; nothing left to continue")
    return warnings


def _declared_count(text: str) -> int:
    """The planner's own 'Short 1: N shots' line — it is asked for, so it is checked."""
    match = re.search(r"Short\s+\d+\s*:\s*(\d+)\s*shots?", text or "", re.IGNORECASE)
    return int(match.group(1)) if match else 0


# Words that cannot end a title. Cutting a narration clause at a fixed word count lands on one
# of these often enough that "Rain drummed on the tin roof while" became a real upload title.
_DANGLING_ENDINGS = frozenset("""
a an the and but or nor so yet for because as if when while that which who whom whose
of to in on at by with from into onto over under after before during through across about
against between without within upon than then where why how
""".split())


def _strip_dangling(words: list[str]) -> list[str]:
    """Drop words that leave a phrase hanging off the end of a title."""
    out = list(words)
    while out and out[-1].strip("\"'`,").lower() in _DANGLING_ENDINGS:
        out.pop()
    return out


# Words that cannot open a standalone title phrase: a clause starting with one of these is
# dependent ("because she had promised"), not a name.
_LEADING_DEPENDENT = frozenset("""
and but or nor so yet because when while if though although as since until after before then
than that which who whom whose
""".split())


def title_from_line(line: str, *, max_words: int = 7) -> str:
    """A short, complete title phrase from a narration line, or "" when there isn't one.

    Splits on clause punctuation, trims to `max_words`, then drops a dangling conjunction or
    preposition — a fixed word count on its own cut mid-phrase. Clauses that open on a
    subordinator are dependent, so they are skipped rather than published as a title.
    """
    for clause in re.split(r"[,.;:!?—–]|\s-\s", line or ""):
        words = _strip_dangling(clause.split()[:max_words])
        if len(words) < 2:
            continue
        if words[0].strip("\"'`").lower() in _LEADING_DEPENDENT:
            continue
        return " ".join(words).strip("\"'` ")
    return ""


def _validate_title(title: str) -> str:
    """Return `title` if it can be published as a story name, else "".

    A title is a name, not a sentence: it must have letters, stay short, and not stop on a word
    that leaves it hanging ("The Girl Who Waited Because").
    """
    title = re.sub(r"\s+", " ", (title or "").strip().strip("*_`\"'")).rstrip(".")
    if not title or not re.search(r"[A-Za-z]", title):
        return ""
    words = title.split()
    if len(title) > 90 or len(words) > 12:
        return ""
    if words[-1].strip("\"'`,").lower() in _DANGLING_ENDINGS:
        return ""
    return title


def parse_title(text: str) -> str:
    """Read the story title the planner states as `Title: ...`.

    The title is what the upload is named after, so it has to be the story's own name rather
    than the series id (which used to be a date, producing "Anime Short #1/2 — 2026-11-12").
    Returns "" when the planner did not state one; the caller then falls back to a title built
    from the opening narration.
    """
    match = re.search(r"^\s*\**\s*title\s*\**\s*[:=]\s*\**\s*(.+?)\s*\**\s*$",
                      text or "", re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    title = _validate_title(match.group(1))
    if not title:
        logger.info("Planner title unusable (%r); ignoring it", match.group(1)[:60])
    return title


# The hook is spoken before the story moves, so it has to fit in one breath at 158 wpm: 16 words is
# ~6 s, a sixth of a 55 s short. Below 7 words it cannot carry who + what is wrong; above 26 it is
# no longer a hook, it is the first paragraph.
HOOK_MIN_WORDS = 7
HOOK_MAX_WORDS = 26

# Openers that spend the hook on setup instead of tension. Each one was measured as a weak opening
# in the hook research: they ask the viewer to wait for the story to start, and the swipe is cheaper
# than the wait. `This is the story of` is deliberately absent — it works when the clause after it
# carries a contradiction, which the prompt requires and the length floor leaves room for.
_HOOK_BANNED_OPENERS = (
    "once upon a time", "in a world", "in a land", "long ago", "far away", "far in the future",
    "have you ever", "what if", "imagine if", "this is a story about",
)


def parse_hook(text: str) -> str:
    """Read the narrator's opening line the planner states as `Hook: "..."`.

    The hook is not decoration: it is the first sentence of Shot 1, so a plan that states a hook
    and then opens the shot on weather has not been followed. Returns "" when the planner stated
    none, which `_plan_gaps` reports rather than guessing one.
    """
    match = re.search(r"^\s*[-*#\s]*\**\s*hook\s*\**\s*[:=]\s*(.+?)\s*$",
                      text or "", re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    hook = match.group(1).strip().strip("*_`").strip()
    while len(hook) >= 2 and hook[0] in "\"'\u201c" and hook[-1] in "\"'\u201d":
        hook = hook[1:-1].strip()
    hook = " ".join(hook.split())
    if not re.search(r"[A-Za-z]", hook):
        return ""
    return hook


def _hook_problem(hook: str) -> str:
    """Why this hook will not hold a scrolling viewer, phrased as a repair instruction, or ""."""
    words = hook.split()
    low = hook.lower().strip(" \"'\u201c\u201d")
    for opener in _HOOK_BANNED_OPENERS:
        if low.startswith(opener):
            return (f"the Hook opens with `{opener}`, which is setup rather than tension: start "
                    f"inside this story with a person, what is wrong, and what it will cost")
    if len(words) < HOOK_MIN_WORDS:
        return (f"the Hook is {len(words)} words: it needs {HOOK_MIN_WORDS}-16 to name someone, "
                f"the thing that is wrong, and the cost")
    if len(words) > HOOK_MAX_WORDS:
        return (f"the Hook is {len(words)} words, too long to be spoken before the story moves: "
                f"cut it to {HOOK_MIN_WORDS}-16 and leave the explanation for later shots")
    return ""


def _opens_with(narration: str, hook: str, *, words: int = 6) -> bool:
    """True when `narration` starts with the hook (punctuation, case and quotes ignored)."""
    def norm(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()

    heard, stated = norm(narration), norm(hook)
    if not heard or not stated:
        return False
    span = min(words, len(stated), len(heard))
    return span >= 4 and heard[:span] == stated[:span]


TITLE_INSTRUCTIONS = (
    "You name short films. Reply with the title only — no quotes, no 'Title:', no explanation. "
    "Two to five words, concrete, naming this story rather than describing the picture.")


def _clean_title_line(text: str) -> str:
    """Pull a title out of a reply that was asked for one and gave a line."""
    line = next((raw.strip() for raw in (text or "").splitlines() if raw.strip()), "")
    line = line.strip("*_`\"' ")
    if ":" in line:  # "Sure, here is a title: The Lantern" -> "The Lantern"
        line = line.rsplit(":", 1)[1]
    line = line.split(".")[0]
    return _validate_title(line)


def request_title(shots: list[Shot], api_key: str, *, gemini_key: str = "",
                  attempts: int = 2) -> str:
    """Ask for the story's name when the plan came back without a `Title:` line.

    Cutting a title out of narration prose is lossy — it produced "Mira has spent three long
    years trapping". One short follow-up is cheaper than putting that name on every part of the
    arc, so the planner is asked directly, and the prose cut is only the last resort.
    """
    beats = "\n".join(f"- {shot.speech}" for shot in shots if shot.speech)[:1500]
    if not beats:
        return ""
    messages = [
        {"role": "system", "content": TITLE_INSTRUCTIONS},
        {"role": "user", "content": f"Story beats:\n{beats}\n\nReply with the title only."},
    ]
    for attempt in range(1, attempts + 1):
        # A title is short by design, so the storyboard length floor must not apply here.
        title = _clean_title_line(
            _plan_request(messages, api_key, min_words=1, gemini_key=gemini_key) or "")
        if title:
            logger.info("Titled by a follow-up request: %s", title)
            return title
        logger.warning("Title request %d/%d returned nothing usable", attempt, attempts)
    return ""


# Roles whose `Story object` makes the best title, in order: the payload beat names the thing the
# story is about, the emotional beat names what it cost.
_TITLE_ROLES = ("deliver_payload", "emotional_beat", "evidence", "resolution", "build_tension")


_SMALL_WORDS = frozenset("a an the of and or in on at to from with for among over under".split())


def _title_case(phrase: str) -> str:
    """`the iron lantern` -> `The Iron Lantern`: a story name, not a sentence."""
    words = phrase.split()
    return " ".join(
        word if (i and word.lower() in _SMALL_WORDS) else word[:1].upper() + word[1:]
        for i, word in enumerate(words))


def title_from_plan(shots: list[Shot]) -> str:
    """Name the story from the plan's own structured fields when the planner did not name it.

    `Story object` is the one field written as a short noun phrase ("iron lantern", "fresh yellow
    ribbon among faded ones"), which is what a title needs. Prose only ever yields a cut sentence,
    and a word count cannot tell a noun phrase from a gerund.
    """
    for role in _TITLE_ROLES:
        for shot in shots:
            if shot.narrative_role != role:
                continue
            words = " ".join((shot.story_object or "").split()).split()[:3]
            while words and (words[-1].strip(",").lower() in _DANGLING_ENDINGS
                             or words[-1].lower().endswith("ing")):
                words.pop()
            if not words:
                continue
            phrase = " ".join(words).strip("\"'` ,")
            if not phrase or phrase.lower() in ("no object", "none", "n a", "n/a"):
                continue
            if phrase.split()[0].lower() not in ("the", "a", "an"):
                phrase = f"The {phrase}"
            return _validate_title(_title_case(phrase))
    return ""


def fallback_title(shots: list[Shot]) -> str:
    """Build a title from the opening narration when the planner did not name the story.

    Uses the first shot whose narration yields a complete phrase, so an unnamed story still
    uploads as something readable instead of a date — or, worse, a fragment cut mid-clause.
    Returns "" when no shot carries usable narration.
    """
    for shot in shots:
        title = title_from_line(shot.speech)
        if title:
            return title
    return ""


def _declared_arc(text: str, lo: int, hi: int) -> int:
    """Read the arc length the planner chose (`Arc: N parts`), clamped to what the run allows."""
    match = (re.search(r"^\s*\**\s*arc\s*\**\s*[:=]\s*\**\s*(\d+)\s*parts?\b", text, re.I | re.M)
             or re.search(r"\barc\s+length\s*[:=]?\s*(\d+)\s*parts?\b", text, re.I))
    if not match:
        logger.info("Planner did not state an arc length; keeping the story to one part")
        return lo
    return max(lo, min(int(match.group(1)), hi))


def _signature_instruction(character: str) -> str:
    """The per-arc identity instruction the detailed planner is told to follow.

    With a signature it is pinned verbatim (so the anchor step has a stable string to hold on
    to); without one it only asks for a consistent protagonist, for callers that plan ad hoc.
    """
    if character:
        return (f"Any prompt that shows the protagonist must describe them with exactly this "
                f"signature, placed last and verbatim: `{character}`. Do not paraphrase the "
                f"signature and do not add meta-language about engagement hooks — describe only "
                f"what is in frame.")
    return ("Any prompt that shows a character on screen must keep one consistent protagonist "
            "with a single short signature placed last. Do not add meta-language about "
            "engagement hooks — describe only what is in frame.")


def plan_shorts_detailed(
    *,
    theme: str,
    character: str = "",
    api_key: str = "",
    want: int = 0,
    target_seconds: float = 0.0,
    part: int = 1,
    total_parts: int | None = 1,
    story_so_far: list[str] | None = None,
    parts_range: tuple[int, int] = (1, MAX_PARTS),
    gemini_key: str = "",
    max_words: int = 0,
) -> tuple[list[Shot], dict[int, str], bool, int]:
    """Run the detailed prompt and parse its structured output.

    `want` is the shot count the pacing needs. It is stated in the request, checked against the
    planner's own count line, and the plan is trimmed to fit — an unpinned count came back as 7-8
    shots when the pacing wanted 13.

    `part` / `total_parts` decide whether the short is allowed to end: a middle part must close on
    a hook, only the last part may resolve. They also decide what the opening line has to do: a
    new story or a single-part short states the story's question, while a later part re-hooks a
    viewer who never saw part 1 instead of recapping it. `story_so_far` is the narration of the
    earlier parts, so a later part continues the same story — and the same narrator — instead of
    inventing a new one.

    `max_words` is the hard word ceiling for this part. Narration is read at a fixed pace, so the
    word count is the runtime: a plan above the ceiling is re-asked with the cut named, because the
    renderer will extend the video rather than speed the narrator up.

    `total_parts=None` means `SERIES_PARTS=auto`: the planner decides the arc length itself, from
    `parts_range`, and states it as an `Arc: N parts` line in the same reply — one request, not two.
    Its answer comes back as the fourth element and is then pinned in `state.json`.

    Returns (shots, image_prompts, used_fallback, total_parts, title).
    """
    decide_arc = total_parts is None
    if decide_arc:
        lo, hi = parts_range
        lo = max(1, min(int(lo), MAX_PARTS))
        hi = max(lo, min(int(hi), MAX_PARTS))
        total_parts, part = hi, 1
    else:
        lo = hi = 1
        total_parts = max(1, min(int(total_parts), MAX_PARTS))
        part = max(1, min(int(part), total_parts))
    system = PLAN_SYSTEM_PROMPT_DETAILED.replace("__CHARACTER_SIGNATURE__",
                                                 _signature_instruction(character))
    user = f"Theme/story idea: {theme}"
    if want:
        system += (f"\n\n## This request\nProduce exactly {want} shots and exactly {want} numbered "
                   f"image prompts, numbered 1-{want} with no gaps, reusing an image number when "
                   f"two shots share a place. State 'Short 1: {want} shots, {want} images.' as the "
                   f"count line.")
        user += f"\nShots required: {want}"
        if target_seconds:
            total_words = words_for_duration(target_seconds)
            per_shot = max(12, round(total_words / want))
            system += (f"\n\n## Narration length\nThe voice-over is read aloud over about "
                       f"{target_seconds:.0f} seconds at roughly {NARRATOR_WPM} words per minute, so "
                       f"the whole short needs about {total_words} words: {per_shot - 3} to "
                       f"{per_shot + 4} words in each of the {want} Narration lines. Count them. A "
                       f"4-8 word line is a caption and will be rejected; the story has to be "
                       f"told, so spend the words on tension, consequence and what she is "
                       f"thinking — never on describing the picture, which the viewer can see. The "
                       f"Hook is the first sentence of shot 1, so shot 1 spends its words on the "
                       f"hook plus one clause that moves the story.")
            user += (f"\nNarration: {per_shot - 3}-{per_shot + 4} words per shot, "
                     f"about {total_words} words in total.")
        if max_words:
            system += (f"\n\n## Word ceiling\nThe narration is read at a fixed pace and is never "
                       f"sped up, so every word you write is a second of video. This short has room "
                       f"for at most {max_words} spoken words in total. Do not exceed it: tighten "
                       f"the lines rather than cutting a beat's meaning, and if the story genuinely "
                       f"cannot be told inside it, raise the part count (the arc) instead of "
                       f"writing longer parts.")
            user += f"\nHard ceiling: {max_words} spoken words in total."

    # Position in the arc decides the ending: hook in the middle, resolution only at the end.
    if decide_arc:
        system += (f"\n\n## Story position\nThis is part 1 of an arc whose length only you can "
                   f"judge. First decide how many shorts this story actually needs, from {lo} to "
                   f"{hi}, and state it as the line `Arc: N parts` directly above the count line. "
                   f"Then write part 1 only. A tight, single-scene idea is 1 part; an idea that "
                   f"needs a hunt, a descent and a reunion is 3. If N is 1, the last shot must "
                   f"resolve the story (Narrative role: resolution). If N is more than 1, part 1 "
                   f"does not resolve: end on a hook, so the last shot is build_tension or "
                   f"deliver_payload, never resolution.\n\nThe Hook opens the arc: it puts the "
                   f"story's question on the table, and part 1 only deepens it.")
        user += (f"\nStory position: part 1 of an arc you choose ({lo}-{hi} parts)."
                 f"\nStart with `Arc: N parts`, then the count line.")
    elif total_parts > 1 and part < total_parts:
        ending = (f"This is part {part} of {total_parts}, NOT the end — the story continues in "
                  f"part {part + 1}. Do not resolve it: end on a hook, and the last shot must be "
                  "build_tension or deliver_payload, never resolution.")
        system += (f"\n\n## Story position\nThis short is part {part} of {total_parts}. {ending}"
                   f"\n\nThe Hook is a re-hook, not a recap: assume the viewer has not seen part "
                   f"{part - 1}. Give one clause of load-bearing context — who they are, what they "
                   f"are chasing, what it now costs them — then the new turn. No 'previously on', "
                   f"no montage of earlier beats, never more than one clause of background.")
        user += f"\nStory position: part {part} of {total_parts}."
    else:
        ending = ("This is the final part: the last shot must resolve the story "
                  "(Narrative role: resolution).")
        if total_parts > 1:
            system += (f"\n\n## Story position\nThis short is part {part} of {total_parts}. "
                       f"{ending}\n\nThe Hook still has to stand on its own, because most viewers "
                       f"will meet the story here: one clause of who and what is at stake, then "
                       f"the ending's turn.")
            user += f"\nStory position: part {part} of {total_parts}."
        else:
            system += (f"\n\n## Story position\nThis is a complete single-part short. {ending}"
                       f"\n\nThe Hook poses the story's question and the last shot answers it.")
            user += "\nStory position: the whole story, one part."
    if story_so_far:
        # Two parts of recap: a third part that only sees the last 12 lines forgets how the story
        # opened. ~24 lines is ~350 input tokens, far inside the completion budget.
        recap = "\n".join(f"- {line}" for line in story_so_far[-24:])
        user += (f"\n\nStory so far (already told, do not repeat):\n{recap}\n\nKeep the same "
                 f"narrator: the person and tense of these lines decide the person and tense of "
                 f"the Hook and every Narration line here — a story that switches from 'he' to 'I' "
                 f"between parts sounds like a different narrator took over.")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    text = _plan_request(messages, api_key, gemini_key=gemini_key)

    if not text:
        logger.warning("Detailed planner unavailable; falling back to basic plan_shots")
        return [], {}, True, 1 if decide_arc else total_parts, ""

    shots, image_prompts = parse_detailed_response(text)

    # A reply that is missing detail is re-asked, not salvaged: the parser has a default for every
    # field, so a skipped `Action` or `Story object` looks exactly like one the planner wrote, and
    # the fabrication then ships in the video. Two bounded rounds, and a round that does not reduce
    # the gap count is discarded — an unattended run cannot spend the day retrying.
    for round_no in range(1, PLAN_REPAIR_ROUNDS + 1):
        gaps = _plan_gaps(text, shots, image_prompts, want=want, need_arc=decide_arc,
                          max_words=max_words)
        if not gaps:
            break
        logger.info("Plan is incomplete (round %d/%d): %s",
                    round_no, PLAN_REPAIR_ROUNDS, "; ".join(gaps))
        repaired = _request_repair(messages, text, gaps, api_key, gemini_key=gemini_key)
        if not repaired:
            break
        shots2, image_prompts2 = parse_detailed_response(repaired)
        remaining = _plan_gaps(repaired, shots2, image_prompts2, want=want, need_arc=decide_arc,
                               max_words=max_words)
        if len(remaining) >= len(gaps):
            logger.warning("Repair left %d gap(s) of %d; keeping the first plan",
                           len(remaining), len(gaps))
            break
        shots, image_prompts, text = shots2, image_prompts2, repaired

    if not shots:
        logger.warning("Detailed planner returned 0 shots; falling back")
        return [], {}, True, 1 if decide_arc else total_parts, ""

    if not image_prompts:
        logger.warning("Planner returned no IMAGE GENERATION PROMPTS section; "
                       "every shot will use the built prompt instead")

    if decide_arc:
        total_parts = _declared_arc(text, lo, hi)
        logger.info("Arc length chosen by the planner: %d part(s) of at most %d",
                    total_parts, hi)

    declared = _declared_count(text)
    if want and declared and declared != len(shots):
        logger.warning("Planner declared %d shots but wrote %d", declared, len(shots))
    shots = fit_shot_count(shots, want)

    # If some shots lack their image prompt but refs exist, fill them in
    for s in shots:
        if s.image_prompt_ref and s.image_prompt_ref in image_prompts:
            s.image_generation_prompt = image_prompts[s.image_prompt_ref]

    # A prompt can serve several shots, so anchor by the strongest need: character if any shot
    # using that ref has the girl on screen.
    needs_character: dict[int, bool] = {}
    for s in shots:
        if s.image_prompt_ref:
            needs_character[s.image_prompt_ref] = (
                needs_character.get(s.image_prompt_ref, False) or s.character)
    image_prompts = {ref: anchor_prompt(p, needs_character.get(ref, False), character)
                     for ref, p in image_prompts.items()}
    for s in shots:
        if s.image_prompt_ref in image_prompts:
            s.image_generation_prompt = image_prompts[s.image_prompt_ref]

    logger.info("Detailed scene plan: %d shots, %d unique image prompts",
                len(shots), len(image_prompts))
    # The hook is the first thing a viewer hears, so it is logged on its own line: a run whose
    # first line is scene-setting is visible in the log instead of only in the retention graph.
    hook = parse_hook(text)
    if hook:
        logger.info("Opening hook: %s", hook)
    else:
        logger.warning("Plan stated no `Hook:` line; the short opens on whatever shot 1 says")
    title = (parse_title(text) or request_title(shots, api_key, gemini_key=gemini_key)
             or title_from_plan(shots) or fallback_title(shots))
    if title:
        logger.info("Story title: %s", title)
    else:
        logger.warning("Planner named no story and no narration to derive one from")
    return shots, image_prompts, False, total_parts, title


PLANNER_MAX_TOKENS = 4096  # Workers AI path only; the OpenRouter path is deliberately uncapped
# No output cap on OpenRouter: every free model there is now a reasoning model, and a 4096-token
# cap is spent on thinking before the storyboard starts. Measured on the same request: cap=4096 ->
# 0 usable words (reasoning ate it), cap=16000 -> 6,955 reasoning tokens then a complete 13-shot
# plan. Uncapped is the only setting that survives models whose thinking length varies per request.
OPENROUTER_MAX_TOKENS = 0  # 0 = send no max_tokens at all


def _plan_request(messages, api_key: str, *, min_words: int = 40,
                  gemini_key: str = "") -> str | None:
    """Planning goes to Antigravity first, then OpenRouter.

    Antigravity is the first-priority text provider whenever credentials are configured; OpenRouter's free
    models and then the local template are the fallbacks. Workers AI's 10,000 free neurons per day
    stay reserved for image generation: a storyboard is the largest text request the pipeline
    makes, and spending neurons on it is what starved the renders. No planner in this module
    accepts Workers AI credentials, so the split is enforced by the signatures rather than by this
    comment.
    """
    if gemini_key:
        content = gemini.text(messages, key=gemini_key, min_words=min_words)
        if content:
            logger.info("Planner answered via Antigravity (%d words)", len(content.split()))
            return content
        logger.warning("Antigravity planner produced nothing; trying the next text provider")
    if not api_key:
        logger.warning("No text provider credential (Antigravity OAuth / OpenRouter): "
                       "planning falls back to the local template")
        return None
    text = _openrouter(messages, api_key, min_words=min_words)
    if not text:
        logger.warning("Planner produced nothing; falling back to the local template")
    return text


def _choice_detail(body: dict) -> str:
    """Why a completion looks the way it does: truncated, thinking-only, or fine."""
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return (f"finish_reason={choice.get('finish_reason')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"reasoning_tokens={reasoning}")


PLANNER_MODELS = ("@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-3.1-8b-instruct-fp8")

# A 429 from Workers AI is not always a rate limit: 4006 is the daily free allocation
# (10,000 neurons per account, shared by every model) being spent. That one is worth stopping on.
QUOTA_MARKERS = ("daily free allocation", "neurons", "insufficient credit", "quota")

# Walked in order: the largest free models first, the auto-router last as a hail-mary.
# `openrouter/free` can land on a reasoning model that returns empty content, and individual
# free upstreams 429 whenever they are busy, so one model is never enough.
DEFAULT_OPENROUTER_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "cohere/north-mini-code:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
)


def api_error_detail(resp: requests.Response) -> str:
    """`code: message` from a Workers AI or OpenRouter error body.

    Logging just the status code made "quota spent", "rate limited" and "bad key" look
    identical in an unattended run's log.
    """
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:200] or "no body"
    for key in ("errors", "error"):
        value = body.get(key)
        first = value[0] if isinstance(value, list) and value else value
        if isinstance(first, dict) and first.get("message"):
            return f"{first.get('code')}: {first['message']}"[:300]
    return (resp.text or "").strip()[:200] or "no body"


def is_quota_error(detail: str) -> bool:
    low = (detail or "").lower()
    return any(marker in low for marker in QUOTA_MARKERS)


def _workers_ai(messages, cfa_token: str, account_id: str) -> str | None:
    if not (cfa_token and account_id):
        return None
    for model in PLANNER_MODELS:
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
                headers={"Authorization": f"Bearer {cfa_token}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": PLANNER_MAX_TOKENS, "temperature": 0.8},
                timeout=120)
        except requests.RequestException as exc:
            logger.warning("Workers AI planner %s request failed: %s", model, exc)
            continue
        if r.status_code != 200:
            detail = api_error_detail(r)
            logger.warning("Workers AI planner %s -> %s (%s)", model, r.status_code, detail)
            if is_quota_error(detail):
                logger.warning("Workers AI free allocation is spent (per account, not per model): "
                               "skipping the other model and falling back to OpenRouter")
                return None
            continue
        text = (r.json().get("result") or {}).get("response")
        if text and len(text.split()) > 40:
            logger.info("Planner answered via Workers AI %s", model)
            return text
        logger.warning("Workers AI planner %s returned %d usable word(s)",
                       model, len((text or "").split()))
    return None


def _openrouter(messages, api_key: str, *, min_words: int = 40) -> str | None:
    """Ask the OpenRouter model list for a completion, rejecting degenerate replies.

    `min_words` guards against a thinking model that spends its whole budget on reasoning and
    returns two words of answer. It is a parameter because not every request wants a storyboard:
    a title is three words by design, and the 40-word floor used to make every title reply look
    like a failure.
    """
    if not api_key:
        return None
    raw = os.environ.get("OPENROUTER_MODELS", "").strip()
    models = tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_OPENROUTER_MODELS
    for model in models:
        payload = {"model": model, "messages": messages, "temperature": 0.8}
        if OPENROUTER_MAX_TOKENS:
            payload["max_tokens"] = OPENROUTER_MAX_TOKENS
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=900)  # uncapped reasoning can think for minutes before writing
        except requests.RequestException as exc:
            logger.warning("OpenRouter planner %s request failed: %s", model, exc)
            continue
        if r.status_code != 200:
            logger.warning("OpenRouter planner %s -> %s (%s)", model, r.status_code,
                           api_error_detail(r))
            continue
        body = r.json()
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
        detail = _choice_detail(body)
        if content and len(content.split()) >= min_words:
            if (body.get("choices") or [{}])[0].get("finish_reason") == "length":
                logger.warning("OpenRouter planner %s reply was cut off (%s); the storyboard may "
                               "be short of the end", model, detail)
            logger.info("Planner answered via OpenRouter %s (%d words, %s)",
                        model, len(content.split()), detail)
            return content
        logger.warning("OpenRouter planner %s returned %d usable word(s), wanted %d (%s)",
                       model, len((content or "").split()), min_words, detail)
    logger.warning("OpenRouter planner produced nothing for %s", ", ".join(models))
    return None
