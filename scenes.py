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
- Narration: "<2-3 full sentences of connected prose: what she does, what it costs her, what she has just understood>"     (or "none" if silent)
- Audio notes: "<music/SFX/ambient or silence>"
- Transition: <cut|fade|dissolve>
- Image prompt ref: Image N
```

Begin with a count line: "Short 1: N shots, M images."

**Before the count line, state the story's name:**
```
Title: <3-7 words, the story's own name, no date, no "Anime Short", no part number>
```
The title is what the video is published as, so it must name this story ("The Lantern She
Left Burning"), not describe the format. Do not put "Part 1" in it — the publisher adds the
part number itself.

**After all shots, add this section header exactly:**
```
--- IMAGE GENERATION PROMPTS ---
```
Then list every image prompt, one per line, numbered sequentially (Image 1:, Image 2:, …). Write
**70-110 words** per prompt, because a viewer holds this frame for six or seven seconds and the
frame has to keep rewarding that attention. Each prompt must be fully self-contained and contain,
in this order: (1) art style, (2) setting, time, lighting and mood, (3) composition as a vertical
9:16 portrait, (4) emotional beat, (5) one or two concrete physical details that sell the place
(weather on a surface, wear on an object, what the light is landing on).
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
) -> tuple[list[Shot], bool]:
    """Return (shots, used_fallback). Text planning takes no Workers AI credentials."""
    system = PLAN_SYSTEM_PROMPT.format(
        n=count, sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERA_MOVES),
        roles=", ".join(NARRATIVE_ROLES))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Write the {count} shots for this story: {theme}"},
    ]

    text = _plan_request(messages, api_key)
    shots = parse_plan(text, count) if text else []
    if len(shots) >= 6:
        logger.info("Scene plan: %d shots (%d without the character)",
                    len(shots), sum(1 for s in shots if not s.character))
        return shots, False

    logger.warning("Scene planner returned %d usable shots; using the template plan", len(shots))
    return _template_shots(count), True


def plan_arc(*, theme: str, parts: int, per_part: int, api_key: str = ""
             ) -> tuple[list[list[Shot]], bool]:
    parts = max(1, min(parts, MAX_PARTS))
    if parts == 1:
        shots, fallback = plan_shots(theme=theme, count=per_part, api_key=api_key)
        return [shots], fallback

    head = PLAN_SYSTEM_PROMPT.format(
        n=per_part, sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERA_MOVES),
            roles=", ".join(NARRATIVE_ROLES))
    messages = [
        {"role": "system", "content": f"{head}\nWrite {parts} parts. Put a line 'PART <n>' before "
                                      f"each part's shots. Keep the story continuous across parts."},
        {"role": "user", "content": f"Write PART 1 through PART {parts} for: {theme}"},
    ]
    text = _plan_request(messages, api_key)
    if text:
        parsed = _split_parts(text, parts, per_part)
        if parsed and all(len(p) >= 4 for p in parsed):
            return parsed, False
    shots, fallback = plan_shots(theme=theme, count=per_part, api_key=api_key)
    return [shots], fallback


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


def anchor_prompt(prompt: str, character: bool, character_sig: str = "") -> str:
    """Force the art direction and identity anchor onto a planner-written image prompt.

    The detailed planner writes prompts that share no style words between frames ('Cyberpunk art
    style', 'Vibrant cyberpunk art') and name no costume accent, so identity has nothing to
    hold onto. Order is preserved from `Shot.prompt()`: style, scene, identity last, format last.
    The shot's `character` flag wins over what the prompt happens to mention. `character_sig` is
    the arc's own signature, so each arc anchors to its own protagonist.
    """
    text = " ".join((prompt or "").split()).strip()
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
    title = match.group(1).strip().strip("\"'`*")
    # A title is a name, not a sentence: drop a trailing period and any stray markdown.
    title = re.sub(r"\s+", " ", title).rstrip(".")
    if len(title) > 90 or len(title.split()) > 12:
        logger.info("Planner title too long (%d chars); ignoring it", len(title))
        return ""
    return title


def fallback_title(shots: list[Shot]) -> str:
    """Build a title from the opening narration when the planner did not name the story.

    Takes the first clause of the first spoken line and title-cases it, so an unnamed story
    still uploads as something readable instead of a date.
    """
    for shot in shots:
        line = shot.speech
        if not line:
            continue
        clause = re.split(r"[,.;:!?]", line, maxsplit=1)[0].strip()
        words = clause.split()[:7]
        if words:
            return " ".join(words).strip("\"'")
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
) -> tuple[list[Shot], dict[int, str], bool, int]:
    """Run the detailed prompt and parse its structured output.

    `want` is the shot count the pacing needs. It is stated in the request, checked against the
    planner's own count line, and the plan is trimmed to fit — an unpinned count came back as 7-8
    shots when the pacing wanted 13.

    `part` / `total_parts` decide whether the short is allowed to end: a middle part must close on
    a hook, only the last part may resolve. `story_so_far` is the narration of the earlier parts,
    so a later part continues the same story instead of inventing a new one.

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
                       f"thinking — never on describing the picture, which the viewer can see.")
            user += (f"\nNarration: {per_shot - 3}-{per_shot + 4} words per shot, "
                     f"about {total_words} words in total.")

    # Position in the arc decides the ending: hook in the middle, resolution only at the end.
    if decide_arc:
        system += (f"\n\n## Story position\nThis is part 1 of an arc whose length only you can "
                   f"judge. First decide how many shorts this story actually needs, from {lo} to "
                   f"{hi}, and state it as the line `Arc: N parts` directly above the count line. "
                   f"Then write part 1 only. A tight, single-scene idea is 1 part; an idea that "
                   f"needs a hunt, a descent and a reunion is 3. If N is 1, the last shot must "
                   f"resolve the story (Narrative role: resolution). If N is more than 1, part 1 "
                   f"does not resolve: end on a hook, so the last shot is build_tension or "
                   f"deliver_payload, never resolution.")
        user += (f"\nStory position: part 1 of an arc you choose ({lo}-{hi} parts)."
                 f"\nStart with `Arc: N parts`, then the count line.")
    elif total_parts > 1 and part < total_parts:
        ending = (f"This is part {part} of {total_parts}, NOT the end — the story continues in "
                  f"part {part + 1}. Do not resolve it: end on a hook, and the last shot must be "
                  "build_tension or deliver_payload, never resolution.")
        system += (f"\n\n## Story position\nThis short is part {part} of {total_parts}. {ending}")
        user += f"\nStory position: part {part} of {total_parts}."
    else:
        ending = ("This is the final part: the last shot must resolve the story "
                  "(Narrative role: resolution).")
        if total_parts > 1:
            system += (f"\n\n## Story position\nThis short is part {part} of {total_parts}. "
                       f"{ending}")
            user += f"\nStory position: part {part} of {total_parts}."
        else:
            system += f"\n\n## Story position\nThis is a complete single-part short. {ending}"
            user += "\nStory position: the whole story, one part."
    if story_so_far:
        # Two parts of recap: a third part that only sees the last 12 lines forgets how the story
        # opened. ~24 lines is ~350 input tokens, far inside the completion budget.
        recap = "\n".join(f"- {line}" for line in story_so_far[-24:])
        user += f"\n\nStory so far (already told, do not repeat):\n{recap}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    text = _plan_request(messages, api_key)

    if not text:
        logger.warning("Detailed planner unavailable; falling back to basic plan_shots")
        return [], {}, True, 1 if decide_arc else total_parts, ""

    shots, image_prompts = parse_detailed_response(text)

    # One bounded repair attempt: a plan far short of the pacing window, or narration written as
    # 8-word captions instead of a story, is worth re-asking for — but never more than once, an
    # unattended run cannot spend the day retrying.
    repair = ""
    if want and 0 < len(shots) < want - 2:
        repair = (f"That was {len(shots)} shots. Output the whole storyboard again with exactly "
                  f"{want} shots and exactly {want} numbered image prompts.")
    elif shots and _spoken_words(shots) < 11:
        logger.info("Narration averages %.1f words per shot; asking once for prose",
                    _spoken_words(shots))
        repair = ("The narration reads as captions: it averages "
                  f"{_spoken_words(shots):.0f} words per shot. Keep the same shots and image "
                  "prompts, but rewrite every Narration line as 2-3 connected sentences of a "
                  "storyteller's voice-over, 16-20 words each, each line continuing the one "
                  "before it. Do not describe the picture.")
    if repair and text:
        logger.info("Repairing the plan: %s", repair.split(".")[0])
        retry = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": repair},
        ]
        text2 = _plan_request(retry, api_key)
        if text2:
            shots2, image_prompts2 = parse_detailed_response(text2)
            better_count = want and len(shots2) > len(shots)
            better_words = shots2 and _spoken_words(shots2) > _spoken_words(shots) + 3
            if better_count or better_words:
                shots, image_prompts, text = shots2, image_prompts2, text2

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
    title = parse_title(text) or fallback_title(shots)
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


def _plan_request(messages, api_key: str) -> str | None:
    """Planning goes to OpenRouter only.

    Workers AI's 10,000 free neurons per day are reserved for image generation: a storyboard is
    the largest text request the pipeline makes, and spending neurons on it is what starved the
    renders. No planner in this module accepts Workers AI credentials, so the split is enforced
    by the signatures rather than by this comment.
    """
    if not api_key:
        logger.warning("No OPENROUTER_API_KEY: planning falls back to the local template")
        return None
    text = _openrouter(messages, api_key)
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


def _openrouter(messages, api_key: str) -> str | None:
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
        if content and len(content.split()) > 40:
            if (body.get("choices") or [{}])[0].get("finish_reason") == "length":
                logger.warning("OpenRouter planner %s reply was cut off (%s); the storyboard may "
                               "be short of the end", model, detail)
            logger.info("Planner answered via OpenRouter %s (%d words, %s)",
                        model, len(content.split()), detail)
            return content
        logger.warning("OpenRouter planner %s returned %d usable word(s) (%s)",
                       model, len((content or "").split()), detail)
    logger.warning("OpenRouter planner produced nothing for %s", ", ".join(models))
    return None
