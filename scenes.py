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
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

MAX_PARTS = 4

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
CHARACTER = "a girl in a black hooded raincoat with a red scarf"

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

    def prompt(self) -> str:
        """Scene and action lead; the character signature is short and comes last."""
        size = SHOT_SIZES.get(self.shot_size, "Medium shot")
        obj = "" if self.story_object in ("", "no object") else f" Key object: {self.story_object}."
        who = f" {CHARACTER}, seen as part of the scene." if self.character else ""
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
    picks = rows if count >= len(rows) else rows[:count]
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
    """13 shots for 45s: ~3.5s per cut, inside the 2-4s short-form change rate."""
    return max(8, min(16, int(round(duration_seconds / 3.4))))


def plan_shots(
    *,
    theme: str,
    count: int,
    cfa_token: str = "",
    account_id: str = "",
    api_key: str = "",
) -> tuple[list[Shot], bool]:
    """Return (shots, used_fallback)."""
    system = PLAN_SYSTEM_PROMPT.format(
        n=count, sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERA_MOVES),
        roles=", ".join(NARRATIVE_ROLES))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Write the {count} shots for this story: {theme}"},
    ]

    text = _workers_ai(messages, cfa_token, account_id)
    if not text:
        text = _openrouter(messages, api_key)
    shots = parse_plan(text, count) if text else []
    if len(shots) >= 6:
        logger.info("Scene plan: %d shots (%d without the character)",
                    len(shots), sum(1 for s in shots if not s.character))
        return shots, False

    logger.warning("Scene planner returned %d usable shots; using the template plan", len(shots))
    return _template_shots(count), True


def plan_arc(*, theme: str, parts: int, per_part: int,
             cfa_token: str = "", account_id: str = "", api_key: str = ""
             ) -> tuple[list[list[Shot]], bool]:
    parts = max(1, min(parts, MAX_PARTS))
    if parts == 1:
        shots, fallback = plan_shots(theme=theme, count=per_part, cfa_token=cfa_token,
                                     account_id=account_id, api_key=api_key)
        return [shots], fallback

    head = PLAN_SYSTEM_PROMPT.format(
        n=per_part, sizes=", ".join(SHOT_SIZES), cameras=", ".join(CAMERA_MOVES),
            roles=", ".join(NARRATIVE_ROLES))
    messages = [
        {"role": "system", "content": f"{head}\nWrite {parts} parts. Put a line 'PART <n>' before "
                                      f"each part's shots. Keep the story continuous across parts."},
        {"role": "user", "content": f"Write PART 1 through PART {parts} for: {theme}"},
    ]
    text = _workers_ai(messages, cfa_token, account_id) or _openrouter(messages, api_key)
    if text:
        parsed = _split_parts(text, parts, per_part)
        if parsed and all(len(p) >= 4 for p in parsed):
            return parsed, False
    shots, fallback = plan_shots(theme=theme, count=per_part, cfa_token=cfa_token,
                                 account_id=account_id, api_key=api_key)
    return [shots], fallback


def _workers_ai(messages, cfa_token: str, account_id: str) -> str | None:
    if not (cfa_token and account_id):
        return None
    for model in ("@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-3.1-8b-instruct-fp8"):
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
                headers={"Authorization": f"Bearer {cfa_token}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": 1200, "temperature": 0.8},
                timeout=90)
        except requests.RequestException as exc:
            logger.warning("Workers AI planner request failed: %s", exc)
            continue
        if r.status_code != 200:
            logger.warning("Workers AI planner %s -> %s", model, r.status_code)
            continue
        text = (r.json().get("result") or {}).get("response")
        if text and len(text.split()) > 40:
            return text
    return None


def _openrouter(messages, api_key: str) -> str | None:
    if not api_key:
        return None
    raw = os.environ.get("OPENROUTER_MODELS", "").strip()
    models = tuple(m.strip() for m in raw.split(",") if m.strip()) or (
        "google/gemma-4-31b-it:free", "openrouter/free")
    for model in models:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": 1200,
                      "temperature": 0.8},
                timeout=90)
        except requests.RequestException as exc:
            logger.warning("OpenRouter planner %s failed: %s", model, exc)
            continue
        if r.status_code != 200:
            continue
        content = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content")
        if content and len(content.split()) > 40:
            return content
    return None
