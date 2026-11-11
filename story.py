"""Story generation.

Provider chain, first success wins:
  1. Cloudflare Workers AI (same CFAI_TOKEN as the images)
  2. OpenRouter free tier
  3. Local template (always available, keeps the pipeline unblocked)
"""

import logging
import os
import random
import re

import requests

logger = logging.getLogger(__name__)

# Verified against this account: both are available on the Workers Free plan.
# @cf/zai-org/glm-5.3-flash returns 403 "not available on the Workers Free plan".
WORKERS_AI_MODELS: tuple[str, ...] = (
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct-fp8",
)

SYSTEM_PROMPT = (
    "You write 50-80 word anime-style short-video narration scripts in English. "
    "Keep it engaging, punchy, and suitable for a vertical video caption. "
    "Reply with the narration only: no title, no quotes, no preamble."
)

TEMPLATES = [
    "In a world where {theme}, nothing is what it seems. Follow our protagonist as they uncover secrets hidden in plain sight.",
    "Meet {protagonist}, a {role} who discovers a mysterious artifact that changes everything. What starts as an ordinary day turns into an extraordinary journey through {setting}.",
    "{setting} holds ancient secrets. When {protagonist} stumbles upon them, the line between reality and imagination blurs forever.",
]

_MAX_CHARS = 700

BEAT_SYSTEM_PROMPT = (
    "You write narration for a vertical anime short. Output exactly {n} numbered lines "
    "(1., 2., ...) and nothing else. Each line is one sentence of 18-26 words that will be "
    "spoken over one image. Every line must be a complete sentence ending with punctuation. "
    "Lines must flow as one continuous story, use vivid concrete imagery a painter could "
    "draw, and avoid meta commentary, headers, quotes, or emoji."
)

MAX_PARTS = 7

# Reasoning-style free models leak their chain of thought into the reply. These markers
# identify narration lines that are really meta text and must never reach the voice track.
_META = re.compile(
    "(?:analy[sz]e|format|requirement|instruction|constraint|output exactly|"
    "nothing else|topic|step |word count|numbered lines)",
    re.IGNORECASE,
)
_MARKDOWN = ("**", "##", "```", "|", "->", "***", "[", "]")


def _looks_like_narration(line: str) -> bool:
    """True only for a clean spoken sentence: terminated, unformatted, 4-48 words."""
    if not re.search(r"[.!?][\"'”’]?$", line):
        return False
    if any(tok in line for tok in _MARKDOWN):
        return False
    if _META.search(line):
        return False
    return 4 <= len(line.split()) <= 48

ARC_SYSTEM_PROMPT = (
    "You write narration for a series of vertical anime shorts. Output exactly {parts} parts and "
    "nothing else. Each part starts with a line \"PART <number>\" followed by exactly {beats} "
    "numbered lines. Each line is one complete sentence of 18-26 words, spoken over one image. "
    "All parts continue one single story in order, and each part ends on a hook. Use vivid "
    "concrete imagery a painter could draw. No meta commentary, headers, quotes, or emoji."
)


def generate_arc(
    *,
    theme: str,
    parts: int = 1,
    beats_per_part: int = 6,
    target_words_per_part: int = 130,
    cfa_token: str = "",
    account_id: str = "",
    api_key: str = "",
) -> tuple[list[list[str]], bool]:
    """Return (parts, used_fallback); each part is a list of narration beats.

    A story that cannot fit inside one short is split into up to MAX_PARTS shorts, one
    uploaded per day, all drawn from the same stored series.
    """
    parts = max(1, min(parts, MAX_PARTS))
    if parts == 1:
        beats, fallback = generate_beats(
            theme=theme, count=beats_per_part, target_words=target_words_per_part,
            cfa_token=cfa_token, account_id=account_id, api_key=api_key,
        )
        return [beats], fallback

    instruction = (
        ARC_SYSTEM_PROMPT.format(parts=parts, beats=beats_per_part)
        + f" About {target_words_per_part} words per part. Theme: {theme}."
    )
    best: list[list[str]] = []
    for attempt in range(1, 3):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": (
                f"Write PART 1 through PART {parts} now." if attempt == 1 else
                f"Write exactly {parts} parts, each starting with 'PART n' and exactly "
                f"{beats_per_part} numbered lines."
            )},
        ]
        raw = None
        if cfa_token and account_id:
            raw = _call_workers_ai_messages(messages, cfa_token, account_id, max_tokens=900)
        if not raw and api_key:
            ok, out = _call_openrouter_messages(messages, api_key)
            raw = out if ok else None
        if not raw:
            break
        candidate = _split_arc(raw, parts, beats_per_part)
        if sum(len(p) for p in candidate) > sum(len(p) for p in best):
            best = candidate
        if len(best) >= 1 and all(len(p) >= 4 for p in best) and len(best) == parts:
            break
        logger.warning("Arc pass %d gave beats %s; retrying", attempt, [len(p) for p in best])

    usable = [p for p in best if len(p) >= 3]
    if usable:
        logger.info("Got %d-part arc: %s beats", len(usable), [len(p) for p in usable])
        return usable, False

    logger.warning("Arc generation failed; falling back to template beats")
    return [_template_beats(theme, beats_per_part)], True


def _split_arc(text: str, parts: int, beats_per_part: int) -> list[list[str]]:
    """Split an LLM reply into PART n sections, degrading to one part if unmarked."""
    chunks = re.split(r"^[\W_]*(?:PART|Part)\s*\d+[\W_]*$", text or "", flags=re.MULTILINE)
    if len(chunks) < 2:  # no PART markers — treat the whole reply as part one
        return [_split_beats(text, beats_per_part)]

    result: list[list[str]] = []
    for body in chunks:
        beats = _split_beats(body, beats_per_part)
        if beats:
            result.append(beats)
        if len(result) == parts:
            break
    return result or [_split_beats(text, beats_per_part)]


def generate_beats(
    *,
    theme: str,
    count: int = 6,
    target_words: int = 140,
    cfa_token: str = "",
    account_id: str = "",
    api_key: str = "",
) -> tuple[list[str], bool]:
    """Return (beats, used_fallback) where each beat is one narration line."""
    prompt = (
        BEAT_SYSTEM_PROMPT.format(n=count)
        + f" Total length about {target_words} words. Theme: {theme}."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Write the {count} narration lines now."},
    ]

    text = None
    best: list[str] = []
    for attempt in range(1, 3):  # one retry if the model under-delivers
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Write the {count} narration lines now." if attempt == 1 else
                f"Write exactly {count} numbered narration lines, {count} lines only."
            )},
        ]
        raw = None
        if cfa_token and account_id:
            raw = _call_workers_ai_messages(messages, cfa_token, account_id)
        if not raw and api_key:
            ok, out = _call_openrouter_messages(messages, api_key)
            raw = out if ok else None
        if not raw:
            break
        candidate = _split_beats(raw, count)
        if len(candidate) > len(best):
            best = candidate
            text = raw
        if len(best) >= count:
            break
        logger.warning("Beat pass %d returned %d/%d lines; retrying", attempt, len(best), count)

    if len(best) >= 3:
        logger.info("Got %d narration beats from cloud", len(best))
        return best, False
    logger.warning("Cloud returned only %d usable beats, falling back", len(best))

    logger.info("Falling back to local story template beats")
    return _template_beats(theme, count), True


def _split_beats(text: str, count: int) -> list[str]:
    """Pull numbered lines out of an LLM reply, tolerating bullets and stray formatting."""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:\d+[.)\]]|[-*•])\s*", "", line)
        line = _clean(line)
        # Always split on sentence ends: a numbered line may pack several beats, and each
        # sentence is judged on its own so meta text cannot ride along with narration.
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = _clean(sentence)
            if _looks_like_narration(sentence):
                lines.append(sentence)
    if len(lines) > count:
        lines = lines[:count]
    # LLMs sometimes stop mid-sentence; never narrate a truncated beat.
    while len(lines) > 3 and not re.search(r"[.!?\"']$", lines[-1]):
        lines.pop()
    return lines


def _template_beats(theme: str, count: int) -> list[str]:
    motif = (theme.split(",")[0] or "neon").strip()
    base = [
        f"A city of {motif} keeps its promises to nobody, and the night knows why.",
        "A lone figure walks the wet street, following a light no one else can see.",
        "Lanterns drift above the rooftops, carrying wishes that were never spoken aloud.",
        "Somewhere behind the neon, an old door opens onto a world without maps.",
        "The silence breaks, and everything the city forgot begins to glow again.",
        "By dawn the light is gone, but the one who saw it will never be the same.",
    ]
    return (base * ((count // len(base)) + 1))[:count]


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip("\"'“”‘’ ")[:_MAX_CHARS].strip()


def generate_story(
    *,
    theme: str,
    api_key: str = "",
    attempts: int = 0,
    cfa_token: str = "",
    account_id: str = "",
) -> tuple[str, bool]:
    """Return (story, used_fallback)."""
    if cfa_token and account_id:
        text = _call_workers_ai(theme, cfa_token, account_id)
        if text:
            logger.info("Story from Cloudflare Workers AI")
            return text, False
        logger.warning("Workers AI story generation failed, trying next provider")

    if api_key and attempts < _daily_limit_guard():
        ok, text = _call_openrouter(theme, api_key)
        if ok:
            logger.info("Story from OpenRouter")
            return _clean(text) or _pick_template(theme), False

    logger.info("Falling back to local story template")
    return _pick_template(theme), True


def _call_workers_ai(theme: str, cfa_token: str, account_id: str) -> str | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Write the narration script. Theme: {theme}."},
    ]
    return _call_workers_ai_messages(messages, cfa_token, account_id)


def _call_workers_ai_messages(
    messages: list[dict], cfa_token: str, account_id: str, *, max_tokens: int = 400
) -> str | None:
    for model in WORKERS_AI_MODELS:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {cfa_token}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.9},
                timeout=90,
            )
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Workers AI story request failed on %s: %s", model, exc)
            continue

        if resp.status_code in (401, 403):
            logger.warning("Workers AI story model rejected (%s): %s", resp.status_code, resp.text[:160])
            continue
        if resp.status_code != 200:
            logger.warning("Workers AI story model %s returned %s: %s", model, resp.status_code, resp.text[:160])
            continue

        result = resp.json().get("result", {})
        text = result.get("response") if isinstance(result, dict) else None
        text = _clean(text or "")
        if len(text.split()) >= 10:
            return text
        logger.warning("Workers AI story model %s returned too little text: %r", model, text[:80])
    return None


def _call_openrouter(theme: str, api_key: str) -> tuple[bool, str | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Write the narration script. Theme: {theme}."},
    ]
    return _call_openrouter_messages(messages, api_key)


# Free OpenRouter routing is unreliable: "openrouter/free" can land on a reasoning model
# that spends the whole token budget thinking and returns content=None. Try a list instead.
DEFAULT_OPENROUTER_MODELS = (
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
)


def _openrouter_models() -> tuple[str, ...]:
    raw = os.environ.get("OPENROUTER_MODELS", "").strip()
    return tuple(m.strip() for m in raw.split(",") if m.strip()) if raw else DEFAULT_OPENROUTER_MODELS


def _call_openrouter_messages(messages: list[dict], api_key: str) -> tuple[bool, str | None]:
    for model in _openrouter_models():
        ok, text = _call_openrouter_once(messages, api_key, model)
        if ok and text:
            return True, text
    return False, None


def _call_openrouter_once(messages: list[dict], api_key: str, model: str) -> tuple[bool, str | None]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max(400, int(os.environ.get("OPENROUTER_MAX_TOKENS") or 400)),
        "temperature": 0.9,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cloud-anime-shorts.example",
        "X-Title": "Cloud Anime Shorts",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=45)
        if resp.status_code != 200:
            logger.warning("OpenRouter returned %s: %s", resp.status_code, resp.text[:200])
            return False, None
        choice = resp.json().get("choices", [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        cleaned = _clean(text) or None
        if not cleaned:
            logger.warning("OpenRouter %s returned no content (finish=%s) — trying next model",
                           model, choice.get("finish_reason"))
        return True, cleaned
        if resp.status_code == 429:
            logger.warning("OpenRouter %s rate-limited; trying next model", model)
            return False, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenRouter %s failed: %s", model, exc)
        return False, None


def _pick_template(theme: str) -> str:
    tmpl = random.choice(TEMPLATES)  # noqa: S311
    return tmpl.format(theme=theme, protagonist="Ren", role="astronomer", setting="Tokyo at midnight")


def _daily_limit_guard() -> int:
    # OpenRouter free tier limit enforced externally; caller decrements before use.
    return 50
