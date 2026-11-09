"""State management for duplicate prevention and daily quota tracking."""

import hashlib
import json
from datetime import date


def compute_fingerprint(story: str, image_urls: list[str]) -> str:
    """Deterministic fingerprint of story + generated image URLs."""
    raw = "|".join([story.strip().lower()] + [u.strip().lower() for u in image_urls])
    return "hash32:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def today_key() -> str:
    return date.today().isoformat()


def parse_state(raw: str | None) -> dict:
    if not raw:
        return {"processed": [], "daily": {}}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"processed": [], "daily": {}}


def save_state(state: dict) -> str:
    return json.dumps(state, separators=(",", ":"))
