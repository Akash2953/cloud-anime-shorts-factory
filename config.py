"""Configuration and constants for the Cloud Anime Shorts Factory."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from story import MAX_PARTS


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so secrets never have to live on the command line.

    Real environment variables win over file values.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Limits:
    """Hard limits enforcing the zero-spend, one-short-per-day policy."""

    max_images_per_short: int = 6
    max_image_attempts_per_day: int = 8
    max_daily_uploads: int = 1
    openrouter_free_daily_limit: int = 50
    openrouter_free_minute_limit: int = 20


@dataclass(frozen=True)
class Quotas:
    """Runtime quota counters persisted in state."""

    images_attempted_today: int = 0
    stories_requested_today: int = 0
    uploads_today: int = 0
    last_reset_date: str = ""


@dataclass(frozen=True)
class Defaults:
    model_ids: dict = field(default_factory=lambda: {
        "image": "black-forest-labs/FLUX.1-schnell-Free",
    })
    workflow_url_template: str = (
        "https://api.github.com/repos/{owner}/{repo}/actions/workflows"
    )


def load_config() -> dict:
    load_dotenv()
    return {
        "CFAI_TOKEN": os.environ.get("CFAI_TOKEN", ""),
        # Cloudflare account ID for Workers AI. R2_ACCOUNT_ID kept as a fallback
        # name so existing GitHub secrets keep working; no R2 credentials needed.
        "CF_ACCOUNT_ID": os.environ.get("CF_ACCOUNT_ID") or os.environ.get("R2_ACCOUNT_ID", ""),
        "YOUTUBE_CLIENT_ID": os.environ.get("YOUTUBE_CLIENT_ID", ""),
        "YOUTUBE_CLIENT_SECRET": os.environ.get("YOUTUBE_CLIENT_SECRET", ""),
        "YOUTUBE_REFRESH_TOKEN": os.environ.get("YOUTUBE_REFRESH_TOKEN", ""),
        # private until you trust the output; set to "public" for hands-off publishing.
        "PUBLISH_PRIVACY": os.environ.get("PUBLISH_PRIVACY", "private").lower(),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
        # Optional local gateway (OmniRoute). Not used by the daily pipeline today.
        "OMNIROUTE_API_KEY": os.environ.get("OMNIROUTE_API_KEY", ""),
        "OMNIROUTE_BASE": os.environ.get("OMNIROUTE_BASE", "http://localhost:20128"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HF_ENABLED": os.environ.get("HF_ENABLED", "false").lower() == "true",
        "GITHUB_OWNER": os.environ.get("GITHUB_OWNER", ""),
        "GITHUB_REPO": os.environ.get("GITHUB_REPO", ""),
        "GITHUB_PAT": os.environ.get("GITHUB_PAT", ""),
        "FFMPEG_AVAILABLE": True,
        "KOKORO_VOICE": os.environ.get("KOKORO_VOICE", "af_heart"),
        # Daily image ceiling. Raise for interactive development; the unattended run keeps 8.
        "MAX_IMAGE_ATTEMPTS": os.environ.get("MAX_IMAGE_ATTEMPTS") or 8,
        # Image strategy: edit (canonical + conditioned edits) | sheet (one sheet, cropped) | fresh
        "IMAGE_STRATEGY": os.environ.get("IMAGE_STRATEGY", "edit"),
        # Runtime is emergent, not fixed: speech is only re-timed to land inside this window.
        "MIN_SECONDS": float(os.environ.get("MIN_SECONDS", 30)),
        "MAX_SECONDS": float(os.environ.get("MAX_SECONDS", 68)),
        # A story may span up to MAX_PARTS shorts, one uploaded per day.
        "SERIES_PARTS": max(1, min(int(os.environ.get("SERIES_PARTS") or 1), MAX_PARTS)),
        # Shots per part. 0 = derive from PREFERRED_SECONDS (~3.4s per cut).
        "SHOTS_PER_PART": int(os.environ.get("SHOTS_PER_PART") or 0),
        "PREFERRED_SECONDS": float(os.environ.get("PREFERRED_SECONDS") or 45),
    }
