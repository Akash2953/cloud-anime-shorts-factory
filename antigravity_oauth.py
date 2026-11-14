"""OAuth helpers for Antigravity's Google Code Assist API."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import platform
import re
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

ANTIGRAVITY_OAUTH_CREDENTIAL = "__antigravity_oauth__"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
LOAD_CODE_ASSIST_URL = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
IDE_RELEASE_FEED_URL = (
    "https://antigravity-auto-updater-974169037036.us-central1.run.app/releases"
)
ANTIGRAVITY_IDE_FALLBACK_VERSION = "2.5.5"
SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
)
REQUEST_TIMEOUT = 30
_IDE_VERSION_CACHE: str | None = None
logger = logging.getLogger(__name__)


class AntigravityOAuthError(RuntimeError):
    """Raised when Google OAuth or Antigravity project discovery fails."""


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str
    expires_in: int
    refresh_token: str = ""


def create_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge.decode("ascii").rstrip("=")


def antigravity_user_agent() -> str:
    override = os.environ.get("ANTIGRAVITY_USER_AGENT", "").strip()
    if override:
        return override

    global _IDE_VERSION_CACHE
    if _IDE_VERSION_CACHE is None:
        version = ANTIGRAVITY_IDE_FALLBACK_VERSION
        try:
            response = requests.get(
                IDE_RELEASE_FEED_URL,
                headers={"Accept": "application/json"},
                timeout=5,
            )
            response.raise_for_status()
            releases = response.json()
            versions = (
                [entry.get("version") for entry in releases if isinstance(entry, dict)]
                if isinstance(releases, list)
                else []
            )
            valid_versions = [
                candidate for candidate in versions
                if isinstance(candidate, str) and re.fullmatch(r"\d+\.\d+\.\d+", candidate)
            ]
            if valid_versions:
                version = max(
                    valid_versions,
                    key=lambda candidate: tuple(int(part) for part in candidate.split(".")),
                )
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Could not resolve current Antigravity IDE version: %s", exc)
        _IDE_VERSION_CACHE = version

    return f"antigravity/ide/{_IDE_VERSION_CACHE} darwin/arm64"


def build_authorization_url(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    query = urlencode({
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': ' '.join(SCOPES),
        'access_type': 'offline',
        'prompt': 'consent',
        'state': state,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
    })
    return f"{AUTHORIZE_URL}?{query}"


def _token_request(
    data: dict[str, str],
    *,
    operation: str,
) -> OAuthTokens:
    response = requests.post(
        TOKEN_URL,
        data=data,
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if not response.ok:
        error = body.get("error_description") or body.get("error") or "request rejected"
        raise AntigravityOAuthError(f"{operation} failed ({response.status_code}): {error}")
    if not body.get("access_token"):
        raise AntigravityOAuthError(f"{operation} response did not contain an access token")
    try:
        expires_in = max(1, int(body.get("expires_in", 3600)))
    except (TypeError, ValueError):
        expires_in = 3600
    return OAuthTokens(
        access_token=str(body["access_token"]),
        expires_in=expires_in,
        refresh_token=str(body.get("refresh_token") or ""),
    )


def exchange_authorization_code(
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    client_id: str,
    client_secret: str = "",
) -> OAuthTokens:
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret
    return _token_request(data, operation="Authorization-code exchange")


def refresh_access_token(
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str = "",
) -> OAuthTokens:
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        data["client_secret"] = client_secret
    return _token_request(data, operation="Antigravity token refresh")


def _platform_enum() -> int:
    if platform.system() == "Windows":
        return 5
    if platform.system() == "Darwin":
        return 2 if platform.machine().lower() in ("arm64", "aarch64") else 1
    if platform.system() == "Linux":
        return 4 if platform.machine().lower() in ("arm64", "aarch64") else 3
    return 0


def extract_project_id(body: dict) -> str:
    project = body.get("cloudaicompanionProject")
    if isinstance(project, str):
        return project.strip()
    if isinstance(project, dict):
        value = project.get("id")
        return value.strip() if isinstance(value, str) else ""
    return ""


def discover_project_id(access_token: str) -> str:
    response = requests.post(
        LOAD_CODE_ASSIST_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": antigravity_user_agent(),
        },
        json={"metadata": {"ideType": 9, "platform": _platform_enum(), "pluginType": 2}},
        timeout=REQUEST_TIMEOUT,
    )
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not response.ok:
        error = body.get("error", {})
        detail = error.get("message") if isinstance(error, dict) else ""
        raise AntigravityOAuthError(
            f"Antigravity project discovery failed ({response.status_code})"
            + (f": {detail}" if detail else "")
        )
    return extract_project_id(body if isinstance(body, dict) else {})
