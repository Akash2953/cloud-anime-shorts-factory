
"""Gemini API and Antigravity OAuth provider.

Two entry points, both plain `requests` (no extra SDK, so requirements.txt is unchanged):

    text()   chat completion for the storyboard / narration planner
    image()  text-to-image and image-to-image (identity-conditioned) rendering

The pipeline uses a Gemini API key or Antigravity OAuth first, and falls back on any failure —
unavailable model, exhausted quota, safety block, timeout — to OpenRouter (text) or Workers AI
(images). Antigravity OAuth refreshes access tokens lazily from its configured refresh token.

Errors are classified into the vocabulary the image pipeline already stops on
(`quota_exhausted`, `auth_401`, `model_missing`, `status_N`) so a Gemini refusal is logged in the
same terms as a Workers AI one.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from io import BytesIO
from dataclasses import dataclass

import requests
from PIL import Image

from antigravity_oauth import (
    ANTIGRAVITY_OAUTH_CREDENTIAL,
    AntigravityOAuthError,
    antigravity_user_agent,
    refresh_access_token,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
ANTIGRAVITY_BASE_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal"

# Current, generally-available models. `GEMINI_TEXT_MODELS` / `GEMINI_IMAGE_MODEL`
# override them (comma separated), so a renamed or upgraded model never needs a code change.
DEFAULT_TEXT_MODELS: tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.0-flash")
DEFAULT_IMAGE_MODELS: tuple[str, ...] = ("gemini-2.5-flash-image",)

# Antigravity OAuth models. `ANTIGRAVITY_TEXT_MODEL` / `ANTIGRAVITY_IMAGE_MODEL`
# override them (comma separated), so a renamed or upgraded model never needs a code change.
DEFAULT_ANTIGRAVITY_TEXT_MODELS: tuple[str, ...] = ("gemini-3-flash",)
DEFAULT_ANTIGRAVITY_IMAGE_MODELS: tuple[str, ...] = ("gemini-3.1-flash-image",)
_ACCESS_TOKEN_CACHE: dict[str, str | float] = {
    "refresh_token": "",
    "access_token": "",
    "expires_at": 0.0,
}
_ACCESS_TOKEN_LOCK = threading.Lock()

# `gemini-2.5-flash-image` documents ["IMAGE"]; the older preview image models require both.
DEFAULT_IMAGE_MODALITIES: tuple[str, ...] = ("IMAGE",)
_WIDENED_IMAGE_MODALITIES = ("TEXT", "IMAGE")
ANTIGRAVITY_IMAGE_ASPECT_RATIO = "9:16"

TEXT_TIMEOUT = 180
IMAGE_TIMEOUT = 300

# A 429 from Google is either a rate limit or the plan's quota; both mean "stop asking this key".
QUOTA_MARKERS = ("resource_exhausted", "quota", "rate limit", "billing", "exceeded")
AUTH_MARKERS = ("api_key_invalid", "api key not valid", "permission_denied", "unauthenticated")


@dataclass
class Result:
    """Image result, shaped like `images.ImageResult` so callers can pass it straight through."""

    ok: bool
    data: bytes | None = None
    error: str | None = None


def api_key() -> str:
    """`GEMINI_API_KEY`, falling back to the `GOOGLE_API_KEY` alias AI Studio exports."""
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def antigravity_access_token() -> str:
    """Return the configured Antigravity OAuth access token without logging it."""
    return os.environ.get("ANTIGRAVITY_ACCESS_TOKEN", "").strip()


def _uses_antigravity(key: str) -> bool:
    if not key:
        return False
    configured_key = os.environ.get("ANTIGRAVITY_API_KEY", "").strip()
    configured_access = antigravity_access_token()
    configured_refresh = os.environ.get("ANTIGRAVITY_REFRESH_TOKEN", "").strip()
    return (
        key == ANTIGRAVITY_OAUTH_CREDENTIAL
        or bool(configured_key and key == configured_key)
        or bool(configured_access and key == configured_access)
        or bool(configured_refresh and key == configured_refresh)
    )


def _resolve_antigravity_access_token(key: str) -> str:
    configured_key = os.environ.get("ANTIGRAVITY_API_KEY", "").strip()
    configured_token = antigravity_access_token()
    refresh_token = os.environ.get("ANTIGRAVITY_REFRESH_TOKEN", "").strip()
    if configured_token and key == configured_token:
        return configured_token
    if configured_key and key == configured_key and key != refresh_token:
        return configured_key
    if not refresh_token:
        if configured_token:
            return configured_token
        logger.warning("Antigravity OAuth is selected but ANTIGRAVITY_REFRESH_TOKEN is missing")
        return ""

    client_id = os.environ.get("ANTIGRAVITY_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("ANTIGRAVITY_OAUTH_CLIENT_SECRET", "").strip()
    if not client_id:
        logger.warning("Antigravity token refresh requires ANTIGRAVITY_OAUTH_CLIENT_ID")
        return ""

    with _ACCESS_TOKEN_LOCK:
        if (
            _ACCESS_TOKEN_CACHE["refresh_token"] == refresh_token
            and float(_ACCESS_TOKEN_CACHE["expires_at"]) > time.monotonic() + 60
        ):
            return str(_ACCESS_TOKEN_CACHE["access_token"])
        try:
            tokens = refresh_access_token(
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
            )
        except (AntigravityOAuthError, requests.RequestException) as exc:
            logger.warning("Antigravity OAuth token refresh failed: %s", exc)
            return ""
        if tokens.refresh_token and tokens.refresh_token != refresh_token:
            logger.warning(
                "Google rotated the Antigravity refresh token; update "
                "ANTIGRAVITY_REFRESH_TOKEN in the environment or GitHub secret"
            )
        _ACCESS_TOKEN_CACHE.update({
            "refresh_token": refresh_token,
            "access_token": tokens.access_token,
            "expires_at": time.monotonic() + tokens.expires_in,
        })
        return tokens.access_token


def text_models(key: str = "") -> tuple[str, ...]:
    if _uses_antigravity(key):
        raw = os.environ.get("ANTIGRAVITY_TEXT_MODEL", "").strip()
        configured = tuple(m.strip() for m in raw.split(",") if m.strip())
        return configured or DEFAULT_ANTIGRAVITY_TEXT_MODELS
    raw = os.environ.get("GEMINI_TEXT_MODELS", "").strip()
    return tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_TEXT_MODELS


def image_models(key: str = "") -> tuple[str, ...]:
    if _uses_antigravity(key):
        raw = os.environ.get("ANTIGRAVITY_IMAGE_MODEL", "").strip()
        configured = tuple(m.strip() for m in raw.split(",") if m.strip())
        return configured or DEFAULT_ANTIGRAVITY_IMAGE_MODELS
    raw = os.environ.get("GEMINI_IMAGE_MODEL", "").strip()
    return tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_IMAGE_MODELS


class _JsonResponse:
    """JSON-like response for a collected Antigravity SSE stream."""

    def __init__(self, status_code: int, body: dict, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self) -> dict:
        return self._body


def _collect_antigravity_stream(response: requests.Response) -> dict:
    """Collect streamed Gemini candidates into the shape consumed by this module."""
    candidates: dict[int, dict] = {}
    prompt_feedback = None
    usage_metadata = None
    for raw_line in response.iter_lines():
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else raw_line
        )
        line = (line or "").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            logger.warning("Antigravity returned a malformed SSE event")
            continue
        body = event.get("response", event)
        if not isinstance(body, dict):
            continue
        prompt_feedback = body.get("promptFeedback", prompt_feedback)
        usage_metadata = body.get("usageMetadata", usage_metadata)
        for index, candidate in enumerate(body.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            candidate_index = candidate.get("index")
            if not isinstance(candidate_index, int):
                candidate_index = index
            target = candidates.setdefault(
                candidate_index,
                {"content": {"role": "model", "parts": []}},
            )
            target["finishReason"] = candidate.get("finishReason", target.get("finishReason"))
            parts = (candidate.get("content") or {}).get("parts") or []
            for part in parts:
                if isinstance(part, dict):
                    target["content"]["parts"].append(part)

    result: dict = {"candidates": [candidates[i] for i in sorted(candidates)]}
    if prompt_feedback is not None:
        result["promptFeedback"] = prompt_feedback
    if usage_metadata is not None:
        result["usageMetadata"] = usage_metadata
    return result


def _antigravity_request(
    model: str,
    token: str,
    request: dict,
    *,
    request_type: str,
    timeout: float,
) -> requests.Response | _JsonResponse:
    project_id = os.environ.get("ANTIGRAVITY_PROJECT_ID", "").strip()
    if not project_id:
        return _JsonResponse(
            400,
            {"error": {"status": "FAILED_PRECONDITION",
                       "message": "ANTIGRAVITY_PROJECT_ID is required"}},
        )
    access_token = _resolve_antigravity_access_token(token)
    if not access_token:
        return _JsonResponse(
            401,
            {"error": {"status": "UNAUTHENTICATED",
                       "message": "Antigravity OAuth access token is unavailable"}},
        )

    response = requests.post(
        f"{ANTIGRAVITY_BASE_URL}:streamGenerateContent?alt=sse",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": antigravity_user_agent(),
        },
        json={
            "project": project_id,
            "requestId": str(uuid.uuid4()),
            "request": request,
            "model": model,
            "userAgent": "antigravity",
            "requestType": request_type,
        },
        timeout=timeout,
        stream=True,
    )
    if response.status_code != 200:
        try:
            body = response.json()
        except ValueError:
            body = {}
        return _JsonResponse(response.status_code, body, response.text or "")
    return _JsonResponse(response.status_code, _collect_antigravity_stream(response))


def error_detail(resp: requests.Response) -> str:
    """`STATUS: message` from a Google API error body, so "quota" never looks like "rate limit"."""
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:200] or "no body"
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return f"{err.get('status') or err.get('code')}: {err.get('message')}"[:300]
    return (resp.text or "").strip()[:200] or "no body"


def classify(status: int, detail: str) -> str:
    """Map a failed Gemini call to the pipeline's failure vocabulary."""
    low = (detail or "").lower()
    if status == 429 or any(marker in low for marker in QUOTA_MARKERS):
        return "quota_exhausted"
    if status in (401, 403) or any(marker in low for marker in AUTH_MARKERS):
        return f"auth_{status}"
    if status == 404 or "not found" in low:
        return "model_missing"
    return f"status_{status}"


def _headers(key: str) -> dict:
    # Header auth, not ?key=: the key never lands in a URL that a log line might echo.
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def _payload(messages: list[dict], *, temperature: float, max_tokens: int = 0) -> dict:
    """OpenAI-style messages -> Gemini `systemInstruction` + `contents`."""
    system = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system").strip()
    contents = [
        {"role": "model" if m.get("role") == "assistant" else "user",
         "parts": [{"text": str(m.get("content") or "")}]}
        for m in messages if m.get("role") != "system"
    ]
    body: dict = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    generation: dict = {"temperature": temperature}
    if max_tokens:
        generation["maxOutputTokens"] = int(max_tokens)
    body["generationConfig"] = generation
    return body


def text_from(body: dict) -> str:
    """Concatenate answer parts, skipping `thought` parts (thinking models echo their reasoning)."""
    chunks: list[str] = []
    for candidate in body.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and part.get("text") and not part.get("thought"):
                chunks.append(part["text"])
    return "".join(chunks).strip()


def image_bytes(body: dict) -> bytes | None:
    """First inline image in a generateContent reply."""
    for candidate in body.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data") if isinstance(inline, dict) else None
            if data:
                return base64.b64decode(data)
    return None


def _crop_image_aspect_ratio(data: bytes, aspect_ratio: str) -> bytes:
    """Center-crop an image to an exact integer aspect ratio."""
    width_ratio, height_ratio = (int(value) for value in aspect_ratio.split(":"))
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
        scale = min(width // width_ratio, height // height_ratio)
        if scale < 1:
            raise ValueError(f"image is too small for {aspect_ratio}")
        target_width, target_height = width_ratio * scale, height_ratio * scale
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        cropped = image.crop((left, top, left + target_width, top + target_height))
        image_format = image.format or "PNG"
        if image_format == "JPEG" and cropped.mode not in ("RGB", "L"):
            cropped = cropped.convert("RGB")
        output = BytesIO()
        cropped.save(output, format=image_format)
        return output.getvalue()


def _mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def text(
    messages: list[dict],
    *,
    key: str,
    min_words: int = 0,
    temperature: float = 0.9,
    max_tokens: int = 0,
    timeout: float = TEXT_TIMEOUT,
) -> str | None:
    """Ask Gemini for a completion. Returns None when every configured model fails.

    `min_words` rejects a reply that spent its budget thinking and returned a stub — the same
    guard the OpenRouter path uses, because a thinking model can finish with 2 usable words.
    """
    if not key:
        return None
    antigravity = _uses_antigravity(key)
    for model in text_models(key):
        try:
            payload = _payload(messages, temperature=temperature, max_tokens=max_tokens)
            if antigravity:
                resp = _antigravity_request(
                    model, key, payload, request_type="agent", timeout=timeout
                )
            else:
                resp = requests.post(
                    f"{BASE_URL}/models/{model}:generateContent",
                    headers=_headers(key),
                    json=payload,
                    timeout=timeout,
                )
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Gemini %s request failed: %s", model, exc)
            continue

        if resp.status_code != 200:
            detail = error_detail(resp)
            kind = classify(resp.status_code, detail)
            logger.warning("Gemini %s -> %s (%s)", model, resp.status_code, detail)
            if kind.startswith("auth_") or kind == "quota_exhausted":
                # Same key for every model: walking the list only repeats the refusal.
                return None
            continue

        try:
            body = resp.json()
        except ValueError:
            logger.warning("Gemini %s returned a non-JSON body", model)
            continue
        content = text_from(body)
        if content and len(content.split()) >= min_words:
            candidate = (body.get("candidates") or [{}])[0]
            logger.info("Gemini %s answered (%d words, finish=%s)", model, len(content.split()),
                        candidate.get("finishReason"))
            return content
        blocked = (body.get("promptFeedback") or {}).get("blockReason")
        logger.warning("Gemini %s returned %d usable word(s), wanted %d%s",
                       model, len(content.split()), min_words,
                       f", blocked={blocked}" if blocked else "")
    return None


def _image_request(model: str, key: str, parts: list[dict], modalities: list[str],
                   timeout: float) -> requests.Response:
    payload = {"contents": [{"role": "user", "parts": parts}],
               "generationConfig": {"responseModalities": modalities}}
    if _uses_antigravity(key):
        payload["generationConfig"]["imageConfig"] = {
            "aspectRatio": ANTIGRAVITY_IMAGE_ASPECT_RATIO,
        }
        return _antigravity_request(
            model, key, payload, request_type="image_gen", timeout=timeout
        )
    return requests.post(
        f"{BASE_URL}/models/{model}:generateContent",
        headers=_headers(key),
        json=payload,
        timeout=timeout,
    )


def image(
    prompt: str,
    *,
    key: str,
    reference: bytes | None = None,
    timeout: float = IMAGE_TIMEOUT,
) -> Result:
    """Render one image, optionally editing `reference` (the arc's canonical character frame)."""
    if not key:
        return Result(ok=False, error="no_key")

    parts: list[dict] = []
    if reference:
        parts.append({"inlineData": {"mimeType": _mime(reference),
                                     "data": base64.b64encode(reference).decode()}})
    parts.append({"text": prompt})

    last = "status_0"
    for model in image_models(key):
        modalities = list(DEFAULT_IMAGE_MODALITIES)
        try:
            resp = _image_request(model, key, parts, modalities, timeout)
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Gemini image %s request failed: %s", model, exc)
            last = f"request_failed:{exc}"
            continue

        if resp.status_code == 400:
            detail = error_detail(resp)
            # Older image models reject ["IMAGE"] alone; widen once before giving up on the model.
            if "modalit" in detail.lower():
                logger.info("Gemini image %s wants TEXT+IMAGE modalities; retrying", model)
                modalities = list(_WIDENED_IMAGE_MODALITIES)
                try:
                    resp = _image_request(model, key, parts, modalities, timeout)
                except requests.RequestException as exc:  # noqa: BLE001
                    logger.warning("Gemini image %s retry failed: %s", model, exc)
                    last = f"request_failed:{exc}"
                    continue

        if resp.status_code != 200:
            detail = error_detail(resp)
            last = classify(resp.status_code, detail)
            logger.warning("Gemini image %s -> %s (%s)", model, resp.status_code, detail)
            if last.startswith("auth_") or last == "quota_exhausted":
                return Result(ok=False, error=last)
            continue

        try:
            body = resp.json()
        except ValueError:
            logger.warning("Gemini image %s returned a non-JSON body", model)
            last = "bad_response"
            continue
        data = image_bytes(body)
        if data:
            if _uses_antigravity(key):
                try:
                    data = _crop_image_aspect_ratio(data, ANTIGRAVITY_IMAGE_ASPECT_RATIO)
                except (OSError, ValueError) as exc:
                    logger.warning("Antigravity image %s returned an invalid image: %s", model, exc)
                    last = "bad_response"
                    continue
                logger.info("Antigravity %s rendered a 9:16 image (%d KB)",
                            model, len(data) // 1024)
            else:
                logger.info("Gemini %s rendered %d KB", model, len(data) // 1024)
            return Result(ok=True, data=data)

        reason = ((body.get("candidates") or [{}])[0].get("finishReason")
                  or (body.get("promptFeedback") or {}).get("blockReason") or "no image part")
        logger.warning("Gemini image %s returned no image (%s)", model, reason)
        last = "blocked" if "SAFETY" in str(reason).upper() else "bad_response"
    return Result(ok=False, error=last)
