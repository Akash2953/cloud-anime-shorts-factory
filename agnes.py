"""Agnes AI image provider (OpenAI-compatible).

`POST /v1/images/generations` on `apihub.agnes-ai.com` returns either a URL or Base64 for the
`agnes-image-2.*-flash` models, which are free at every resolution tier at the time of writing.
The pipeline reaches for Agnes before Antigravity because of the pixels: the same 9:16 frame
comes back at 1472x2624 (2K), which `render.py` *down*samples to 1080x1920, where
`gemini-3.1-flash-image` through Antigravity answers ~765x1360 and gets upscaled 1.4-1.6x.

Text-to-image and identity-conditioned edits share the endpoint. An edit passes the canonical
character frame in `extra_body.image` (public URL or data URI), which is what makes the `edit`
strategy work here: the same girl, coat and boots survive a new pose and a new composition.

Two quirks of this API live in the request builder rather than in the callers:

  * `size` is a resolution tier ("2K"), not a pixel size; `ratio` carries the aspect ratio.
  * `response_format` must sit inside `extra_body`. Text-to-image additionally accepts
    top-level `return_base64`, which is the documented route for Base64 out.

Errors are classified into the same vocabulary the image pipeline already stops on
(`quota_exhausted`, `auth_401`, `model_missing`, `status_N`) so an Agnes refusal is logged in the
same terms as an Antigravity or Workers AI one.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass

import requests

from gemini import classify

logger = logging.getLogger(__name__)

BASE_URL = "https://apihub.agnes-ai.com/v1"

# `AGNES_IMAGE_MODEL` / `AGNES_IMAGE_SIZE` / `AGNES_TEXT_MODEL` override these, so an upgrade
# never needs a code change.
DEFAULT_IMAGE_MODELS: tuple[str, ...] = ("agnes-image-2.5-flash",)
DEFAULT_TEXT_MODELS: tuple[str, ...] = ("agnes-2.5-flash",)
DEFAULT_IMAGE_SIZE = "2K"
DEFAULT_RATIO = "9:16"

IMAGE_TIMEOUT = 300
TEXT_TIMEOUT = 300
# A 2K PNG lands around 4 MB; a URL response is fetched with the same budget as the render.
DOWNLOAD_TIMEOUT = 120


@dataclass
class Result:
    """Image result, shaped like `gemini.Result` / `images.ImageResult`."""

    ok: bool
    data: bytes | None = None
    error: str | None = None


def image_models() -> tuple[str, ...]:
    raw = os.environ.get("AGNES_IMAGE_MODEL", "").strip()
    return tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_IMAGE_MODELS


def text_models() -> tuple[str, ...]:
    raw = os.environ.get("AGNES_TEXT_MODEL", "").strip()
    return tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_TEXT_MODELS


def image_size() -> str:
    return os.environ.get("AGNES_IMAGE_SIZE", "").strip() or DEFAULT_IMAGE_SIZE


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _detail(resp: requests.Response) -> str:
    """Error text from an OpenAI-style error body, falling back to the raw body."""
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "")[:300]
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or err)
    return str(err or body)[:300]


def _payload(model: str, prompt: str, *, size: str, ratio: str,
             reference: bytes | None) -> dict:
    payload: dict = {"model": model, "prompt": prompt, "size": size, "ratio": ratio}
    if reference:
        payload["extra_body"] = {
            "image": [f"data:image/png;base64,{base64.b64encode(reference).decode()}"],
            # Base64 out for edits: the documented route, and it saves a second round-trip.
            "response_format": "b64_json",
        }
        return payload
    payload["return_base64"] = True
    payload["extra_body"] = {"response_format": "b64_json"}
    return payload


def _image_bytes(resp: requests.Response) -> bytes | None:
    """Bytes from `data[0].b64_json`, else download `data[0].url`."""
    try:
        body = resp.json()
    except ValueError:
        return None
    data = body.get("data") or []
    if not data or not isinstance(data[0], dict):
        return None
    first = data[0]
    if first.get("b64_json"):
        try:
            return base64.b64decode(first["b64_json"])
        except (ValueError, TypeError):
            return None
    url = first.get("url")
    if url:
        try:
            download = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Agnes image download failed: %s", exc)
            return None
        if download.status_code == 200 and download.content:
            return download.content
        logger.warning("Agnes image download -> %d", download.status_code)
    return None


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

    last = "status_0"
    for model in image_models():
        payload = _payload(model, prompt, size=image_size(), ratio=DEFAULT_RATIO,
                           reference=reference)
        try:
            resp = requests.post(f"{BASE_URL}/images/generations", headers=_headers(key),
                                 json=payload, timeout=timeout)
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Agnes image %s request failed: %s", model, exc)
            last = f"request_failed:{exc}"
            continue

        if resp.status_code != 200:
            detail = _detail(resp)
            last = classify(resp.status_code, detail)
            logger.warning("Agnes image %s -> %s (%s)", model, resp.status_code, detail)
            # A spend or credential refusal will refuse the next model too.
            if last.startswith("auth_") or last == "quota_exhausted":
                return Result(ok=False, error=last)
            continue

        data = _image_bytes(resp)
        if data:
            logger.info("Agnes %s rendered %s %d KB", model, image_size(), len(data) // 1024)
            return Result(ok=True, data=data)
        logger.warning("Agnes image %s returned no usable image", model)
        last = "bad_response"
    return Result(ok=False, error=last)


def text(
    messages: list[dict],
    *,
    key: str,
    min_words: int = 0,
    temperature: float = 0.8,
    max_tokens: int = 0,
    timeout: float = TEXT_TIMEOUT,
) -> str | None:
    """Ask Agnes for a completion. Returns None when every configured model fails.

    OpenAI-shaped request and response, so the storyboard calls it exactly like the OpenRouter
    path. `min_words` rejects a reply that spent its budget thinking and returned a stub; the
    reply's `reasoning_content` is never returned, only the answer text, and it is stripped
    because the service pads it with leading newlines.
    """
    if not key:
        return None
    last = "status_0"
    for model in text_models():
        payload: dict = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        try:
            resp = requests.post(f"{BASE_URL}/chat/completions", headers=_headers(key),
                                 json=payload, timeout=timeout)
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Agnes text %s request failed: %s", model, exc)
            last = f"request_failed:{exc}"
            continue

        if resp.status_code != 200:
            detail = _detail(resp)
            last = classify(resp.status_code, detail)
            logger.warning("Agnes text %s -> %s (%s)", model, resp.status_code, detail)
            # One key for every model: a credential or quota refusal will refuse the rest too.
            if last.startswith("auth_") or last == "quota_exhausted":
                return None
            continue

        try:
            body = resp.json()
        except ValueError:
            logger.warning("Agnes text %s returned a non-JSON body", model)
            last = "bad_response"
            continue

        choice = (body.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        words = len(content.split())
        if content and words >= min_words:
            logger.info("Agnes %s answered (%d words, finish=%s)", model, words,
                        choice.get("finish_reason"))
            return content
        logger.warning("Agnes text %s returned %d usable word(s), wanted %d (finish=%s)",
                       model, words, min_words, choice.get("finish_reason"))
        last = "too_short"
    logger.warning("Agnes text produced nothing (%s)", last)
    return None
