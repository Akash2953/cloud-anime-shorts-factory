"""YouTube Data API publisher ensuring private/unlisted first upload."""

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

# Everything here used to be single-shot, so one 503 from Google or one dropped connection on a
# slow runner cost the day's upload — the queue retried *tomorrow*, not the request. These are the
# statuses worth a second try; 400/401/403 are permanent (bad credentials, quota, unverified
# channel) and retrying only burns time on a run that cannot succeed.
PUBLISH_ATTEMPTS = 3
BACKOFF_S = 2.0
TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
OK_STATUS = (200, 201)


def _is_transient(status: int) -> bool:
    return status in TRANSIENT_STATUS


def _backoff(attempt: int, delay: float | None = None) -> None:
    """Exponential pause between attempts: 2s, 4s, ... (resolved at call time so it is tunable)."""
    delay = BACKOFF_S if delay is None else delay
    if delay:
        time.sleep(delay * (2 ** (attempt - 1)))


def _json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {}


def exchange_refresh_token(client_id: str, client_secret: str, refresh_token: str) -> str | None:
    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        try:
            resp = requests.post(url, data=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Token exchange attempt %d/%d failed: %s",
                           attempt, PUBLISH_ATTEMPTS, exc)
        else:
            if resp.status_code == 200:
                token = _json(resp).get("access_token")
                if token:
                    return token
                logger.error("Token exchange returned no access_token: %s", resp.text[:300])
                return None
            if not _is_transient(resp.status_code):
                # invalid_grant, revoked refresh token, wrong client: not fixable by retrying.
                logger.error("Token exchange rejected (%d): %s",
                             resp.status_code, resp.text[:300])
                return None
            logger.warning("Token exchange attempt %d/%d -> %d",
                           attempt, PUBLISH_ATTEMPTS, resp.status_code)
        if attempt < PUBLISH_ATTEMPTS:
            _backoff(attempt)
    logger.error("Token exchange failed after %d attempts", PUBLISH_ATTEMPTS)
    return None


def _start_session(meta: dict, video_bytes: bytes, access_token: str) -> str | None:
    """Open a resumable upload session and return its URL."""
    init_url = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(len(video_bytes)),
        "X-Upload-Content-Type": "video/mp4",
    }
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        try:
            resp = requests.post(init_url, headers=headers, json=meta, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Upload init attempt %d/%d failed: %s", attempt, PUBLISH_ATTEMPTS, exc)
        else:
            if resp.status_code in OK_STATUS:
                url = resp.headers.get("Location")
                if not url:
                    logger.error("Upload session started without a Location header")
                    return None
                return url
            if not _is_transient(resp.status_code):
                logger.error("Upload init rejected (%d): %s", resp.status_code, resp.text[:300])
                return None
            logger.warning("Upload init attempt %d/%d -> %d",
                           attempt, PUBLISH_ATTEMPTS, resp.status_code)
        if attempt < PUBLISH_ATTEMPTS:
            _backoff(attempt)
    logger.error("Could not start an upload session after %d attempts", PUBLISH_ATTEMPTS)
    return None


def _resume_offset(resp: requests.Response, start: int, total: int) -> int:
    """Where the server says the upload stopped: `Range: bytes=0-N` means N+1 bytes are stored."""
    match = re.search(r"bytes=0-(\d+)", resp.headers.get("Range") or "")
    return min(int(match.group(1)) + 1, total) if match else start


def send_upload(upload_url: str, video_bytes: bytes, *,
                attempts: int = PUBLISH_ATTEMPTS) -> str | None:
    """Send the file over the resumable session, continuing from wherever it stopped.

    A 308 from YouTube is not an error: it is the protocol's "resume incomplete" answer, carrying
    the byte range the server already holds. Treating it as a failure threw away an upload that
    was nearly finished, and re-sending the whole file from byte 0 wastes the whole transfer.
    """
    total = len(video_bytes)
    start = 0
    for attempt in range(1, attempts + 1):
        chunk = video_bytes[start:]
        headers = {"Content-Type": "video/mp4", "Content-Length": str(len(chunk))}
        if start:
            headers["Content-Range"] = f"bytes {start}-{total - 1}/{total}"
        resp: requests.Response | None
        try:
            resp = requests.put(upload_url, data=chunk, headers=headers, timeout=300)
        except requests.RequestException as exc:
            logger.warning("Upload attempt %d/%d failed at byte %d/%d: %s",
                           attempt, attempts, start, total, exc)
            resp = None
        if resp is not None:
            if resp.status_code in OK_STATUS:
                video_id = _json(resp).get("id")
                logger.info("Successfully uploaded video ID: %s", video_id)
                return video_id
            if resp.status_code == 308:
                resumed = _resume_offset(resp, start, total)
                if resumed > start:
                    start = resumed
                    logger.info("Upload incomplete (%d/%d bytes stored); resuming",
                                start, total)
                    continue
                logger.warning("Upload attempt %d/%d returned 308 with no progress", attempt, attempts)
            elif not _is_transient(resp.status_code):
                logger.error("Chunk upload rejected (%d): %s",
                             resp.status_code, resp.text[:300])
                return None
            else:
                logger.warning("Upload attempt %d/%d -> %d",
                               attempt, attempts, resp.status_code)
        if attempt < attempts:
            _backoff(attempt)
    logger.error("Upload failed after %d attempts (%d of %d bytes stored)",
                 attempts, start, total)
    return None


def upload_video(
    video_bytes: bytes,
    title: str,
    description: str,
    access_token: str,
    *,
    privacy_status: str = "private",
    tags: list[str] | None = None,
) -> str | None:
    """Upload MP4 bytes to YouTube Shorts as private (default) to allow manual validation."""
    meta = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "categoryId": "24",  # Entertainment
            "tags": tags or ["anime", "shorts", "animation"],
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }
    upload_url = _start_session(meta, video_bytes, access_token)
    if not upload_url:
        return None
    return send_upload(upload_url, video_bytes)


def upload_thumbnail(video_id: str, access_token: str, image_bytes: bytes) -> bool:
    """Attach a custom thumbnail to an uploaded video.

    YouTube's `thumbnails.set` takes the image as the raw request body (`uploadType=media`), not
    as multipart. It costs ~50 quota units and needs a phone-verified channel; a channel that
    is not verified returns 403, which is logged and treated as "no thumbnail" rather than a
    failed upload — the video is already live at that point, so this never raises.

    Limits: JPEG/PNG, under 2 MB, and at least 1280x720.
    """
    url = ("https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
           f"?videoId={video_id}&uploadType=media")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "image/jpeg",
        "Content-Length": str(len(image_bytes)),
    }
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        try:
            resp = requests.post(url, headers=headers, data=image_bytes, timeout=120)
        except requests.RequestException as exc:
            logger.warning("Thumbnail upload exception for %s: %s", video_id, exc)
        else:
            if resp.status_code in OK_STATUS:
                logger.info("Thumbnail set for video %s", video_id)
                return True
            if not _is_transient(resp.status_code):
                logger.warning("Thumbnail upload failed for %s: %s", video_id, resp.text[:400])
                return False
            logger.warning("Thumbnail attempt %d/%d -> %d", attempt, PUBLISH_ATTEMPTS,
                           resp.status_code)
        if attempt < PUBLISH_ATTEMPTS:
            _backoff(attempt, delay=BACKOFF_S / 2)   # the cheap calls need less patience
    logger.warning("Thumbnail upload gave up for %s", video_id)
    return False


def set_privacy(video_id: str, access_token: str, privacy_status: str = "public") -> bool:
    """Flip an already-uploaded video's visibility after manual validation.

    Costs ~50 quota units and needs no local copy of the file.
    """
    url = "https://www.googleapis.com/youtube/v3/videos?part=status"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {"id": video_id, "status": {"privacyStatus": privacy_status}}
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        try:
            resp = requests.put(url, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Privacy update attempt %d/%d failed: %s",
                           attempt, PUBLISH_ATTEMPTS, exc)
        else:
            if resp.status_code == 200:
                logger.info("Video %s is now %s", video_id, privacy_status)
                return True
            if not _is_transient(resp.status_code):
                logger.error("Privacy update rejected (%d): %s",
                             resp.status_code, resp.text[:300])
                return False
            logger.warning("Privacy update attempt %d/%d -> %d", attempt, PUBLISH_ATTEMPTS,
                           resp.status_code)
        if attempt < PUBLISH_ATTEMPTS:
            _backoff(attempt, delay=BACKOFF_S / 2)
    logger.error("Privacy update failed after %d attempts", PUBLISH_ATTEMPTS)
    return False
