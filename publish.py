"""YouTube Data API publisher ensuring private/unlisted first upload."""

import logging
import requests

logger = logging.getLogger(__name__)


def exchange_refresh_token(client_id: str, client_secret: str, refresh_token: str) -> str | None:
    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        resp = requests.post(url, data=payload, timeout=30)
        if resp.status_code != 200:
            logger.error("Token exchange failed: %s", resp.text)
            return None
        return resp.json().get("access_token")
    except Exception as exc:  # noqa: BLE001
        logger.error("Token exchange exception: %s", exc)
        return None


def upload_video(
    video_bytes: bytes,
    title: str,
    description: str,
    access_token: str,
    *,
    privacy_status: str = "private",
) -> str | None:
    """Upload MP4 bytes to YouTube Shorts as private (default) to allow manual validation."""
    init_url = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(len(video_bytes)),
        "X-Upload-Content-Type": "video/mp4",
    }
    meta = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "categoryId": "24",  # Entertainment
            "tags": ["anime", "shorts", "animation"],
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    try:
        init_resp = requests.post(init_url, headers=headers, json=meta, timeout=30)
        if init_resp.status_code not in (200, 201):
            logger.error("Init upload failed: %s", init_resp.text)
            return None

        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            return None

        upload_resp = requests.put(
            upload_url,
            data=video_bytes,
            headers={"Content-Type": "video/mp4", "Content-Length": str(len(video_bytes))},
            timeout=300,
        )
        if upload_resp.status_code not in (200, 201):
            logger.error("Chunk upload failed: %s", upload_resp.text)
            return None

        data = upload_resp.json()
        video_id = data.get("id")
        logger.info("Successfully uploaded video ID: %s", video_id)
        return video_id
    except Exception as exc:  # noqa: BLE001
        logger.error("YouTube upload exception: %s", exc)
        return None


def set_privacy(video_id: str, access_token: str, privacy_status: str = "public") -> bool:
    """Flip an already-uploaded video's visibility after manual validation.

    Costs ~50 quota units and needs no local copy of the file.
    """
    url = "https://www.googleapis.com/youtube/v3/videos?part=status"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {"id": video_id, "status": {"privacyStatus": privacy_status}}
    try:
        resp = requests.put(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logger.error("Privacy update failed: %s", resp.text)
            return False
        logger.info("Video %s is now %s", video_id, privacy_status)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Privacy update exception: %s", exc)
        return False
