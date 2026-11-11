"""Mint a YouTube refresh token for the daily uploader.

    python tools/youtube_auth.py

Opens Google's consent screen on 127.0.0.1, captures the code, trades it for a refresh token,
verifies it against your channel, and writes the three YOUTUBE_* values into .env (never echoing
the token to the terminal). No dependencies beyond requests.

Prerequisites, all in the same Google Cloud project:
  1. YouTube Data API v3  -> ENABLED
  2. OAuth consent screen -> External, and PUBLISHED (in "Testing", refresh tokens die after 7 days)
  3. OAuth client         -> type "Desktop app"   <- must be Desktop, not Web
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socket
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import requests

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
# force-ssl covers upload + videos.update (set_privacy) + channel reads, in one consent.
SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
ENV_PATH = Path(".env")
ENV_KEYS = ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")


class _Catcher(http.server.BaseHTTPRequestHandler):
    """Single-shot loopback listener for the OAuth redirect."""

    # Class-level storage for the auth result
    _result: dict[str, str] = {}

    def do_GET(self):  # noqa: N802
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "code" in params:
            type(self)._result["code"] = params["code"][0]
            body = b"<h2>Authorised.</h2><p>Close this tab and return to the terminal.</p>"
        else:
            type(self)._result["error"] = params.get("error", ["no code"])[0]
            body = b"<h2>Authorisation failed.</h2><p>See the terminal.</p>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr noise
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def set_env_values(values: dict[str, str], path: Path = ENV_PATH) -> None:
    """Upsert keys in .env, preserving every other line untouched."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for key, value in values.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        replaced = False
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask(value: str) -> str:
    return f"{value[:6]}…{value[-4:]} ({len(value)} chars)" if len(value) > 12 else "(set)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", default="")
    ap.add_argument("--client-secret", default="")
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--url-only", action="store_true", help="print the consent URL and exit")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for the redirect")
    args = ap.parse_args()

    env = load_env()
    client_id = args.client_id or env.get("YOUTUBE_CLIENT_ID") or os.environ.get("YOUTUBE_CLIENT_ID", "")
    client_secret = (args.client_secret or env.get("YOUTUBE_CLIENT_SECRET")
                     or os.environ.get("YOUTUBE_CLIENT_SECRET", ""))
    if not client_id or not client_secret:
        print("Missing OAuth client. Pass --client-id/--client-secret or put YOUTUBE_CLIENT_ID and\n"
              "YOUTUBE_CLIENT_SECRET in .env (Google Cloud -> APIs & Services -> Credentials ->\n"
              "OAuth 2.0 Client IDs -> type: Desktop app).", file=sys.stderr)
        return 2

    port = args.port or free_port()
    redirect_uri = f"http://127.0.0.1:{port}"
    url = f"{AUTH_URL}?" + urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",          # forces a refresh_token even on re-authorisation
        "include_granted_scopes": "true",
    })

    print("1. Open this URL and approve access:\n")
    print(f"   {url}\n")
    if args.url_only:
        print(f"   (redirect URI for this run: {redirect_uri})")
        return 0

    server = http.server.HTTPServer(("127.0.0.1", port), _Catcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"2. Waiting for the redirect on {redirect_uri} ...")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass

    waited = 0.0
    while "code" not in _Catcher._result and "error" not in _Catcher._result and waited < args.timeout:
        threading.Event().wait(0.5)
        waited += 0.5
    server.shutdown()

    if _Catcher._result.get("error") or not _Catcher._result.get("code"):
        print(f"   No authorisation code received ({_Catcher._result.get('error') or 'timed out'}).", file=sys.stderr)
        return 1

    token = requests.post(TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret, "code": _Catcher._result["code"],
        "grant_type": "authorization_code", "redirect_uri": redirect_uri,
    }, timeout=30)
    if token.status_code != 200:
        print(f"   Code exchange failed: {token.text[:300]}", file=sys.stderr)
        return 1
    refresh_token = token.json().get("refresh_token")
    if not refresh_token:
        print("   Google returned no refresh_token. Remove the app's prior grant at\n"
              "   https://myaccount.google.com/permissions and run this again.", file=sys.stderr)
        return 1

    # Prove the token works and show which channel it belongs to.
    access = token.json()["access_token"]
    ch = requests.get("https://www.googleapis.com/youtube/v3/channels",
                      params={"part": "snippet,statistics", "mine": "true"},
                      headers={"Authorization": f"Bearer {access}"}, timeout=30)
    if ch.status_code == 200:
        items = ch.json().get("items") or []
        if items:
            snip = items[0]["snippet"]
            print(f"3. Authorised channel: {snip.get('title')!r} ({items[0]['id']})")
        else:
            print("3. Token works, but this Google account has no YouTube channel yet.\n"
                  "   Create one at https://www.youtube.com/create_channel and re-run.")
    else:
        print(f"3. Token obtained, but the channel check failed: {ch.text[:200]}")

    set_env_values({"YOUTUBE_CLIENT_ID": client_id, "YOUTUBE_CLIENT_SECRET": client_secret,
                    "YOUTUBE_REFRESH_TOKEN": refresh_token})
    print(f"4. Wrote {', '.join(ENV_KEYS)} to {ENV_PATH.resolve()}")
    print(f"   refresh token: {mask(refresh_token)}")
    print("\n   Verify with:  python verify.py --skip-images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
