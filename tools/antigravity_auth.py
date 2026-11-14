"""Sign in to Antigravity and save a refresh token to the ignored local .env file."""

from __future__ import annotations

import argparse
import http.server
import os
import re
import secrets
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from antigravity_oauth import (
    AntigravityOAuthError,
    build_authorization_url,
    create_pkce_pair,
    discover_project_id,
    exchange_authorization_code,
)  # noqa: E402

ENV_PATH = PROJECT_ROOT / ".env"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip("'\"")
    return values


def save_env_values(values: dict[str, str], path: Path = ENV_PATH) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for name, value in values.items():
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=")
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = f"{name}={value}"
                break
        else:
            lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _callback_handler(expected_state: str, result: dict[str, str]):
    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed_path = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_path.query)
            received_state = params.get("state", [""])[0]
            if parsed_path.path != "/callback":
                result["error"] = "Unexpected OAuth callback path"
                message = "Authorization failed. Return to the terminal."
                self.send_response(404)
            elif received_state != expected_state:
                result["error"] = "OAuth state validation failed"
                message = "Authorization failed. Return to the terminal."
                self.send_response(400)
            elif params.get("error"):
                result["error"] = params["error"][0]
                message = "Authorization was not completed. Return to the terminal."
                self.send_response(400)
            elif params.get("code"):
                result["code"] = params["code"][0]
                message = "Authorization complete. Close this tab and return to the terminal."
                self.send_response(200)
            else:
                result["error"] = "No authorization code was returned"
                message = "Authorization failed. Return to the terminal."
                self.send_response(400)
            body = f"<h2>{message}</h2>".encode()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return

    return CallbackHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for sign-in")
    args = parser.parse_args()

    env = load_env()
    client_id = os.environ.get("ANTIGRAVITY_OAUTH_CLIENT_ID") or env.get(
        "ANTIGRAVITY_OAUTH_CLIENT_ID", ""
    )
    client_secret = os.environ.get("ANTIGRAVITY_OAUTH_CLIENT_SECRET") or env.get(
        "ANTIGRAVITY_OAUTH_CLIENT_SECRET", ""
    )
    if not client_id:
        print(
            "Set ANTIGRAVITY_OAUTH_CLIENT_ID in .env or the environment before signing in.",
            file=sys.stderr,
        )
        return 2

    state = secrets.token_urlsafe(32)
    code_verifier, code_challenge = create_pkce_pair()
    result: dict[str, str] = {}
    server = http.server.HTTPServer(
        ("127.0.0.1", 0), _callback_handler(state, result)
    )
    server.timeout = 0.5
    redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
    url = build_authorization_url(client_id, redirect_uri, state, code_challenge)

    print("Open this URL and approve the Antigravity access request:\n")
    print(f"{url}\n")
    try:
        if not webbrowser.open(url):
            print("The browser did not open automatically; use the URL above.")
    except (OSError, webbrowser.Error):
        print("The browser could not be opened automatically; use the URL above.")

    deadline = time.monotonic() + max(1, args.timeout)
    try:
        while not result and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    if result.get("error") or not result.get("code"):
        print(
            f"Sign-in failed: {result.get('error') or 'timed out waiting for callback'}",
            file=sys.stderr,
        )
        return 1

    try:
        tokens = exchange_authorization_code(
            code=result["code"],
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            client_id=client_id,
            client_secret=client_secret,
        )
    except (AntigravityOAuthError, requests.RequestException) as exc:
        print(f"OAuth exchange failed: {exc}", file=sys.stderr)
        return 1

    if not tokens.refresh_token:
        print(
            "Google returned no refresh token. Revoke this app's grant in Google Account "
            "permissions, then run the sign-in command again.",
            file=sys.stderr,
        )
        return 1

    project_id = env.get("ANTIGRAVITY_PROJECT_ID", "")
    try:
        project_id = discover_project_id(tokens.access_token) or project_id
    except (AntigravityOAuthError, requests.RequestException) as exc:
        print(f"Project discovery failed: {exc}", file=sys.stderr)

    values = {
        "ANTIGRAVITY_OAUTH_CLIENT_ID": client_id,
        "ANTIGRAVITY_REFRESH_TOKEN": tokens.refresh_token,
    }
    if client_secret:
        values["ANTIGRAVITY_OAUTH_CLIENT_SECRET"] = client_secret
    if project_id:
        values["ANTIGRAVITY_PROJECT_ID"] = project_id
    save_env_values(values)

    print("Saved the OAuth refresh token and available project ID to the ignored .env file.")
    if not project_id:
        print(
            "Project ID was not discovered. Set ANTIGRAVITY_PROJECT_ID in .env before "
            "generating content."
        )
    print("Access tokens are refreshed automatically when the pipeline runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
