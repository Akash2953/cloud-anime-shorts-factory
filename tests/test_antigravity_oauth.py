"""Antigravity OAuth URL, token exchange, and project discovery contracts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from antigravity_oauth import (
    AUTHORIZE_URL,
    LOAD_CODE_ASSIST_URL,
    TOKEN_URL,
    AntigravityOAuthError,
    antigravity_user_agent,
    build_authorization_url,
    create_pkce_pair,
    discover_project_id,
    exchange_authorization_code,
    extract_project_id,
    refresh_access_token,
)
from tools.antigravity_auth import load_env, save_env_values


class TestAuthorizationUrl(unittest.TestCase):
    def test_pkce_challenge_matches_verifier(self):
        import base64
        import hashlib

        verifier, challenge = create_pkce_pair()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).decode("ascii").rstrip("=")
        self.assertEqual(challenge, expected)

    def test_authorization_url_requests_offline_pkce_consent(self):
        _verifier, challenge = create_pkce_pair()
        url = build_authorization_url(
            "client-id", "http://127.0.0.1:12345/callback", "state-value", challenge
        )
        self.assertTrue(url.startswith(AUTHORIZE_URL + "?"))
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["client_id"], ["client-id"])
        self.assertEqual(query["redirect_uri"], ["http://127.0.0.1:12345/callback"])
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["access_type"], ["offline"])
        self.assertEqual(query["prompt"], ["consent"])
        self.assertEqual(query["state"], ["state-value"])
        self.assertEqual(query["code_challenge"], [challenge])
        self.assertEqual(query["code_challenge_method"], ["S256"])

    def test_user_agent_uses_latest_ide_release_version(self):
        response = MagicMock()
        response.json.return_value = [
            {"version": "2.9.9"},
            {"version": "3.10.1"},
            {"version": "3.9.12"},
        ]
        response.raise_for_status.return_value = None
        with patch("antigravity_oauth._IDE_VERSION_CACHE", None), \
             patch.dict("os.environ", {"ANTIGRAVITY_USER_AGENT": ""}), \
             patch("antigravity_oauth.requests.get", return_value=response):
            user_agent = antigravity_user_agent()
        self.assertEqual(user_agent, "antigravity/ide/3.10.1 darwin/arm64")

    def test_user_agent_can_be_overridden_without_release_lookup(self):
        with patch.dict("os.environ", {
            "ANTIGRAVITY_USER_AGENT": "custom-antigravity-agent"
        }), patch("antigravity_oauth.requests.get") as get:
            self.assertEqual(antigravity_user_agent(), "custom-antigravity-agent")
        get.assert_not_called()


class TestTokenExchange(unittest.TestCase):
    @staticmethod
    def _response(status=200, body=None):
        response = MagicMock()
        response.status_code = status
        response.ok = status < 400
        response.json.return_value = body or {}
        return response

    def test_authorization_code_exchange_sends_pkce_and_returns_tokens(self):
        with patch("antigravity_oauth.requests.post",
                   return_value=self._response(body={
                       "access_token": "access",
                       "refresh_token": "refresh",
                       "expires_in": 1800,
                   })) as post:
            tokens = exchange_authorization_code(
                code="auth-code",
                redirect_uri="http://127.0.0.1:12345/callback",
                code_verifier="verifier",
                client_id="client-id",
                client_secret="client-secret",
            )
        self.assertEqual(tokens.access_token, "access")
        self.assertEqual(tokens.refresh_token, "refresh")
        self.assertEqual(tokens.expires_in, 1800)
        self.assertEqual(post.call_args.args[0], TOKEN_URL)
        self.assertEqual(post.call_args.kwargs["data"]["code_verifier"], "verifier")
        self.assertEqual(post.call_args.kwargs["data"]["client_secret"], "client-secret")

    def test_refresh_exchange_uses_refresh_grant(self):
        with patch("antigravity_oauth.requests.post",
                   return_value=self._response(body={"access_token": "new-access"})) as post:
            tokens = refresh_access_token(
                refresh_token="refresh",
                client_id="client-id",
            )
        self.assertEqual(tokens.access_token, "new-access")
        self.assertEqual(post.call_args.kwargs["data"]["grant_type"], "refresh_token")
        self.assertEqual(post.call_args.kwargs["data"]["refresh_token"], "refresh")

    def test_failed_exchange_surfaces_provider_error(self):
        with patch("antigravity_oauth.requests.post",
                   return_value=self._response(
                       status=400,
                       body={"error": "invalid_grant", "error_description": "expired grant"},
                   )), self.assertRaisesRegex(AntigravityOAuthError, "expired grant"):
            refresh_access_token(refresh_token="refresh", client_id="client-id")


class TestProjectDiscovery(unittest.TestCase):
    def test_extracts_supported_project_id_shapes(self):
        self.assertEqual(extract_project_id({"cloudaicompanionProject": "project-id"}),
                         "project-id")
        self.assertEqual(extract_project_id({
            "cloudaicompanionProject": {"id": " project-id "}
        }), "project-id")
        self.assertEqual(extract_project_id({"cloudaicompanionProject": {"name": "other"}}), "")

    def test_discovery_sends_bearer_token_and_returns_project(self):
        response = MagicMock()
        response.ok = True
        response.json.return_value = {"cloudaicompanionProject": "project-id"}
        with patch("antigravity_oauth.requests.post", return_value=response) as post:
            project = discover_project_id("access-token")
        self.assertEqual(project, "project-id")
        self.assertEqual(post.call_args.args[0], LOAD_CODE_ASSIST_URL)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer access-token")


class TestAuthEnvPersistence(unittest.TestCase):
    def test_saves_rotated_values_without_dropping_existing_env_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("# keep this comment\nOTHER_SETTING=value\nANTIGRAVITY_REFRESH_TOKEN=old\n",
                            encoding="utf-8")
            save_env_values({
                "ANTIGRAVITY_OAUTH_CLIENT_ID": "client-id",
                "ANTIGRAVITY_REFRESH_TOKEN": "new-refresh",
            }, path)
            saved = load_env(path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(saved["OTHER_SETTING"], "value")
        self.assertEqual(saved["ANTIGRAVITY_REFRESH_TOKEN"], "new-refresh")
        self.assertEqual(saved["ANTIGRAVITY_OAUTH_CLIENT_ID"], "client-id")
        self.assertEqual(sum(line.startswith("ANTIGRAVITY_REFRESH_TOKEN=") for line in lines), 1)
        self.assertEqual(lines[0], "# keep this comment")


if __name__ == "__main__":
    unittest.main()
