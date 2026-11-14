"""Gemini provider: request shaping, error vocabulary and first-priority routing.

No network: `requests.post` and the provider call sites are patched, so these assert the
contract the rest of the pipeline depends on — Gemini is asked first, and a refusal degrades to
OpenRouter (text) or Workers AI (images) instead of failing the run.
"""

import base64
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
import gemini
import images
from PIL import Image
from antigravity_oauth import ANTIGRAVITY_OAUTH_CREDENTIAL, OAuthTokens
from scenes import plan_shorts_detailed, plan_shots
from tests.test_scenes import detailed_reply

JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-payload"


def _resp(status: int, body: dict | None = None, text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = body if body is not None else {}
    return resp


def _candidates(*texts: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": t} for t in texts]}}]}


def _sse_resp(*events: dict) -> MagicMock:
    resp = _resp(200)
    resp.iter_lines.return_value = [
        *(f"data: {json.dumps(event)}".encode() for event in events),
        b"data: [DONE]",
    ]
    return resp


class TestAntigravityConfig(unittest.TestCase):
    def test_access_token_is_selected_for_provider_calls(self):
        with patch("config.load_dotenv"), patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "ANTIGRAVITY_API_KEY": "",
            "ANTIGRAVITY_ACCESS_TOKEN": "oauth-token",
            "ANTIGRAVITY_PROJECT_ID": "project-id",
        }):
            cfg = config.load_config()
        self.assertEqual(cfg["ANTIGRAVITY_API_KEY"], "oauth-token")
        self.assertEqual(cfg["ANTIGRAVITY_ACCESS_TOKEN"], "oauth-token")
        self.assertEqual(cfg["ANTIGRAVITY_PROJECT_ID"], "project-id")

    def test_refresh_token_selects_antigravity_without_refreshing_during_config_load(self):
        with patch("config.load_dotenv"), patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "ANTIGRAVITY_API_KEY": "",
            "ANTIGRAVITY_ACCESS_TOKEN": "",
            "ANTIGRAVITY_REFRESH_TOKEN": "refresh-token",
        }), patch("antigravity_oauth.requests.post") as oauth_request:
            cfg = config.load_config()
        self.assertEqual(cfg["ANTIGRAVITY_API_KEY"], ANTIGRAVITY_OAUTH_CREDENTIAL)
        oauth_request.assert_not_called()

    def test_gemini_api_key_does_not_enable_the_image_provider(self):
        with patch("config.load_dotenv"), patch.dict(os.environ, {
            "GEMINI_API_KEY": "ai-studio-key",
            "GOOGLE_API_KEY": "",
            "ANTIGRAVITY_ACCESS_TOKEN": "",
            "ANTIGRAVITY_REFRESH_TOKEN": "",
            "ANTIGRAVITY_API_KEY": "",
        }):
            cfg = config.load_config()
        self.assertEqual(cfg["ANTIGRAVITY_API_KEY"], "")


class TestTextRequest(unittest.TestCase):
    def test_system_message_becomes_system_instruction(self):
        with patch("gemini.requests.post", return_value=_resp(200, _candidates("Shot 1: ..."))) as post:
            out = gemini.text([{"role": "system", "content": "Be a planner."},
                               {"role": "user", "content": "Write it."}], key="secret")
        self.assertEqual(out, "Shot 1: ...")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "Be a planner.")
        self.assertEqual([c["role"] for c in payload["contents"]], ["user"])
        self.assertNotIn("secret", post.call_args.args[0])          # key never in the URL
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "secret")

    def test_assistant_turns_are_sent_as_model_turns(self):
        with patch("gemini.requests.post", return_value=_resp(200, _candidates("ok"))) as post:
            gemini.text([{"role": "user", "content": "a"},
                         {"role": "assistant", "content": "b"},
                         {"role": "user", "content": "c"}], key="k")
        roles = [c["role"] for c in post.call_args.kwargs["json"]["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])

    def test_thought_parts_are_not_returned_as_story(self):
        body = {"candidates": [{"content": {"parts": [
            {"text": "thinking out loud about the story", "thought": True},
            {"text": "A real narration line lands here cleanly."}]}}]}
        with patch("gemini.requests.post", return_value=_resp(200, body)):
            self.assertEqual(gemini.text([{"role": "user", "content": "x"}], key="k"),
                             "A real narration line lands here cleanly.")

    def test_short_reply_is_rejected_and_the_next_model_is_tried(self):
        replies = [_resp(200, _candidates("two words")),
                   _resp(200, _candidates("word " * 40))]
        with patch.object(gemini, "text_models", return_value=("first", "second")), \
             patch("gemini.requests.post", side_effect=replies) as post:
            out = gemini.text([{"role": "user", "content": "x"}], key="k", min_words=20)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(out.split()), 40)

    def test_quota_stops_the_model_walk(self):
        quota = _resp(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}})
        with patch.object(gemini, "text_models", return_value=("a", "b")), \
             patch("gemini.requests.post", return_value=quota) as post:
            self.assertIsNone(gemini.text([{"role": "user", "content": "x"}], key="k"))
        self.assertEqual(post.call_count, 1)   # same key: the other model cannot help

    def test_no_key_makes_no_request(self):
        with patch("gemini.requests.post") as post:
            self.assertIsNone(gemini.text([{"role": "user", "content": "x"}], key=""))
        post.assert_not_called()

    def test_antigravity_oauth_posts_bearer_request_and_collects_streamed_text(self):
        with patch.dict(os.environ, {
            "ANTIGRAVITY_ACCESS_TOKEN": "oauth-token",
            "ANTIGRAVITY_PROJECT_ID": "project-id",
            "ANTIGRAVITY_TEXT_MODEL": "gemini-3-flash",
        }), patch("gemini.requests.post", return_value=_sse_resp(
            {"response": {"candidates": [
                {"content": {"parts": [{"text": "A story from "}]}}
            ]}},
            {"response": {"candidates": [
                {"content": {"parts": [{"text": "Antigravity."}]}}
            ]}},
        )) as post, patch(
            "gemini.antigravity_user_agent",
            return_value="antigravity/ide/3.10.1 darwin/arm64",
        ):
            out = gemini.text([{"role": "user", "content": "Write a story."}], key="oauth-token")

        self.assertEqual(out, "A story from Antigravity.")
        args = post.call_args
        self.assertEqual(args.args[0],
                         "https://daily-cloudcode-pa.googleapis.com/v1internal:"
                         "streamGenerateContent?alt=sse")
        self.assertEqual(args.kwargs["headers"]["Authorization"], "Bearer oauth-token")
        self.assertEqual(
            args.kwargs["headers"]["User-Agent"],
            "antigravity/ide/3.10.1 darwin/arm64",
        )
        self.assertNotIn("oauth-token", args.args[0])
        self.assertEqual(args.kwargs["json"]["requestType"], "agent")
        self.assertEqual(args.kwargs["json"]["project"], "project-id")
        self.assertEqual(args.kwargs["json"]["model"], "gemini-3-flash")
        self.assertTrue(args.kwargs["stream"])

    def test_antigravity_requires_a_project_id_without_sending_the_token(self):
        with patch.dict(os.environ, {
            "ANTIGRAVITY_ACCESS_TOKEN": "oauth-token",
            "ANTIGRAVITY_PROJECT_ID": "",
        }), patch("gemini.requests.post") as post:
            out = gemini.text([{"role": "user", "content": "Write a story."}], key="oauth-token")
        self.assertIsNone(out)
        post.assert_not_called()

    def test_antigravity_refreshes_and_reuses_access_token(self):
        cache = {"refresh_token": "", "access_token": "", "expires_at": 0.0}
        with patch.object(gemini, "_ACCESS_TOKEN_CACHE", cache), patch.dict(os.environ, {
            "ANTIGRAVITY_ACCESS_TOKEN": "",
            "ANTIGRAVITY_REFRESH_TOKEN": "refresh-token",
            "ANTIGRAVITY_OAUTH_CLIENT_ID": "client-id",
            "ANTIGRAVITY_OAUTH_CLIENT_SECRET": "client-secret",
            "ANTIGRAVITY_PROJECT_ID": "project-id",
            "ANTIGRAVITY_TEXT_MODEL": "gemini-3-flash",
        }), patch("gemini.refresh_access_token",
                 return_value=OAuthTokens("fresh-access-token", 3600)) as refresh, \
             patch("gemini.requests.post", return_value=_sse_resp(
                 {"response": {"candidates": [
                     {"content": {"parts": [{"text": "A refreshed OAuth answer."}]}}
                 ]}},
             )) as post:
            for _ in range(2):
                out = gemini.text(
                    [{"role": "user", "content": "Write a story."}],
                    key=ANTIGRAVITY_OAUTH_CREDENTIAL,
                )

        self.assertEqual(out, "A refreshed OAuth answer.")
        refresh.assert_called_once_with(
            refresh_token="refresh-token",
            client_id="client-id",
            client_secret="client-secret",
        )
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer fresh-access-token",
        )

    def test_antigravity_refresh_requires_client_id(self):
        with patch.object(gemini, "_ACCESS_TOKEN_CACHE", {
            "refresh_token": "", "access_token": "", "expires_at": 0.0
        }), patch.dict(os.environ, {
            "ANTIGRAVITY_ACCESS_TOKEN": "",
            "ANTIGRAVITY_REFRESH_TOKEN": "refresh-token",
            "ANTIGRAVITY_OAUTH_CLIENT_ID": "",
            "ANTIGRAVITY_PROJECT_ID": "project-id",
        }), patch("gemini.refresh_access_token") as refresh, \
             patch("gemini.requests.post") as post:
            out = gemini.text(
                [{"role": "user", "content": "Write a story."}],
                key=ANTIGRAVITY_OAUTH_CREDENTIAL,
            )
        self.assertIsNone(out)
        refresh.assert_not_called()
        post.assert_not_called()


class TestImageRequest(unittest.TestCase):
    @staticmethod
    def _image_body(data: bytes = JPEG) -> dict:
        return {"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(data).decode()}}]}}]}

    def test_image_is_decoded(self):
        with patch("gemini.requests.post", return_value=_resp(200, self._image_body())) as post:
            res = gemini.image("a lantern in the rain", key="k")
        self.assertTrue(res.ok)
        self.assertEqual(res.data, JPEG)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["generationConfig"]["responseModalities"], ["IMAGE"])

    def test_reference_frame_is_sent_as_inline_data(self):
        reference = b"\x89PNG\r\n\x1a\ncanonical-frame"
        with patch("gemini.requests.post", return_value=_resp(200, self._image_body())) as post:
            gemini.image("keep the same character", key="k", reference=reference)
        parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(base64.b64decode(parts[0]["inlineData"]["data"]), reference)
        self.assertEqual(parts[1]["text"], "keep the same character")

    def test_rejected_modalities_are_retried_widened(self):
        narrow = _resp(400, {"error": {"status": "INVALID_ARGUMENT",
                                       "message": "response modalities not supported"}})
        with patch.object(gemini, "image_models", return_value=("m",)), \
             patch("gemini.requests.post",
                   side_effect=[narrow, _resp(200, self._image_body())]) as post:
            res = gemini.image("x", key="k")
        self.assertTrue(res.ok)
        widened = post.call_args_list[1].kwargs["json"]["generationConfig"]["responseModalities"]
        self.assertEqual(widened, ["TEXT", "IMAGE"])

    def test_quota_is_reported_in_the_image_vocabulary(self):
        quota = _resp(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}})
        with patch("gemini.requests.post", return_value=quota):
            res = gemini.image("x", key="k")
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "quota_exhausted")

    def test_a_reply_without_an_image_is_a_failure(self):
        body = {"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"parts": []}}]}
        with patch("gemini.requests.post", return_value=_resp(200, body)):
            res = gemini.image("x", key="k")
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "blocked")

    def test_antigravity_image_uses_image_gen_envelope_and_decodes_stream(self):
        image = Image.new("RGB", (24, 32), "orange")
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_b64 = base64.b64encode(image_bytes.getvalue()).decode()
        with patch.dict(os.environ, {
            "ANTIGRAVITY_ACCESS_TOKEN": "oauth-token",
            "ANTIGRAVITY_PROJECT_ID": "project-id",
            "ANTIGRAVITY_IMAGE_MODEL": "gemini-3.1-flash-image",
        }), patch("gemini.requests.post", return_value=_sse_resp(
            {"response": {"candidates": [
                {"content": {"parts": [{"inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_b64,
                }}]}}
            ]}},
        )) as post, patch(
            "gemini.antigravity_user_agent",
            return_value="antigravity/ide/3.10.1 darwin/arm64",
        ):
            res = gemini.image("A cozy cabin", key="oauth-token")

        self.assertTrue(res.ok)
        with Image.open(BytesIO(res.data)) as rendered:
            self.assertEqual(rendered.size, (18, 32))
        args = post.call_args
        self.assertEqual(args.kwargs["headers"]["Authorization"], "Bearer oauth-token")
        self.assertEqual(
            args.kwargs["headers"]["User-Agent"],
            "antigravity/ide/3.10.1 darwin/arm64",
        )
        self.assertEqual(args.kwargs["json"]["requestType"], "image_gen")
        self.assertEqual(args.kwargs["json"]["model"], "gemini-3.1-flash-image")
        self.assertEqual(
            args.kwargs["json"]["request"]["generationConfig"]["imageConfig"]["aspectRatio"],
            "9:16",
        )


class TestStoryRouting(unittest.TestCase):
    def test_gemini_plans_the_story_before_openrouter(self):
        with patch("scenes.gemini.text", return_value=detailed_reply(8)) as gem, \
             patch("scenes._openrouter") as router, \
             patch("scenes.request_title", return_value=""):
            shots, _, fallback, _, _ = plan_shorts_detailed(
                theme="t", api_key="or", gemini_key="gk", want=8)
        self.assertEqual(len(shots), 8)
        self.assertFalse(fallback)
        gem.assert_called_once()
        router.assert_not_called()

    def test_openrouter_still_answers_when_gemini_returns_nothing(self):
        with patch("scenes.gemini.text", return_value=None), \
             patch("scenes._openrouter", return_value=detailed_reply(8)) as router, \
             patch("scenes.request_title", return_value=""):
            _shots, _, fallback, _, _ = plan_shorts_detailed(
                theme="t", api_key="or", gemini_key="gk", want=8)
        self.assertFalse(fallback)
        router.assert_called()

    def test_no_keys_at_all_falls_back_to_the_template(self):
        with patch("scenes.gemini.text") as gem, patch("scenes._openrouter") as router:
            shots, fallback = plan_shots(theme="t", count=8)
        gem.assert_not_called()
        router.assert_not_called()
        self.assertTrue(fallback)
        self.assertEqual(len(shots), 8)


class TestImageRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_antigravity_renders_before_workers_ai(self):
        with patch("images.gemini.image", return_value=gemini.Result(ok=True, data=JPEG)) as gem, \
             patch("images.generate_fresh") as workers:
            frames, _ = images.produce_images(
                ["a lantern in the rain"], cfa_token="t", account_id="a",
                img_dir=self.dir, strategy="fresh", antigravity_key="oauth-token")
        self.assertEqual(len(frames), 1)
        gem.assert_called_once()
        self.assertEqual(gem.call_args.kwargs["key"], "oauth-token")
        workers.assert_not_called()

    def test_workers_ai_takes_over_when_antigravity_fails(self):
        with patch("images.gemini.image",
                   return_value=gemini.Result(ok=False, error="quota_exhausted")), \
             patch("images.generate_fresh",
                   return_value=images.ImageResult(ok=True, data=JPEG)) as workers:
            frames, _ = images.produce_images(
                ["a lantern in the rain"], cfa_token="t", account_id="a",
                img_dir=self.dir, strategy="fresh", antigravity_key="oauth-token")
        self.assertEqual(len(frames), 1)
        workers.assert_called_once()

    def test_identity_edit_also_prefers_antigravity(self):
        with patch("images.gemini.image", return_value=gemini.Result(ok=True, data=JPEG)) as gem, \
             patch("images.generate_edit") as workers_edit, \
             patch("images.generate_fresh") as workers_fresh:
            frames, used = images.produce_images(
                ["first shot", "second shot"], cfa_token="t", account_id="a",
                img_dir=self.dir, strategy="edit", antigravity_key="oauth-token")
        self.assertEqual(used, "edit")
        self.assertEqual(len(frames), 2)
        self.assertEqual(gem.call_count, 2)
        self.assertIsNotNone(gem.call_args_list[1].kwargs.get("reference"))
        workers_edit.assert_not_called()
        workers_fresh.assert_not_called()

    def test_without_an_antigravity_key_only_workers_ai_runs(self):
        with patch("images.gemini.image") as gem, \
             patch("images.generate_fresh",
                   return_value=images.ImageResult(ok=True, data=JPEG)) as workers:
            images.produce_images(["a lantern"], cfa_token="t", account_id="a",
                                  img_dir=self.dir, strategy="fresh")
        gem.assert_not_called()
        workers.assert_called_once()


if __name__ == "__main__":
    unittest.main()
