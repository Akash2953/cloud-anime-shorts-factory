"""Agnes image provider: request shape, response parsing and provider ordering.

No network: `requests` is patched, so these assert the contract this module owes `images.py`.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest

import agnes
import images

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
PNG_B64 = base64.b64encode(PNG).decode()


class _Resp:
    def __init__(self, status: int, body: dict | None = None, content: bytes = b""):
        self.status_code = status
        self._body = body
        self.content = content
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _sent(monkeypatch):
    """Capture the JSON body of the POST this module makes."""
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Resp(200, {"data": [{"b64_json": PNG_B64, "url": None}]})

    monkeypatch.setattr(agnes.requests, "post", fake_post)
    return calls


def test_text_to_image_asks_for_base64(monkeypatch):
    calls = _sent(monkeypatch)
    res = agnes.image("a girl in a red raincoat", key="k")

    assert res.ok and res.data == PNG
    url, kwargs = calls[0]
    assert url == "https://apihub.agnes-ai.com/v1/images/generations"
    body = kwargs["json"]
    assert body["model"] == "agnes-image-2.5-flash"
    assert body["size"] == "2K"
    assert body["ratio"] == "9:16"
    assert body["return_base64"] is True
    assert "image" not in body["extra_body"]
    assert kwargs["headers"]["Authorization"] == "Bearer k"


def test_edit_sends_reference_as_data_uri(monkeypatch):
    calls = _sent(monkeypatch)
    res = agnes.image("same girl, new pose", key="k", reference=PNG)

    assert res.ok and res.data == PNG
    body = calls[0][1]["json"]
    ref = body["extra_body"]["image"][0]
    assert ref.startswith("data:image/png;base64,")
    assert base64.b64decode(ref.split(",", 1)[1]) == PNG
    # Edits take Base64 out through extra_body, per the API's contract.
    assert body["extra_body"]["response_format"] == "b64_json"
    assert "return_base64" not in body


def test_url_response_is_downloaded(monkeypatch):
    monkeypatch.setattr(agnes.requests, "post", lambda url, **kw: _Resp(
        200, {"data": [{"b64_json": None, "url": "https://cdn.example/x.png"}]}))
    downloaded = _Resp(200, content=PNG)
    monkeypatch.setattr(agnes.requests, "get", lambda url, **kw: downloaded)

    res = agnes.image("prompt", key="k")

    assert res.ok and res.data == PNG


def test_quota_refusal_stops_before_the_next_model(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"]["model"])
        return _Resp(429, {"error": {"message": "rate limit exceeded", "type": "rate_limit"}})

    monkeypatch.setattr(agnes.requests, "post", fake_post)
    monkeypatch.setenv("AGNES_IMAGE_MODEL", "m-one,m-two")

    res = agnes.image("prompt", key="k")

    assert not res.ok and res.error == "quota_exhausted"
    assert calls == ["m-one"]  # a spend refusal will refuse the second model too


def test_model_failure_falls_through_to_the_next_model(monkeypatch):
    seen = []

    def fake_post(url, **kwargs):
        model = kwargs["json"]["model"]
        seen.append(model)
        if model == "m-one":
            return _Resp(500, {"error": {"message": "boom", "type": "server_error"}})
        return _Resp(200, {"data": [{"b64_json": PNG_B64}]})

    monkeypatch.setattr(agnes.requests, "post", fake_post)
    monkeypatch.setenv("AGNES_IMAGE_MODEL", "m-one,m-two")

    res = agnes.image("prompt", key="k")

    assert res.ok
    assert seen == ["m-one", "m-two"]


def test_no_key_never_calls_the_api(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("should not reach the network")

    monkeypatch.setattr(agnes.requests, "post", boom)
    res = agnes.image("prompt", key="")
    assert not res.ok and res.error == "no_key"


def test_size_and_model_are_env_overridable(monkeypatch):
    monkeypatch.setenv("AGNES_IMAGE_SIZE", "4K")
    monkeypatch.setenv("AGNES_IMAGE_MODEL", "agnes-image-2.1-flash")
    assert agnes.image_size() == "4K"
    assert agnes.image_models() == ("agnes-image-2.1-flash",)


def test_fresh_image_prefers_antigravity_so_the_short_is_one_style(monkeypatch):
    """The canonical frame must come from the same model family that renders the edits."""
    monkeypatch.setattr(images.gemini, "image",
                        lambda *a, **k: images.gemini.Result(ok=True, data=b"gemini"))
    monkeypatch.setattr(agnes, "image",
                        lambda *a, **k: pytest.fail("Agnes should not define the canonical frame"))
    monkeypatch.setattr(images, "generate_fresh",
                        lambda *a, **k: pytest.fail("Workers AI should not run"))

    res = images.fresh_image("prompt", "cf", "acct", antigravity_key="ag", agnes_key="ak")

    assert res.ok and res.data == b"gemini"


def test_fresh_image_falls_back_to_agnes(monkeypatch):
    monkeypatch.setattr(images.gemini, "image",
                        lambda *a, **k: images.gemini.Result(ok=False, error="status_500"))
    monkeypatch.setattr(agnes, "image", lambda *a, **k: agnes.Result(ok=True, data=PNG))
    res = images.fresh_image("prompt", "cf", "acct", antigravity_key="ag", agnes_key="ak")
    assert res.ok and res.data == PNG


def test_agnes_alone_still_renders_fresh(monkeypatch):
    """No Antigravity key: Agnes is the only renderer, and nothing else is called."""
    monkeypatch.setattr(agnes, "image", lambda *a, **k: agnes.Result(ok=True, data=PNG))
    monkeypatch.setattr(images.gemini, "image",
                        lambda *a, **k: pytest.fail("no Antigravity key, so no Antigravity call"))
    res = images.fresh_image("prompt", "cf", "acct", agnes_key="ak")
    assert res.ok and res.data == PNG


def test_fresh_image_falls_back_when_agnes_refuses(monkeypatch):
    monkeypatch.setattr(agnes, "image", lambda *a, **k: agnes.Result(ok=False, error="status_500"))
    monkeypatch.setattr(images.gemini, "image",
                        lambda *a, **k: images.gemini.Result(ok=True, data=b"gemini"))
    res = images.fresh_image("prompt", "cf", "acct", antigravity_key="ag", agnes_key="ak")
    assert res.ok and res.data == b"gemini"


def test_edit_image_prefers_antigravity_because_it_follows_the_pose(monkeypatch):
    """Measured: Agnes reproduced the canonical pose from the same prompt and reference."""
    seen = {}

    def fake_gemini(prompt, *, key, reference=None, **kw):
        seen["reference"] = reference
        return images.gemini.Result(ok=True, data=b"gemini")

    monkeypatch.setattr(images.gemini, "image", fake_gemini)
    monkeypatch.setattr(agnes, "image",
                        lambda *a, **k: pytest.fail("Agnes should not render the edit"))

    res = images.edit_image(PNG, "prompt", "cf", "acct", antigravity_key="ag")

    assert res.ok and res.data == b"gemini" and seen["reference"] == PNG


def test_edit_image_does_not_fall_back_to_agnes(monkeypatch):
    """A pose-freezing edit is not a fallback: the caller renders the shot fresh instead."""
    monkeypatch.setattr(images.gemini, "image",
                        lambda *a, **k: images.gemini.Result(ok=False, error="quota_exhausted"))
    monkeypatch.setattr(agnes, "image",
                        lambda *a, **k: pytest.fail("Agnes would repeat the reference's pose"))
    monkeypatch.setattr(images, "generate_edit",
                        lambda *a, **k: images.ImageResult(ok=False, error="quota_exhausted"))

    res = images.edit_image(PNG, "prompt", "cf", "acct", antigravity_key="ag")

    assert not res.ok and res.error == "quota_exhausted"


def test_blank_agnes_key_keeps_the_old_chain(monkeypatch):
    """No Agnes key must leave today's Antigravity -> Workers AI behaviour untouched."""
    monkeypatch.setattr(images.gemini, "image",
                        lambda *a, **k: images.gemini.Result(ok=True, data=b"gemini"))
    res = images.fresh_image("prompt", "cf", "acct", antigravity_key="ag")
    assert res.ok and res.data == b"gemini"


def test_text_strips_padding_and_ignores_reasoning(monkeypatch):
    body = {"choices": [{"message": {"content": "\n\nA short answer.", "reasoning_content": "think"},
                         "finish_reason": "stop"}]}
    monkeypatch.setattr(agnes.requests, "post", lambda url, **kw: _Resp(200, body))

    assert agnes.text([{"role": "user", "content": "hi"}], key="k") == "A short answer."


def test_text_rejects_a_stub_below_min_words(monkeypatch):
    body = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(agnes.requests, "post", lambda url, **kw: _Resp(200, body))
    monkeypatch.setenv("AGNES_TEXT_MODEL", "only-one")

    assert agnes.text([{"role": "user", "content": "hi"}], key="k", min_words=40) is None


def test_text_sends_temperature_but_no_cap_by_default(monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        sent.update(kwargs["json"])
        return _Resp(200, {"choices": [{"message": {"content": "fine answer here"}}]})

    monkeypatch.setattr(agnes.requests, "post", fake_post)
    agnes.text([{"role": "user", "content": "hi"}], key="k", temperature=0.7)

    assert sent["temperature"] == 0.7
    assert "max_tokens" not in sent  # an uncapped reply cannot be eaten by reasoning tokens
    assert sent["model"] == "agnes-2.5-flash"
