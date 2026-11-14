"""Publish retries: transient failures are retried in-run, permanent ones are not.

The failure these lock down: one 503 from Google, or a connection dropped mid-upload, ended the
run with "queue untouched, retrying tomorrow". The account was fine and the file was fine — the
request just needed a second try. Equally, retrying a 401/403 wastes the run, so those must stop
immediately.
"""

import unittest
from unittest.mock import MagicMock, patch

import publish


def response(status: int, *, body: dict | None = None, headers: dict | None = None,
             text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.text = text or str(body or "")
    resp.json.return_value = body or {}
    return resp


class TestTokenExchange(unittest.TestCase):
    def setUp(self):
        self._backoff = publish.BACKOFF_S
        publish.BACKOFF_S = 0

    def tearDown(self):
        publish.BACKOFF_S = self._backoff

    def test_transient_error_is_retried(self):
        with patch("publish.requests.post",
                   side_effect=[response(503), response(200, body={"access_token": "tok"})]) as post:
            self.assertEqual(publish.exchange_refresh_token("i", "s", "r"), "tok")
        self.assertEqual(post.call_count, 2)

    def test_a_dropped_connection_is_retried(self):
        import requests
        with patch("publish.requests.post",
                   side_effect=[requests.ConnectionError("reset"),
                                response(200, body={"access_token": "tok"})]) as post:
            self.assertEqual(publish.exchange_refresh_token("i", "s", "r"), "tok")
        self.assertEqual(post.call_count, 2)

    def test_invalid_grant_is_not_retried(self):
        with patch("publish.requests.post",
                   return_value=response(400, text="invalid_grant")) as post:
            self.assertIsNone(publish.exchange_refresh_token("i", "s", "r"))
        self.assertEqual(post.call_count, 1)

    def test_a_long_outage_gives_up(self):
        with patch("publish.requests.post", return_value=response(503)) as post:
            self.assertIsNone(publish.exchange_refresh_token("i", "s", "r"))
        self.assertEqual(post.call_count, publish.PUBLISH_ATTEMPTS)


class TestUploadVideo(unittest.TestCase):
    def setUp(self):
        self._backoff = publish.BACKOFF_S
        publish.BACKOFF_S = 0
        self.video = b"x" * 1000

    def tearDown(self):
        publish.BACKOFF_S = self._backoff

    def upload(self, *, init=None, put=None):
        patches = []
        if init is not None:
            patches.append(patch("publish.requests.post", side_effect=init))
        if put is not None:
            patches.append(patch("publish.requests.put", side_effect=put))
        started = [p.start() for p in patches]
        try:
            return publish.upload_video(self.video, "Title", "desc", "token"), started
        finally:
            for p in patches:
                p.stop()

    def test_a_happy_upload_returns_the_video_id(self):
        video_id, (init, put) = self.upload(
            init=[response(200, headers={"Location": "https://upload"})],
            put=[response(201, body={"id": "abc123"})])
        self.assertEqual(video_id, "abc123")
        self.assertEqual(init.call_count, 1)
        self.assertEqual(put.call_count, 1)

    def test_a_transient_init_failure_is_retried(self):
        video_id, (init, _put) = self.upload(
            init=[response(503), response(200, headers={"Location": "https://upload"})],
            put=[response(200, body={"id": "abc123"})])
        self.assertEqual(video_id, "abc123")
        self.assertEqual(init.call_count, 2)

    def test_an_auth_failure_stops_immediately(self):
        video_id, (init, put) = self.upload(
            init=[response(401, text="invalid credentials")],
            put=[response(200, body={"id": "nope"})])
        self.assertIsNone(video_id)
        self.assertEqual(init.call_count, 1)
        self.assertEqual(put.call_count, 0)

    def test_a_transient_put_failure_is_retried(self):
        video_id, (_init, put) = self.upload(
            init=[response(200, headers={"Location": "https://upload"})],
            put=[response(503), response(200, body={"id": "abc123"})])
        self.assertEqual(video_id, "abc123")
        self.assertEqual(put.call_count, 2)

    def test_a_308_resumes_from_the_stored_byte(self):
        # 308 = "resume incomplete": 400 of the 1000 bytes are already stored.
        video_id, (_init, put) = self.upload(
            init=[response(200, headers={"Location": "https://upload"})],
            put=[response(308, headers={"Range": "bytes=0-399"}),
                 response(200, body={"id": "abc123"})])
        self.assertEqual(video_id, "abc123")
        self.assertEqual(put.call_count, 2)
        resumed = put.call_args_list[1].kwargs
        self.assertEqual(resumed["headers"]["Content-Range"], "bytes 400-999/1000")
        self.assertEqual(len(resumed["data"]), 600)

    def test_the_first_chunk_has_no_content_range(self):
        _video_id, (_init, put) = self.upload(
            init=[response(200, headers={"Location": "https://upload"})],
            put=[response(200, body={"id": "abc123"})])
        self.assertNotIn("Content-Range", put.call_args_list[0].kwargs["headers"])

    def test_a_session_without_a_location_is_a_failure(self):
        video_id, (_init, put) = self.upload(
            init=[response(200)], put=[response(200, body={"id": "abc123"})])
        self.assertIsNone(video_id)
        self.assertEqual(put.call_count, 0)


class TestThumbnailAndPrivacy(unittest.TestCase):
    def setUp(self):
        self._backoff = publish.BACKOFF_S
        publish.BACKOFF_S = 0

    def tearDown(self):
        publish.BACKOFF_S = self._backoff

    def test_an_unverified_channel_is_not_retried(self):
        with patch("publish.requests.post", return_value=response(403, text="forbidden")) as post:
            self.assertFalse(publish.upload_thumbnail("v", "tok", b"jpg"))
        self.assertEqual(post.call_count, 1)

    def test_a_transient_thumbnail_failure_is_retried(self):
        with patch("publish.requests.post",
                   side_effect=[response(500), response(200)]) as post:
            self.assertTrue(publish.upload_thumbnail("v", "tok", b"jpg"))
        self.assertEqual(post.call_count, 2)

    def test_privacy_flip_retries_transient_errors(self):
        with patch("publish.requests.put",
                   side_effect=[response(502), response(200)]) as put:
            self.assertTrue(publish.set_privacy("v", "tok", "public"))
        self.assertEqual(put.call_count, 2)

    def test_privacy_flip_gives_up_on_a_permanent_error(self):
        with patch("publish.requests.put", return_value=response(403)) as put:
            self.assertFalse(publish.set_privacy("v", "tok", "public"))
        self.assertEqual(put.call_count, 1)


if __name__ == "__main__":
    unittest.main()
