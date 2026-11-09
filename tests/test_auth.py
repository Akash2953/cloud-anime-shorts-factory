"""OAuth helper tests: .env upsert must never clobber other credentials."""

import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from tools.youtube_auth import _Catcher, load_env, mask, set_env_values


class TestEnvUpsert(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / ".env"
        self.path.write_text(
            "# comment stays\nCFAI_TOKEN=abc\nYOUTUBE_CLIENT_ID=old-id\n\nYOUTUBE_REFRESH_TOKEN=old-token\n",
            encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def test_replaces_existing_keys_in_place(self):
        set_env_values({"YOUTUBE_CLIENT_ID": "new-id"}, self.path)
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("YOUTUBE_CLIENT_ID=new-id", text)
        self.assertNotIn("old-id", text)
        self.assertIn("# comment stays", text)
        self.assertIn("CFAI_TOKEN=abc", text)

    def test_appends_missing_key_without_touching_others(self):
        set_env_values({"YOUTUBE_CLIENT_SECRET": "shh"}, self.path)
        env = load_env(self.path)
        self.assertEqual(env["YOUTUBE_CLIENT_SECRET"], "shh")
        self.assertEqual(env["CFAI_TOKEN"], "abc")
        self.assertEqual(env["YOUTUBE_REFRESH_TOKEN"], "old-token")
        self.assertTrue(self.path.read_text(encoding="utf-8").startswith("# comment stays"))

    def test_creates_file_when_absent(self):
        fresh = Path(self.dir.name) / "new.env"
        set_env_values({"A": "1"}, fresh)
        self.assertEqual(load_env(fresh), {"A": "1"})

    def test_mask_hides_the_body(self):
        token = "1//0gVeryLongSecretRefreshTokenValue"
        shown = mask(token)
        self.assertNotIn("VeryLongSecret", shown)
        self.assertIn(str(len(token)), shown)


class TestLoopbackCatcher(unittest.TestCase):
    def test_captures_code_from_redirect(self):
        import http.server
        _Catcher.code = None
        _Catcher.error = None
        server = http.server.HTTPServer(("127.0.0.1", 0), _Catcher)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/?code=THE_CODE&scope=x", timeout=10) as r:
                self.assertEqual(r.status, 200)
        finally:
            server.shutdown()
        self.assertEqual(_Catcher.code, "THE_CODE")

    def test_records_denial(self):
        import http.server
        _Catcher.code = None
        _Catcher.error = None
        server = http.server.HTTPServer(("127.0.0.1", 0), _Catcher)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/?error=access_denied", timeout=10).read()
        finally:
            server.shutdown()
        self.assertEqual(_Catcher.error, "access_denied")
        self.assertIsNone(_Catcher.code)


if __name__ == "__main__":
    unittest.main()
