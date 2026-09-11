from __future__ import annotations

from http.client import HTTPConnection
from pathlib import Path
import tempfile
import threading
import unittest

from lusas_ai.web_ui import create_server


class FakeAgent:
    def chat(self, prompt: str) -> str:
        return f"Echo: {prompt}"


class WebUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.server = create_server(
            Path(self.temporary.name),
            host="127.0.0.1",
            port=0,
            agent=FakeAgent(),  # type: ignore[arg-type]
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def test_serves_console_and_health_endpoint(self) -> None:
        self.connection.request("GET", "/")
        response = self.connection.getresponse()
        content = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn("LUSAS AI", content)
        self.assertIn("/styles.css", content)

        self.connection.request("GET", "/api/health")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('"ok": true', response.read().decode("utf-8"))

    def test_chat_endpoint_uses_existing_agent(self) -> None:
        self.connection.request(
            "POST",
            "/api/chat",
            body='{"prompt":"hello"}',
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('"response": "Echo: hello"', response.read().decode("utf-8"))

    def test_chat_endpoint_rejects_empty_prompt(self) -> None:
        self.connection.request(
            "POST",
            "/api/chat",
            body='{"prompt":"  "}',
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
