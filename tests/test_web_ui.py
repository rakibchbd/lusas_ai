from __future__ import annotations

from http.client import HTTPConnection
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from lusas_ai.web_ui import create_server


class FakeAgent:
    def chat(self, prompt: str, model_id: str | None = None) -> str:
        return f"Echo: {prompt} ({model_id})"


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
        self.assertIn("styles.css", content)

        self.connection.request("GET", "/api/health")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('"ok": true', response.read().decode("utf-8"))

        self.connection.request("GET", "/admin.html")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn("Official models", response.read().decode("utf-8"))

    def test_model_catalog_and_selection_endpoint(self) -> None:
        self.connection.request("GET", "/api/models")
        response = self.connection.getresponse()
        body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn('"model_id": "sara-1.0"', body)
        self.assertIn('"display_name": "Lira 1.0"', body)

        self.connection.request(
            "POST", "/api/models/select", body='{"model_id":"lira-1.0"}',
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('"selected_model_id": "lira-1.0"', response.read().decode("utf-8"))

    def test_admin_api_requires_token(self) -> None:
        with patch.dict("os.environ", {"LUSAS_ADMIN_TOKEN": "test-secret"}, clear=False):
            self.connection.request("GET", "/api/admin/overview")
            response = self.connection.getresponse()
            self.assertEqual(response.status, 403)

            self.connection.request("GET", "/api/admin/overview", headers={"X-LUSAS-Admin-Token": "test-secret"})
            response = self.connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("training_jobs", response.read().decode("utf-8"))

    def test_chat_endpoint_uses_existing_agent(self) -> None:
        self.connection.request(
            "POST",
            "/api/chat",
            body='{"prompt":"hello","model_id":"sara-1.0"}',
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('"response": "Echo: hello (sara-1.0)"', response.read().decode("utf-8"))

    def test_chat_endpoint_rejects_empty_prompt(self) -> None:
        self.connection.request(
            "POST",
            "/api/chat",
            body='{"prompt":"  ","model_id":"sara-1.0"}',
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
