"""Local web console for chatting with and inspecting LUSAS AI."""

from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .agent import LusasAgent
from .config import Settings
from .local_model import LocalModelError
from .monitor import report


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
MAX_BODY_BYTES = 64 * 1024
MAX_PROMPT_CHARS = 8_000
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class LusasRequestHandler(BaseHTTPRequestHandler):
    """Serve the bundled UI and a small JSON API."""

    server: "LusasWebServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        # Keep the terminal useful by logging only API errors and lifecycle output.
        if self.path.startswith("/api/") and args and str(args[0]).startswith("4"):
            super().log_message(format, *args)

    def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.path.startswith("/api/"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if urlsplit(self.path).path != "/api/chat":
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "lusas-web"})
            return
        if path == "/api/status":
            try:
                self._send_json(HTTPStatus.OK, report(self.server.settings, recent=12))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        asset = _STATIC_FILES.get(path)
        if asset is None:
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        filename, content_type = asset
        try:
            body = (WEB_ROOT / filename).read_bytes()
        except OSError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_bytes(HTTPStatus.OK, content_type, body)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlsplit(self.path).path
        if path != "/api/chat":
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length or "0")
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid request body length")
            return
        if body_length < 1 or body_length > MAX_BODY_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "Request body is too large or empty")
            return
        try:
            payload = json.loads(self.rfile.read(body_length))
        except (json.JSONDecodeError, OSError):
            self._error(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON")
            return
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            self._error(HTTPStatus.BAD_REQUEST, "prompt must be a non-empty string")
            return
        prompt = prompt.strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            self._error(HTTPStatus.BAD_REQUEST, "prompt is too long")
            return
        try:
            with self.server.chat_lock:
                response = self.server.agent.chat(prompt)
        except LocalModelError as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        except (OSError, RuntimeError, ValueError) as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_json(HTTPStatus.OK, {"response": response})


class LusasWebServer(ThreadingHTTPServer):
    """Threaded server with one shared, lazily-loaded local agent."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        settings: Settings,
        agent: LusasAgent | None = None,
    ) -> None:
        self.settings = settings
        self.agent = agent or LusasAgent(settings.root)
        self.chat_lock = threading.Lock()
        super().__init__(server_address, LusasRequestHandler)


def create_server(
    root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    agent: LusasAgent | None = None,
) -> LusasWebServer:
    """Create a local server, optionally with an injected agent for tests."""
    if not 0 <= port <= 65_535:
        raise ValueError("web port must be between 0 and 65535")
    return LusasWebServer((host, port), Settings.load(root), agent=agent)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local web console until interrupted."""
    server = create_server(root, host=host, port=port)
    address = server.server_address
    print(f"LUSAS AI web UI: http://{address[0]}:{address[1]}", flush=True)
    print("Press Ctrl-C to stop the web UI.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.shutdown()
        server.server_close()
