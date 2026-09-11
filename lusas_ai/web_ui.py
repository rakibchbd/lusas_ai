"""Local LUSAS AI console and administrator control API."""

from __future__ import annotations

from http import HTTPStatus
import hmac
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .admin_db import AdminStore
from .agent import LusasAgent
from .config import Settings
from .local_model import LocalModelError
from .model_registry import MODEL_IDS, catalog, select_model, selected_model_id
from .monitor import report
from training.model_lifecycle import promote, restore_backup


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
MAX_BODY_BYTES = 64 * 1024
MAX_PROMPT_CHARS = 8_000
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/admin": ("admin.html", "text/html; charset=utf-8"),
    "/admin.html": ("admin.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/admin.js": ("admin.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class LusasRequestHandler(BaseHTTPRequestHandler):
    server: "LusasWebServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
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
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-LUSAS-Admin-Token")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=True).encode("utf-8"))

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def _payload(self) -> dict[str, Any] | None:
        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length or "0")
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid request body length")
            return None
        if body_length < 1 or body_length > MAX_BODY_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "Request body is too large or empty")
            return None
        try:
            payload = json.loads(self.rfile.read(body_length))
        except (json.JSONDecodeError, OSError):
            self._error(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON")
            return None
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")
            return None
        return payload

    def _admin_actor(self) -> str | None:
        expected = os.environ.get(self.server.settings.admin_token_env, "")
        supplied = self.headers.get("X-LUSAS-Admin-Token", "")
        if not expected or not supplied or not hmac.compare_digest(expected, supplied):
            return None
        return "administrator"

    def _require_admin(self) -> str | None:
        actor = self._admin_actor()
        if actor is None:
            self._error(HTTPStatus.FORBIDDEN, "Administrator authentication is required.")
        return actor

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not urlsplit(self.path).path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-LUSAS-Admin-Token")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "lusas-web"})
            return
        if path == "/api/models":
            self._send_json(HTTPStatus.OK, {"selected_model_id": selected_model_id(self.server.settings), "models": catalog(self.server.settings)})
            return
        if path == "/api/status":
            try:
                self._send_json(HTTPStatus.OK, report(self.server.settings, recent=12))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        if path.startswith("/api/admin/"):
            actor = self._require_admin()
            if actor is None:
                return
            store = self.server.admin_store
            try:
                if path == "/api/admin/overview":
                    self._send_json(HTTPStatus.OK, store.overview())
                elif path == "/api/admin/knowledge":
                    self._send_json(HTTPStatus.OK, {"records": store.knowledge()})
                elif path == "/api/admin/training":
                    self._send_json(HTTPStatus.OK, {"jobs": store.jobs()})
                elif path == "/api/admin/audit":
                    self._send_json(HTTPStatus.OK, {"events": store.audit_events()})
                else:
                    self._error(HTTPStatus.NOT_FOUND, "Not found")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        asset = _STATIC_FILES.get(path)
        if asset is None:
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        filename, content_type = asset
        try:
            self._send_bytes(HTTPStatus.OK, content_type, (WEB_ROOT / filename).read_bytes())
        except OSError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        payload = self._payload()
        if payload is None:
            return
        try:
            if path == "/api/chat":
                prompt = payload.get("prompt")
                model_id = payload.get("model_id")
                if not isinstance(prompt, str) or not prompt.strip():
                    self._error(HTTPStatus.BAD_REQUEST, "prompt must be a non-empty string")
                    return
                if not isinstance(model_id, str) or model_id not in MODEL_IDS:
                    self._error(HTTPStatus.BAD_REQUEST, "model_id must be sara-1.0 or lira-1.0")
                    return
                prompt = prompt.strip()
                if len(prompt) > MAX_PROMPT_CHARS:
                    self._error(HTTPStatus.BAD_REQUEST, "prompt is too long")
                    return
                with self.server.chat_lock:
                    response = self.server.agent.chat(prompt, model_id=model_id)
                self._send_json(HTTPStatus.OK, {"response": response, "model_id": model_id})
                return
            if path == "/api/models/select":
                model_id = payload.get("model_id")
                if not isinstance(model_id, str) or model_id not in MODEL_IDS:
                    self._error(HTTPStatus.BAD_REQUEST, "model_id must be sara-1.0 or lira-1.0")
                    return
                select_model(self.server.settings, model_id)
                self.server.admin_store.audit("model_selected", model_id, "local-user")
                self._send_json(HTTPStatus.OK, {"selected_model_id": model_id})
                return
            actor = self._require_admin()
            if actor is None:
                return
            store = self.server.admin_store
            if path == "/api/admin/knowledge/review":
                result = store.review_knowledge(str(payload.get("record_id", "")), str(payload.get("status", "")), actor)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/admin/sources":
                self._send_json(HTTPStatus.OK, store.add_source(str(payload.get("url", "")), actor))
                return
            if path == "/api/admin/training/start":
                model_id = str(payload.get("model_id", ""))
                if model_id not in MODEL_IDS:
                    self._error(HTTPStatus.BAD_REQUEST, "Unknown official model ID")
                    return
                approved_count = len(store.training_records())
                self._send_json(HTTPStatus.OK, store.start_training(model_id, actor, approved_count))
                return
            if path == "/api/admin/deployments/approve":
                result = store.approve_deployment(
                    str(payload.get("model_id", "")),
                    str(payload.get("candidate_path", "")),
                    payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {},
                    actor,
                )
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/admin/deployments/deploy":
                model_id = str(payload.get("model_id", ""))
                candidate_path = str(payload.get("candidate_path", ""))
                approval = store.approved_deployment(model_id, candidate_path)
                if approval is None:
                    raise ValueError("This candidate has no recorded administrator approval. Approve it first.")
                evaluation = approval["evaluation"]
                approved_candidate = Path(approval["candidate_path"])
                backup = promote(approved_candidate, self.server.settings.root, True, model_id=model_id, admin_approved=True, evaluation=evaluation, approved_by=actor)
                store.audit("deployment_applied", model_id, actor, candidate_path=str(approved_candidate), backup=str(backup))
                self._send_json(HTTPStatus.OK, {"model_id": model_id, "status": "stable", "backup": str(backup)})
                return
            if path == "/api/admin/deployments/rollback":
                model_id = str(payload.get("model_id", ""))
                backup_value = Path(str(payload.get("backup_path", ""))).expanduser()
                backup = backup_value if backup_value.is_absolute() else self.server.settings.root / backup_value
                restore_backup(backup, self.server.settings.root, model_id=model_id)
                store.audit("deployment_rollback", model_id, actor, backup=str(backup))
                self._send_json(HTTPStatus.OK, {"model_id": model_id, "status": "rolled_back"})
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found")
        except LocalModelError as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))


class LusasWebServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], settings: Settings, agent: LusasAgent | None = None) -> None:
        self.settings = settings
        self.agent = agent or LusasAgent(settings.root)
        self.admin_store = AdminStore(settings)
        self.chat_lock = threading.Lock()
        super().__init__(server_address, LusasRequestHandler)


def create_server(root: Path, host: str = "127.0.0.1", port: int = 8765, *, agent: LusasAgent | None = None) -> LusasWebServer:
    if not 0 <= port <= 65_535:
        raise ValueError("web port must be between 0 and 65535")
    return LusasWebServer((host, port), Settings.load(root), agent=agent)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
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
