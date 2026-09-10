from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class NotificationResult:
    recorded: bool
    native_attempted: bool
    native_delivered: bool
    backend: str | None = None
    error: str | None = None


def _apple_script_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _run_native(message: str) -> tuple[bool, str | None, str | None]:
    """Best-effort native notification with an explicit result."""
    if sys.platform == "darwin":
        if not shutil.which("osascript"):
            return False, "osascript", "osascript is not available"
        script = (
            f'display notification "{_apple_script_escape(message)}" '
            'with title "LUSAS AI"'
        )
        backend = "macos"
        command = ["osascript", "-e", script]
    elif sys.platform.startswith("win"):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            return False, "windows-toast", "PowerShell is not available"
        # Windows.Data.Xml.Dom.XmlDocument avoids a third-party dependency.
        escaped = html.escape(message, quote=True)
        script = (
            "[Windows.UI.Notifications.ToastNotificationManager, "
            "Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; "
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
            "ContentType = WindowsRuntime] > $null; "
            f"$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; "
            f"$xml.LoadXml('<toast><visual><binding template=\"ToastText01\">"
            f"<text>{escaped}</text></binding></visual></toast>'); "
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier"
            "('LUSAS AI').Show($toast)"
        )
        backend = "windows-toast"
        command = [powershell, "-NoProfile", "-NonInteractive", "-Command", script]
    elif sys.platform.startswith("linux"):
        notify_send = shutil.which("notify-send")
        if not notify_send:
            return False, "linux-notify-send", "notify-send is not available"
        backend = "linux-notify-send"
        command = [notify_send, "LUSAS AI", message]
    else:
        return False, None, f"unsupported platform: {sys.platform}"

    try:
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, backend, str(exc)
    if completed.returncode:
        return False, backend, (
            completed.stderr.strip() or f"native notifier exited {completed.returncode}"
        )
    return True, backend, None


class NotificationService:
    """Cross-platform native desktop notification adapter."""

    def send(self, message: str) -> NotificationResult:
        delivered, backend, error = _run_native(message)
        return NotificationResult(
            recorded=False,
            native_attempted=backend is not None,
            native_delivered=delivered,
            backend=backend,
            error=error,
        )


def notify(root: Path, notification_path: Path, message: str, **details: Any) -> NotificationResult:
    """Persist every event and attempt a platform-native notification.

    Native delivery is deliberately best effort: a missing desktop session
    must not make an otherwise safe upgrade appear to have failed.
    """
    del root
    native = NotificationService().send(message)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "details": details,
        "native": {
            "attempted": native.native_attempted,
            "delivered": native.native_delivered,
            "backend": native.backend,
            "error": native.error,
        },
    }
    try:
        notification_path.parent.mkdir(parents=True, exist_ok=True)
        with notification_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        return NotificationResult(
            recorded=False,
            native_attempted=native.native_attempted,
            native_delivered=native.native_delivered,
            backend=native.backend,
            error=f"notification log failed: {exc}",
        )
    return NotificationResult(
        recorded=True,
        native_attempted=native.native_attempted,
        native_delivered=native.native_delivered,
        backend=native.backend,
        error=native.error,
    )
