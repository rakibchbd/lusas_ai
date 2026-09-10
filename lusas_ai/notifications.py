from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _apple_script_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def notify(root: Path, notification_path: Path, message: str, **details: Any) -> None:
    """Record a notification and optionally show it in the macOS UI."""
    del root
    notification_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "details": details,
    }
    with notification_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if sys.platform == "darwin":
        safe_message = _apple_script_escape(message)
        script = (
            f'display notification "{safe_message}" '
            'with title "LUSAS AI"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
