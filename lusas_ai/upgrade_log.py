from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_version(count: int) -> str:
    """Format the user-visible monotonically increasing LUSAS version."""
    return f"v0.{count:03d}"


def current_count(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return int(payload.get("upgrade_count", 0))


def next_version(state_path: Path) -> str:
    current = current_count(state_path) + 1
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {"upgrade_count": current, "current_version": format_version(current)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return format_version(current)


def record(log_path: Path, **event: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
