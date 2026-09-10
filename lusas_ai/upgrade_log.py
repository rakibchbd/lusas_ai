from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def next_version(state_path: Path) -> str:
    current = 0
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        current = int(payload.get("upgrade_count", 0))
    current += 1
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"upgrade_count": current}, indent=2) + "\n",
        encoding="utf-8",
    )
    return f"{current / 1_000_000:.6f}"


def record(log_path: Path, **event: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
