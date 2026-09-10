"""Persistent failure lessons and regression evidence for future cycles."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def record_failure(root: Path, category: str, message: str, **context: Any) -> dict[str, Any]:
    normalized = f"{category}\n{message}".strip()
    fingerprint = sha256(normalized.encode("utf-8")).hexdigest()
    root = root.expanduser().resolve()
    root.joinpath(".lusas").mkdir(parents=True, exist_ok=True)
    state_path = root / ".lusas" / "failure_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except (OSError, ValueError):
        state = {}
    previous = state.get(fingerprint, {})
    occurrences = int(previous.get("occurrences", 0)) + 1 if isinstance(previous, dict) else 1
    state[fingerprint] = {
        "occurrences": occurrences,
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lesson = {
        "time": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "message": message,
        "fingerprint": fingerprint,
        "occurrences": occurrences,
        "lesson": f"Detect and address {category}: {message}",
        "context": context,
    }
    # Keep the lesson log useful during a persistent failure: record the first
    # occurrence and exponential repeat checkpoints, not one line every five
    # minutes forever.
    should_record = occurrences == 1 or occurrences & (occurrences - 1) == 0
    if should_record:
        with root.joinpath(".lusas", "lessons.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(lesson, ensure_ascii=True) + "\n")
    regression = {
        "created_at": lesson["time"],
        "failure_fingerprint": fingerprint,
        "category": category,
        "assertion": f"A future cycle must not repeat: {message}",
        "source_lesson": str(root / ".lusas" / "lessons.jsonl"),
    }
    if occurrences == 1:
        with root.joinpath(".lusas", "regressions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(regression, ensure_ascii=True) + "\n")
    return lesson
