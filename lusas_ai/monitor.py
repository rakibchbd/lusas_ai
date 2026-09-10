from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import time
from typing import Any

from .config import Settings
from .learning import LearningStore


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"Invalid notification on line {line_number}.")
        events.append(event)
    return events


def snapshot(settings: Settings, recent: int = 10) -> str:
    learned = LearningStore(settings.learning_path).examples()
    events = _read_events(settings.notification_path)
    upgrades = _read_events(settings.upgrade_log_path)
    state = (
        json.loads(settings.upgrade_state_path.read_text(encoding="utf-8"))
        if settings.upgrade_state_path.exists()
        else {"upgrade_count": 0}
    )
    lines = [
        "LUSAS learning monitor",
        f"Model: {settings.local_model_path}",
        f"Learned examples: {len(learned)}",
        f"Recorded events: {len(events)}",
        f"Successful upgrades: {state.get('upgrade_count', 0)}",
        f"Current version: {int(state.get('upgrade_count', 0)) / 1_000_000:.6f}",
    ]
    if learned:
        lines.append("\nLearned examples:")
        for index, example in enumerate(learned[-recent:], start=max(1, len(learned) - recent + 1)):
            lines.append(f"  [{index}] {example['instruction']}")
            lines.append(f"      -> {example['output'][:160]}")
    if events:
        lines.append("\nRecent events:")
        for event in events[-recent:]:
            lines.append(
                f"  {event.get('time', 'unknown time')} "
                f"{event.get('message', 'unknown event')}"
            )
    if upgrades:
        lines.append("\nUpgrade history:")
        for upgrade in upgrades[-recent:]:
            version = upgrade.get("version") or "-"
            lines.append(
                f"  {upgrade.get('time', 'unknown time')} "
                f"version={version} status={upgrade.get('status')} "
                f"score={upgrade.get('score', '-')}, "
                f"learned={upgrade.get('learned_examples', 0)}"
            )
    return "\n".join(lines)


def follow(settings: Settings, interval: float = 1.0) -> Iterator[str]:
    last_signature: tuple[int, int] | None = None
    while True:
        learned_stat = (
            settings.learning_path.stat()
            if settings.learning_path.exists()
            else None
        )
        event_stat = (
            settings.notification_path.stat()
            if settings.notification_path.exists()
            else None
        )
        signature = (
            learned_stat.st_mtime_ns if learned_stat else 0,
            event_stat.st_mtime_ns if event_stat else 0,
        )
        if signature != last_signature:
            last_signature = signature
            yield snapshot(settings)
        time.sleep(interval)
