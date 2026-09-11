from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import time
from typing import Any

from .config import Settings
from .control import read as read_control
from .engine import read_cycles
from .gaps import prioritized
from .knowledge import is_outdated, records as knowledge_records
from .learning import LearningStore
from .scorecard import history as scorecard_history
from .skills import ensure_builtin
from .upgrade_log import current_count, format_version
from .version_registry import bootstrap_legacy


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


def report(settings: Settings, recent: int = 10) -> dict[str, Any]:
    """Return a JSON-serializable, HTTP-free local upgrade dashboard."""
    upgrades = _read_events(settings.upgrade_log_path)
    progress = _read_events(settings.progress_path)
    controls = read_control(settings.root, settings.autonomy_level)
    knowledge = knowledge_records(settings)
    skills = ensure_builtin(settings)
    gaps = prioritized(settings)
    cycles = read_cycles(settings)
    scorecards = scorecard_history(settings)
    state = {"upgrade_count": current_count(settings.upgrade_state_path)}
    versions_path = settings.root / ".lusas" / "versions.jsonl"
    lessons_path = settings.root / ".lusas" / "lessons.jsonl"
    regressions_path = settings.root / ".lusas" / "regressions.jsonl"
    return {
        "policy": {
            "evolution_enabled": settings.evolution_enabled,
            "auto_apply_upgrades": settings.auto_apply_upgrades,
            "auto_code_upgrades": settings.auto_code_upgrades,
            "autonomy_level": controls["autonomy_level"],
            "runtime": controls,
        },
        "upgrade_count": state.get("upgrade_count", 0),
        "current_version": format_version(int(state.get("upgrade_count", 0))),
        "web_learning": {
            "enabled": settings.web_learning_enabled,
            "interval_minutes": settings.web_refresh_interval_minutes,
            "cached_articles": sum(1 for _ in settings.web_cache_path.read_text(encoding="utf-8").splitlines()) if settings.web_cache_path.exists() else 0,
            "knowledge_items": len(knowledge),
            "verified_items": sum(1 for item in knowledge if item.get("verification") == "corroborated"),
            "outdated_items": sum(1 for item in knowledge if is_outdated(item)),
        },
        "knowledge": {
            "total": len(knowledge),
            "verified": sum(1 for item in knowledge if item.get("verification") == "corroborated"),
            "outdated": sum(1 for item in knowledge if is_outdated(item)),
            "domains": sorted({str(item.get("domain", "general")) for item in knowledge}),
        },
        "knowledge_graph": {
            "nodes": len(knowledge),
            "edges": sum(len(item.get("concepts", [])) for item in knowledge),
        },
        "knowledge_gaps": {
            "open": len([item for item in gaps if item.get("status", "open") == "open"]),
            "items": gaps[:recent],
        },
        "skills": {
            "total": len(skills),
            "verified": sum(1 for item in skills if item.get("last_verified")),
            "items": skills,
        },
        "scorecard": scorecards[-1] if scorecards else None,
        "current_evolution": cycles[-1] if cycles else None,
        "evolution_cycles": cycles[-recent:],
        "versions": bootstrap_legacy(versions_path, upgrades)[-recent:],
        "lessons_count": len(_read_events(lessons_path)),
        "regressions_count": len(_read_events(regressions_path)),
        "total_history": len(upgrades),
        "recent_upgrades": upgrades[-recent:],
        "recent_progress": progress[-recent:],
    }


def snapshot(settings: Settings, recent: int = 10) -> str:
    learned = LearningStore(settings.learning_path).examples()
    knowledge = knowledge_records(settings)
    gaps = prioritized(settings)
    controls = read_control(settings.root, settings.autonomy_level)
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
        f"Current version: {format_version(int(state.get('upgrade_count', 0)))}",
        f"Web articles cached: {settings.web_cache_path.read_text(encoding='utf-8').count(chr(10)) if settings.web_cache_path.exists() else 0}",
        f"Knowledge: {len(knowledge)} items, {sum(1 for item in knowledge if item.get('verification') == 'corroborated')} verified",
        f"Open knowledge gaps: {len([item for item in gaps if item.get('status', 'open') == 'open'])}",
        f"Autonomy level: {controls['autonomy_level']}/5",
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
                f"learned={upgrade.get('learned_examples', 0)} "
                f"deployment={upgrade.get('deployment', '-')}"
            )
    progress = _read_events(settings.progress_path)
    if progress:
        lines.append("\nRecent evolution progress:")
        for event in progress[-recent:]:
            lines.append(
                f"  {event.get('time', 'unknown time')} "
                f"[{event.get('phase', '-')}] {event.get('message', '')} "
                f"({event.get('status', 'running')})"
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
