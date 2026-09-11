"""Evidence-based knowledge-gap detection and research prioritization."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .knowledge import is_outdated, records as knowledge_records


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid gap record on line {line_number}.")
        result.append(payload)
    return result


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=True) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def record(
    settings: Settings,
    topic: str,
    reason: str,
    *,
    priority: float = 0.5,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topic = topic.strip()
    reason = reason.strip()
    if not topic or not reason:
        raise ValueError("A knowledge gap needs a topic and reason.")
    fingerprint = sha256(f"{topic}\n{reason}".encode("utf-8")).hexdigest()
    current = _read(settings.knowledge_gaps_path)
    match = next((item for item in current if item.get("fingerprint") == fingerprint), None)
    now = datetime.now(timezone.utc).isoformat()
    if match is None:
        match = {
            "fingerprint": fingerprint,
            "topic": topic,
            "reason": reason,
            "priority": max(0.0, min(1.0, float(priority))),
            "occurrences": 0,
            "status": "open",
            "created_at": now,
            "last_seen": now,
            "evidence": evidence or {},
        }
        current.append(match)
    match["occurrences"] = int(match.get("occurrences", 0)) + 1
    match["last_seen"] = now
    if evidence:
        match["evidence"] = evidence
    _write(settings.knowledge_gaps_path, current)
    return match


def prioritized(settings: Settings) -> list[dict[str, Any]]:
    return sorted(
        _read(settings.knowledge_gaps_path),
        key=lambda item: (
            float(item.get("priority", 0.0)),
            int(item.get("occurrences", 0)),
            str(item.get("last_seen", "")),
        ),
        reverse=True,
    )


def detect(
    settings: Settings,
    audit_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create objectives only from observed failures, conflicts, or staleness."""
    if audit_result:
        for finding in audit_result.get("findings", []):
            record(
                settings,
                "repository-integrity",
                str(finding),
                priority=0.95,
                evidence={"audit_fingerprint": audit_result.get("fingerprint")},
            )

    knowledge = knowledge_records(settings)
    outdated = sum(1 for item in knowledge if is_outdated(item))
    conflicting = sum(1 for item in knowledge if item.get("verification") == "conflicting")
    if outdated:
        record(
            settings,
            "outdated-knowledge",
            f"{outdated} stored knowledge item(s) exceeded their freshness window.",
            priority=0.7,
            evidence={"outdated_items": outdated},
        )
    if conflicting:
        record(
            settings,
            "conflicting-sources",
            f"{conflicting} stored knowledge item(s) have conflicting source evidence.",
            priority=0.85,
            evidence={"conflicting_items": conflicting},
        )

    upgrade_path = settings.upgrade_log_path
    if upgrade_path.exists():
        events = _read(upgrade_path)
        rejected = [item for item in events if item.get("status") == "rejected"]
        if rejected:
            latest = rejected[-1]
            record(
                settings,
                "autonomous-upgrade-quality",
                str(latest.get("reason", "A candidate was rejected by a quality gate.")),
                priority=0.8,
                evidence={"latest_rejection": latest},
            )
    return prioritized(settings)
