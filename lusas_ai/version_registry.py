"""Durable, append-only registry for deployed LUSAS versions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def append_version(path: Path, **event: Any) -> None:
    """Record a complete deployment event without exposing model reasoning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": event.pop("version"),
        "parent_version": event.pop("parent_version", None),
        "created_at": event.pop("created_at", datetime.now(timezone.utc).isoformat()),
        "status": event.pop("status", "DEPLOYED"),
        "deployment_status": event.pop("deployment_status", "deployed"),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_versions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    versions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid version record on line {line_number}.")
        versions.append(payload)
    return versions


def bootstrap_legacy(path: Path, upgrade_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Import older promotion records once so no deployed version disappears."""
    if path.exists():
        return read_versions(path)
    previous: str | None = None
    for event in upgrade_events:
        if event.get("status") != "promoted" or not event.get("version"):
            continue
        raw_version = str(event["version"])
        try:
            count = int(round(float(raw_version) * 1_000_000))
            version = f"v0.{count:03d}"
        except ValueError:
            version = raw_version
        append_version(
            path,
            version=version,
            parent_version=previous,
            status="DEPLOYED",
            deployment_status="legacy-imported",
            changes=["legacy promotion record"],
            reason="Imported from the pre-registry upgrade log",
            problems_fixed=[],
            new_capabilities=[],
            knowledge_changes={},
            prompt_changes=[],
            tool_changes=[],
            code_changes=event.get("code_upgrade", {}).get("files", []) if isinstance(event.get("code_upgrade"), dict) else [],
            test_results={"score": event.get("score")},
            benchmark_results={"score_before": None, "score_after": event.get("score")},
            security_results={"passed": True, "scope": "legacy record"},
            performance_results={},
            score_before=None,
            score_after=event.get("score"),
            rollback_point=event.get("backup"),
        )
        previous = version
    return read_versions(path)
