"""Persistent, incremental maintenance for the autonomous evolution loop.

This module contains the work that can run between user requests: it records
served interactions, processes them into gap signals, checks knowledge
retention with real retrieval tasks, maintains incremental file fingerprints,
and enforces a small resource budget. It never claims that a model improved
unless a real practice or evaluation operation produced evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any

from .admin_db import AdminStore
from .config import Settings
from .control import read as read_control
from .gaps import record as record_gap
from .knowledge import ensure_seed, records as knowledge_records, retrieve
from .skills import ensure_builtin


_FAILURE_MARKERS = (
    "i don't know",
    "i do not know",
    "not installed",
    "could not",
    "error:",
    "failed",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _file_digest(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def incremental_ingestion(settings: Settings) -> dict[str, Any]:
    """Detect changed inputs and refresh only the affected local indexes."""
    paths = (
        settings.root / "training" / "data" / "examples.jsonl",
        settings.learning_path,
        settings.web_cache_path,
        settings.web_training_path,
        settings.knowledge_path,
    )
    previous = _read_json(settings.incremental_index_path).get("files", {})
    previous = previous if isinstance(previous, dict) else {}
    current: dict[str, str | None] = {}
    changed: list[str] = []
    for path in paths:
        try:
            relative = str(path.relative_to(settings.root))
        except ValueError:
            # macOS temporary directories can resolve through /private while
            # the project root retains its /var spelling. The absolute key is
            # still stable for this installation and avoids losing state.
            relative = str(path)
        digest = _file_digest(path)
        current[relative] = digest
        if relative not in previous or previous.get(relative) != digest:
            changed.append(relative)
    if changed:
        # ensure_seed invalidates the retrieval cache. Learned/web ingestion
        # already normalizes records at write time, so no full reprocessing is
        # needed here.
        ensure_seed(settings)
        try:
            knowledge_key = str(settings.knowledge_path.relative_to(settings.root))
        except ValueError:
            knowledge_key = str(settings.knowledge_path)
        current[knowledge_key] = _file_digest(settings.knowledge_path)
    _write_json(
        settings.incremental_index_path,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": current,
            "changed_files": changed,
        },
    )
    return {"changed_files": changed, "changed_count": len(changed)}


def resource_snapshot(settings: Settings) -> dict[str, Any]:
    """Capture real process and storage usage for background scheduling."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = float(usage.ru_maxrss) / (1024 * 1024 if sys.platform == "darwin" else 1024)
    disk = shutil.disk_usage(settings.root)
    return {
        "rss_mb": round(rss_mb, 2),
        "disk_free_mb": round(disk.free / (1024 * 1024), 2),
        "memory_budget_mb": settings.background_max_memory_mb,
        "within_memory_budget": rss_mb <= settings.background_max_memory_mb,
        "within_time_budget": True,
    }


def _practice(settings: Settings, store: AdminStore, limit: int = 12) -> dict[str, Any]:
    attempted = 0
    passed = 0
    failed = 0
    for item in knowledge_records(settings):
        if attempted >= max(1, limit):
            break
        if item.get("kind") not in {"fact", "learned_fact"}:
            continue
        knowledge_id = str(item.get("id", item.get("knowledge_id", "")))
        if not knowledge_id or store.practice_seen(knowledge_id):
            continue
        topic = str(item.get("topic", item.get("content", ""))).strip()
        if not topic:
            continue
        matches = retrieve(settings, topic, limit=5)
        retained = any(
            str(match.get("id", match.get("knowledge_id", ""))) == knowledge_id
            for match in matches
        )
        status = "passed" if retained else "failed"
        score = 1.0 if retained else 0.0
        practice = store.record_practice(
            knowledge_id,
            f"Retrieve and use the concept: {topic}",
            status,
            score,
            {"retrieved_ids": [str(match.get("id", "")) for match in matches]},
        )
        _append_jsonl(
            settings.practice_path,
            {
                **practice,
                "task": f"Retrieve and use the concept: {topic}",
                "evidence": {"retrieved_ids": [str(match.get("id", "")) for match in matches]},
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        attempted += 1
        if retained:
            passed += 1
        else:
            failed += 1
            record_gap(
                settings,
                topic,
                "Retention practice could not retrieve the stored concept.",
                priority=0.7,
                evidence={"knowledge_id": knowledge_id, "practice_status": status},
            )
    return {"attempted": attempted, "passed": passed, "failed": failed}


def _process_interactions(settings: Settings, store: AdminStore, limit: int = 20) -> dict[str, int]:
    processed = 0
    failures = 0
    for item in store.pending_interactions(limit=limit):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip().lower()
        failed = any(marker in answer for marker in _FAILURE_MARKERS)
        if failed:
            failures += 1
            record_gap(
                settings,
                question or "unanswered interaction",
                "A served interaction contained a failure or uncertainty marker.",
                priority=0.8,
                evidence={
                    "interaction_id": item.get("interaction_id"),
                    "model_id": item.get("model_id"),
                    "web_search_used": bool(item.get("web_search_used")),
                },
            )
        store.complete_interaction(
            str(item.get("interaction_id", "")),
            status="needs_review" if failed else "processed",
            answer_quality="failed" if failed else "unrated",
            confidence=0.2 if failed else 0.7,
        )
        processed += 1
    return {"processed": processed, "failures": failures}


def maintain(settings: Settings, *, evolution_id: str | None = None) -> dict[str, Any]:
    """Run one bounded maintenance slice with durable evidence."""
    started = time.monotonic()
    controls = read_control(settings.root, settings.autonomy_level)
    if not settings.continuous_learning_enabled:
        return {"status": "disabled", "reason": "continuous learning is disabled"}
    if controls.get("emergency_stop") or controls.get("learning_paused"):
        return {"status": "paused", "reason": "runtime learning controls are paused"}

    store = AdminStore(settings)
    incremental = incremental_ingestion(settings)
    interactions = _process_interactions(settings, store)
    resources = resource_snapshot(settings)
    elapsed = time.monotonic() - started
    resources["within_time_budget"] = elapsed <= settings.background_max_seconds
    practice = (
        _practice(settings, store)
        if resources["within_memory_budget"] and resources["within_time_budget"]
        else {"attempted": 0, "passed": 0, "failed": 0, "skipped": True}
    )

    for skill in ensure_builtin(settings):
        failure_rate = skill.get("failure_rate")
        if isinstance(failure_rate, (int, float)) and failure_rate > 0.25:
            task = store.enqueue_task(
                "capability_gap",
                min(1.0, 0.5 + float(failure_rate)),
                {"skill": skill.get("skill"), "failure_rate": failure_rate},
            )
            record_gap(
                settings,
                str(skill.get("skill", "capability")),
                "Capability success rate is below the configured threshold.",
                priority=float(task["priority"]),
                evidence={"failure_rate": failure_rate, "task_id": task["task_id"]},
            )

    result = {
        "status": "completed",
        "evolution_id": evolution_id,
        "incremental": incremental,
        "interactions": interactions,
        "practice": practice,
        "resources": resources,
        "queue": store.continuous_counts(),
    }
    state = _read_json(settings.continuous_state_path)
    state.update(
        {
            "status": result["status"],
            "last_cycle": datetime.now(timezone.utc).isoformat(),
            "last_evolution_id": evolution_id,
            "last_result": result,
        }
    )
    _write_json(settings.continuous_state_path, state)
    return result


def state(settings: Settings) -> dict[str, Any]:
    """Read the last durable maintenance state without changing it."""
    return _read_json(settings.continuous_state_path)
