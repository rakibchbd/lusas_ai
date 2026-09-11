"""Evidence-backed multidimensional scorecards for LUSAS versions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .control import read as read_control
from .knowledge import is_outdated, records as knowledge_records


DIMENSIONS = (
    "intelligence",
    "knowledge",
    "coding",
    "tools",
    "memory",
    "research",
    "autonomy",
    "reliability",
    "security",
    "performance",
)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            result.append(payload)
    return result


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def measure(
    settings: Settings,
    *,
    audit_result: dict[str, Any] | None = None,
    model_upgrade: dict[str, Any] | None = None,
    code_upgrade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    knowledge = knowledge_records(settings)
    controls = read_control(settings.root, settings.autonomy_level)
    lessons_path = settings.root / ".lusas" / "lessons.jsonl"
    regressions_path = settings.root / ".lusas" / "regressions.jsonl"
    lessons = len(_read(lessons_path))
    regressions = len(_read(regressions_path))
    verified = sum(1 for item in knowledge if item.get("verification") == "corroborated")
    outdated = sum(1 for item in knowledge if is_outdated(item))

    dimensions: dict[str, dict[str, Any]] = {
        "intelligence": {
            "score": None,
            "available": False,
            "evidence": "No independent reasoning benchmark is configured.",
        },
        "knowledge": {
            "score": (verified / len(knowledge) * 100) if knowledge else None,
            "available": bool(knowledge),
            "evidence": f"{verified}/{len(knowledge)} records corroborated; {outdated} outdated.",
        },
        "coding": {
            "score": (
                float((model_upgrade or {}).get("score", 0.0)) * 100
                if (model_upgrade or {}).get("score") is not None
                else None
            ),
            "available": (model_upgrade or {}).get("score") is not None,
            "evidence": "Candidate model evaluation score; not a general intelligence score.",
        },
        "tools": {
            "score": None,
            "available": False,
            "evidence": "No independent tool-selection benchmark is configured.",
        },
        "memory": {
            "score": None,
            "available": False,
            "evidence": f"{lessons} lessons and {regressions} regression records exist; retrieval quality is unmeasured.",
        },
        "research": {
            "score": (verified / len(knowledge) * 100) if knowledge else None,
            "available": bool(knowledge),
            "evidence": "Source corroboration coverage of locally ingested web evidence.",
        },
        "autonomy": {
            "score": None,
            "available": False,
            "evidence": f"Configured maximum autonomy level: {controls['autonomy_level']}/5; behavior is not inferred from configuration.",
        },
        "reliability": {
            "score": 100.0 if audit_result and audit_result.get("status") == "healthy" and regressions == 0 else None,
            "available": bool(audit_result),
            "evidence": "Repository syntax audit and recorded regression count.",
        },
        "security": {
            "score": None,
            "available": False,
            "evidence": "Static candidate security checks exist; a complete vulnerability score is not claimed.",
        },
        "performance": {
            "score": (
                100.0
                if (code_upgrade or {}).get("benchmark_seconds") is not None
                and (code_upgrade or {}).get("status") in {"promoted", "staged"}
                else None
            ),
            "available": (code_upgrade or {}).get("benchmark_seconds") is not None,
            "evidence": "Candidate test benchmark timing is recorded when code evolution runs.",
        },
    }
    available_scores = [
        float(item["score"])
        for item in dimensions.values()
        if item.get("available") and item.get("score") is not None
    ]
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "dimensions": dimensions,
        "overall_score": (
            sum(available_scores) / len(available_scores) if available_scores else None
        ),
        "measured_dimensions": len(available_scores),
        "score_note": "Only measured dimensions are averaged; unavailable capabilities remain null.",
    }


def record(
    settings: Settings,
    *,
    version: str,
    evolution_id: str | None = None,
    audit_result: dict[str, Any] | None = None,
    model_upgrade: dict[str, Any] | None = None,
    code_upgrade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = measure(
        settings,
        audit_result=audit_result,
        model_upgrade=model_upgrade,
        code_upgrade=code_upgrade,
    )
    payload.update({"version": version, "evolution_id": evolution_id})
    _append(settings.scorecards_path, payload)
    return payload


def history(settings: Settings) -> list[dict[str, Any]]:
    return _read(settings.scorecards_path)
