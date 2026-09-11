"""Registry of capabilities that have real local implementations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import Settings


BUILTIN_SKILLS: tuple[dict[str, Any], ...] = (
    {
        "skill": "local_chat",
        "description": "Generate responses with the configured local model.",
        "required_tools": ["local model runtime"],
        "knowledge_dependencies": [],
    },
    {
        "skill": "bounded_web_research",
        "description": "Fetch HTTPS content only from administrator-allowlisted sources.",
        "required_tools": ["HTTPS fetch", "feed parser"],
        "knowledge_dependencies": ["source verification", "freshness metadata"],
    },
    {
        "skill": "sandboxed_code_evolution",
        "description": "Propose, security-check, test, benchmark, and stage source changes.",
        "required_tools": ["AST checker", "isolated candidate workspace", "unit tests"],
        "knowledge_dependencies": ["quality gates", "rollback"],
    },
    {
        "skill": "local_model_training",
        "description": "Train and independently evaluate a local adapter when dependencies are installed.",
        "required_tools": ["PyTorch", "Transformers", "PEFT"],
        "knowledge_dependencies": ["training examples", "evaluation set"],
    },
    {
        "skill": "versioned_rollback",
        "description": "Back up deployed files and restore a selected known-good state.",
        "required_tools": ["local filesystem", "version registry"],
        "knowledge_dependencies": ["deployment history"],
    },
)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid skill record on line {line_number}.")
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


def ensure_builtin(settings: Settings) -> list[dict[str, Any]]:
    """Register only capabilities backed by modules in this installation."""
    current = {item.get("skill"): item for item in _read(settings.skills_path)}
    now = datetime.now(timezone.utc).isoformat()
    for builtin in BUILTIN_SKILLS:
        name = builtin["skill"]
        item = current.setdefault(
            name,
            {
                **builtin,
                "success_rate": None,
                "benchmark_score": None,
                "failure_rate": None,
                "last_used": None,
                "last_verified": now,
                "version_introduced": "v0.000",
            },
        )
        item["last_verified"] = item.get("last_verified") or now
    result = list(current.values())
    _write(settings.skills_path, result)
    return result


def record_use(
    settings: Settings,
    skill: str,
    *,
    succeeded: bool,
    benchmark_score: float | None = None,
) -> dict[str, Any]:
    records = ensure_builtin(settings)
    item = next((entry for entry in records if entry.get("skill") == skill), None)
    if item is None:
        raise ValueError(f"Unknown skill: {skill}")
    item["last_used"] = datetime.now(timezone.utc).isoformat()
    uses = int(item.get("uses", 0)) + 1
    successes = int(item.get("successes", 0)) + int(succeeded)
    item["uses"] = uses
    item["successes"] = successes
    item["success_rate"] = successes / uses
    item["failure_rate"] = 1.0 - item["success_rate"]
    if benchmark_score is not None:
        item["benchmark_score"] = benchmark_score
    _write(settings.skills_path, records)
    return item
