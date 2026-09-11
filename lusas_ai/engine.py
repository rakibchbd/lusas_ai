"""Cycle coordinator and durable observability for autonomous evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import Settings


@dataclass
class EvolutionCycle:
    evolution_id: str
    started_at: str
    phases: list[dict[str, Any]] = field(default_factory=list)


class EvolutionEngine:
    """Own one observable cycle; specialist modules perform each phase's work."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def begin(self) -> EvolutionCycle:
        counter_path = self.settings.root / ".lusas" / "evolution-counter.json"
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(counter_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        count = int(payload.get("count", 0)) + 1 if isinstance(payload, dict) else 1
        temporary = counter_path.with_name(counter_path.name + ".tmp")
        temporary.write_text(json.dumps({"count": count}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(counter_path)
        return EvolutionCycle(
            evolution_id=f"EV-{count:06d}",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def phase(self, cycle: EvolutionCycle, name: str, **details: Any) -> None:
        cycle.phases.append(
            {
                "phase": name,
                "time": datetime.now(timezone.utc).isoformat(),
                "details": details,
            }
        )

    def complete(self, cycle: EvolutionCycle, **details: Any) -> dict[str, Any]:
        payload = {
            "evolution_id": cycle.evolution_id,
            "started_at": cycle.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "phases": cycle.phases,
            **details,
        }
        self.settings.evolution_cycles_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.evolution_cycles_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return payload


def read_cycles(settings: Settings) -> list[dict[str, Any]]:
    if not settings.evolution_cycles_path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in settings.evolution_cycles_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            result.append(payload)
    return result
