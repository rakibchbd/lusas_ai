"""Runtime controls for pausing and bounding autonomous behavior."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any


CONTROL_FILE = ".lusas/control.json"
CONTROL_KEYS = {
    "autonomy_level",
    "learning_paused",
    "upgrades_paused",
    "web_research_enabled",
    "self_code_enabled",
    "auto_deploy_enabled",
    "production_locked",
    "emergency_stop",
}


def path_for(root: Path) -> Path:
    return root.expanduser().resolve() / CONTROL_FILE


def defaults(max_autonomy_level: int = 5) -> dict[str, Any]:
    return {
        "autonomy_level": max(0, min(5, int(max_autonomy_level))),
        "learning_paused": False,
        "upgrades_paused": False,
        "web_research_enabled": True,
        "self_code_enabled": True,
        "auto_deploy_enabled": True,
        "production_locked": False,
        "emergency_stop": False,
    }


def read(root: Path, max_autonomy_level: int = 5) -> dict[str, Any]:
    """Read runtime controls, falling back safely when the file is absent."""
    result = defaults(max_autonomy_level)
    control_path = path_for(root)
    if not control_path.exists():
        return result
    try:
        payload = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result
    if not isinstance(payload, dict):
        return result
    result.update({key: payload[key] for key in CONTROL_KEYS if key in payload})
    try:
        result["autonomy_level"] = max(
            0,
            min(defaults(max_autonomy_level)["autonomy_level"], int(result["autonomy_level"])),
        )
    except (TypeError, ValueError):
        result["autonomy_level"] = defaults(max_autonomy_level)["autonomy_level"]
    for key in CONTROL_KEYS - {"autonomy_level"}:
        result[key] = bool(result[key])
    return result


def write(root: Path, state: dict[str, Any], max_autonomy_level: int = 5) -> Path:
    """Atomically persist only recognized control fields."""
    normalized = read_from_payload(
        {**read(root, max_autonomy_level), **state}, max_autonomy_level
    )
    control_path = path_for(root)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = control_path.with_name(control_path.name + ".tmp")
    temporary.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(control_path)
    return control_path


def read_from_payload(payload: dict[str, Any], max_autonomy_level: int = 5) -> dict[str, Any]:
    result = defaults(max_autonomy_level)
    result.update({key: payload[key] for key in CONTROL_KEYS if key in payload})
    try:
        result["autonomy_level"] = max(
            0,
            min(defaults(max_autonomy_level)["autonomy_level"], int(result["autonomy_level"])),
        )
    except (TypeError, ValueError):
        result["autonomy_level"] = defaults(max_autonomy_level)["autonomy_level"]
    for key in CONTROL_KEYS - {"autonomy_level"}:
        result[key] = bool(result[key])
    return result


def update(root: Path, max_autonomy_level: int = 5, **changes: Any) -> dict[str, Any]:
    unknown = set(changes) - CONTROL_KEYS
    if unknown:
        raise ValueError(f"Unknown runtime control(s): {', '.join(sorted(unknown))}")
    state = read(root, max_autonomy_level)
    state.update(changes)
    write(root, state, max_autonomy_level)
    return read(root, max_autonomy_level)


def automatic_deploy_allowed(state: dict[str, Any]) -> bool:
    return (
        int(state.get("autonomy_level", 0)) >= 4
        and not state.get("emergency_stop", False)
        and not state.get("upgrades_paused", False)
        and bool(state.get("auto_deploy_enabled", False))
        and not state.get("production_locked", False)
    )


def code_evolution_allowed(state: dict[str, Any]) -> bool:
    return (
        int(state.get("autonomy_level", 0)) >= 2
        and not state.get("emergency_stop", False)
        and not state.get("upgrades_paused", False)
        and bool(state.get("self_code_enabled", False))
    )


def model_learning_allowed(state: dict[str, Any]) -> bool:
    return (
        int(state.get("autonomy_level", 0)) >= 2
        and not state.get("emergency_stop", False)
        and not state.get("learning_paused", False)
    )


def web_research_allowed(state: dict[str, Any]) -> bool:
    return (
        not state.get("emergency_stop", False)
        and not state.get("learning_paused", False)
        and bool(state.get("web_research_enabled", False))
    )
