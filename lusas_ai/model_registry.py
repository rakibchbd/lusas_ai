"""The official LUSAS AI model catalog and filesystem layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any


class UnknownModelError(ValueError):
    """Raised when a request names a model outside the official catalog."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    foundation_model_env: str
    foundation_path_env: str
    description: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="sara-1.0",
        display_name="Sara 1.0",
        foundation_model_env="LUSAS_SARA_FOUNDATION_MODEL",
        foundation_path_env="LUSAS_SARA_FOUNDATION_PATH",
        description="The first official LUSAS general-purpose local model.",
    ),
    ModelSpec(
        model_id="lira-1.0",
        display_name="Lira 1.0",
        foundation_model_env="LUSAS_LIRA_FOUNDATION_MODEL",
        foundation_path_env="LUSAS_LIRA_FOUNDATION_PATH",
        description="The second official LUSAS general-purpose local model.",
    ),
)

MODEL_IDS = tuple(spec.model_id for spec in MODEL_SPECS)
_SPECS = {spec.model_id: spec for spec in MODEL_SPECS}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _SPECS[model_id]
    except KeyError as exc:
        raise UnknownModelError(
            f"Unknown LUSAS model ID {model_id!r}. Choose Sara 1.0 or Lira 1.0."
        ) from exc


def model_root(settings: Any, model_id: str) -> Path:
    get_model_spec(model_id)
    return (settings.root / settings.model_storage / model_id).resolve()


def stable_path(settings: Any, model_id: str) -> Path:
    return model_root(settings, model_id) / "stable"


def candidates_path(settings: Any, model_id: str) -> Path:
    return model_root(settings, model_id) / "candidates"


def backups_path(settings: Any, model_id: str) -> Path:
    return model_root(settings, model_id) / "backups"


def foundation_config(settings: Any, model_id: str) -> dict[str, str | None]:
    """Resolve a foundation only from explicit per-model configuration.

    There is deliberately no default foundation and no cross-model fallback.
    """
    spec = get_model_spec(model_id)
    configured_models = getattr(settings, "foundation_models", None) or {}
    configured = configured_models.get(model_id, {})
    model = os.environ.get(spec.foundation_model_env) or configured.get("model")
    path = os.environ.get(spec.foundation_path_env) or configured.get("path")
    return {
        "model": model.strip() if isinstance(model, str) and model.strip() else None,
        "path": path.strip() if isinstance(path, str) and path.strip() else None,
    }


def selected_model_id(settings: Any) -> str:
    path = settings.selected_model_path
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model_id = payload.get("model_id") if isinstance(payload, dict) else None
            if isinstance(model_id, str) and model_id in MODEL_IDS:
                return model_id
        except (OSError, json.JSONDecodeError):
            pass
    return settings.default_model_id


def select_model(settings: Any, model_id: str) -> str:
    get_model_spec(model_id)
    path = settings.selected_model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps({"model_id": model_id}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return model_id


def _candidate_count(settings: Any, model_id: str) -> int:
    path = candidates_path(settings, model_id)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_dir())


def catalog(settings: Any) -> list[dict[str, Any]]:
    active = selected_model_id(settings)
    result: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        foundation = foundation_config(settings, spec.model_id)
        stable = stable_path(settings, spec.model_id)
        metadata_path = stable / "model.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, json.JSONDecodeError):
                metadata = {}
        result.append(
            {
                **asdict(spec),
                "active": spec.model_id == active,
                "stable_available": stable.is_dir(),
                "candidate_count": _candidate_count(settings, spec.model_id),
                "foundation_configured": bool(foundation["model"] or foundation["path"]),
                "foundation_model": foundation["model"],
                "foundation_path": foundation["path"],
                "stable_version": metadata.get("version"),
                "deployment_status": metadata.get("deployment_status", "not_installed"),
            }
        )
    return result
