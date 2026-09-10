from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    root: Path
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5-coder:7b"
    workspace: str = "workspace"
    auto_apply_upgrades: bool = False
    auto_model_upgrades: bool = True
    model_upgrade_interval_minutes: int = 1440
    notify_file: str = ".lusas/notifications.jsonl"

    @property
    def workspace_root(self) -> Path:
        return (self.root / self.workspace).resolve()

    @property
    def notification_path(self) -> Path:
        return (self.root / self.notify_file).resolve()

    @classmethod
    def load(cls, root: Path) -> "Settings":
        root = root.expanduser().resolve()
        config_path = root / "config.json"
        if not config_path.exists():
            return cls(root=root)

        with config_path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)

        return cls(
            root=root,
            ollama_url=str(payload.get("ollama_url", cls.ollama_url)),
            model=str(payload.get("model", cls.model)),
            workspace=str(payload.get("workspace", cls.workspace)),
            auto_apply_upgrades=bool(
                payload.get("auto_apply_upgrades", cls.auto_apply_upgrades)
            ),
            auto_model_upgrades=bool(
                payload.get("auto_model_upgrades", cls.auto_model_upgrades)
            ),
            model_upgrade_interval_minutes=int(
                payload.get(
                    "model_upgrade_interval_minutes",
                    cls.model_upgrade_interval_minutes,
                )
            ),
            notify_file=str(payload.get("notify_file", cls.notify_file)),
        )
