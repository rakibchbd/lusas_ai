from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    root: Path
    model_path: str = "models/production"
    learning_file: str = ".lusas/learned.jsonl"
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 8192
    num_predict: int = 1024
    seed: int | None = 42
    workspace: str = "workspace"
    auto_apply_upgrades: bool = False
    auto_model_upgrades: bool = True
    auto_publish_upgrades: bool = True
    model_upgrade_interval_minutes: int = 1440
    notify_file: str = ".lusas/notifications.jsonl"
    web_pending_file: str = ".lusas/web_pending.jsonl"
    upgrade_log_file: str = ".lusas/upgrade_log.jsonl"
    upgrade_state_file: str = ".lusas/upgrade_state.json"

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
            model_path=str(payload.get("model_path", cls.model_path)),
            learning_file=str(payload.get("learning_file", cls.learning_file)),
            temperature=float(payload.get("temperature", cls.temperature)),
            top_p=float(payload.get("top_p", cls.top_p)),
            top_k=int(payload.get("top_k", cls.top_k)),
            repeat_penalty=float(
                payload.get("repeat_penalty", cls.repeat_penalty)
            ),
            num_ctx=int(payload.get("num_ctx", cls.num_ctx)),
            num_predict=int(payload.get("num_predict", cls.num_predict)),
            seed=(
                None
                if payload.get("seed", cls.seed) is None
                else int(payload.get("seed", cls.seed))
            ),
            workspace=str(payload.get("workspace", cls.workspace)),
            auto_apply_upgrades=bool(
                payload.get("auto_apply_upgrades", cls.auto_apply_upgrades)
            ),
            auto_model_upgrades=bool(
                payload.get("auto_model_upgrades", cls.auto_model_upgrades)
            ),
            auto_publish_upgrades=bool(
                payload.get("auto_publish_upgrades", cls.auto_publish_upgrades)
            ),
            model_upgrade_interval_minutes=int(
                payload.get(
                    "model_upgrade_interval_minutes",
                    cls.model_upgrade_interval_minutes,
                )
            ),
            notify_file=str(payload.get("notify_file", cls.notify_file)),
            web_pending_file=str(
                payload.get("web_pending_file", cls.web_pending_file)
            ),
            upgrade_log_file=str(
                payload.get("upgrade_log_file", cls.upgrade_log_file)
            ),
            upgrade_state_file=str(
                payload.get("upgrade_state_file", cls.upgrade_state_file)
            ),
        )

    @property
    def local_model_path(self) -> Path:
        return (self.root / self.model_path).resolve()

    @property
    def learning_path(self) -> Path:
        return (self.root / self.learning_file).resolve()

    @property
    def web_pending_path(self) -> Path:
        return (self.root / self.web_pending_file).resolve()

    @property
    def upgrade_log_path(self) -> Path:
        return (self.root / self.upgrade_log_file).resolve()

    @property
    def upgrade_state_path(self) -> Path:
        return (self.root / self.upgrade_state_file).resolve()
