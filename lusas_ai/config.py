from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _relative_setting(value: Any, default: str) -> str:
    """Keep config-controlled paths inside the project root."""
    if not isinstance(value, str) or not value.strip():
        return default
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return default
    return value


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _autonomy_level(value: Any, default: int) -> int:
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    root: Path
    model_path: str = "models/production"
    learning_file: str = ".lusas/learned.jsonl"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.0
    num_ctx: int = 8192
    num_predict: int = 128
    seed: int | None = 42
    workspace: str = "workspace"
    autonomy_level: int = 5
    auto_apply_upgrades: bool = False
    auto_model_upgrades: bool = True
    auto_code_upgrades: bool = False
    model_upgrade_interval_minutes: int = 5
    evolution_enabled: bool = False
    evolution_interval_minutes: int = 5
    notify_file: str = ".lusas/notifications.jsonl"
    upgrade_log_file: str = ".lusas/upgrade_log.jsonl"
    upgrade_state_file: str = ".lusas/upgrade_state.json"
    cycle_state_file: str = ".lusas/cycle_state.json"
    upgrade_diff_directory: str = ".lusas/upgrade-diffs"
    progress_file: str = ".lusas/evolution-progress.jsonl"
    web_learning_enabled: bool = False
    web_refresh_interval_minutes: int = 60
    web_sources: tuple[str, ...] = ()
    web_allowed_domains: tuple[str, ...] = ()
    web_max_items: int = 100
    web_cache_file: str = ".lusas/web_articles.jsonl"
    web_training_file: str = ".lusas/web_training.jsonl"
    web_state_file: str = ".lusas/web_state.json"
    knowledge_file: str = ".lusas/knowledge.jsonl"
    knowledge_gaps_file: str = ".lusas/knowledge_gaps.jsonl"
    skills_file: str = ".lusas/skills.jsonl"
    audits_file: str = ".lusas/audits.jsonl"
    scorecards_file: str = ".lusas/scorecards.jsonl"
    evolution_cycles_file: str = ".lusas/evolution_cycles.jsonl"

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
            model_path=_relative_setting(payload.get("model_path"), cls.model_path),
            learning_file=_relative_setting(
                payload.get("learning_file"), cls.learning_file
            ),
            temperature=float(payload.get("temperature", cls.temperature)),
            top_p=float(payload.get("top_p", cls.top_p)),
            top_k=_positive_int(payload.get("top_k"), cls.top_k),
            repeat_penalty=float(
                payload.get("repeat_penalty", cls.repeat_penalty)
            ),
            num_ctx=_positive_int(payload.get("num_ctx"), cls.num_ctx),
            num_predict=_positive_int(payload.get("num_predict"), cls.num_predict),
            seed=(
                None
                if payload.get("seed", cls.seed) is None
                else int(payload.get("seed", cls.seed))
            ),
            workspace=_relative_setting(payload.get("workspace"), cls.workspace),
            autonomy_level=_autonomy_level(
                payload.get("autonomy_level"), cls.autonomy_level
            ),
            auto_apply_upgrades=bool(
                payload.get("auto_apply_upgrades", cls.auto_apply_upgrades)
            ),
            auto_model_upgrades=bool(
                payload.get("auto_model_upgrades", cls.auto_model_upgrades)
            ),
            auto_code_upgrades=bool(
                payload.get("auto_code_upgrades", cls.auto_code_upgrades)
            ),
            model_upgrade_interval_minutes=_positive_int(
                payload.get("model_upgrade_interval_minutes"),
                cls.model_upgrade_interval_minutes,
            ),
            evolution_enabled=bool(
                payload.get("evolution_enabled", cls.evolution_enabled)
            ),
            evolution_interval_minutes=_positive_int(
                payload.get("evolution_interval_minutes"),
                cls.evolution_interval_minutes,
            ),
            notify_file=_relative_setting(
                payload.get("notify_file"), cls.notify_file
            ),
            upgrade_log_file=_relative_setting(
                payload.get("upgrade_log_file"), cls.upgrade_log_file
            ),
            upgrade_state_file=_relative_setting(
                payload.get("upgrade_state_file"), cls.upgrade_state_file
            ),
            cycle_state_file=_relative_setting(
                payload.get("cycle_state_file"), cls.cycle_state_file
            ),
            upgrade_diff_directory=_relative_setting(
                payload.get("upgrade_diff_directory"), cls.upgrade_diff_directory
            ),
            progress_file=_relative_setting(
                payload.get("progress_file"), cls.progress_file
            ),
            web_learning_enabled=bool(
                payload.get("web_learning_enabled", cls.web_learning_enabled)
            ),
            web_refresh_interval_minutes=_positive_int(
                payload.get("web_refresh_interval_minutes"),
                cls.web_refresh_interval_minutes,
            ),
            web_sources=_string_tuple(payload.get("web_sources"), cls.web_sources),
            web_allowed_domains=_string_tuple(
                payload.get("web_allowed_domains"), cls.web_allowed_domains
            ),
            web_max_items=_positive_int(
                payload.get("web_max_items"), cls.web_max_items
            ),
            web_cache_file=_relative_setting(
                payload.get("web_cache_file"), cls.web_cache_file
            ),
            web_training_file=_relative_setting(
                payload.get("web_training_file"), cls.web_training_file
            ),
            web_state_file=_relative_setting(
                payload.get("web_state_file"), cls.web_state_file
            ),
            knowledge_file=_relative_setting(
                payload.get("knowledge_file"), cls.knowledge_file
            ),
            knowledge_gaps_file=_relative_setting(
                payload.get("knowledge_gaps_file"), cls.knowledge_gaps_file
            ),
            skills_file=_relative_setting(
                payload.get("skills_file"), cls.skills_file
            ),
            audits_file=_relative_setting(
                payload.get("audits_file"), cls.audits_file
            ),
            scorecards_file=_relative_setting(
                payload.get("scorecards_file"), cls.scorecards_file
            ),
            evolution_cycles_file=_relative_setting(
                payload.get("evolution_cycles_file"), cls.evolution_cycles_file
            ),
        )

    @property
    def local_model_path(self) -> Path:
        return (self.root / self.model_path).resolve()

    @property
    def learning_path(self) -> Path:
        return (self.root / self.learning_file).resolve()

    @property
    def upgrade_log_path(self) -> Path:
        return (self.root / self.upgrade_log_file).resolve()

    @property
    def upgrade_state_path(self) -> Path:
        return (self.root / self.upgrade_state_file).resolve()

    @property
    def cycle_state_path(self) -> Path:
        return (self.root / self.cycle_state_file).resolve()

    @property
    def upgrade_diff_root(self) -> Path:
        return (self.root / self.upgrade_diff_directory).resolve()

    @property
    def progress_path(self) -> Path:
        return (self.root / self.progress_file).resolve()

    @property
    def web_cache_path(self) -> Path:
        return (self.root / self.web_cache_file).resolve()

    @property
    def web_training_path(self) -> Path:
        return (self.root / self.web_training_file).resolve()

    @property
    def web_state_path(self) -> Path:
        return (self.root / self.web_state_file).resolve()

    @property
    def knowledge_path(self) -> Path:
        return (self.root / self.knowledge_file).resolve()

    @property
    def knowledge_gaps_path(self) -> Path:
        return (self.root / self.knowledge_gaps_file).resolve()

    @property
    def skills_path(self) -> Path:
        return (self.root / self.skills_file).resolve()

    @property
    def audits_path(self) -> Path:
        return (self.root / self.audits_file).resolve()

    @property
    def scorecards_path(self) -> Path:
        return (self.root / self.scorecards_file).resolve()

    @property
    def evolution_cycles_path(self) -> Path:
        return (self.root / self.evolution_cycles_file).resolve()
