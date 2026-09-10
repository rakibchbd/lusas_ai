"""Five-minute local evolution scheduler.

The scheduler is intentionally a thin wrapper: training/auto_upgrade.py
remains the long-running service entry point, while this class makes the
evolution cadence testable and prevents overlapping cycles.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import time
from typing import Callable, Iterator

from .config import Settings
from .evolution import EvolutionOrchestrator, EvolutionResult


class EvolutionScheduler:
    def __init__(
        self,
        settings: Settings,
        model_factory: Callable[[], object],
        interval_minutes: int | None = None,
    ):
        self.settings = settings
        self.model_factory = model_factory
        self.interval_minutes = interval_minutes or settings.evolution_interval_minutes
        if self.interval_minutes < 1:
            raise ValueError("Evolution interval must be at least one minute.")

    @property
    def lock_path(self) -> Path:
        return self.settings.root / ".lusas" / "evolution.lock"

    @contextmanager
    def lock(self) -> Iterator[bool]:
        """Use an advisory lock so a crashed worker cannot leave a stale lock."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def run_once(self, goal: str, apply: bool = False) -> EvolutionResult | None:
        if not self.settings.evolution_enabled:
            return None
        with self.lock() as acquired:
            if not acquired:
                return None
            return EvolutionOrchestrator(
                self.settings, self.model_factory()
            ).run(goal, apply=apply)

    def run_forever(self, goal: str, apply: bool = False) -> None:
        while True:
            self.run_once(goal, apply=apply)
            time.sleep(self.interval_minutes * 60)
