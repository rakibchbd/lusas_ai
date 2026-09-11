from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

from .config import Settings
from .control import automatic_deploy_allowed, read as read_control
from .notifications import notify


MANAGED_DIRECTORIES = ("lusas_ai", "tests", "training")
PROTECTED_PARTS = {
    ".git",
    ".lusas",
    ".github",
    "config.json",
    "pyproject.toml",
    "models",
    "workspace",
}


@dataclass(frozen=True)
class TestResult:
    passed: bool
    output: str


@dataclass(frozen=True)
class UpgradeResult:
    backup_path: Path
    staging_path: Path
    tests: TestResult
    applied: bool


def _managed_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory_name in MANAGED_DIRECTORIES:
        directory = root / directory_name
        if directory.exists():
            files.extend(
                path for path in directory.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    return sorted(files)


def _validate_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if (
        path.is_absolute()
        or "\\" in relative_path
        or not path.parts
        or ".." in path.parts
        or path.parts[0] not in MANAGED_DIRECTORIES
        or any(part in PROTECTED_PARTS for part in path.parts)
    ):
        raise ValueError(
            f"Self-upgrades may only change files under {MANAGED_DIRECTORIES}."
        )
    return path


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def create_backup(root: Path) -> Path:
    backup_path = root / ".lusas" / "backups" / _run_id()
    backup_path.mkdir(parents=True, exist_ok=False)
    manifest: list[str] = []
    for source in _managed_files(root):
        relative = source.relative_to(root)
        destination = backup_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append(str(relative))
    (backup_path / "manifest.json").write_text(
        json.dumps({"files": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return backup_path


def stage_candidate(root: Path, changes: Mapping[str, str]) -> Path:
    validated_changes = [
        (_validate_relative_path(relative_name), content)
        for relative_name, content in changes.items()
    ]
    for relative, content in validated_changes:
        existing = root / relative
        if relative.parts[0] == "tests" and existing.is_file():
            if existing.read_text(encoding="utf-8") != content:
                raise ValueError(
                    f"Existing regression test is immutable: {relative}. "
                    "Add a new test file instead of weakening the quality gate."
                )
    staging_path = root / ".lusas" / "staging" / _run_id()
    staging_path.mkdir(parents=True, exist_ok=False)

    for source in _managed_files(root):
        relative = source.relative_to(root)
        destination = staging_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for relative, content in validated_changes:
        destination = staging_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return staging_path


def run_tests(candidate_root: Path) -> TestResult:
    command = [
        sys.executable, "-m", "unittest", "discover",
        "-s", "tests", "-p", "test_*.py",
    ]
    completed = subprocess.run(
        command,
        cwd=candidate_root,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return TestResult(passed=completed.returncode == 0, output=output)


def apply_candidate(root: Path, staging_path: Path) -> None:
    for source in _managed_files(staging_path):
        relative = source.relative_to(staging_path)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def restore_backup(root: Path, backup_path: Path) -> None:
    backup_root = (root / ".lusas" / "backups").resolve()
    backup_path = backup_path.expanduser().resolve()
    if backup_path != backup_root and backup_root not in backup_path.parents:
        raise ValueError("Rollback source must be inside .lusas/backups.")
    if not backup_path.is_dir():
        raise ValueError("Rollback source does not exist.")
    apply_candidate(root, backup_path)


def perform_upgrade(
    settings: Settings,
    changes: Mapping[str, str],
    goal: str,
    apply: bool = False,
) -> UpgradeResult:
    root = settings.root
    controls = read_control(root, settings.autonomy_level)
    if controls.get("emergency_stop") or controls.get("production_locked"):
        raise ValueError("Self-upgrade is blocked by runtime production controls.")
    if apply and controls.get("upgrades_paused"):
        raise ValueError("Self-upgrade is paused by runtime production controls.")
    backup_path = create_backup(root)
    staging_path = stage_candidate(root, changes)
    tests = run_tests(staging_path)
    applied = False

    automatic_deploy = (
        settings.auto_apply_upgrades and automatic_deploy_allowed(controls)
    )
    if tests.passed and (apply or automatic_deploy):
        apply_candidate(root, staging_path)
        applied = True

    if tests.passed and applied:
        message = "Self-upgrade tested and applied."
    elif tests.passed:
        message = "Self-upgrade staged and tested; not applied."
    else:
        message = "Self-upgrade staged but tests failed; not applied."

    notify(
        root,
        settings.notification_path,
        message,
        goal=goal,
        backup=str(backup_path),
        staging=str(staging_path),
        applied=applied,
        tests_passed=tests.passed,
    )
    return UpgradeResult(
        backup_path=backup_path,
        staging_path=staging_path,
        tests=tests,
        applied=applied,
    )
