"""Deterministic self-auditing for the local LUSAS installation.

The audit deliberately observes the repository without importing project code or
executing files.  Its findings are evidence for later cycles, not a claim that
the system has tested properties it did not measure.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import Settings


AUDIT_ROOTS = ("lusas_ai", "training", "tests")
REQUIRED_FILES = (
    "config.json",
    "pyproject.toml",
    "lusas_ai/cli.py",
    "training/auto_upgrade.py",
)


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid audit record on line {line_number}.")
        records.append(payload)
    return records


def run(settings: Settings, evolution_id: str | None = None) -> dict[str, Any]:
    root = settings.root
    files: list[str] = []
    digest = hashlib.sha256()
    syntax_errors: list[str] = []
    python_files = 0
    test_files = 0
    total_bytes = 0

    for prefix in AUDIT_ROOTS:
        directory = root / prefix
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            files.append(relative)
            content = path.read_bytes()
            total_bytes += len(content)
            digest.update(relative.encode("utf-8"))
            digest.update(content)
            if path.suffix == ".py":
                python_files += 1
                try:
                    compile(content, relative, "exec")
                except (SyntaxError, ValueError) as exc:
                    syntax_errors.append(f"{relative}: {exc}")
            if path.name.startswith("test_") and path.suffix == ".py":
                test_files += 1

    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    findings = [f"missing required file: {item}" for item in missing]
    findings.extend(syntax_errors)
    result: dict[str, Any] = {
        "evolution_id": evolution_id,
        "time": datetime.now(timezone.utc).isoformat(),
        "status": "healthy" if not findings else "findings",
        "files": len(files),
        "python_files": python_files,
        "test_files": test_files,
        "total_bytes": total_bytes,
        "fingerprint": digest.hexdigest(),
        "missing_required_files": missing,
        "syntax_errors": syntax_errors,
        "findings": findings,
        "scope": list(AUDIT_ROOTS),
    }
    _append(settings.audits_path, result)
    return result
