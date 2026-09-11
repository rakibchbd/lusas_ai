"""Approval-gated versioning, deployment, backup, and rollback for official models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from lusas_ai.config import Settings
from lusas_ai.model_registry import (
    MODEL_IDS,
    backups_path,
    candidates_path,
    get_model_spec,
    stable_path,
)


REQUIRED_EVALUATION_GATES = frozenset(
    {"accuracy", "hallucination", "safety", "regression", "english", "bengali", "latency", "resources"}
)


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def backup_stable(root: Path, model_id: str, stable: Path | None = None) -> Path:
    settings = Settings.load(root)
    get_model_spec(model_id)
    source = stable or stable_path(settings, model_id)
    backup = backups_path(settings, model_id) / run_id()
    backup.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copytree(source, backup)
    else:
        backup.mkdir()
    (backup / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "source": str(source),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return backup


def _evaluation_is_complete(evaluation: dict[str, Any] | None) -> bool:
    if not isinstance(evaluation, dict) or evaluation.get("passed") is not True:
        return False
    gates = evaluation.get("gates")
    return isinstance(gates, dict) and REQUIRED_EVALUATION_GATES.issubset(gates) and all(
        gates[name] is True for name in REQUIRED_EVALUATION_GATES
    )


def promote(
    candidate: Path,
    root: Path,
    evaluation_passed: bool,
    *,
    model_id: str = "sara-1.0",
    admin_approved: bool = False,
    evaluation: dict[str, Any] | None = None,
    approved_by: str | None = None,
) -> Path:
    """Deploy only a fully evaluated candidate after explicit admin approval."""
    get_model_spec(model_id)
    if not evaluation_passed or not _evaluation_is_complete(evaluation):
        raise ValueError("Candidate must pass every required evaluation gate before promotion.")
    if not admin_approved or not approved_by:
        raise ValueError("An administrator approval is required before deployment.")
    candidate = candidate.expanduser().resolve()
    settings = Settings.load(root)
    allowed_root = candidates_path(settings, model_id).resolve()
    if allowed_root != candidate and allowed_root not in candidate.parents:
        raise ValueError("Candidate must be inside the selected model's candidates directory.")
    if not candidate.is_dir():
        raise ValueError("Candidate directory does not exist.")
    metadata_path = candidate / "model.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict) and metadata.get("model_id") not in {None, model_id}:
            raise ValueError("Candidate metadata does not match the selected model.")

    stable = stable_path(settings, model_id)
    backup = backup_stable(root, model_id, stable)
    model_root = stable.parent
    incoming = model_root / f".stable-{run_id()}"
    previous = model_root / f".previous-stable-{run_id()}"
    shutil.copytree(candidate, incoming)
    metadata = {
        "model_id": model_id,
        "display_name": get_model_spec(model_id).display_name,
        "version": datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S"),
        "deployment_status": "stable",
        "approved_by": approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation,
    }
    (incoming / "model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    try:
        if stable.exists():
            stable.rename(previous)
        incoming.rename(stable)
    except Exception:
        if stable.exists() and stable != incoming:
            shutil.rmtree(stable, ignore_errors=True)
        if previous.exists() and not stable.exists():
            previous.rename(stable)
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return backup


def restore_backup(backup: Path, root: Path, model_id: str = "sara-1.0") -> None:
    get_model_spec(model_id)
    backup = backup.expanduser().resolve()
    settings = Settings.load(root)
    allowed_root = backups_path(settings, model_id).resolve()
    if allowed_root != backup and allowed_root not in backup.parents:
        raise ValueError("Backup must be inside the selected model's backups directory.")
    if not backup.is_dir():
        raise ValueError("Backup directory does not exist.")
    manifest_path = backup / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and manifest.get("model_id") not in {None, model_id}:
            raise ValueError("Backup metadata does not match the selected model.")

    stable = stable_path(settings, model_id)
    incoming = stable.parent / f".restore-{run_id()}"
    shutil.copytree(backup, incoming, ignore=shutil.ignore_patterns("manifest.json"))
    previous = stable.parent / f".previous-stable-{run_id()}"
    try:
        if stable.exists():
            stable.rename(previous)
        incoming.rename(stable)
    except Exception:
        if stable.exists() and stable != incoming:
            shutil.rmtree(stable, ignore_errors=True)
        if previous.exists() and not stable.exists():
            previous.rename(stable)
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Sara 1.0 and Lira 1.0 versions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    promote_parser.add_argument("--candidate", type=Path, required=True)
    promote_parser.add_argument("--evaluation", type=Path, required=True)
    promote_parser.add_argument("--approved-by", required=True)
    promote_parser.add_argument("--admin-approved", action="store_true")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    rollback_parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    root = project_root()
    if args.command == "promote":
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
        backup = promote(
            args.candidate,
            root,
            evaluation_passed=True,
            model_id=args.model_id,
            admin_approved=args.admin_approved,
            evaluation=evaluation,
            approved_by=args.approved_by,
        )
        print(f"{get_model_spec(args.model_id).display_name} deployed. Backup: {backup}")
        return 0
    restore_backup(args.backup, root, model_id=args.model_id)
    print(f"{get_model_spec(args.model_id).display_name} restored from {args.backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
