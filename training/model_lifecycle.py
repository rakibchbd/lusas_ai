from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def backup_production(root: Path, production: Path) -> Path:
    backup = root / "models" / "backups" / run_id()
    backup.parent.mkdir(parents=True, exist_ok=True)
    if production.exists():
        shutil.copytree(production, backup)
    else:
        backup.mkdir()
    (backup / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(production),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return backup


def promote(candidate: Path, root: Path, evaluation_passed: bool) -> Path:
    if not evaluation_passed:
        raise ValueError("Candidate must pass evaluation before promotion.")
    candidate = candidate.expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError("Candidate directory does not exist.")

    models_root = (root / "models").resolve()
    if models_root != candidate and models_root not in candidate.parents:
        raise ValueError("Candidate must be inside the project models directory.")

    production = models_root / "production"
    backup = backup_production(root, production)
    incoming = models_root / f".production-{run_id()}"
    shutil.copytree(candidate, incoming)
    if production.exists():
        shutil.rmtree(production)
    incoming.rename(production)
    return backup


def restore_backup(backup: Path, root: Path) -> None:
    backup = backup.expanduser().resolve()
    backups_root = (root / "models" / "backups").resolve()
    if backups_root != backup and backups_root not in backup.parents:
        raise ValueError("Backup must be inside models/backups.")
    if not backup.is_dir():
        raise ValueError("Backup directory does not exist.")

    production = root / "models" / "production"
    incoming = root / "models" / f".restore-{run_id()}"
    shutil.copytree(backup, incoming, ignore=shutil.ignore_patterns("manifest.json"))
    if production.exists():
        shutil.rmtree(production)
    incoming.rename(production)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote an evaluated LUSAS model candidate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--candidate", type=Path, required=True)
    promote_parser.add_argument(
        "--evaluation-passed",
        action="store_true",
        help="Confirm that an independent evaluation passed.",
    )
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "promote":
        backup = promote(
            args.candidate,
            project_root(),
            evaluation_passed=args.evaluation_passed,
        )
        print(f"Model promoted. Previous production backup: {backup}")
        return 0
    if args.command == "rollback":
        restore_backup(args.backup, project_root())
        print(f"Model restored from backup: {args.backup}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
