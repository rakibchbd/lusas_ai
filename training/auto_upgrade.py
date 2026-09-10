from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lusas_ai.config import Settings
from lusas_ai.notifications import notify
from lusas_ai.publisher import publish_upgrade
from lusas_ai.upgrade_log import next_version, record
from training.evaluate_model import evaluate
from training.model_lifecycle import promote
from training.train_lora import train


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def run_once(root: Path) -> dict:
    settings = Settings.load(root)
    if not settings.auto_model_upgrades:
        return {"status": "disabled"}

    started_at = datetime.now(timezone.utc).isoformat()
    data_path = root / ".lusas" / f"training-{run_id()}.jsonl"
    eval_path = root / "training" / "data" / "eval.jsonl"
    records = []
    base_data_path = root / "training" / "data" / "examples.jsonl"
    for path in (base_data_path, settings.learning_path):
        if path.exists():
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    candidate = root / "models" / "candidates" / f"auto-{run_id()}"
    train(
        data_path=data_path,
        output_path=candidate,
        base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        epochs=1.0,
        max_length=1024,
    )
    report = evaluate(candidate, eval_path, max_new_tokens=128)
    (candidate / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    if report["passed"]:
        version = next_version(settings.upgrade_state_path)
        backup = promote(candidate, root, evaluation_passed=True)
        publication = None
        if settings.auto_publish_upgrades:
            publication = publish_upgrade(
                root,
                candidate,
                report["score"],
                records,
            )
        notify(
            root,
            settings.notification_path,
            "Automatic model upgrade tested and promoted.",
            candidate=str(candidate),
            backup=str(backup),
            score=report["score"],
            publication=publication,
            version=version,
        )
        record(
            settings.upgrade_log_path,
            time=started_at,
            status="promoted",
            version=version,
            candidate=str(candidate),
            score=report["score"],
            learned_examples=len(records),
            publication=publication,
        )
        data_path.unlink(missing_ok=True)
        return {
            "status": "promoted",
            "candidate": str(candidate),
            "backup": str(backup),
            "score": report["score"],
            "publication": publication,
        }

    notify(
        root,
        settings.notification_path,
        "Automatic model upgrade failed evaluation; not promoted.",
        candidate=str(candidate),
        score=report["score"],
    )
    record(
        settings.upgrade_log_path,
        time=started_at,
        status="rejected",
        version=None,
        candidate=str(candidate),
        score=report["score"],
        learned_examples=len(records),
    )
    data_path.unlink(missing_ok=True)
    return {
        "status": "rejected",
        "candidate": str(candidate),
        "score": report["score"],
    }


def run_locked_once(root: Path) -> dict:
    lock_path = root / ".lusas" / "auto-model-upgrade.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        try:
            return run_once(root)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the automatic LUSAS model-upgrade loop."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one train-evaluate-promote cycle and exit.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=None,
        help="Minutes between cycles; config.json is used by default.",
    )
    args = parser.parse_args()
    root = PROJECT_ROOT
    settings = Settings.load(root)
    interval = args.interval_minutes or settings.model_upgrade_interval_minutes
    if interval < 1:
        raise SystemExit("interval-minutes must be at least 1.")

    while True:
        try:
            result = run_locked_once(root)
            print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:
            notify(
                root,
                settings.notification_path,
                "Automatic model upgrade failed before promotion.",
                error=str(exc),
            )
            print(f"Automatic upgrade error: {exc}", flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(interval * 60)


if __name__ == "__main__":
    raise SystemExit(main())
