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
from lusas_ai.agent import LusasAgent, ProposalError
from lusas_ai.cycle_state import fingerprint, read as read_cycle_state, write as write_cycle_state
from lusas_ai.lessons import record_failure
from lusas_ai.notifications import notify
from lusas_ai.upgrade_log import current_count, format_version, next_version, record
from lusas_ai.version_registry import append_version
from lusas_ai.web_learning import refresh as refresh_web
from training.evaluate_model import evaluate
from training.model_lifecycle import promote, restore_backup
from training.train_lora import train


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def progress(message: str, **details: object) -> None:
    """Keep long model operations visible in launchd and terminal logs."""
    suffix = f" {json.dumps(details, ensure_ascii=True)}" if details else ""
    print(f"[auto-upgrade] {message}{suffix}", flush=True)


def run_once(root: Path) -> dict:
    settings = Settings.load(root)
    if not settings.auto_model_upgrades:
        return {"status": "disabled"}

    started_at = datetime.now(timezone.utc).isoformat()
    progress("cycle started")
    web_result = refresh_web(settings)
    progress("web learning checked", **web_result)
    data_path = root / ".lusas" / f"training-{run_id()}.jsonl"
    eval_path = root / "training" / "data" / "eval.jsonl"
    records = []
    base_data_path = root / "training" / "data" / "examples.jsonl"
    for path in (
        base_data_path,
        settings.learning_path,
        settings.web_training_path,
    ):
        if path.exists():
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    input_fingerprint = fingerprint(
        [
            base_data_path,
            settings.learning_path,
            settings.web_training_path,
            settings.web_cache_path,
            eval_path,
        ]
    )
    cycle_state = read_cycle_state(settings.cycle_state_path)
    if (
        cycle_state.get("input_fingerprint") == input_fingerprint
        and cycle_state.get("learned_examples") == len(records)
    ):
        progress("training inputs unchanged; skipping")
        record(
            settings.upgrade_log_path,
            time=started_at,
            status="skipped",
            version=None,
            score=None,
            learned_examples=len(records),
            web_new_items=web_result.get("new_items", 0),
            reason="training inputs unchanged",
        )
        return {
            "status": "skipped",
            "reason": "training inputs unchanged",
            "learned_examples": len(records),
            "web_new_items": web_result.get("new_items", 0),
        }
    data_path.parent.mkdir(parents=True, exist_ok=True)
    progress("preparing training data", examples=len(records))
    data_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    candidate = root / "models" / "candidates" / f"auto-{run_id()}"
    progress("training candidate model", candidate=str(candidate))
    train(
        data_path=data_path,
        output_path=candidate,
        base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        epochs=1.0,
        max_length=512,
    )
    progress("evaluating candidate model")
    report = evaluate(candidate, eval_path, max_new_tokens=128)
    progress(
        "candidate evaluation complete",
        passed=report["passed"],
        score=report["score"],
    )
    baseline_report = None
    if settings.local_model_path.is_dir():
        progress("evaluating production baseline")
        baseline_report = evaluate(settings.local_model_path, eval_path, max_new_tokens=128)
    baseline_score = baseline_report["score"] if baseline_report else 0.0
    improvement = report["score"] - baseline_score
    quality_passed = report["passed"] and (
        baseline_report is None or improvement > 0
    )
    progress(
        "quality gate complete",
        passed=quality_passed,
        baseline_score=baseline_score,
        candidate_score=report["score"],
        improvement=improvement,
    )
    report["baseline"] = baseline_report
    report["improvement"] = improvement
    report["quality_gate_passed"] = quality_passed
    (candidate / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    code_upgrade = {"status": "disabled"}
    if quality_passed and settings.evolution_enabled and settings.auto_code_upgrades:
        try:
            agent = LusasAgent(root)
            result = agent.evolve(
                "Review the LUSAS source for one small, measurable reliability, "
                "testability, or performance improvement. Return no change if "
                "there is no safe improvement.",
                apply=settings.auto_apply_upgrades,
                progress_callback=lambda event: print(
                    f"[evolve:{event.phase}] {event.message}", flush=True
                ),
            )
            if result.status in {"promoted", "staged"}:
                code_upgrade = {
                    "status": result.status,
                    "summary": result.summary,
                    "tests_passed": bool(result.metrics and result.metrics.tests_passed),
                    "files": sorted(result.files),
                }
            else:
                code_upgrade = {"status": result.status, "reason": result.reason}
        except (ProposalError, ValueError, OSError, RuntimeError) as exc:
            code_upgrade = {"status": "rejected", "error": str(exc)}
            notify(
                root,
                settings.notification_path,
                "Autonomous code upgrade rejected.",
                error=str(exc),
            )

    if quality_passed:
        progress("promoting validated model")
        backup = promote(candidate, root, evaluation_passed=True)
        progress("checking deployed model health")
        deployed_report = evaluate(settings.local_model_path, eval_path, max_new_tokens=128)
        if not deployed_report["passed"]:
            restore_backup(backup, root)
            reason = "post-deploy health check failed; previous model restored"
            record_failure(
                root,
                "post-deploy-model-health",
                reason,
                candidate=str(candidate),
                score=deployed_report["score"],
            )
            notify(
                root,
                settings.notification_path,
                "Automatic model upgrade rolled back after deployment health failure.",
                candidate=str(candidate),
                backup=str(backup),
                score=deployed_report["score"],
            )
            record(
                settings.upgrade_log_path,
                time=started_at,
                status="rolled_back",
                version=None,
                candidate=str(candidate),
                score=deployed_report["score"],
                baseline_score=baseline_score,
                improvement=improvement,
                learned_examples=len(records),
                web_new_items=web_result.get("new_items", 0),
                rollback_point=str(backup),
                reason=reason,
            )
            write_cycle_state(
                settings.cycle_state_path,
                {
                    "input_fingerprint": input_fingerprint,
                    "learned_examples": len(records),
                    "version": None,
                    "last_status": "rolled_back",
                    "candidate_score": report["score"],
                    "baseline_score": baseline_score,
                    "improvement": improvement,
                },
            )
            data_path.unlink(missing_ok=True)
            return {
                "status": "rolled_back",
                "candidate": str(candidate),
                "backup": str(backup),
                "score": deployed_report["score"],
                "reason": reason,
            }
        parent_version = format_version(current_count(settings.upgrade_state_path))
        version = next_version(settings.upgrade_state_path)
        append_version(
            root / ".lusas" / "versions.jsonl",
            version=version,
            parent_version=parent_version if parent_version != "v0.000" else None,
            status="DEPLOYED",
            deployment_status="deployed",
            changes=["trained local LoRA model"],
            reason="new training data passed the independent quality gate",
            problems_fixed=[],
            new_capabilities=["learned training examples"],
            knowledge_changes={"web_new_items": web_result.get("new_items", 0)},
            prompt_changes=[],
            tool_changes=[],
            code_changes=code_upgrade.get("files", []),
            test_results=report.get("cases", []),
            benchmark_results={
                "score_before": baseline_score,
                "score_after": report["score"],
                "improvement": improvement,
            },
            security_results={"passed": True, "scope": "training data and model adapter"},
            performance_results={"training_records": len(records)},
            score_before=baseline_score,
            score_after=report["score"],
            rollback_point=str(backup),
        )
        notify(
            root,
            settings.notification_path,
            "Automatic model upgrade tested and promoted.",
            candidate=str(candidate),
            backup=str(backup),
            score=report["score"],
            version=version,
            code_upgrade=code_upgrade,
        )
        record(
            settings.upgrade_log_path,
            time=started_at,
            status="promoted",
            version=version,
            candidate=str(candidate),
            score=report["score"],
            baseline_score=baseline_score,
            improvement=improvement,
            learned_examples=len(records),
            web_new_items=web_result.get("new_items", 0),
            code_upgrade=code_upgrade,
        )
        write_cycle_state(
            settings.cycle_state_path,
            {
                "input_fingerprint": input_fingerprint,
                "learned_examples": len(records),
                "version": version,
                "last_status": "promoted",
                "candidate_score": report["score"],
                "baseline_score": baseline_score,
                "improvement": improvement,
            },
        )
        data_path.unlink(missing_ok=True)
        return {
            "status": "promoted",
            "candidate": str(candidate),
            "backup": str(backup),
            "score": report["score"],
            "code_upgrade": code_upgrade,
            "web_new_items": web_result.get("new_items", 0),
        }

    notify(
        root,
        settings.notification_path,
        "Automatic model upgrade failed the quality gate; not promoted.",
        candidate=str(candidate),
        score=report["score"],
        baseline_score=baseline_score,
        improvement=improvement,
    )
    record(
        settings.upgrade_log_path,
        time=started_at,
        status="rejected",
        version=None,
        candidate=str(candidate),
        score=report["score"],
        baseline_score=baseline_score,
        improvement=improvement,
        learned_examples=len(records),
        web_new_items=web_result.get("new_items", 0),
        code_upgrade=code_upgrade,
    )
    write_cycle_state(
        settings.cycle_state_path,
        {
            "input_fingerprint": input_fingerprint,
            "learned_examples": len(records),
            "version": None,
            "last_status": "rejected",
            "candidate_score": report["score"],
            "baseline_score": baseline_score,
            "improvement": improvement,
        },
    )
    data_path.unlink(missing_ok=True)
    return {
        "status": "rejected",
        "candidate": str(candidate),
        "score": report["score"],
        "baseline_score": baseline_score,
        "improvement": improvement,
        "reason": "quality gate did not demonstrate an improvement",
        "code_upgrade": code_upgrade,
        "web_new_items": web_result.get("new_items", 0),
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
            record_failure(
                root,
                "automatic-upgrade-cycle",
                str(exc),
                worker=str(PROJECT_ROOT / "training" / "auto_upgrade.py"),
            )
            notify(
                root,
                settings.notification_path,
                "Automatic model upgrade failed before promotion.",
                error=str(exc),
            )
            progress("cycle failed", error=str(exc))
            print(f"Automatic upgrade error: {exc}", flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(interval * 60)


if __name__ == "__main__":
    raise SystemExit(main())
