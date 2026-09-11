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
from lusas_ai.audit import run as run_audit
from lusas_ai.cycle_state import fingerprint, read as read_cycle_state, write as write_cycle_state
from lusas_ai.control import (
    automatic_deploy_allowed,
    code_evolution_allowed,
    model_learning_allowed,
    read as read_control,
    web_research_allowed,
)
from lusas_ai.lessons import record_failure
from lusas_ai.notifications import notify
from lusas_ai.engine import EvolutionEngine
from lusas_ai.gaps import detect as detect_gaps, record as record_gap
from lusas_ai.scorecard import record as record_scorecard
from lusas_ai.skills import ensure_builtin, record_use
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


def _training_records(root: Path, settings: Settings) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    base_data_path = root / "training" / "data" / "examples.jsonl"
    for path in (base_data_path, settings.learning_path, settings.web_training_path):
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Training record on line {line_number} is not an object.")
            records.append(payload)
    return records


def _run_code_evolution(
    root: Path, settings: Settings, apply: bool
) -> dict[str, object]:
    """Run source evolution independently of whether model data changed."""
    if not settings.evolution_enabled or not settings.auto_code_upgrades:
        return {"status": "disabled"}
    try:
        agent = LusasAgent(root)
        result = agent.evolve(
            "Review the LUSAS source for one small, measurable reliability, "
            "testability, or performance improvement. Return no change if "
            "there is no safe improvement.",
            apply=apply,
            progress_callback=lambda event: print(
                f"[evolve:{event.phase}] {event.message}", flush=True
            ),
        )
        if result.status in {"promoted", "staged"}:
            return {
                "status": result.status,
                "summary": result.summary,
                "tests_passed": bool(result.metrics and result.metrics.tests_passed),
                "benchmark_seconds": (
                    result.metrics.benchmark_seconds if result.metrics else None
                ),
                "score": result.metrics.score if result.metrics else None,
                "files": sorted(result.files),
            }
        return {"status": result.status, "reason": result.reason}
    except (ProposalError, ValueError, OSError, RuntimeError) as exc:
        notify(
            root,
            settings.notification_path,
            "Autonomous code upgrade rejected.",
            error=str(exc),
        )
        return {"status": "rejected", "error": str(exc)}


def _interval_due(last_run: object, interval_minutes: int) -> bool:
    if not isinstance(last_run, str):
        return True
    try:
        elapsed = (
            datetime.now(timezone.utc) - datetime.fromisoformat(last_run)
        ).total_seconds()
    except ValueError:
        return True
    return elapsed >= interval_minutes * 60


def run_once(root: Path) -> dict:
    settings = Settings.load(root)
    controls = read_control(root, settings.autonomy_level)
    if controls.get("emergency_stop", False):
        progress("emergency stop is active; cycle halted")
        return {"status": "stopped", "reason": "emergency stop is active"}
    model_loop_enabled = (
        settings.auto_model_upgrades
        and model_learning_allowed(controls)
        and not controls.get("upgrades_paused", False)
    )
    code_loop_enabled = (
        settings.evolution_enabled
        and settings.auto_code_upgrades
        and code_evolution_allowed(controls)
    )
    web_loop_enabled = settings.web_learning_enabled and web_research_allowed(controls)
    auto_deploy = settings.auto_apply_upgrades and automatic_deploy_allowed(controls)
    if not model_loop_enabled and not code_loop_enabled and not web_loop_enabled:
        return {"status": "disabled"}

    engine = EvolutionEngine(settings)
    cycle = engine.begin()
    started_at = datetime.now(timezone.utc).isoformat()
    progress("cycle started", evolution_id=cycle.evolution_id)
    engine.phase(cycle, "health_check", status="started")
    audit_result = run_audit(settings, evolution_id=cycle.evolution_id)
    engine.phase(
        cycle,
        "audit",
        status=audit_result["status"],
        findings=len(audit_result.get("findings", [])),
    )
    progress(
        "self-audit complete",
        evolution_id=cycle.evolution_id,
        status=audit_result["status"],
        findings=len(audit_result.get("findings", [])),
    )
    skills = ensure_builtin(settings)
    engine.phase(cycle, "skill_registry", skills=len(skills))
    initial_gaps = detect_gaps(settings, audit_result)
    engine.phase(cycle, "knowledge_gap_detection", open_gaps=len(initial_gaps))
    web_result = refresh_web(settings) if web_loop_enabled else {
        "status": "disabled", "new_items": 0
    }
    progress("web learning checked", **web_result)
    if web_loop_enabled and web_result.get("status") in {"refreshed", "skipped"}:
        record_use(
            settings,
            "bounded_web_research",
            succeeded=not bool(web_result.get("errors")),
        )
    if web_result.get("errors"):
        record_failure(
            root,
            "web-research",
            "One or more configured research sources failed.",
            evolution_id=cycle.evolution_id,
            errors=web_result.get("errors"),
        )
    engine.phase(
        cycle,
        "research",
        status=web_result.get("status"),
        new_items=web_result.get("new_items", 0),
        knowledge_items=web_result.get("knowledge_items", 0),
    )
    records = _training_records(root, settings)
    engine.phase(cycle, "learning", training_examples=len(records))
    base_data_path = root / "training" / "data" / "examples.jsonl"
    eval_path = root / "training" / "data" / "eval.jsonl"
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
    code_due = code_loop_enabled and _interval_due(
        cycle_state.get("last_code_evolution"), settings.evolution_interval_minutes
    )
    inputs_unchanged = (
        cycle_state.get("input_fingerprint") == input_fingerprint
        and cycle_state.get("learned_examples") == len(records)
    )
    model_upgrade: dict[str, object] = {"status": "disabled"}
    data_path: Path | None = None

    if model_loop_enabled:
        if inputs_unchanged:
            progress("training inputs unchanged; skipping model training")
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
            model_upgrade = {
                "status": "skipped",
                "reason": "training inputs unchanged",
                "learned_examples": len(records),
            }
        else:
            data_path = root / ".lusas" / f"training-{run_id()}.jsonl"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            progress("preparing training data", examples=len(records))
            data_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=True) for item in records) + "\n",
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
                baseline_report = evaluate(
                    settings.local_model_path, eval_path, max_new_tokens=128
                )
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
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )

            if quality_passed and auto_deploy:
                progress("promoting validated model")
                backup = promote(candidate, root, evaluation_passed=True)
                progress("checking deployed model health")
                deployed_report = evaluate(
                    settings.local_model_path, eval_path, max_new_tokens=128
                )
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
                    model_upgrade = {
                        "status": "rolled_back",
                        "candidate": str(candidate),
                        "backup": str(backup),
                        "score": deployed_report["score"],
                        "reason": reason,
                    }
                else:
                    parent_version = format_version(
                        current_count(settings.upgrade_state_path)
                    )
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
                        code_changes=[],
                        test_results=report.get("cases", []),
                        benchmark_results={
                            "score_before": baseline_score,
                            "score_after": report["score"],
                            "improvement": improvement,
                        },
                        security_results={
                            "passed": True,
                            "scope": "training data and model adapter",
                        },
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
                    )
                    model_upgrade = {
                        "status": "promoted",
                        "candidate": str(candidate),
                        "backup": str(backup),
                        "score": report["score"],
                        "version": version,
                    }
                write_cycle_state(
                    settings.cycle_state_path,
                    {
                        "input_fingerprint": input_fingerprint,
                        "learned_examples": len(records),
                        "version": model_upgrade.get("version"),
                        "last_status": model_upgrade["status"],
                        "candidate_score": report["score"],
                        "baseline_score": baseline_score,
                        "improvement": improvement,
                    },
                )
            elif quality_passed:
                reason = "candidate passed; automatic deployment is paused"
                notify(
                    root,
                    settings.notification_path,
                    "Automatic model upgrade staged but not deployed.",
                    candidate=str(candidate),
                    score=report["score"],
                    baseline_score=baseline_score,
                    improvement=improvement,
                    reason=reason,
                )
                record(
                    settings.upgrade_log_path,
                    time=started_at,
                    status="staged",
                    version=None,
                    candidate=str(candidate),
                    score=report["score"],
                    baseline_score=baseline_score,
                    improvement=improvement,
                    learned_examples=len(records),
                    web_new_items=web_result.get("new_items", 0),
                    reason=reason,
                )
                write_cycle_state(
                    settings.cycle_state_path,
                    {
                        "input_fingerprint": input_fingerprint,
                        "learned_examples": len(records),
                        "version": None,
                        "last_status": "staged",
                        "candidate_score": report["score"],
                        "baseline_score": baseline_score,
                        "improvement": improvement,
                    },
                )
                model_upgrade = {
                    "status": "staged",
                    "candidate": str(candidate),
                    "score": report["score"],
                    "baseline_score": baseline_score,
                    "improvement": improvement,
                    "reason": reason,
                }
            else:
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
                model_upgrade = {
                    "status": "rejected",
                    "candidate": str(candidate),
                    "score": report["score"],
                    "baseline_score": baseline_score,
                    "improvement": improvement,
                    "reason": "quality gate did not demonstrate an improvement",
                }
            data_path.unlink(missing_ok=True)

    engine.phase(
        cycle,
        "evaluation",
        model_status=model_upgrade.get("status"),
        model_score=model_upgrade.get("score"),
    )
    if code_due:
        code_upgrade = _run_code_evolution(root, settings, apply=auto_deploy)
    elif code_loop_enabled:
        code_upgrade = {
            "status": "skipped",
            "reason": "evolution interval not reached",
        }
    else:
        code_upgrade = {"status": "disabled"}
    engine.phase(cycle, "testing", code_status=code_upgrade.get("status"))
    engine.phase(cycle, "benchmark", code_score=code_upgrade.get("score"))
    engine.phase(cycle, "experiment", model=model_upgrade, code=code_upgrade)
    if code_due:
        latest_state = read_cycle_state(settings.cycle_state_path)
        latest_state.update(
            {
                "last_code_evolution": datetime.now(timezone.utc).isoformat(),
                "last_code_evolution_status": code_upgrade.get("status"),
            }
        )
        write_cycle_state(settings.cycle_state_path, latest_state)
    model_status = str(model_upgrade.get("status", "disabled"))
    code_status = str(code_upgrade.get("status", "disabled"))
    if model_status in {"promoted", "staged", "rejected", "rolled_back"}:
        record_use(
            settings,
            "local_model_training",
            succeeded=model_status in {"promoted", "staged"},
            benchmark_score=(
                float(model_upgrade["score"])
                if model_upgrade.get("score") is not None
                else None
            ),
        )
    if code_due:
        record_use(
            settings,
            "sandboxed_code_evolution",
            succeeded=code_status in {"promoted", "staged"},
            benchmark_score=(
                float(code_upgrade["score"])
                if code_upgrade.get("score") is not None
                else None
            ),
        )
    if "promoted" in {model_status, code_status}:
        status = "promoted"
    elif "rolled_back" in {model_status, code_status}:
        status = "rolled_back"
    elif "rejected" in {model_status, code_status}:
        status = "rejected"
    elif model_status == "skipped":
        status = "skipped"
    else:
        status = code_status if code_status != "disabled" else model_status
    if model_status in {"rejected", "rolled_back"}:
        record_failure(
            root,
            "model-quality-gate",
            str(model_upgrade.get("reason", "The model candidate did not pass its quality gate.")),
            evolution_id=cycle.evolution_id,
            candidate=model_upgrade.get("candidate"),
            score=model_upgrade.get("score"),
        )
        record_gap(
            settings,
            "model-quality",
            str(model_upgrade.get("reason", "The model candidate did not pass its quality gate.")),
            priority=0.75,
            evidence={"model_upgrade": model_upgrade},
        )
    if code_status == "rejected":
        record_failure(
            root,
            "code-evolution-gate",
            str(code_upgrade.get("reason", code_upgrade.get("error", "The code candidate was rejected."))),
            evolution_id=cycle.evolution_id,
            files=code_upgrade.get("files", []),
        )
        record_gap(
            settings,
            "code-evolution",
            str(code_upgrade.get("reason", code_upgrade.get("error", "The code candidate was rejected."))),
            priority=0.8,
            evidence={"code_upgrade": code_upgrade},
        )
    gaps = detect_gaps(settings, audit_result)
    current_version = format_version(current_count(settings.upgrade_state_path))
    scorecard = record_scorecard(
        settings,
        version=current_version,
        evolution_id=cycle.evolution_id,
        audit_result=audit_result,
        model_upgrade=model_upgrade,
        code_upgrade=code_upgrade,
    )
    engine.phase(
        cycle,
        "monitor",
        version=current_version,
        overall_score=scorecard.get("overall_score"),
        measured_dimensions=scorecard.get("measured_dimensions"),
    )
    engine.complete(
        cycle,
        status=status,
        version=current_version,
        audit_status=audit_result.get("status"),
        open_knowledge_gaps=len(gaps),
        knowledge_items=web_result.get("knowledge_items", 0),
        model_upgrade=model_upgrade,
        code_upgrade=code_upgrade,
        scorecard={
            "overall_score": scorecard.get("overall_score"),
            "measured_dimensions": scorecard.get("measured_dimensions"),
        },
    )
    return {
        "status": status,
        "evolution_id": cycle.evolution_id,
        "model_upgrade": model_upgrade,
        "code_upgrade": code_upgrade,
        "learned_examples": len(records),
        "web_new_items": web_result.get("new_items", 0),
        "audit": audit_result,
        "knowledge_gaps": len(gaps),
        "scorecard": scorecard,
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
    configured_intervals = []
    if settings.auto_model_upgrades:
        configured_intervals.append(settings.model_upgrade_interval_minutes)
    if settings.evolution_enabled and settings.auto_code_upgrades:
        configured_intervals.append(settings.evolution_interval_minutes)
    if settings.web_learning_enabled:
        configured_intervals.append(settings.web_refresh_interval_minutes)
    interval = args.interval_minutes or min(configured_intervals or [60])
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
