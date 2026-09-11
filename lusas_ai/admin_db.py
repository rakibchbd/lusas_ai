"""Local SQLite administration store with explicit migrations and audit logging."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid
from urllib.parse import urlparse

from .data_pipeline import approved_training_records
from .model_registry import MODEL_IDS, catalog, get_model_spec, foundation_config, candidates_path
from training.model_lifecycle import REQUIRED_EVALUATION_GATES


MIGRATION_ROOT = Path(__file__).resolve().parents[1] / "migrations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdminStore:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.path = settings.admin_database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()
        self.sync_models()
        self.sync_configured_sources()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _project_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (self.settings.root / path if not path.is_absolute() else path).resolve()

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
            applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for path in sorted(MIGRATION_ROOT.glob("*.sql")):
                if path.name in applied:
                    continue
                connection.executescript(path.read_text(encoding="utf-8"))
                connection.execute("INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)", (path.name, _now()))

    def sync_models(self) -> None:
        with self._connect() as connection:
            for item in catalog(self.settings):
                connection.execute(
                    """INSERT INTO model_catalog(model_id, display_name, foundation_model, foundation_path, status, stable_version, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(model_id) DO UPDATE SET display_name=excluded.display_name,
                       foundation_model=excluded.foundation_model, foundation_path=excluded.foundation_path,
                       status=excluded.status, stable_version=excluded.stable_version, updated_at=excluded.updated_at""",
                    (
                        item["model_id"], item["display_name"], item["foundation_model"],
                        item["foundation_path"],
                        "stable" if item["stable_available"] else "not_installed",
                        item["stable_version"], _now(),
                    ),
                )

    def audit(self, action: str, subject: str, actor: str, **details: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_events(action, subject, details, actor, created_at) VALUES (?, ?, ?, ?, ?)",
                (action, subject, json.dumps(details, ensure_ascii=True), actor, _now()),
            )

    def record_interaction(
        self,
        model_id: str,
        question: str,
        answer: str,
        *,
        knowledge_used: bool = False,
        web_search_used: bool = False,
        sources: list[str] | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Persist a served answer for asynchronous quality and gap analysis."""
        get_model_spec(model_id)
        created_at = _now()
        interaction_id = f"interaction-{uuid.uuid4().hex}"
        source_list = [str(item) for item in (sources or []) if str(item).strip()]
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO interaction_events(
                   interaction_id, model_id, question, answer, knowledge_used,
                   web_search_used, sources, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    interaction_id,
                    model_id,
                    question[:8_000],
                    answer[:20_000],
                    int(knowledge_used),
                    int(web_search_used),
                    json.dumps(source_list, ensure_ascii=True),
                    confidence,
                    created_at,
                ),
            )
        return {
            "interaction_id": interaction_id,
            "model_id": model_id,
            "status": "queued",
            "created_at": created_at,
        }

    def pending_interactions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM interaction_events WHERE status = 'queued' ORDER BY created_at LIMIT ?",
                (max(1, min(100, int(limit))),),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["sources"] = json.loads(item.get("sources", "[]"))
            except (TypeError, json.JSONDecodeError):
                item["sources"] = []
            result.append(item)
        return result

    def complete_interaction(
        self,
        interaction_id: str,
        *,
        status: str = "processed",
        answer_quality: str = "unrated",
        confidence: float | None = None,
        feedback: str | None = None,
    ) -> None:
        if status not in {"processed", "failed", "needs_review"}:
            raise ValueError("Invalid interaction status.")
        with self._connect() as connection:
            connection.execute(
                """UPDATE interaction_events SET status = ?, answer_quality = ?,
                   confidence = COALESCE(?, confidence), feedback = ?, processed_at = ?
                   WHERE interaction_id = ?""",
                (status, answer_quality, confidence, feedback, _now(), interaction_id),
            )

    def add_interaction_feedback(self, interaction_id: str, feedback: str) -> None:
        feedback = feedback.strip()
        if not feedback:
            raise ValueError("Feedback cannot be empty.")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE interaction_events SET feedback = ?, status = 'queued',
                   answer_quality = 'needs_review', processed_at = NULL
                   WHERE interaction_id = ?""",
                (feedback[:4_000], interaction_id),
            )
        if cursor.rowcount != 1:
            raise ValueError("Interaction was not found.")

    def record_practice(
        self,
        knowledge_id: str,
        task: str,
        status: str,
        score: float,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in {"passed", "failed"}:
            raise ValueError("Practice status must be passed or failed.")
        run_id = f"practice-{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO practice_runs(run_id, knowledge_id, task, status, score, evidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, knowledge_id, task, status, score, json.dumps(evidence, ensure_ascii=True), _now()),
            )
        return {"run_id": run_id, "knowledge_id": knowledge_id, "status": status, "score": score}

    def practice_seen(self, knowledge_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM practice_runs WHERE knowledge_id = ? LIMIT 1", (knowledge_id,)
            ).fetchone()
        return row is not None

    def record_research(
        self,
        query: str,
        *,
        sources_selected: list[str],
        source_quality: dict[str, Any],
        elapsed_ms: float,
        information_found: bool,
        verification_success: bool,
        usefulness: str = "unrated",
    ) -> dict[str, Any]:
        research_id = f"research-{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO research_events(
                   research_id, query, sources_selected, source_quality, elapsed_ms,
                   information_found, verification_success, usefulness, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    research_id,
                    query[:8_000],
                    json.dumps(sources_selected, ensure_ascii=True),
                    json.dumps(source_quality, ensure_ascii=True),
                    float(elapsed_ms),
                    int(information_found),
                    int(verification_success),
                    usefulness,
                    _now(),
                ),
            )
        return {"research_id": research_id, "information_found": information_found}

    def enqueue_task(self, kind: str, priority: float, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        task_id = "task-" + sha256(f"{kind}\n{normalized}".encode("utf-8")).hexdigest()[:24]
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO evolution_tasks(task_id, kind, priority, payload, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET priority = MAX(priority, excluded.priority),
                   updated_at = excluded.updated_at WHERE evolution_tasks.status IN ('queued', 'paused')""",
                (task_id, kind, max(0.0, min(1.0, float(priority))), normalized, now, now),
            )
        return {"task_id": task_id, "kind": kind, "priority": priority, "status": "queued"}

    def continuous_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            counts: dict[str, int] = {}
            for table, key in (
                ("interaction_events", "interactions"),
                ("practice_runs", "practice_runs"),
                ("research_events", "research_events"),
                ("evolution_tasks", "evolution_tasks"),
            ):
                counts[key] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            counts["queued_interactions"] = int(connection.execute("SELECT COUNT(*) FROM interaction_events WHERE status = 'queued'").fetchone()[0])
            counts["failed_practice_runs"] = int(connection.execute("SELECT COUNT(*) FROM practice_runs WHERE status = 'failed'").fetchone()[0])
        return counts

    def continuous_overview(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        state_path = getattr(self.settings, "continuous_state_path", None)
        if state_path and state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except (OSError, json.JSONDecodeError):
                state = {}
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT interaction_id, model_id, question, answer, answer_quality,
                   feedback, status, created_at, processed_at
                   FROM interaction_events ORDER BY created_at DESC LIMIT 25"""
            ).fetchall()
            research = connection.execute(
                """SELECT research_id, query, sources_selected, source_quality,
                   elapsed_ms, information_found, verification_success, usefulness,
                   created_at FROM research_events ORDER BY created_at DESC LIMIT 25"""
            ).fetchall()
        return {
            "counts": self.continuous_counts(),
            "state": state,
            "interactions": [dict(row) for row in rows],
            "research": [dict(row) for row in research],
        }

    def overview(self) -> dict[str, Any]:
        self.sync_models()
        with self._connect() as connection:
            jobs = [dict(row) for row in connection.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 20")]
            approvals = [dict(row) for row in connection.execute("SELECT * FROM deployment_approvals ORDER BY created_at DESC LIMIT 20")]
            audits = [dict(row) for row in connection.execute("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT 30")]
            sources = [dict(row) for row in connection.execute("SELECT * FROM approved_sources ORDER BY created_at DESC")]
        return {
            "models": catalog(self.settings),
            "candidates": self.candidate_reports(),
            "training_jobs": jobs,
            "deployment_approvals": approvals,
            "sources": sources,
            "audit_events": audits,
            "continuous": self.continuous_overview(),
        }

    def candidate_reports(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for model in catalog(self.settings):
            candidate_root = candidates_path(self.settings, model["model_id"])
            if not candidate_root.is_dir():
                continue
            for candidate in sorted((item for item in candidate_root.iterdir() if item.is_dir()), reverse=True):
                evaluation: dict[str, Any] = {}
                evaluation_path = candidate / "evaluation.json"
                if evaluation_path.exists():
                    try:
                        loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            evaluation = loaded
                    except (OSError, json.JSONDecodeError):
                        evaluation = {}
                result.append({
                    "model_id": model["model_id"],
                    "display_name": model["display_name"],
                    "candidate_path": str(candidate),
                    "passed": evaluation.get("passed", False),
                    "score": evaluation.get("score"),
                    "gates": evaluation.get("gates", {}),
                    "metrics": evaluation.get("metrics", {}),
                })
        return result[:100]

    def knowledge(self) -> list[dict[str, Any]]:
        if not self.settings.knowledge_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.settings.knowledge_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record.setdefault("approval_status", "pending" if record.get("kind") == "web_evidence" else "approved")
                records.append(record)
        return records

    def review_knowledge(self, record_id: str, status: str, actor: str) -> dict[str, Any]:
        if status not in {"approved", "rejected"}:
            raise ValueError("Review status must be approved or rejected.")
        records = self.knowledge()
        found = next((item for item in records if str(item.get("id", item.get("knowledge_id"))) == record_id), None)
        if found is None:
            raise ValueError("Knowledge record was not found.")
        found["approval_status"] = status
        found["reviewed_by"] = actor
        found["reviewed_at"] = _now()
        temporary = self.settings.knowledge_path.with_name(self.settings.knowledge_path.name + ".tmp")
        temporary.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in records), encoding="utf-8")
        temporary.replace(self.settings.knowledge_path)
        # Keep the derived web-training queue in sync with the administrator's
        # decision. The queue remains inert until this approved status is present.
        for path in (self.settings.web_cache_path, self.settings.web_training_path):
            if not path.exists():
                continue
            try:
                queued = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except json.JSONDecodeError:
                continue
            changed = False
            for item in queued:
                item_id = item.get("id", item.get("source_id")) if isinstance(item, dict) else None
                if str(item_id) != record_id:
                    continue
                item["approval_status"] = status
                item["reviewed_by"] = actor
                item["reviewed_at"] = _now()
                changed = True
            if changed:
                path_tmp = path.with_name(path.name + ".tmp")
                path_tmp.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in queued), encoding="utf-8")
                path_tmp.replace(path)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO data_reviews(record_id, category, status, malicious_findings, reviewed_by, reviewed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (record_id, found.get("domain", "general"), status, json.dumps(found.get("malicious_findings", [])), actor, _now()),
            )
        self.audit("knowledge_review", record_id, actor, status=status)
        return found

    def add_source(self, url: str, actor: str) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Only HTTPS administrator-approved sources are allowed.")
        hostname = parsed.hostname.lower().rstrip(".")
        if self.settings.web_allowed_domains and not any(
            hostname == domain.lower().lstrip(".").rstrip(".")
            or hostname.endswith("." + domain.lower().lstrip(".").rstrip("."))
            for domain in self.settings.web_allowed_domains
        ):
            raise ValueError("Source domain is not in the configured administrator allowlist.")
        item = {"url": url, "domain": hostname, "admin_approved": 1}
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO approved_sources(url, domain, enabled, admin_approved, approved_by, created_at) VALUES (?, ?, 1, 1, ?, ?)",
                (item["url"], item["domain"], actor, _now()),
            )
        self.audit("source_approved", url, actor, domain=item["domain"])
        return item

    def sync_configured_sources(self) -> None:
        """Treat explicitly configured HTTPS feeds as administrator-approved."""
        for url in self.settings.web_sources:
            try:
                parsed = urlparse(url)
                hostname = parsed.hostname.lower().rstrip(".") if parsed.hostname else ""
                if parsed.scheme != "https" or not hostname:
                    continue
                with self._connect() as connection:
                    connection.execute(
                        "INSERT OR IGNORE INTO approved_sources(url, domain, enabled, admin_approved, approved_by, created_at) VALUES (?, ?, 1, 1, 'config', ?)",
                        (url, hostname, _now()),
                    )
            except (OSError, sqlite3.Error):
                continue

    def approved_sources(self) -> list[str]:
        with self._connect() as connection:
            return [row[0] for row in connection.execute("SELECT url FROM approved_sources WHERE enabled = 1 AND admin_approved = 1 ORDER BY created_at")]

    def start_training(self, model_id: str, actor: str, dataset_count: int) -> dict[str, Any]:
        spec = get_model_spec(model_id)
        foundation = foundation_config(self.settings, model_id)
        job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        with self._connect() as connection:
            active_jobs = int(
                connection.execute(
                    "SELECT COUNT(*) FROM training_jobs WHERE status IN ('queued', 'running')"
                ).fetchone()[0]
            )
        has_capacity = active_jobs < self.settings.background_max_concurrent_jobs
        status = (
            "queued"
            if (foundation["model"] or foundation["path"]) and dataset_count and has_capacity
            else "blocked"
        )
        error = None
        if not (foundation["model"] or foundation["path"]):
            error = f"Configure {spec.foundation_model_env} or {spec.foundation_path_env} before training."
        elif not dataset_count:
            error = "No clean, approved training records are available."
        elif not has_capacity:
            error = "The background training concurrency budget is full; retry after the active job finishes."
        candidate = candidates_path(self.settings, model_id) / job_id
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO training_jobs(job_id, model_id, status, dataset_count, candidate_path, error, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, model_id, status, dataset_count, str(candidate), error, actor, _now(), _now()),
            )
        self.audit("training_started", model_id, actor, job_id=job_id, status=status)
        if status == "queued":
            threading.Thread(target=self._run_training_job, args=(job_id,), daemon=True).start()
        return {"job_id": job_id, "model_id": model_id, "status": status, "candidate_path": str(candidate), "error": error}

    def training_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        paths = (
            self.settings.root / "training" / "data" / "examples.jsonl",
            self.settings.learning_path,
            self.settings.web_training_path,
        )
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
        return approved_training_records(records)

    def _run_training_job(self, job_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return
        model_id = row["model_id"]
        candidate = Path(row["candidate_path"])
        data_path = self.settings.root / ".lusas" / f"admin-{job_id}.jsonl"
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            records = self.training_records()
            data_path.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in records), encoding="utf-8")
            foundation = foundation_config(self.settings, model_id)
            from training.evaluate_model import evaluate
            from training.train_lora import train
            with self._connect() as connection:
                connection.execute("UPDATE training_jobs SET status = 'running', updated_at = ? WHERE job_id = ?", (_now(), job_id))
            train(
                data_path=data_path,
                output_path=candidate,
                model_id=model_id,
                foundation_model=foundation["model"],
                foundation_path=Path(foundation["path"]) if foundation["path"] else None,
                epochs=1.0,
                max_length=512,
            )
            evaluation = evaluate(model_id, candidate, self.settings.root / "training" / "data" / "eval.jsonl", settings=self.settings)
            status = "evaluated_pass" if evaluation["passed"] else "evaluated_fail"
            with self._connect() as connection:
                connection.execute("UPDATE training_jobs SET status = ?, metrics = ?, updated_at = ? WHERE job_id = ?", (status, json.dumps(evaluation), _now(), job_id))
            self.audit("training_completed", model_id, "training-worker", job_id=job_id, status=status, candidate_path=str(candidate))
        except Exception as exc:  # a failed job must be visible and never promoted
            with self._connect() as connection:
                connection.execute("UPDATE training_jobs SET status = 'failed', error = ?, updated_at = ? WHERE job_id = ?", (str(exc), _now(), job_id))
            self.audit("training_failed", model_id, "training-worker", job_id=job_id, error=str(exc))
        finally:
            data_path.unlink(missing_ok=True)

    def approve_deployment(self, model_id: str, candidate_path: str, evaluation: dict[str, Any], actor: str) -> dict[str, Any]:
        get_model_spec(model_id)
        candidate = self._project_path(candidate_path)
        allowed = candidates_path(self.settings, model_id).resolve()
        if allowed != candidate and allowed not in candidate.parents:
            raise ValueError("Candidate is outside the selected model's candidate directory.")
        gates = evaluation.get("gates")
        if evaluation.get("passed") is not True or not isinstance(gates, dict) or not REQUIRED_EVALUATION_GATES.issubset(gates) or any(gates[name] is not True for name in REQUIRED_EVALUATION_GATES):
            raise ValueError("Only a candidate that passes every required evaluation gate can be approved.")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO deployment_approvals(model_id, candidate_path, evaluation, status, approved_by, created_at, approved_at) VALUES (?, ?, ?, 'approved', ?, ?, ?)",
                (model_id, str(candidate), json.dumps(evaluation), actor, _now(), _now()),
            )
        self.audit("deployment_approved", model_id, actor, candidate_path=str(candidate), approval_id=cursor.lastrowid)
        return {"approval_id": cursor.lastrowid, "model_id": model_id, "candidate_path": str(candidate), "status": "approved", "approved_by": actor}

    def approved_deployment(self, model_id: str, candidate_path: str) -> dict[str, Any] | None:
        get_model_spec(model_id)
        candidate = str(self._project_path(candidate_path))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM deployment_approvals WHERE model_id = ? AND candidate_path = ? AND status = 'approved' ORDER BY approved_at DESC LIMIT 1",
                (model_id, candidate),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["evaluation"] = json.loads(result["evaluation"])
        return result

    def jobs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 50")]

    def audit_events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT 100")]
