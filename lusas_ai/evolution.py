"""Guarded, local-only source evolution pipeline.

The pipeline deliberately treats model output as untrusted input.  A proposal
is inspected, copied into an isolated candidate directory, statically checked,
tested, and compared with the current checkout before it can be deployed.
There is no Git or network operation in this module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol

from .config import Settings
from .git_integration import GitCommitResult, GitIntegration
from .notifications import notify
from .updater import (
    apply_candidate,
    create_backup,
    restore_backup,
)
from .upgrade_log import next_version, record


ALLOWED_PATHS = ("lusas_ai", "tests", "training")
PROTECTED_PATHS = {
    ".git",
    ".github",
    ".lusas",
    "config.json",
    "pyproject.toml",
    "models",
    "workspace",
}
FORBIDDEN_IMPORTS = {
    "ctypes",
    "ftplib",
    "http",
    "httpx",
    "paramiko",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
FORBIDDEN_CALLS = {"eval", "exec", "__import__"}
FORBIDDEN_ATTRIBUTE_CALLS = {"execv", "execve", "popen", "spawn", "system"}
SAFE_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


class EvolutionRejected(ValueError):
    """Raised when a proposal does not satisfy a safety gate."""


class ModelBackend(Protocol):
    def chat(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class RepositoryReport:
    root: Path
    files: tuple[str, ...]
    fingerprint: str
    python_files: int
    tests: int


class RepositoryAnalyzer:
    """Create a bounded, deterministic view of the configured repository."""

    def __init__(self, root: Path, allowed_paths: tuple[str, ...] = ALLOWED_PATHS):
        self.root = root.expanduser().resolve()
        self.allowed_paths = allowed_paths

    def analyze(self) -> RepositoryReport:
        files: list[str] = []
        digest = hashlib.sha256()
        python_files = 0
        tests = 0
        for prefix in self.allowed_paths:
            directory = self.root / prefix
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or any(part in PROTECTED_PATHS for part in path.parts):
                    continue
                relative = path.relative_to(self.root).as_posix()
                files.append(relative)
                digest.update(relative.encode("utf-8"))
                digest.update(path.read_bytes())
                if path.suffix == ".py":
                    python_files += 1
                if path.name.startswith("test_") and path.suffix == ".py":
                    tests += 1
        return RepositoryReport(
            root=self.root,
            files=tuple(files),
            fingerprint=digest.hexdigest(),
            python_files=python_files,
            tests=tests,
        )

    def snapshot(self, max_chars: int = 100_000) -> str:
        chunks: list[str] = []
        used = 0
        for relative in self.analyze().files:
            path = self.root / relative
            if path.suffix not in {".py", ".json", ".md", ".toml", ".txt"}:
                continue
            remaining = max_chars - used
            if remaining <= 0:
                break
            content = path.read_text(encoding="utf-8", errors="replace")[:remaining]
            chunks.append(f"--- {relative} ---\n{content}")
            used += len(content)
        return "\n\n".join(chunks)


class ProtectedPathValidator:
    """Validate model-provided paths before they reach the filesystem."""

    def __init__(
        self,
        allowed_paths: tuple[str, ...] = ALLOWED_PATHS,
        protected_paths: set[str] = PROTECTED_PATHS,
    ):
        self.allowed_paths = allowed_paths
        self.protected_paths = protected_paths

    def validate(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 2
            or path.parts[0] not in self.allowed_paths
            or any(part in self.protected_paths for part in path.parts)
            or path.suffix.lower() not in SAFE_TEXT_SUFFIXES
        ):
            raise EvolutionRejected(
                f"Proposed path is outside the protected evolution boundary: "
                f"{relative_path!r}"
            )
        return path

    def validate_changes(self, changes: Mapping[str, str]) -> None:
        if not changes:
            raise EvolutionRejected("The proposal contains no file changes.")
        for relative, content in changes.items():
            self.validate(relative)
            if not isinstance(content, str):
                raise EvolutionRejected(f"Content for {relative!r} is not text.")
            if len(content.encode("utf-8")) > 1_000_000:
                raise EvolutionRejected(f"Content for {relative!r} is too large.")


class ImprovementPlanner:
    """Use the existing local model to produce a constrained JSON proposal."""

    prompt = """You are a guarded local software-improvement planner.
Return ONLY JSON: {"summary":"...", "files":{"lusas_ai/file.py":"complete text"}}.
Only change files under lusas_ai/, tests/, or training/. Never add network,
credential, shell, persistence, or deployment behavior. Include complete file
contents and tests for behavior changes. If there is no safe improvement,
return {"summary":"no safe improvement","files":{}}."""

    def __init__(self, model: ModelBackend):
        self.model = model

    def plan(self, report: RepositoryReport, snapshot: str, goal: str) -> tuple[str, dict[str, str]]:
        response = self.model.chat(
            f"{self.prompt}\n\nGoal:\n{goal}\n\n"
            f"Repository fingerprint: {report.fingerprint}\nSource:\n{snapshot}"
        )
        start = response.find("{")
        if start < 0:
            raise EvolutionRejected("The local model did not return a JSON proposal.")
        try:
            payload = json.JSONDecoder().raw_decode(response[start:])[0]
        except json.JSONDecodeError as exc:
            raise EvolutionRejected("The local model returned invalid proposal JSON.") from exc
        if not isinstance(payload, dict):
            raise EvolutionRejected("The proposal must be a JSON object.")
        summary = payload.get("summary")
        changes = payload.get("files")
        if not isinstance(summary, str) or not summary.strip():
            raise EvolutionRejected("The proposal needs a summary.")
        if not isinstance(changes, dict):
            raise EvolutionRejected("The proposal needs a files object.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in changes.items()):
            raise EvolutionRejected("Proposal paths and contents must be text.")
        return summary.strip(), changes


class CandidateWorkspace:
    """Materialize a candidate under .lusas, never in the active checkout."""

    def __init__(self, root: Path, path: Path, changed_files: tuple[str, ...]):
        self.root = root
        self.path = path
        self.changed_files = changed_files

    @classmethod
    def create(
        cls,
        root: Path,
        changes: Mapping[str, str],
        validator: ProtectedPathValidator | None = None,
    ) -> "CandidateWorkspace":
        root = root.expanduser().resolve()
        validator = validator or ProtectedPathValidator()
        validator.validate_changes(changes)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = root / ".lusas" / "evolution-candidates" / stamp
        path.mkdir(parents=True, exist_ok=False)
        for prefix in ALLOWED_PATHS:
            source = root / prefix
            if source.is_dir():
                if source.is_symlink():
                    raise EvolutionRejected(f"Cannot copy symlinked directory: {prefix}")
                destination_root = path / prefix
                for source_path in source.rglob("*"):
                    if source_path.is_symlink():
                        continue
                    relative = source_path.relative_to(source)
                    destination = destination_root / relative
                    if source_path.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif source_path.is_file():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, destination)
        for relative, content in changes.items():
            destination = path / validator.validate(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        return cls(root, path, tuple(sorted(changes)))

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    output: str = ""
    checks: tuple[str, ...] = ()


class SecurityChecker:
    """Run conservative AST checks using only Python's standard library."""

    def check(self, candidate: CandidateWorkspace) -> CheckResult:
        failures: list[str] = []
        for relative in candidate.changed_files:
            path = candidate.path / relative
            if path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (OSError, SyntaxError) as exc:
                failures.append(f"{relative}: {exc}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".", 1)[0] for alias in node.names]
                    failures.extend(
                        f"{relative}: forbidden import {name}" for name in names if name in FORBIDDEN_IMPORTS
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    name = node.module.split(".", 1)[0]
                    if name in FORBIDDEN_IMPORTS:
                        failures.append(f"{relative}: forbidden import {name}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        failures.append(f"{relative}: forbidden call {node.func.id}")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in FORBIDDEN_ATTRIBUTE_CALLS
                ):
                    failures.append(f"{relative}: forbidden call {node.func.attr}")
        return CheckResult(not failures, "\n".join(failures), ("static-security",))


class TestRunner:
    def run(self, candidate: CandidateWorkspace, timeout: int = 300) -> CheckResult:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
        try:
            completed = subprocess.run(
                command, cwd=candidate.path, text=True, capture_output=True,
                timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CheckResult(False, str(exc), ("unit-tests",))
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return CheckResult(completed.returncode == 0, output, ("unit-tests",))


@dataclass(frozen=True)
class BenchmarkResult:
    passed: bool
    duration_seconds: float
    score: float


class BenchmarkRunner:
    """Turn the deterministic unit-test gate into a measurable benchmark."""

    def run(self, candidate: CandidateWorkspace, runner: TestRunner | None = None) -> BenchmarkResult:
        started = time.monotonic()
        result = (runner or TestRunner()).run(candidate)
        duration = time.monotonic() - started
        return BenchmarkResult(
            passed=result.passed,
            duration_seconds=duration,
            score=1.0 if result.passed else 0.0,
        )


@dataclass(frozen=True)
class QualityMetrics:
    tests_passed: bool
    security_passed: bool
    changed_files: int
    score: float
    benchmark_passed: bool = True
    benchmark_seconds: float = 0.0

    @classmethod
    def measure(
        cls,
        tests: CheckResult,
        security: CheckResult,
        changed_files: int,
        baseline: float = 0.0,
        benchmark: BenchmarkResult | None = None,
    ) -> "QualityMetrics":
        score = 0.0
        if tests.passed:
            score += 0.7
        if security.passed:
            score += 0.3
        benchmark_passed = benchmark.passed if benchmark else tests.passed
        benchmark_seconds = benchmark.duration_seconds if benchmark else 0.0
        return cls(
            tests.passed, security.passed, changed_files, score,
            benchmark_passed, benchmark_seconds,
        )

    @property
    def passed(self) -> bool:
        return self.tests_passed and self.security_passed and self.benchmark_passed


class CandidateComparator:
    def better_or_equal(self, candidate: QualityMetrics, baseline: QualityMetrics | None = None) -> bool:
        return candidate.passed and (baseline is None or candidate.score >= baseline.score)


class AuditHistory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def append(self, **event: Any) -> None:
        record(self.settings.upgrade_log_path, time=datetime.now(timezone.utc).isoformat(), **event)


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    message: str
    status: str = "running"
    details: Mapping[str, Any] = field(default_factory=dict)
    time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProgressReporter:
    """Persist progress for dashboards and optionally stream it to a caller."""

    def __init__(
        self,
        settings: Settings,
        callback: Callable[[ProgressEvent], None] | None = None,
    ):
        self.settings = settings
        self.callback = callback

    def emit(
        self, phase: str, message: str, status: str = "running", **details: Any
    ) -> ProgressEvent:
        event = ProgressEvent(phase, message, status, details)
        try:
            self.settings.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with self.settings.progress_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "time": event.time,
                            "phase": event.phase,
                            "message": event.message,
                            "status": event.status,
                            "details": dict(event.details),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except OSError:
            # Progress must never turn a safe candidate into a failed upgrade.
            pass
        if self.callback:
            try:
                self.callback(event)
            except Exception:
                # A terminal/dashboard consumer is observational only.
                pass
        return event


def capture_diff(
    root: Path, candidate: CandidateWorkspace, files: Mapping[str, str] | tuple[str, ...]
) -> str:
    """Capture a deterministic unified diff before a candidate is deployed."""
    diff: list[str] = []
    for relative in sorted(files):
        before_path = root / relative
        after_path = candidate.path / relative
        before = (
            before_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
            if before_path.exists()
            else []
        )
        after = (
            after_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
            if after_path.exists()
            else []
        )
        diff.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                lineterm="\n",
            )
        )
    return "".join(diff)


def persist_diff(settings: Settings, diff: str) -> Path:
    digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]
    path = settings.upgrade_diff_root / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{digest}.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(diff, encoding="utf-8")
    return path


class DeploymentHooks:
    """Apply only a validated candidate and expose an explicit rollback hook."""

    def deploy(self, settings: Settings, candidate: CandidateWorkspace) -> Path:
        backup = create_backup(settings.root)
        try:
            apply_candidate(settings.root, candidate.path)
        except Exception:
            # Never leave a partially copied candidate active.
            restore_backup(settings.root, backup)
            raise
        return backup

    def rollback(
        self,
        settings: Settings,
        backup: Path,
        changed_files: tuple[str, ...] = (),
    ) -> None:
        restore_backup(settings.root, backup)
        manifest_path = backup / "manifest.json"
        backed_up = set()
        if manifest_path.exists():
            backed_up = set(json.loads(manifest_path.read_text(encoding="utf-8")).get("files", ()))
        for relative in changed_files:
            if relative not in backed_up:
                (settings.root / relative).unlink(missing_ok=True)


@dataclass(frozen=True)
class EvolutionResult:
    status: str
    summary: str = ""
    candidate: Path | None = None
    backup: Path | None = None
    metrics: QualityMetrics | None = None
    reason: str | None = None
    files: tuple[str, ...] = field(default_factory=tuple)
    diff_path: Path | None = None
    git_commit: str | None = None
    deployment: str = "none"


class EvolutionOrchestrator:
    """Coordinate analysis, generation, gates, comparison, and deployment."""

    def __init__(
        self,
        settings: Settings,
        model: ModelBackend,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ):
        self.settings = settings
        self.model = model
        self.validator = ProtectedPathValidator()
        self.history = AuditHistory(settings)
        self.deployer = DeploymentHooks()
        self.git = GitIntegration()
        self.progress_callback = progress_callback

    def run(
        self,
        goal: str,
        apply: bool = False,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> EvolutionResult:
        progress = ProgressReporter(
            self.settings, progress_callback or self.progress_callback
        )
        analyzer = RepositoryAnalyzer(self.settings.root)
        progress.emit("analysis", "Analyzing repository")
        report = analyzer.analyze()
        try:
            progress.emit("planning", "Asking the local model for a proposal")
            summary, changes = ImprovementPlanner(self.model).plan(
                report, analyzer.snapshot(), goal
            )
            progress.emit("validation", "Validating proposed paths", files=sorted(changes))
            self.validator.validate_changes(changes)
            candidate = CandidateWorkspace.create(self.settings.root, changes, self.validator)
            diff = capture_diff(self.settings.root, candidate, changes)
            if not diff:
                reason = "candidate does not change any file"
                self.history.append(
                    status="rejected", reason=reason, goal=goal, files=sorted(changes)
                )
                progress.emit("complete", reason, "rejected")
                return EvolutionResult("rejected", summary, candidate.path, reason=reason)
            diff_path = persist_diff(self.settings, diff)
            progress.emit("security", "Running static security checks")
            security = SecurityChecker().check(candidate)
            progress.emit("tests", "Running candidate unit tests")
            runner = TestRunner()
            tests = runner.run(candidate)
            progress.emit("benchmark", "Measuring candidate test benchmark")
            benchmark = BenchmarkRunner().run(candidate, runner=runner)
            metrics = QualityMetrics.measure(
                tests, security, len(changes), benchmark=benchmark
            )
            if not metrics.passed:
                reason = "security checks or unit tests failed"
                self.history.append(
                    status="rejected",
                    reason=reason,
                    goal=goal,
                    summary=summary,
                    files=sorted(changes),
                    diff=diff,
                    diff_path=str(diff_path),
                    metrics=metrics.__dict__,
                    deployment="none",
                )
                progress.emit("complete", reason, "rejected", metrics=metrics.__dict__)
                return EvolutionResult("rejected", summary, candidate.path, metrics=metrics,
                                       reason=reason, files=tuple(sorted(changes)),
                                       diff_path=diff_path)
            baseline = QualityMetrics(True, True, 0, 1.0)
            if not CandidateComparator().better_or_equal(metrics, baseline):
                raise EvolutionRejected(
                    "Candidate did not meet or improve the baseline quality gates."
                )
            backup = None
            deployed = False
            git_result = GitCommitResult(False, error="not requested")
            deployment = "none"
            if apply or self.settings.auto_apply_upgrades:
                progress.emit("deployment", "Applying validated candidate")
                backup = self.deployer.deploy(self.settings, candidate)
                deployed = True
                deployment = "applied"
                if self.settings.git_commit_upgrades:
                    progress.emit("git", "Creating local Git commit")
                    git_result = self.git.commit_applied(
                        self.settings.root, sorted(changes), summary
                    )
                    if not git_result.committed:
                        reason = f"local Git commit failed: {git_result.error}"
                        self.deployer.rollback(
                            self.settings, backup, changed_files=tuple(changes)
                        )
                        deployment = "rolled_back"
                        self.history.append(
                            status="rejected",
                            reason=reason,
                            goal=goal,
                            summary=summary,
                            files=sorted(changes),
                            diff=diff,
                            diff_path=str(diff_path),
                            metrics=metrics.__dict__,
                            deployment=deployment,
                            git_commit=None,
                        )
                        progress.emit("complete", reason, "rejected")
                        notify(
                            self.settings.root,
                            self.settings.notification_path,
                            "Evolution candidate rolled back after Git failure.",
                            reason=reason,
                            diff_path=str(diff_path),
                        )
                        return EvolutionResult(
                            "rejected", summary, candidate.path, backup, metrics,
                            reason, tuple(sorted(changes)), diff_path, None, deployment
                        )
                elif not git_result.committed:
                    git_result = GitCommitResult(False, error="Git commits disabled")
            version = next_version(self.settings.upgrade_state_path) if deployed else None
            progress.emit(
                "history", "Recording permanent upgrade history",
                deployment=deployment, git_commit=git_result.commit,
            )
            self.history.append(
                status="promoted" if deployed else "staged",
                version=version,
                summary=summary,
                goal=goal,
                files=sorted(changes),
                score=metrics.score,
                metrics=metrics.__dict__,
                diff=diff,
                diff_path=str(diff_path),
                deployment=deployment,
                git_commit=git_result.commit,
                git_commit_enabled=self.settings.git_commit_upgrades,
            )
            notify(
                self.settings.root, self.settings.notification_path,
                "Evolution candidate deployed." if deployed else "Evolution candidate staged.",
                summary=summary, files=sorted(changes), score=metrics.score,
                version=version, diff_path=str(diff_path),
                git_commit=git_result.commit,
            )
            progress.emit(
                "complete",
                "Evolution candidate deployed." if deployed else "Evolution candidate staged.",
                "succeeded",
            )
            return EvolutionResult(
                "promoted" if deployed else "staged", summary,
                candidate.path, backup, metrics, files=tuple(sorted(changes)),
                diff_path=diff_path, git_commit=git_result.commit, deployment=deployment,
            )
        except (EvolutionRejected, OSError, ValueError, RuntimeError) as exc:
            self.history.append(status="rejected", reason=str(exc), goal=goal)
            progress.emit("complete", str(exc), "rejected")
            notify(self.settings.root, self.settings.notification_path,
                   "Evolution candidate rejected.", reason=str(exc), goal=goal)
            return EvolutionResult("rejected", reason=str(exc))
