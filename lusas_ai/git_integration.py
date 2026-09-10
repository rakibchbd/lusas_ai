"""Opt-in local Git commits for accepted source upgrades.

This module intentionally has no remote operations.  It refuses to commit
when the repository already has staged changes so an upgrade cannot silently
include unrelated user work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class GitCommitResult:
    committed: bool
    commit: str | None = None
    error: str | None = None


class GitIntegration:
    """Create one local commit containing only an applied candidate."""

    def commit_applied(
        self, root: Path, files: Sequence[str], summary: str
    ) -> GitCommitResult:
        root = root.expanduser().resolve()
        if not files:
            return GitCommitResult(False, error="no changed files to commit")
        if not (root / ".git").exists():
            return GitCommitResult(False, error="repository has no .git directory")

        try:
            staged = self._run(root, ["diff", "--cached", "--name-only"]).stdout.strip()
            if staged:
                return GitCommitResult(
                    False,
                    error="repository index already contains staged changes",
                )
            status = self._run(
                root, ["status", "--porcelain", "--", *files]
            ).stdout.strip()
            if not status:
                return GitCommitResult(False, error="applied candidate has no Git changes")
            self._run(root, ["add", "--", *files])
            message = f"LUSAS autonomous upgrade: {summary.strip()}"[:200]
            self._run(root, ["commit", "-m", message])
            commit = self._run(root, ["rev-parse", "HEAD"]).stdout.strip()
            if not commit:
                return GitCommitResult(False, error="Git did not report the new commit")
            return GitCommitResult(True, commit=commit)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            self._unstage(root, files)
            return GitCommitResult(False, error=detail.strip())

    @staticmethod
    def _run(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )

    @classmethod
    def _unstage(cls, root: Path, files: Sequence[str]) -> None:
        try:
            cls._run(root, ["reset", "--", *files])
        except (OSError, subprocess.CalledProcessError):
            # The original commit failure is the useful error.  A failed
            # cleanup is still visible in the next `git status` invocation.
            pass
