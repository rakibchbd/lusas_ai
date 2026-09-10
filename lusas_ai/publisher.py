from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


class PublishError(RuntimeError):
    """Raised when an upgrade cannot be published as a pull request."""


def publish_upgrade(
    root: Path,
    candidate: Path,
    score: float,
    learned_examples: list[dict[str, str]],
) -> dict[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            list(args),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise PublishError((completed.stderr or completed.stdout).strip())
        return completed.stdout.strip()

    status = run("git", "status", "--porcelain")
    unexpected = [
        line[3:]
        for line in status.splitlines()
        if line and line[3:] not in {
            "training/data/learned.jsonl",
            "training/upgrade_history.jsonl",
        }
        and not line[3:].startswith(("lusas_ai/", "tests/", "training/"))
    ]
    if unexpected:
        raise PublishError(
            "Cannot publish while unrelated working-tree changes exist: "
            + ", ".join(unexpected)
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch = f"lusas/upgrade-{timestamp}"
    history_path = root / "training" / "upgrade_history.jsonl"
    learned_path = root / "training" / "data" / "learned.jsonl"
    learned_path.parent.mkdir(parents=True, exist_ok=True)
    learned_path.write_text(
        "\n".join(
            json.dumps(example, ensure_ascii=True)
            for example in learned_examples
        )
        + ("\n" if learned_examples else ""),
        encoding="utf-8",
    )
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "candidate": str(candidate),
                    "score": score,
                },
                ensure_ascii=True,
            )
            + "\n"
        )

    run("git", "switch", "-c", branch)
    try:
        run("git", "add", "lusas_ai", "tests", "training/data/learned.jsonl", "training/upgrade_history.jsonl")
        run("git", "commit", "-m", f"LUSAS model upgrade {timestamp}")
        run("git", "push", "--set-upstream", "origin", branch)
        pr_url = run(
            "gh",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            f"LUSAS model upgrade {timestamp}",
            "--body",
            f"Automated local model upgrade passed evaluation with score {score:.3f}.",
        )
    except Exception:
        run("git", "switch", "-")
        raise
    return {"branch": branch, "pull_request": pr_url}
