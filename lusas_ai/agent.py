from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .ollama import OllamaClient
from .updater import UpgradeResult, perform_upgrade
from .workspace import Workspace


AGENT_SYSTEM_PROMPT = """You are LUSAS AI, a local-first coding agent.
You help with software in the configured workspace. Keep responses focused on
code, implementation decisions, tests, and concise change summaries.
Do not access credentials, attack third-party systems, bypass security
controls, or make changes outside the configured workspace."""

UPGRADE_SYSTEM_PROMPT = """You are the self-upgrade planner for LUSAS AI.
Return ONLY one JSON object with this shape:
{"summary": "short summary", "files": {"lusas_ai/example.py": "complete file text"}}

You may change only files under lusas_ai/, tests/, or training/. Include complete contents
for every file you change. Preserve the existing CLI behavior unless the goal
requires changing it. Do not add credential access, exploit code, arbitrary
network access, shell execution, persistence outside the project, or security
bypass behavior. Add or update tests for behavior you change."""


class ProposalError(ValueError):
    """Raised when the model returns an invalid self-upgrade proposal."""


def _parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ProposalError("The model did not return a JSON object.")
    try:
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ProposalError("The self-upgrade proposal was not valid JSON.") from exc
    if not isinstance(result, dict):
        raise ProposalError("The self-upgrade proposal must be a JSON object.")
    return result


class LusasAgent:
    def __init__(self, root: Path) -> None:
        self.settings = Settings.load(root)
        self.workspace = Workspace(self.settings.workspace_root)
        self.ollama = OllamaClient(
            self.settings.ollama_url,
            self.settings.model,
        )

    def chat(self, prompt: str) -> str:
        context = self.workspace.snapshot()
        return self.ollama.chat(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Workspace context:\n{context}\n\n"
                        f"User request:\n{prompt}"
                    ),
                },
            ]
        )

    def propose_self_upgrade(self, goal: str) -> tuple[str, dict[str, str]]:
        current_source = self._source_snapshot()
        response = self.ollama.chat(
            [
                {"role": "system", "content": UPGRADE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Upgrade goal:\n{goal}\n\n"
                        "Current LUSAS AI source:\n"
                        f"{current_source}"
                    ),
                },
            ]
        )
        proposal = _parse_json_object(response)
        summary = proposal.get("summary")
        files = proposal.get("files")
        if not isinstance(summary, str) or not summary.strip():
            raise ProposalError("The proposal needs a non-empty summary.")
        if not isinstance(files, dict) or not files:
            raise ProposalError("The proposal needs a non-empty files object.")
        if not all(
            isinstance(path, str) and isinstance(content, str)
            for path, content in files.items()
        ):
            raise ProposalError("Every proposed file path and content must be text.")
        return summary, files

    def self_upgrade(self, goal: str, apply: bool = False) -> UpgradeResult:
        _, changes = self.propose_self_upgrade(goal)
        return perform_upgrade(self.settings, changes, goal=goal, apply=apply)

    def _source_snapshot(self, max_chars: int = 100_000) -> str:
        chunks: list[str] = []
        used = 0
        for directory_name in ("lusas_ai", "tests"):
            directory = self.settings.root / directory_name
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.py")):
                content = path.read_text(encoding="utf-8", errors="replace")
                remaining = max_chars - used
                if remaining <= 0:
                    return "\n\n".join(chunks)
                relative = path.relative_to(self.settings.root)
                clipped = content[:remaining]
                chunks.append(f"--- {relative} ---\n{clipped}")
                used += len(clipped)
        return "\n\n".join(chunks)
