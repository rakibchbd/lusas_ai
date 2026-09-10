from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .agent import LusasAgent
from .local_model import LocalModelError
from .monitor import follow, snapshot
from .web_learning import DEFAULT_SOURCES, WebCollector
from .updater import UpgradeResult, restore_backup


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_upgrade_result(result: UpgradeResult) -> None:
    print(f"Backup:  {result.backup_path}")
    print(f"Staged:  {result.staging_path}")
    print(f"Tests:   {'passed' if result.tests.passed else 'failed'}")
    print(f"Applied: {'yes' if result.applied else 'no'}")
    if result.tests.output:
        print("\nTest output:\n" + result.tests.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lusas")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show local configuration.")
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Inspect learned data and follow learning/upgrade events.",
    )
    monitor_parser.add_argument(
        "--follow",
        action="store_true",
        help="Refresh when local learning or notification files change.",
    )
    monitor_parser.add_argument("--interval", type=float, default=1.0)
    monitor_parser.add_argument("--recent", type=int, default=10)
    web_parser = subparsers.add_parser(
        "collect-web",
        help="Collect bounded documentation samples for review.",
    )
    web_parser.add_argument(
        "--source",
        action="append",
        help="Allowed documentation URL; defaults to Python, MDN, and PyTorch.",
    )

    chat_parser = subparsers.add_parser("chat", help="Chat with the local agent.")
    chat_parser.add_argument("prompt", nargs="*", help="One-shot prompt.")
    learn_parser = subparsers.add_parser(
        "learn",
        help="Add an approved instruction and answer to the next training cycle.",
    )
    learn_parser.add_argument("--instruction", required=True)
    learn_parser.add_argument("--output", required=True)

    upgrade_parser = subparsers.add_parser(
        "self-upgrade",
        help="Prepare and optionally apply a tested self-upgrade.",
    )
    upgrade_parser.add_argument("--goal", required=True, help="Upgrade objective.")
    upgrade_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the candidate only after its tests pass.",
    )
    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Restore a previous self-upgrade backup.",
    )
    rollback_parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help="Path to a backup directory under .lusas/backups.",
    )
    return parser


def _interactive_chat(agent: LusasAgent) -> None:
    print("LUSAS AI local chat. Type /exit to quit.")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"/exit", "/quit"}:
            return
        if not prompt:
            continue
        print("\nLUSAS AI> " + agent.chat(prompt))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = LusasAgent(_project_root())

    try:
        if args.command == "status":
            print(f"Model:     {agent.settings.local_model_path}")
            print("Backend:   local LUSAS model")
            print(f"Workspace: {agent.workspace.root}")
            print(
                "Auto-apply self-upgrades: "
                f"{'enabled' if agent.settings.auto_apply_upgrades else 'disabled'}"
            )
            return 0

        if args.command == "monitor":
            if args.interval <= 0 or args.recent < 1:
                raise ValueError("monitor interval must be positive and recent must be at least 1.")
            if args.follow:
                for update in follow(agent.settings, interval=args.interval):
                    print("\033[2J\033[H" + update, flush=True)
            else:
                print(snapshot(agent.settings, recent=args.recent))
            return 0

        if args.command == "collect-web":
            sources = tuple(args.source) if args.source else None
            count = WebCollector(agent.settings.web_pending_path).collect(
                sources or DEFAULT_SOURCES
            )
            print(
                f"Collected {count} sample(s) for review at "
                f"{agent.settings.web_pending_path}"
            )
            return 0

        if args.command == "chat":
            if args.prompt:
                print(agent.chat(" ".join(args.prompt)))
            else:
                _interactive_chat(agent)
            return 0

        if args.command == "learn":
            agent.learn(args.instruction, args.output)
            print(f"Learned example saved to {agent.settings.learning_path}")
            return 0

        if args.command == "self-upgrade":
            result = agent.self_upgrade(args.goal, apply=args.apply)
            _print_upgrade_result(result)
            return 0 if result.tests.passed else 1

        if args.command == "rollback":
            restore_backup(agent.settings.root, args.backup)
            print(f"Restored backup: {args.backup}")
            return 0
    except (LocalModelError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1
