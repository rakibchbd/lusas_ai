from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent import LusasAgent
from .control import read as read_control, update as update_control
from .dashboard import write as write_dashboard
from .local_model import LocalModelError
from .monitor import follow, report, snapshot
from .service import install_service, remove_service, service_state
from .updater import UpgradeResult, restore_backup
from .web_learning import refresh as refresh_web


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
    control_parser = subparsers.add_parser(
        "control", help="Pause, resume, or bound autonomous evolution."
    )
    control_parser.add_argument(
        "action",
        choices=(
            "status",
            "set-autonomy",
            "stop",
            "resume",
            "pause-learning",
            "resume-learning",
            "pause-upgrades",
            "resume-upgrades",
            "disable-web",
            "enable-web",
            "disable-self-code",
            "enable-self-code",
            "disable-auto-deploy",
            "enable-auto-deploy",
            "lock",
            "unlock",
        ),
    )
    control_parser.add_argument(
        "--level",
        type=int,
        help="Runtime autonomy level from 0 to the configured maximum (for set-autonomy).",
    )
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
    report_parser = subparsers.add_parser(
        "report", help="Show the inspectable local upgrade dashboard."
    )
    report_parser.add_argument("--recent", type=int, default=10)
    report_parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable report JSON."
    )
    subparsers.add_parser(
        "dashboard", help="Generate the local HTML control dashboard."
    )
    web_parser = subparsers.add_parser(
        "web-refresh",
        help="Fetch newly published items from configured allowlisted web feeds.",
    )
    web_parser.add_argument(
        "--force", action="store_true", help="Refresh even before the interval."
    )
    service_parser = subparsers.add_parser(
        "service",
        help="Install or remove automatic macOS startup for model upgrades.",
    )
    service_parser.add_argument("action", choices=("install", "remove"))

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
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Run the guarded analyzer, quality, security, and deployment pipeline.",
    )
    evolve_parser.add_argument("--goal", required=True, help="Improvement objective.")
    evolve_parser.add_argument(
        "--apply", action="store_true",
        help="Deploy only after every evolution gate passes.",
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
            print(
                "Evolution scheduler: "
                f"{'enabled' if agent.settings.evolution_enabled else 'disabled'} "
                f"({agent.settings.evolution_interval_minutes} minute interval)"
            )
            print(
                "Web learning: "
                f"{'enabled' if agent.settings.web_learning_enabled else 'disabled'} "
                f"({agent.settings.web_refresh_interval_minutes} minute interval)"
            )
            print(f"Background worker: {service_state()}")
            print(f"Runtime controls: {json.dumps(read_control(agent.settings.root, agent.settings.autonomy_level), sort_keys=True)}")
            return 0

        if args.command == "control":
            if args.action == "status":
                print(json.dumps(read_control(agent.settings.root, agent.settings.autonomy_level), indent=2))
                return 0
            if args.action == "set-autonomy":
                if args.level is None or not 0 <= args.level <= agent.settings.autonomy_level:
                    raise ValueError(
                        f"autonomy level must be between 0 and {agent.settings.autonomy_level}"
                    )
                state = update_control(
                    agent.settings.root,
                    agent.settings.autonomy_level,
                    autonomy_level=args.level,
                )
                print(json.dumps(state, indent=2))
                return 0
            changes = {
                "stop": {"emergency_stop": True, "learning_paused": True, "upgrades_paused": True},
                "resume": {"emergency_stop": False, "learning_paused": False, "upgrades_paused": False},
                "pause-learning": {"learning_paused": True},
                "resume-learning": {"learning_paused": False},
                "pause-upgrades": {"upgrades_paused": True},
                "resume-upgrades": {"upgrades_paused": False},
                "disable-web": {"web_research_enabled": False},
                "enable-web": {"web_research_enabled": True},
                "disable-self-code": {"self_code_enabled": False},
                "enable-self-code": {"self_code_enabled": True},
                "disable-auto-deploy": {"auto_deploy_enabled": False},
                "enable-auto-deploy": {"auto_deploy_enabled": True},
                "lock": {"production_locked": True},
                "unlock": {"production_locked": False},
            }[args.action]
            state = update_control(
                agent.settings.root, agent.settings.autonomy_level, **changes
            )
            print(json.dumps(state, indent=2))
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

        if args.command == "report":
            if args.recent < 1:
                raise ValueError("report recent must be at least 1.")
            payload = report(agent.settings, recent=args.recent)
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=True))
            else:
                print(snapshot(agent.settings, recent=args.recent))
            return 0

        if args.command == "dashboard":
            print(f"Dashboard written to {write_dashboard(agent.settings)}")
            return 0

        if args.command == "web-refresh":
            print(json.dumps(refresh_web(agent.settings, force=args.force), indent=2))
            return 0

        if args.command == "service":
            path = (
                install_service(agent.settings.root)
                if args.action == "install"
                else remove_service()
            )
            print(f"Service {args.action}ed: {path}")
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

        if args.command == "evolve":
            def show_progress(event: object) -> None:
                phase = getattr(event, "phase", "evolution")
                message = getattr(event, "message", "")
                print(f"[evolve:{phase}] {message}", flush=True)

            result = agent.evolve(
                args.goal, apply=args.apply, progress_callback=show_progress
            )
            print(f"Evolution: {result.status}")
            if result.reason:
                print(f"Reason: {result.reason}")
            if result.candidate:
                print(f"Candidate: {result.candidate}")
            if result.backup:
                print(f"Backup: {result.backup}")
            if result.diff_path:
                print(f"Diff: {result.diff_path}")
            return 0 if result.status in {"staged", "promoted"} else 1

        if args.command == "rollback":
            restore_backup(agent.settings.root, args.backup)
            print(f"Restored backup: {args.backup}")
            return 0
    except (LocalModelError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1
