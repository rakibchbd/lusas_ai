from __future__ import annotations

import os
from pathlib import Path
import plistlib
import subprocess
import sys


SERVICE_LABEL = "com.lusas.ai.autoupgrade"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"


def build_plist(root: Path, python_executable: str | None = None) -> dict:
    root = root.expanduser().resolve()
    executable = python_executable or sys.executable
    log_dir = root / ".lusas"
    return {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [
            executable,
            str(root / "training" / "auto_upgrade.py"),
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": {"NetworkState": True},
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": str(log_dir / "auto-upgrade.log"),
        "StandardErrorPath": str(log_dir / "auto-upgrade-error.log"),
    }


def install_service(root: Path) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("The automatic startup service is supported only on macOS.")
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = root.expanduser().resolve()
    (root / ".lusas").mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(build_plist(root)))
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootout", domain, str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    completed = subprocess.run(
        ["launchctl", "bootstrap", domain, str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Could not start {SERVICE_LABEL}: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return path


def remove_service() -> Path:
    path = plist_path()
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootout", domain, str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    path.unlink(missing_ok=True)
    return path
