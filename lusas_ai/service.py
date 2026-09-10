from __future__ import annotations

import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time


SERVICE_LABEL = "com.lusas.ai.autoupgrade"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"


def log_directory() -> Path:
    """Return a LaunchAgent-safe log directory outside macOS protected folders."""
    return Path.home() / "Library" / "Logs" / "LUSAS_AI"


def service_state() -> str:
    """Return the observable LaunchAgent state without changing it."""
    if sys.platform != "darwin":
        return "unsupported"
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{SERVICE_LABEL}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return "not-installed"
    if "state = running" in result.stdout:
        return "running"
    return "not-running"


def build_plist(root: Path, python_executable: str | None = None) -> dict:
    root = root.expanduser().resolve()
    executable = python_executable or sys.executable
    log_dir = log_directory()
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
    root = root.expanduser().resolve()
    worker = root / "training" / "auto_upgrade.py"
    if not worker.is_file():
        raise RuntimeError(f"The LUSAS worker was not found at {worker}.")
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_directory().mkdir(parents=True, exist_ok=True)
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
    time.sleep(1)
    state = subprocess.run(
        ["launchctl", "print", f"{domain}/{SERVICE_LABEL}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if state.returncode or "state = not running" in state.stdout:
        raise RuntimeError(
            f"{SERVICE_LABEL} was installed but is not running. "
            f"Check {log_directory()} for the worker error log."
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
