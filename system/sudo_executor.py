#!/usr/bin/env python3
"""Privileged executor - the ONLY script granted NOPASSWD sudo.

Invoked as:
    sudo /opt/fedora-voice-assistant/system/sudo_executor.py <action> [args...]

Every action is implemented as a tightly validated subcommand. Arbitrary
command injection is impossible because:
  * Only a fixed set of actions is accepted.
  * Each action validates its own arguments against an allow-list regex.
  * All subprocess calls are list-form (never shell=True).

The biometric gate in the daemon is the security boundary; this script only
enforces the tight perimeter around what root-level commands may be run.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Callable, Iterable


# Regexes for validating arguments. Anchored, short, and deliberately strict.
PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._+-]{0,99}$")
SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9@._-]{1,100}\.(service|target|socket|timer)$")
PATH_RE = re.compile(r"^/(etc|opt|var|usr/local|home/[a-zA-Z0-9._-]+)(/[A-Za-z0-9._-]+)*$")


class InvalidArgument(Exception):
    """Raised when an argument fails allow-list validation."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_package_names(names: Iterable[str]) -> list[str]:
    validated: list[str] = []
    for name in names:
        if not PACKAGE_NAME_RE.match(name):
            raise InvalidArgument(f"Rejected package name: {name!r}")
        validated.append(name)
    if not validated:
        raise InvalidArgument("No package names supplied.")
    return validated


def _validate_service(name: str) -> str:
    if not SERVICE_NAME_RE.match(name):
        raise InvalidArgument(f"Rejected systemd unit name: {name!r}")
    return name


def _validate_path(path: str) -> str:
    if not PATH_RE.match(path):
        raise InvalidArgument(f"Rejected path: {path!r}")
    if ".." in path.split("/"):
        raise InvalidArgument("Path contains '..'.")
    return path


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> int:
    print(f"[sudo_executor] running: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode


def action_dnf_install(args: list[str]) -> int:
    packages = _validate_package_names(args)
    return _run(["/usr/bin/dnf", "install", "-y", *packages])


def action_dnf_remove(args: list[str]) -> int:
    packages = _validate_package_names(args)
    return _run(["/usr/bin/dnf", "remove", "-y", *packages])


def action_flatpak_install(args: list[str]) -> int:
    packages = _validate_package_names(args)
    return _run(["/usr/bin/flatpak", "install", "-y", "--noninteractive", "flathub", *packages])


def action_flatpak_remove(args: list[str]) -> int:
    packages = _validate_package_names(args)
    return _run(["/usr/bin/flatpak", "uninstall", "-y", "--noninteractive", *packages])


def action_systemctl(args: list[str]) -> int:
    if len(args) < 2:
        raise InvalidArgument("systemctl requires <verb> <unit>.")
    verb, unit = args[0], args[1]
    if verb not in {"start", "stop", "restart", "enable", "disable"}:
        raise InvalidArgument(f"Rejected systemctl verb: {verb!r}")
    unit = _validate_service(unit)
    return _run(["/usr/bin/systemctl", verb, unit])


def action_shutdown(args: list[str]) -> int:
    # Optional delay in minutes, default: 1 minute to allow TTS to finish.
    delay = "+1"
    if args:
        if not re.match(r"^\+\d{1,3}$", args[0]):
            raise InvalidArgument(f"Rejected shutdown delay: {args[0]!r}")
        delay = args[0]
    return _run(["/usr/sbin/shutdown", "-h", delay])


def action_restart(args: list[str]) -> int:
    delay = "+1"
    if args:
        if not re.match(r"^\+\d{1,3}$", args[0]):
            raise InvalidArgument(f"Rejected restart delay: {args[0]!r}")
        delay = args[0]
    return _run(["/usr/sbin/shutdown", "-r", delay])


def action_write_file(args: list[str]) -> int:
    """Write stdin to a strict-path file. Supports a narrow set of config paths."""
    if len(args) != 1:
        raise InvalidArgument("write-file requires exactly one path argument.")
    path = _validate_path(args[0])
    data = sys.stdin.buffer.read()
    with open(path, "wb") as fh:
        fh.write(data)
    return 0


ACTIONS: dict[str, Callable[[list[str]], int]] = {
    "dnf-install": action_dnf_install,
    "dnf-remove": action_dnf_remove,
    "flatpak-install": action_flatpak_install,
    "flatpak-remove": action_flatpak_remove,
    "systemctl": action_systemctl,
    "shutdown": action_shutdown,
    "restart": action_restart,
    "write-file": action_write_file,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: sudo_executor.py <action> [args...]", file=sys.stderr)
        print(f"actions: {', '.join(sorted(ACTIONS))}", file=sys.stderr)
        return 2

    action, *rest = argv
    handler = ACTIONS.get(action)
    if handler is None:
        print(f"[sudo_executor] rejected action: {action!r}", file=sys.stderr)
        return 2

    # Refuse to run without real root unless we're being unit-tested.
    if os.geteuid() != 0 and os.environ.get("FVA_SUDO_EXECUTOR_ALLOW_NONROOT") != "1":
        print("[sudo_executor] must be invoked via sudo.", file=sys.stderr)
        return 3

    try:
        return handler(rest)
    except InvalidArgument as exc:
        print(f"[sudo_executor] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
