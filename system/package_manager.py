"""DNF + Flatpak search/install/remove orchestration.

Read-only searches run directly as the current user. Installs and removals are
delegated to `sudo_executor.py` so the only privileged code path remains the
NOPASSWD-scoped helper script.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.config import PROJECT_ROOT
from system.executor import ExecutionResult

logger = logging.getLogger(__name__)

SUDO_EXECUTOR = PROJECT_ROOT / "system" / "sudo_executor.py"

PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._+-]{0,99}$")


@dataclass(frozen=True)
class PackageSearchHit:
    source: str          # "dnf" | "flatpak"
    name: str
    summary: str


def _require_valid_name(name: str) -> None:
    if not PACKAGE_NAME_RE.match(name):
        raise ValueError(f"Invalid package name: {name!r}")


# ---------------------------------------------------------------------------
# Search (unprivileged)
# ---------------------------------------------------------------------------

def search(query: str) -> list[PackageSearchHit]:
    hits: list[PackageSearchHit] = []
    hits.extend(_search_dnf(query))
    hits.extend(_search_flatpak(query))
    return hits


def _search_dnf(query: str) -> Iterable[PackageSearchHit]:
    if shutil.which("dnf") is None:
        return []
    try:
        proc = subprocess.run(
            ["dnf", "--quiet", "search", query],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        logger.exception("dnf search failed")
        return []

    hits: list[PackageSearchHit] = []
    for line in proc.stdout.splitlines():
        # Expected format: "name.arch : summary"
        m = re.match(r"^([A-Za-z0-9._+-]+)\.[A-Za-z0-9_]+\s*:\s*(.+)$", line)
        if m:
            hits.append(PackageSearchHit(source="dnf", name=m.group(1), summary=m.group(2)))
    return hits[:15]


def _search_flatpak(query: str) -> Iterable[PackageSearchHit]:
    if shutil.which("flatpak") is None:
        return []
    try:
        proc = subprocess.run(
            ["flatpak", "search", "--columns=application,description", query],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        logger.exception("flatpak search failed")
        return []

    hits: list[PackageSearchHit] = []
    for line in proc.stdout.splitlines()[1:]:  # skip header
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip():
            hits.append(PackageSearchHit(
                source="flatpak", name=parts[0].strip(), summary=parts[1].strip(),
            ))
    return hits[:15]


# ---------------------------------------------------------------------------
# Install / remove (privileged, routed through sudo_executor)
# ---------------------------------------------------------------------------

def install(name: str, *, source: str = "dnf") -> ExecutionResult:
    _require_valid_name(name)
    if source not in {"dnf", "flatpak"}:
        return ExecutionResult(ok=False, spoken=f"Unknown package source {source}.")
    action = "dnf-install" if source == "dnf" else "flatpak-install"
    return _sudo(action, name, human_verb="installed")


def remove(name: str, *, source: str = "dnf") -> ExecutionResult:
    _require_valid_name(name)
    if source not in {"dnf", "flatpak"}:
        return ExecutionResult(ok=False, spoken=f"Unknown package source {source}.")
    action = "dnf-remove" if source == "dnf" else "flatpak-remove"
    return _sudo(action, name, human_verb="removed")


def _sudo(action: str, *args: str, human_verb: str) -> ExecutionResult:
    cmd = ["sudo", "-n", str(SUDO_EXECUTOR), action, *args]
    logger.info("Invoking sudo_executor: %s", cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(ok=False, spoken="That command took too long. Cancelled.",
                               stderr=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("sudo_executor invocation failed")
        return ExecutionResult(ok=False, spoken="I could not call the privileged helper.",
                               stderr=str(exc))

    if proc.returncode == 0:
        return ExecutionResult(ok=True, spoken=f"Done. Package {human_verb}.",
                               stdout=proc.stdout, stderr=proc.stderr)

    # Summarise stderr for the user.
    short_err = _first_line(proc.stderr) or "unknown error"
    return ExecutionResult(
        ok=False,
        spoken=f"The command failed: {short_err}",
        stdout=proc.stdout, stderr=proc.stderr,
    )


def _first_line(text: str) -> str:
    """Return the most informative line of stderr.

    Skips our own `[sudo_executor]` informational banners and DNF's progress
    decoration so the user hears the actual error cause (e.g. "no match for
    argument"), not "running /usr/bin/dnf install -y ...".
    """
    meaningful: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[sudo_executor]"):
            continue
        if line.startswith(">>>") or line.startswith("==="):
            continue
        meaningful.append(line)
    if not meaningful:
        return ""
    for line in meaningful:
        low = line.lower()
        if "error" in low or "no match" in low or "failed" in low or "not found" in low:
            return line
    return meaningful[-1]
