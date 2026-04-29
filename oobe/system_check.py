"""Pre-flight system checks for the OOBE.

Runs before the onboarding GUI starts. Each check returns a Status row; the UI
renders them and blocks enrolment until every required check passes.
"""
from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_all_checks() -> list[CheckResult]:
    return [
        _check_microphone(),
        _check_dnf(),
        _check_python_dependencies(),
        _check_disk_space(),
        _check_ollama(),
        _check_pipewire(),
    ]


def all_required_passed(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results if r.required)


# ---------------------------------------------------------------------------

def _check_microphone() -> CheckResult:
    try:
        import pyaudio  # type: ignore
    except ImportError:
        return CheckResult("Microphone", False, "PyAudio not installed.", required=True)

    pa = pyaudio.PyAudio()
    try:
        count = pa.get_device_count()
        input_devices = [
            pa.get_device_info_by_index(i)["name"]
            for i in range(count)
            if int(pa.get_device_info_by_index(i).get("maxInputChannels", 0)) > 0
        ]
    except Exception as exc:  # noqa: BLE001
        pa.terminate()
        return CheckResult("Microphone", False, f"Enumeration failed: {exc}")
    pa.terminate()

    if not input_devices:
        return CheckResult("Microphone", False, "No input devices detected.")
    return CheckResult(
        "Microphone", True, f"{len(input_devices)} input device(s): {input_devices[0]}"
    )


def _check_dnf() -> CheckResult:
    if shutil.which("dnf") is None:
        return CheckResult("DNF", False, "dnf not on PATH.")
    try:
        proc = subprocess.run(
            ["dnf", "--version"], capture_output=True, text=True, timeout=5,
        )
        version = (proc.stdout.strip().splitlines() or ["unknown"])[0]
    except Exception as exc:  # noqa: BLE001
        return CheckResult("DNF", False, f"dnf invocation failed: {exc}")
    return CheckResult("DNF", True, version)


def _check_python_dependencies() -> CheckResult:
    missing: list[str] = []
    for module in (
        "numpy", "scipy", "yaml",
        "pyaudio", "faster_whisper", "speechbrain",
        "ollama", "langchain",
    ):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        return CheckResult(
            "Python dependencies", False,
            f"Missing modules: {', '.join(missing)}",
        )
    return CheckResult("Python dependencies", True, "All required modules importable.")


def _check_disk_space() -> CheckResult:
    total, used, free = shutil.disk_usage(Path.home())
    free_gb = free / (1024 ** 3)
    ok = free_gb >= 5.0
    return CheckResult(
        "Disk space", ok,
        f"{free_gb:.1f} GB free (5 GB required for models).",
        required=True,
    )


def _check_ollama() -> CheckResult:
    if shutil.which("ollama") is None:
        return CheckResult(
            "Ollama", False,
            "ollama not installed; run bootstrap_fedora.sh",
            required=False,
        )
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Ollama", False, f"ollama list failed: {exc}", required=False)

    if proc.returncode != 0:
        return CheckResult(
            "Ollama", False, "ollama daemon not running. Start with: systemctl --user start ollama",
            required=False,
        )
    return CheckResult(
        "Ollama", True,
        proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "running",
        required=False,
    )


def _check_pipewire() -> CheckResult:
    if shutil.which("pactl") is None:
        return CheckResult("PipeWire/Pulse", False, "pactl not on PATH.")
    try:
        proc = subprocess.run(
            ["pactl", "info"], capture_output=True, text=True, timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("PipeWire/Pulse", False, f"pactl info failed: {exc}")
    if proc.returncode != 0:
        return CheckResult("PipeWire/Pulse", False, proc.stderr.strip() or "pactl info failed")
    # First line: "Server String: ..."
    first_line = (proc.stdout.splitlines() or ["running"])[0]
    return CheckResult("PipeWire/Pulse", True, first_line)
