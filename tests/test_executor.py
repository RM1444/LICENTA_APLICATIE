"""Executor dispatch + sudo_executor argument validation tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system import executor, sudo_executor


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_dispatch_unknown_action_fails_gracefully() -> None:
    result = executor.dispatch("conjure_dragon", {})
    assert result.ok is False
    assert "conjure_dragon" in result.spoken


def test_get_current_time_returns_hh_mm() -> None:
    result = executor.get_current_time({})
    assert result.ok is True
    assert ":" in result.spoken


def test_open_application_missing_name_fails() -> None:
    result = executor.open_application({})
    assert result.ok is False
    assert "which application" in result.spoken.lower()


def test_close_application_missing_name_fails() -> None:
    result = executor.close_application({})
    assert result.ok is False


def test_dispatch_handler_exception_is_contained() -> None:
    with mock.patch.dict(executor.HANDLERS, {"boom": lambda _p: (_ for _ in ()).throw(RuntimeError("kaboom"))}):
        result = executor.dispatch("boom", {})
    assert result.ok is False
    assert "unexpectedly" in result.spoken.lower()


# ---------------------------------------------------------------------------
# sudo_executor argument validation
# ---------------------------------------------------------------------------

def test_sudo_validate_package_names_accepts_valid() -> None:
    assert sudo_executor._validate_package_names(["firefox", "gcc-c++"]) == ["firefox", "gcc-c++"]


def test_sudo_validate_package_names_rejects_shell_metachars() -> None:
    for bad in ("foo;rm -rf /", "foo bar", "$(reboot)", "--help", ""):
        with pytest.raises(sudo_executor.InvalidArgument):
            sudo_executor._validate_package_names([bad])


def test_sudo_validate_service_accepts_typed_units() -> None:
    assert sudo_executor._validate_service("ollama.service") == "ollama.service"
    assert sudo_executor._validate_service("timers@user.timer") == "timers@user.timer"


def test_sudo_validate_service_rejects_bare_names() -> None:
    for bad in ("ollama", "ollama.service;ls", "/etc/passwd"):
        with pytest.raises(sudo_executor.InvalidArgument):
            sudo_executor._validate_service(bad)


def test_sudo_validate_path_accepts_allowed_roots(tmp_path, monkeypatch) -> None:
    assert sudo_executor._validate_path("/etc/hosts") == "/etc/hosts"
    assert sudo_executor._validate_path("/opt/fedora-voice-assistant/config") == "/opt/fedora-voice-assistant/config"


def test_sudo_validate_path_rejects_traversal() -> None:
    for bad in ("/etc/../root/.ssh", "/tmp/x", "relative/path", "/etc/*", "/etc/"):
        with pytest.raises(sudo_executor.InvalidArgument):
            sudo_executor._validate_path(bad)


def test_sudo_main_rejects_unknown_action() -> None:
    os.environ["FVA_SUDO_EXECUTOR_ALLOW_NONROOT"] = "1"
    try:
        rc = sudo_executor.main(["definitely-not-allowed"])
    finally:
        os.environ.pop("FVA_SUDO_EXECUTOR_ALLOW_NONROOT", None)
    assert rc == 2


def test_sudo_main_rejects_shutdown_with_garbage_delay() -> None:
    os.environ["FVA_SUDO_EXECUTOR_ALLOW_NONROOT"] = "1"
    try:
        # Don't actually run shutdown; replace _run with a no-op.
        with mock.patch.object(sudo_executor, "_run", return_value=0):
            rc_ok = sudo_executor.main(["shutdown"])
            assert rc_ok == 0
        with mock.patch.object(sudo_executor, "_run", return_value=0):
            rc_bad = sudo_executor.main(["shutdown", "now;reboot"])
    finally:
        os.environ.pop("FVA_SUDO_EXECUTOR_ALLOW_NONROOT", None)
    assert rc_bad == 2


def test_sudo_main_requires_root_without_override() -> None:
    # The env var is not set, and pytest is not running as root.
    rc = sudo_executor.main(["dnf-install", "vim"])
    assert rc == 3
