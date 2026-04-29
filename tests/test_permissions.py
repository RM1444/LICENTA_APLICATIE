"""Permissions matrix tests - covers every cell from the skill manifest."""
from __future__ import annotations

import pytest

from core.permissions import Category, Role, category_for, is_allowed, requires_approval


@pytest.mark.parametrize(
    "role,action,expected",
    [
        # General - everyone can ask the time
        (Role.OWNER, "get_time", True),
        (Role.GUEST, "get_time", True),
        (Role.UNREGISTERED, "get_time", True),

        # Calendar read - owner + guest yes, unregistered no
        (Role.OWNER, "get_calendar", True),
        (Role.GUEST, "get_calendar", True),
        (Role.UNREGISTERED, "get_calendar", False),

        # Calendar write - only owner
        (Role.OWNER, "add_calendar_event", True),
        (Role.GUEST, "add_calendar_event", False),
        (Role.UNREGISTERED, "add_calendar_event", False),

        # App launching - owner + guest
        (Role.OWNER, "open_app", True),
        (Role.GUEST, "open_app", True),
        (Role.UNREGISTERED, "open_app", False),

        # Package install - owner only
        (Role.OWNER, "install_package", True),
        (Role.GUEST, "install_package", False),
        (Role.UNREGISTERED, "install_package", False),

        # System config - owner only (shutdown/restart are system_config)
        (Role.OWNER, "shutdown", True),
        (Role.GUEST, "shutdown", False),
        (Role.UNREGISTERED, "shutdown", False),
    ],
)
def test_matrix_access(role: Role, action: str, expected: bool) -> None:
    assert is_allowed(role, action) is expected


def test_unknown_action_default_denies() -> None:
    assert is_allowed(Role.OWNER, "conjure_dragon") is False
    assert is_allowed(Role.UNREGISTERED, "conjure_dragon") is False


def test_requires_approval_owner_never() -> None:
    for action in ("install_package", "shutdown", "add_calendar_event", "get_time"):
        assert requires_approval(Role.OWNER, action) is False


def test_requires_approval_guest_for_denied_actions() -> None:
    assert requires_approval(Role.GUEST, "install_package") is True
    assert requires_approval(Role.GUEST, "get_time") is False


def test_requires_approval_unregistered_for_denied_actions() -> None:
    assert requires_approval(Role.UNREGISTERED, "open_app") is True
    assert requires_approval(Role.UNREGISTERED, "get_weather") is False


def test_category_mapping_covers_critical_actions() -> None:
    assert category_for("install_package") == Category.PACKAGE_INSTALL
    assert category_for("modify_file") == Category.FILE_SYSTEM
    assert category_for("shutdown") == Category.SYSTEM_CONFIG
    assert category_for("nonexistent_action") is None


def test_role_accepts_string() -> None:
    assert is_allowed("Owner", "shutdown") is True
    assert is_allowed("Guest", "shutdown") is False
