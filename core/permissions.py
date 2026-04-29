"""Role-based access control matrix.

Mirrors Section 2.4 of the skill manifest. The daemon calls `is_allowed(role,
action)` before executing every parsed intent, and `requires_approval(role,
action)` to decide whether an owner-approval window is required for a
guest/unregistered speaker.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class Role(str, Enum):
    OWNER = "Owner"
    GUEST = "Guest"
    UNREGISTERED = "Unregistered"


# Action categories used by the permissions matrix.
class Category(str, Enum):
    GENERAL = "general"                # time, weather, general knowledge
    CALENDAR_READ = "calendar_read"
    CALENDAR_WRITE = "calendar_write"
    APP_LAUNCH = "app_launch"
    FILE_SYSTEM = "file_system"
    PACKAGE_INSTALL = "package_install"
    SYSTEM_CONFIG = "system_config"
    USER_MANAGEMENT = "user_management"


# Map brain-parsed action names to permission categories.
ACTION_CATEGORY: dict[str, Category] = {
    "get_time":            Category.GENERAL,
    "get_weather":         Category.GENERAL,
    "search_web":          Category.GENERAL,
    "system_info":         Category.GENERAL,
    "set_alarm":           Category.GENERAL,

    "get_calendar":        Category.CALENDAR_READ,
    "add_calendar_event":  Category.CALENDAR_WRITE,

    "open_app":            Category.APP_LAUNCH,
    "close_app":           Category.APP_LAUNCH,
    "screenshot":          Category.APP_LAUNCH,

    "lock_screen":         Category.SYSTEM_CONFIG,
    "volume_control":      Category.SYSTEM_CONFIG,
    "brightness_control":  Category.SYSTEM_CONFIG,

    "modify_file":         Category.FILE_SYSTEM,

    "install_package":     Category.PACKAGE_INSTALL,
    "remove_package":      Category.PACKAGE_INSTALL,

    "shutdown":            Category.SYSTEM_CONFIG,
    "restart":             Category.SYSTEM_CONFIG,
    "run_command":         Category.SYSTEM_CONFIG,
}


# The matrix from the skill manifest. Columns: Owner, Guest, Unregistered.
MATRIX: dict[Category, dict[Role, bool]] = {
    Category.GENERAL:          {Role.OWNER: True,  Role.GUEST: True,  Role.UNREGISTERED: True},
    Category.CALENDAR_READ:    {Role.OWNER: True,  Role.GUEST: True,  Role.UNREGISTERED: False},
    Category.CALENDAR_WRITE:   {Role.OWNER: True,  Role.GUEST: False, Role.UNREGISTERED: False},
    Category.APP_LAUNCH:       {Role.OWNER: True,  Role.GUEST: True,  Role.UNREGISTERED: False},
    Category.FILE_SYSTEM:      {Role.OWNER: True,  Role.GUEST: False, Role.UNREGISTERED: False},
    Category.PACKAGE_INSTALL:  {Role.OWNER: True,  Role.GUEST: False, Role.UNREGISTERED: False},
    Category.SYSTEM_CONFIG:    {Role.OWNER: True,  Role.GUEST: False, Role.UNREGISTERED: False},
    Category.USER_MANAGEMENT:  {Role.OWNER: True,  Role.GUEST: False, Role.UNREGISTERED: False},
}


def category_for(action: str) -> Optional[Category]:
    return ACTION_CATEGORY.get(action)


def is_allowed(role: Role | str, action: str) -> bool:
    role = Role(role) if isinstance(role, str) else role
    category = category_for(action)
    if category is None:
        # Unknown action defaults to DENY for safety.
        return False
    return MATRIX[category].get(role, False)


def requires_approval(role: Role | str, action: str) -> bool:
    """Return True if this action, for this role, demands owner verbal approval.

    Owners never need approval; other roles need approval for anything beyond
    their default grant.
    """
    role = Role(role) if isinstance(role, str) else role
    if role == Role.OWNER:
        return False
    return not is_allowed(role, action)
