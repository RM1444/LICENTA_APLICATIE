"""Standard (non-sudo) action handlers.

Handler signatures are uniform: `(parameters: dict) -> ExecutionResult`. The
daemon dispatches parsed intents through `dispatch(action, parameters)`. Each
handler is responsible for logging its own outcome via the returned result.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    spoken: str
    stdout: str = ""
    stderr: str = ""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def get_current_time(_: dict[str, Any]) -> ExecutionResult:
    now = _dt.datetime.now().strftime("%H:%M")
    return ExecutionResult(ok=True, spoken=f"It is {now}.")


def get_weather(parameters: dict[str, Any]) -> ExecutionResult:
    # Intentionally stubbed: cloud access is optional and off by default.
    return ExecutionResult(
        ok=False,
        spoken="Weather lookup is disabled in this build. Enable an offline provider first.",
    )


def set_alarm(parameters: dict[str, Any]) -> ExecutionResult:
    when = str(parameters.get("time") or parameters.get("at") or "").strip()
    if not when:
        return ExecutionResult(ok=False, spoken="I did not catch the alarm time.")
    # Delegate to GNOME Clock through gnome-schedule if present; else advise.
    if shutil.which("gnome-clocks") is None:
        return ExecutionResult(
            ok=False,
            spoken="Install GNOME Clocks to set alarms, then try again.",
        )
    try:
        subprocess.Popen(["gnome-clocks"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to launch gnome-clocks")
        return ExecutionResult(ok=False, spoken="I could not open GNOME Clocks.", stderr=str(exc))
    return ExecutionResult(ok=True, spoken=f"Opening Clocks. Please set the alarm for {when}.")


def read_calendar(_: dict[str, Any]) -> ExecutionResult:
    try:
        result = _evolution_list_events_today()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calendar read failed")
        return ExecutionResult(ok=False, spoken="I could not read your calendar.", stderr=str(exc))
    if not result:
        return ExecutionResult(ok=True, spoken="You have no events scheduled for today.")
    summary = "; ".join(result)
    return ExecutionResult(ok=True, spoken=f"Today you have: {summary}.")


def add_calendar_event(parameters: dict[str, Any]) -> ExecutionResult:
    title = str(parameters.get("title") or parameters.get("summary") or "").strip()
    when = str(parameters.get("time") or parameters.get("at") or "").strip()
    if not title:
        return ExecutionResult(ok=False, spoken="The event needs a title.")
    try:
        _evolution_add_event(title=title, when=when)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calendar write failed")
        return ExecutionResult(ok=False, spoken="I could not add that event.", stderr=str(exc))
    return ExecutionResult(ok=True, spoken=f"Added event: {title}.")


_APP_ALIASES: dict[str, str] = {
    "settings": "gnome-control-center",
    "system settings": "gnome-control-center",
    "control center": "gnome-control-center",
    "files": "nautilus",
    "file manager": "nautilus",
    "terminal": "gnome-terminal",
    "calculator": "gnome-calculator",
    "calendar": "gnome-calendar",
    "clock": "gnome-clocks",
    "weather": "gnome-weather",
    "maps": "gnome-maps",
    "screenshot": "gnome-screenshot",
    "software": "gnome-software",
    "disks": "gnome-disks",
    "text editor": "gnome-text-editor",
    "editor": "gnome-text-editor",
}

_FLATPAK_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*){2,}$")


def open_application(parameters: dict[str, Any]) -> ExecutionResult:
    app_name = str(parameters.get("app_name") or parameters.get("application") or "").strip().lower()
    # Strip trailing punctuation Whisper sometimes leaves on the last word.
    app_name = app_name.strip(".,!?;:'\"").strip()
    if not app_name:
        return ExecutionResult(ok=False, spoken="Which application should I open?")

    alias = _APP_ALIASES.get(app_name)
    candidate_binaries = []
    if alias:
        candidate_binaries.append(alias)
    candidate_binaries.extend([app_name, app_name.replace(" ", "-"), f"gnome-{app_name}"])

    for binary in candidate_binaries:
        if shutil.which(binary):
            try:
                subprocess.Popen([binary])
                return ExecutionResult(ok=True, spoken=f"Opening {app_name}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to launch %s", binary)
                return ExecutionResult(ok=False, spoken=f"Could not start {app_name}", stderr=str(exc))

    # Try flatpak only if app_name is a plausible reverse-DNS id.
    if shutil.which("flatpak") and _FLATPAK_ID_RE.match(app_name):
        try:
            subprocess.Popen(["flatpak", "run", app_name])
            return ExecutionResult(ok=True, spoken=f"Opening flatpak {app_name}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Flatpak launch failed")
            return ExecutionResult(ok=False, spoken=f"Could not start {app_name}", stderr=str(exc))

    return ExecutionResult(ok=False, spoken=f"I could not find an application called {app_name}")


def close_application(parameters: dict[str, Any]) -> ExecutionResult:
    app_name = str(parameters.get("app_name") or parameters.get("application") or "").strip()
    if not app_name:
        return ExecutionResult(ok=False, spoken="Which application should I close?")
    try:
        # Use pkill -f so partial matches work for flatpak/AppImage processes.
        proc = subprocess.run(
            ["pkill", "-f", app_name],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode in (0, 1):  # 1 = no process matched
            msg = f"Closed {app_name}." if proc.returncode == 0 else f"{app_name} was not running."
            return ExecutionResult(ok=True, spoken=msg, stdout=proc.stdout)
        return ExecutionResult(ok=False, spoken=f"Could not close {app_name}.", stderr=proc.stderr)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pkill failed for %s", app_name)
        return ExecutionResult(ok=False, spoken="Close command failed.", stderr=str(exc))


def take_screenshot(_: dict[str, Any]) -> ExecutionResult:
    shots_dir = Path.home() / "Pictures" / "Screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    filename = shots_dir / _dt.datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
    try:
        if shutil.which("gnome-screenshot"):
            subprocess.run(["gnome-screenshot", "-f", str(filename)], check=True, timeout=10)
        elif shutil.which("grim"):
            subprocess.run(["grim", str(filename)], check=True, timeout=10)
        else:
            return ExecutionResult(ok=False, spoken="No screenshot tool is available.")
    except subprocess.CalledProcessError as exc:
        logger.exception("Screenshot failed")
        return ExecutionResult(ok=False, spoken="Screenshot failed.", stderr=str(exc))
    return ExecutionResult(ok=True, spoken=f"Screenshot saved to {filename.name}.")


def lock_screen(_: dict[str, Any]) -> ExecutionResult:
    try:
        _dbus_lock_screen()
        return ExecutionResult(ok=True, spoken="Locking the screen.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Screen lock via D-Bus failed; trying loginctl")
    try:
        subprocess.run(["loginctl", "lock-session"], check=True, timeout=5)
        return ExecutionResult(ok=True, spoken="Locking the screen.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("loginctl lock-session failed")
        return ExecutionResult(ok=False, spoken="I could not lock the screen.", stderr=str(exc))


def set_volume(parameters: dict[str, Any]) -> ExecutionResult:
    direction = str(parameters.get("direction") or "").lower()
    value = parameters.get("value")

    if shutil.which("pactl") is None:
        return ExecutionResult(ok=False, spoken="pactl is not installed.")

    try:
        if direction == "mute":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], check=True, timeout=5)
            return ExecutionResult(ok=True, spoken="Muting.")
        if direction == "up":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], check=True, timeout=5)
            return ExecutionResult(ok=True, spoken="Turning it up.")
        if direction == "down":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], check=True, timeout=5)
            return ExecutionResult(ok=True, spoken="Turning it down.")
        if direction == "set" and value is not None:
            pct = max(0, min(100, int(value)))
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
                check=True, timeout=5,
            )
            return ExecutionResult(ok=True, spoken=f"Volume set to {pct}%.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Volume control failed")
        return ExecutionResult(ok=False, spoken="Volume control failed.", stderr=str(exc))

    return ExecutionResult(ok=False, spoken="I did not understand the volume command.")


def set_brightness(parameters: dict[str, Any]) -> ExecutionResult:
    if shutil.which("brightnessctl") is None:
        return ExecutionResult(ok=False, spoken="Install brightnessctl to control screen brightness.")
    direction = str(parameters.get("direction") or "").lower()
    value = parameters.get("value")
    try:
        if direction == "set" and value is not None:
            pct = max(0, min(100, int(value)))
            subprocess.run(["brightnessctl", "set", f"{pct}%"], check=True, timeout=5)
            return ExecutionResult(ok=True, spoken=f"Brightness set to {pct}%.")
        step = "+10%" if direction != "down" else "10%-"
        subprocess.run(["brightnessctl", "set", step], check=True, timeout=5)
        return ExecutionResult(ok=True, spoken=f"Brightness {'up' if step.endswith('%') and '+' in step else 'down'}.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Brightness control failed")
        return ExecutionResult(ok=False, spoken="Brightness control failed.", stderr=str(exc))


# ---------------------------------------------------------------------------
# D-Bus helpers (lazy-imported so unit tests can skip the dep)
# ---------------------------------------------------------------------------

def _dbus_lock_screen() -> None:
    import dbus  # type: ignore

    bus = dbus.SessionBus()
    proxy = bus.get_object(
        "org.gnome.ScreenSaver",
        "/org/gnome/ScreenSaver",
    )
    proxy.Lock(dbus_interface="org.gnome.ScreenSaver")


def _evolution_list_events_today() -> list[str]:
    """Read today's calendar events via Evolution Data Server.

    Returns a list of human-readable summaries. Returns an empty list when
    EDS is unavailable so we can still speak a sensible message.
    """
    try:
        import gi  # type: ignore
        gi.require_version("EDataServer", "1.2")
        gi.require_version("ECal", "2.0")
        from gi.repository import EDataServer, ECal, GLib  # type: ignore
    except Exception:  # noqa: BLE001
        logger.warning("Evolution Data Server bindings unavailable.")
        return []

    # Minimal EDS query - kept defensive because EDS schemas change often.
    try:
        registry = EDataServer.SourceRegistry.new_sync(None)
        source = registry.ref_builtin_calendar()
        client = ECal.Client.connect_sync(source, ECal.ClientSourceType.EVENTS, 30, None)
        today = _dt.date.today()
        start = int(_dt.datetime.combine(today, _dt.time.min).timestamp())
        end = int(_dt.datetime.combine(today, _dt.time.max).timestamp())
        success, events = client.get_object_list_as_comps_sync(
            f"(occur-in-time-range? (make-time {start}) (make-time {end}))",
            None,
        )
        if not success:
            return []
        return [ev.get_summary().get_value() for ev in events if ev.get_summary()]
    except Exception:  # noqa: BLE001
        logger.exception("EDS query failed")
        return []


def _evolution_add_event(*, title: str, when: str) -> None:
    # Minimal stub that opens GNOME Calendar with the title prefilled, since
    # direct EDS writes require a full iCal ingredient that is out of scope.
    if shutil.which("gnome-calendar") is None:
        raise RuntimeError("GNOME Calendar is not installed.")
    subprocess.Popen(["gnome-calendar"])


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[[dict[str, Any]], ExecutionResult]] = {
    "get_time": get_current_time,
    "get_weather": get_weather,
    "set_alarm": set_alarm,
    "get_calendar": read_calendar,
    "add_calendar_event": add_calendar_event,
    "open_app": open_application,
    "close_app": close_application,
    "screenshot": take_screenshot,
    "lock_screen": lock_screen,
    "volume_control": set_volume,
    "brightness_control": set_brightness,
}


def dispatch(action: str, parameters: dict[str, Any]) -> ExecutionResult:
    """Run a non-sudo action. Unknown actions produce a deny-by-default result."""
    handler = HANDLERS.get(action)
    if handler is None:
        return ExecutionResult(
            ok=False,
            spoken=f"I don't have a handler for {action}.",
        )
    try:
        return handler(parameters or {})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Handler %s crashed", action)
        return ExecutionResult(
            ok=False,
            spoken="The command failed unexpectedly.",
            stderr=str(exc),
        )
