"""Default (hardcoded) response router.

Handles simple, well-defined commands that do NOT require any LLM call.
These are resolved purely by pattern matching and execute immediately,
saving API tokens and reducing latency for common queries.

Two categories:
    1. Intent shortcuts - map directly to an Intent without calling an LLM.
    2. Conversational shortcuts - return a spoken string for greetings,
       thanks, etc. without needing the LLM chat endpoint.

Usage:
    from core.default_responses import try_default_intent, try_default_chat

    intent = try_default_intent(transcript)
    if intent is not None:
        # skip LLM intent parsing entirely
        ...

    reply = try_default_chat(transcript)
    if reply is not None:
        # skip LLM chat entirely
        ...
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent shortcuts — bypass LLM intent parsing
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, action, parameters_factory, requires_sudo, confidence)
# parameters_factory is a callable that takes the regex match and returns a dict.

def _no_params(_match) -> dict:
    return {}


def _volume_params(match) -> dict:
    text = match.string.lower()
    if "mute" in text or "unmute" in text:
        return {"direction": "mute"}
    if "up" in text or "louder" in text or "raise" in text or "increase" in text:
        return {"direction": "up"}
    if "down" in text or "quieter" in text or "lower" in text or "decrease" in text:
        return {"direction": "down"}
    m = re.search(r"\b(\d{1,3})\s*%?", text)
    if m:
        return {"direction": "set", "value": max(0, min(100, int(m.group(1))))}
    return {"direction": "up"}


def _brightness_params(match) -> dict:
    text = match.string.lower()
    m = re.search(r"\b(\d{1,3})\s*%?", text)
    if m:
        return {"direction": "set", "value": max(0, min(100, int(m.group(1))))}
    if "down" in text or "dim" in text or "lower" in text or "decrease" in text:
        return {"direction": "down"}
    return {"direction": "up"}


# Words that signal a compound / multi-step command which the LLM should parse.
_COMPOUND_MARKERS = re.compile(
    r"\b(and\s+(then\s+)?(open|launch|start|close|kill|run|go|load|play|visit))|"
    r"\b(then\s+(open|launch|start|close|kill|run|go|load|play|visit))|"
    r"\b(after\s+that)|"
    r"\bfrom\s+there\b",
    re.I,
)


def _open_app_params(match) -> dict | None:
    """Extract the app name. Returns None for compound commands so the
    match is rejected and the transcript falls through to the LLM."""
    text = match.string.strip()
    # Compound command? Let the LLM handle it.
    if _COMPOUND_MARKERS.search(text):
        return None
    # Remove the command prefix
    cleaned = re.sub(
        r"^(please\s+)?(can you\s+)?(open|launch|start|run)\s+",
        "", text, flags=re.I,
    ).strip().strip(".,!?;:'\"")
    return {"app_name": cleaned} if cleaned else {}


def _close_app_params(match) -> dict | None:
    """Extract the app name. Returns None for compound commands."""
    text = match.string.strip()
    if _COMPOUND_MARKERS.search(text):
        return None
    cleaned = re.sub(
        r"^(please\s+)?(can you\s+)?(close|kill|quit|stop|exit)\s+",
        "", text, flags=re.I,
    ).strip().strip(".,!?;:'\"")
    return {"app_name": cleaned} if cleaned else {}


# (pattern, action, params_factory, requires_sudo, confidence)
_INTENT_SHORTCUTS: list[tuple[re.Pattern, str, callable, bool, float]] = [
    # Time
    (re.compile(
        r"^(what('?s| is) the (current )?time|"
        r"tell me the time|"
        r"what time is it|"
        r"(current )?time[.!?]?)$",
        re.I,
    ), "get_time", _no_params, False, 0.99),

    # Screenshot
    (re.compile(
        r"^(take (a )?screenshot|capture (the )?screen|screenshot)[.!?]?$",
        re.I,
    ), "screenshot", _no_params, False, 0.99),

    # Lock screen
    (re.compile(
        r"^(lock (the )?screen|lock it|lock my (computer|pc|screen))[.!?]?$",
        re.I,
    ), "lock_screen", _no_params, False, 0.99),

    # Shutdown
    (re.compile(
        r"^(shut ?down|power off|turn off (the )?(computer|pc|system))[.!?]?$",
        re.I,
    ), "shutdown", _no_params, True, 0.99),

    # Restart
    (re.compile(
        r"^(restart|reboot|reboot (the )?(computer|pc|system))[.!?]?$",
        re.I,
    ), "restart", _no_params, True, 0.99),

    # Volume
    (re.compile(
        r"^(volume\s+(up|down|mute|unmute)|"
        r"(turn|set)\s+(the\s+)?volume\s+(up|down|to\s+\d+)|"
        r"mute|unmute|"
        r"(louder|quieter|raise|lower)\s*(the\s+)?volume|"
        r"(turn it|make it)\s+(louder|quieter))[.!?]?$",
        re.I,
    ), "volume_control", _volume_params, False, 0.95),

    # Brightness
    (re.compile(
        r"^((set\s+)?(the\s+)?brightness\s+(up|down|to\s+\d+)|"
        r"(turn|set)\s+(the\s+)?brightness\s+(up|down|to\s+\d+)|"
        r"(brighter|dimmer|dim)\s*(the\s+)?screen|"
        r"(increase|decrease)\s+(the\s+)?brightness)[.!?]?$",
        re.I,
    ), "brightness_control", _brightness_params, False, 0.95),

    # Open app (broad but still deterministic)
    (re.compile(
        r"^(please\s+)?(can you\s+)?(open|launch|start|run)\s+\S+",
        re.I,
    ), "open_app", _open_app_params, False, 0.92),

    # Close app
    (re.compile(
        r"^(please\s+)?(can you\s+)?(close|kill|quit|stop|exit)\s+\S+",
        re.I,
    ), "close_app", _close_app_params, False, 0.92),

    # Calendar read
    (re.compile(
        r"^(what('?s| is) on my calendar|"
        r"(show|read|check) (my )?calendar|"
        r"(do I|what do I) have (today|scheduled)|"
        r"my (schedule|events|appointments)( today)?)[.!?]?$",
        re.I,
    ), "get_calendar", _no_params, False, 0.95),

    # Switch LLM provider — multiple phrasings
    (re.compile(
        r"^(switch|change|use|set)\s+"
        r"((to\s+|the\s+)?(llm|model|provider|ai|brain)\s+(to\s+)?)?"
        r"(to\s+)?"
        r"(anthropic|claude|gemini|google|groq|ollama|local)\b",
        re.I,
    ), "switch_llm", lambda m: {"provider": _extract_provider(m)}, False, 0.99),
]


def _extract_provider(match) -> str:
    """Extract the target provider name from a switch command."""
    text = match.string.lower()
    if "claude" in text or "anthropic" in text:
        return "anthropic"
    if "gemini" in text or "google" in text:
        return "gemini"
    if "groq" in text:
        return "groq"
    if "ollama" in text or "local" in text:
        return "ollama"
    return "gemini"  # fallback default


def try_default_intent(text: str) -> Optional[object]:
    """Attempt to resolve `text` to an Intent via hardcoded patterns.

    Returns an Intent dataclass (imported lazily to avoid circular imports)
    or None if no default pattern matches.
    """
    text = (text or "").strip()
    if not text:
        return None

    for pattern, action, params_factory, requires_sudo, confidence in _INTENT_SHORTCUTS:
        match = pattern.search(text)
        if match:
            params = params_factory(match)
            # params_factory returns None to reject compound commands
            if params is None:
                logger.debug("Default pattern matched %s but params rejected (compound command).", action)
                continue
            from core.brain import Intent  # lazy import avoids circular
            logger.info("Default intent matched: %s (pattern shortcut, conf=%.2f)", action, confidence)
            return Intent(
                action=action,
                parameters=params,
                requires_sudo=requires_sudo,
                confidence=confidence,
            )
    return None


# ---------------------------------------------------------------------------
# Conversational shortcuts — bypass LLM chat
# ---------------------------------------------------------------------------

_CHAT_SHORTCUTS: list[tuple[re.Pattern, str | callable]] = [
    # Greetings
    (re.compile(r"^(hi|hello|hey|howdy|good (morning|afternoon|evening))[.!?]?$", re.I),
     "Hello! How can I help you?"),

    # Thanks
    (re.compile(r"^(thanks?|thank you|cheers|much appreciated)[.!?]?$", re.I),
     "You're welcome!"),

    # How are you
    (re.compile(r"^(how are you|how('?s| is) it going|what('?s| is) up)[.!?]?$", re.I),
     "I'm running smoothly. What can I do for you?"),

    # Identity
    (re.compile(r"^(who are you|what are you|what('?s| is) your name)[.!?]?$", re.I),
     "I'm your Fedora voice assistant. I can open apps, control your system, and more."),

    # Goodbye
    (re.compile(r"^(bye|goodbye|see you|good night|that('?s| is) all)[.!?]?$", re.I),
     "Goodbye! I'll still be listening for the wake word."),

    # Date
    (re.compile(
        r"^(what('?s| is) (the )?(today('?s)?|current) date|"
        r"what day is (it|today)|"
        r"(today('?s)?|the) date)[.!?]?$",
        re.I,
    ), lambda: f"Today is {_dt.datetime.now().strftime('%A, %B %d, %Y')}."),

    # Day of week
    (re.compile(r"^what day (of the week )?(is it|is today)[.!?]?$", re.I),
     lambda: f"Today is {_dt.datetime.now().strftime('%A')}."),

    # Capabilities
    (re.compile(
        r"^(what can you do|what are your (capabilities|features)|help me|"
        r"(list|show) (your )?(commands|features))[.!?]?$",
        re.I,
    ), "I can open and close apps, control volume and brightness, take screenshots, "
       "lock the screen, check your calendar, install packages, and answer questions. "
       "Just ask!"),
]


def try_default_chat(text: str) -> Optional[str]:
    """Attempt to resolve `text` with a canned conversational response.

    Returns a spoken-ready string or None if no shortcut matches.
    """
    text = (text or "").strip()
    if not text:
        return None

    for pattern, response in _CHAT_SHORTCUTS:
        if pattern.search(text):
            if callable(response):
                result = response()
            else:
                result = response
            logger.info("Default chat matched: %r -> %r", text, result[:50])
            return result
    return None
