"""Shared per-model cooldown for Groq API rate-limit errors.

Groq returns HTTP 429 when a model's per-day/per-minute token budget is
exhausted, with a "Please try again in 14m0s" message. Retrying inside that
window just produces more 429s that flood the logs and stall the user.

`should_skip(model)` returns True while the cooldown is active.
`record_rate_limit(model, exc)` parses the retry-after window from the
exception message and stores it so future calls short-circuit.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cooldowns: dict[str, float] = {}  # model -> monotonic deadline

_RETRY_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.I)
_DEFAULT_COOLDOWN_S = 60.0


def should_skip(model: str) -> bool:
    with _lock:
        deadline = _cooldowns.get(model, 0.0)
        if deadline == 0.0:
            return False
        if time.monotonic() >= deadline:
            _cooldowns.pop(model, None)
            return False
        return True


def seconds_remaining(model: str) -> float:
    with _lock:
        deadline = _cooldowns.get(model, 0.0)
    return max(0.0, deadline - time.monotonic())


def record_rate_limit(model: str, exc: BaseException) -> float:
    """Mark `model` as rate-limited, parsing the retry-after from `exc`.

    Returns the number of seconds the model will be skipped for.
    """
    retry_after = _parse_retry_after(exc) or _DEFAULT_COOLDOWN_S
    deadline = time.monotonic() + retry_after
    with _lock:
        existing = _cooldowns.get(model, 0.0)
        _cooldowns[model] = max(existing, deadline)
    logger.warning("Groq model %s rate-limited; cooling down for %.0fs.", model, retry_after)
    return retry_after


def _parse_retry_after(exc: BaseException) -> Optional[float]:
    message = str(exc)
    match = _RETRY_RE.search(message)
    if not match:
        return None
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60.0 + seconds
