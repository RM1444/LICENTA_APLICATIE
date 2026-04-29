"""Command manager: JSON-based voice commands with fuzzy matching.

Loads voice commands from `commands.json` and dispatches them via
`thefuzz` fuzzy string matching — the same approach used by
linux-voice-control. Commands that don't match any entry are routed
to the AI assistant for a conversational response.

Usage:
    from core.command_manager import command_mgr

    command_mgr.load()
    result = command_mgr.match_and_execute(transcript)
    if result is None:
        # No command matched — send to AI
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from thefuzz import fuzz, process  # type: ignore

from core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Minimum fuzzy score to consider a match (0–100).
MATCH_THRESHOLD = 75
# Secondary validation threshold for multi-word commands.
TOKEN_SORT_THRESHOLD = 70


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a matched command."""
    matched_phrase: str
    feedback: str
    ok: bool
    builtin: Optional[str] = None


class CommandManager:
    """Loads commands from JSON, fuzzy-matches transcripts, executes shell commands."""

    def __init__(self) -> None:
        self._commands: dict = {}
        self._choices: list[str] = []
        self._loaded = False

    def load(self, path: Optional[Path] = None) -> None:
        """Load commands from the JSON file."""
        path = path or (PROJECT_ROOT / "commands.json")
        if not path.exists():
            logger.warning("commands.json not found at %s; no commands loaded.", path)
            self._commands = {}
            self._choices = []
            self._loaded = True
            return

        with path.open("r", encoding="utf-8") as fh:
            self._commands = json.load(fh)

        self._choices = list(self._commands.keys())
        self._loaded = True
        logger.info("Loaded %d voice commands from %s", len(self._choices), path)

    def match_and_execute(self, text: str) -> Optional[CommandResult]:
        """Try to match `text` against known commands and execute.

        Returns a CommandResult if a command matched, or None if the
        transcript should be routed to the AI assistant.
        """
        if not self._loaded:
            self.load()

        text = (text or "").strip().lower()
        text = "".join(ch for ch in text if ch.isalpha() or ch.isdigit() or ch == " ").strip()
        if not text:
            return None

        if not self._choices:
            return None

        # Primary fuzzy match
        best_match = process.extractOne(text, self._choices)
        if best_match is None:
            return None

        matched_phrase, score = best_match[0], best_match[1]
        logger.info("Fuzzy match: %r -> %r (score=%d)", text, matched_phrase, score)

        if score < MATCH_THRESHOLD:
            logger.info("Score %d below threshold %d; routing to AI.", score, MATCH_THRESHOLD)
            return None

        # Secondary validation for multi-word commands
        if " " in matched_phrase:
            token_ratio = fuzz.token_sort_ratio(text, matched_phrase)
            if token_ratio < TOKEN_SORT_THRESHOLD:
                logger.info("Token sort ratio %d below %d; routing to AI.",
                            token_ratio, TOKEN_SORT_THRESHOLD)
                return None

        cmd_data = self._commands[matched_phrase]

        # Handle built-in actions
        builtin = cmd_data.get("builtin")
        if builtin:
            return self._handle_builtin(builtin, matched_phrase)

        # Speak feedback
        feedback = cmd_data.get("feedback", "")

        # Execute shell command
        exec_cmd = cmd_data.get("exec", "")
        if not exec_cmd:
            return CommandResult(
                matched_phrase=matched_phrase,
                feedback=feedback or f"Matched: {matched_phrase}.",
                ok=True,
            )

        try:
            subprocess.Popen(
                shlex.split(exec_cmd),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Executed: %s", exec_cmd)
            return CommandResult(
                matched_phrase=matched_phrase,
                feedback=feedback or f"Done.",
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to execute: %s", exec_cmd)
            return CommandResult(
                matched_phrase=matched_phrase,
                feedback=f"Failed to run {matched_phrase}.",
                ok=False,
            )

    def _handle_builtin(self, builtin: str, phrase: str) -> CommandResult:
        """Handle built-in actions that don't need a shell command."""
        if builtin == "get_time":
            now = _dt.datetime.now().strftime("%H:%M")
            return CommandResult(
                matched_phrase=phrase,
                feedback=f"It is {now}.",
                ok=True,
                builtin=builtin,
            )
        return CommandResult(
            matched_phrase=phrase,
            feedback=f"Unknown builtin: {builtin}.",
            ok=False,
            builtin=builtin,
        )

    @property
    def command_count(self) -> int:
        return len(self._choices)


# Module-level singleton
command_mgr = CommandManager()
