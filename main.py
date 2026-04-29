#!/usr/bin/env python3
"""Fedora Voice Assistant - daemon entry point.

Simplified pipeline (no biometrics, no approval flow):
    audio capture -> STT -> fuzzy match commands.json -> execute
                                   \-> no match -> AI chat -> TTS response

A small GTK4 + libadwaita status window runs on the main thread while the
audio/dispatch pipeline runs in a worker thread. The window exposes a
Hold-to-Talk button (also bound to Space).

Signal handling:
    SIGTERM / SIGINT -> flag the pipeline to stop, close all resources,
    quit the GTK app, exit 0.
"""
from __future__ import annotations

import logging
import logging.handlers
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

from core.config import PROJECT_ROOT, get_config, resolve_path

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from core import brain, db_manager, tts_engine
from core.audio_pipeline import AudioPipeline, CapturedUtterance
from core.command_manager import command_mgr
from core.provider_manager import provider_mgr
from core.ui import AssistantApplication, AssistantWindow

logger = logging.getLogger("fva")


# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    cfg = get_config()["logging"]
    log_path = resolve_path(get_config()["paths"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=int(cfg.get("max_bytes", 10_485_760)),
        backupCount=int(cfg.get("backup_count", 5)),
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(cfg.get("level", "INFO"))
    root.addHandler(handler)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class AssistantDaemon:
    def __init__(self, *, ptt_event: threading.Event) -> None:
        self._stop_event = threading.Event()
        self._pipeline: Optional[AudioPipeline] = None
        self._ptt_event = ptt_event
        self._window: Optional[AssistantWindow] = None

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def attach_window(self, window: AssistantWindow) -> None:
        self._window = window

    def _ui_status(self, status: str, detail: Optional[str] = None) -> None:
        if self._window is not None:
            self._window.push_status(status, detail)

    def _ui_transcript(self, text: str) -> None:
        if self._window is not None:
            self._window.push_transcript(text)

    def _ui_action(self, text: str) -> None:
        if self._window is not None:
            self._window.push_action(text)

    def _say(self, text: str) -> None:
        self._ui_status("Speaking", text)
        tts_engine.speak(text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()
        if self._pipeline is not None:
            self._pipeline._stop_event.set()

    def run(self) -> int:
        logger.info("Fedora Voice Assistant starting up.")

        db_manager.initialize_database()
        command_mgr.load()
        logger.info("Loaded %d voice commands.", command_mgr.command_count)

        self._pipeline = AudioPipeline()
        try:
            self._pipeline.start()
            self._ui_status("Idle", 'Say "Hey Jarvis" or hold the button below.')
            tts_engine.speak("Assistant online.")
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled daemon error")
            self._ui_status("Error", "Daemon crashed - see logs.")
            tts_engine.speak("The assistant hit an error and is shutting down.")
            return 1
        finally:
            if self._pipeline is not None:
                self._pipeline.stop()
        logger.info("Daemon exited cleanly.")
        return 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        assert self._pipeline is not None

        def on_wake() -> None:
            """Called by the pipeline after a hot word is recognised."""
            self._ui_status("Listening", "Wake word detected.")
            tts_engine.speak("listening")

        for utterance in self._pipeline.listen_for_utterances(
            ptt_event=self._ptt_event,
            on_wake=on_wake,
        ):
            if self._stop_event.is_set():
                break
            try:
                self._handle_utterance(utterance)
            except Exception:  # noqa: BLE001
                logger.exception("Error handling utterance")
                self._say("Something went wrong. I am still listening.")
            # Small delay after response TTS to let speaker echo fade
            # before the pipeline starts recording the next hot-word clip.
            time.sleep(0.3)
            self._ui_status("Idle", 'Say "Hey Jarvis" or hold the button below.')

    # ------------------------------------------------------------------
    # Per-utterance processing
    # ------------------------------------------------------------------

    def _handle_utterance(self, utterance: CapturedUtterance) -> None:
        # Step 1: Transcribe
        self._ui_status("Processing", "Transcribing…")
        transcript = brain.transcribe(utterance.audio_f32, utterance.sample_rate)
        logger.info("Transcript [source=%s]: %r", utterance.source, transcript.text)
        self._ui_transcript(transcript.text or "(empty)")

        if not transcript.text:
            self._say("I did not hear anything.")
            db_manager.log_event(
                speaker_id=None,
                transcript=None, action=None, result="error",
                similarity_score=None,
                error_message="Empty transcript",
            )
            return

        # Step 2: Try fuzzy match against commands.json
        self._ui_status("Processing", "Matching command…")
        cmd_result = command_mgr.match_and_execute(transcript.text)

        if cmd_result is not None:
            # Command matched — speak feedback
            logger.info("Command matched: %r -> %s", transcript.text, cmd_result.matched_phrase)
            self._ui_action(f"cmd: {cmd_result.matched_phrase}")
            if cmd_result.feedback:
                self._say(cmd_result.feedback)
            db_manager.log_event(
                speaker_id=None,
                transcript=transcript.text,
                action=f"cmd:{cmd_result.matched_phrase}",
                result="success" if cmd_result.ok else "error",
                similarity_score=None,
            )
            return

        # Step 3: No command matched — send to AI for conversational response
        logger.info("No command match; routing to AI chat.")
        self._ui_status("Processing", "Asking AI…")
        self._ui_action("ai chat")

        reply = brain.chat_response(transcript.text)
        self._ui_action(f"ai: {reply[:60]}")
        self._say(reply)

        db_manager.log_event(
            speaker_id=None,
            transcript=transcript.text,
            action="ai_chat",
            result="success",
            similarity_score=None,
        )


# ---------------------------------------------------------------------------
# Entry point - GTK on main thread, daemon on worker thread
# ---------------------------------------------------------------------------

def main() -> int:
    _configure_logging()
    provider_mgr.initialize()

    ptt_event = threading.Event()
    daemon = AssistantDaemon(ptt_event=ptt_event)

    daemon_thread = threading.Thread(target=daemon.run, name="fva-daemon", daemon=True)

    app_holder: dict = {"app": None}

    def on_window_ready(window: AssistantWindow) -> None:
        daemon.attach_window(window)
        if not daemon_thread.is_alive():
            daemon_thread.start()

    def on_quit() -> None:
        logger.info("Window closed; stopping daemon.")
        daemon.stop()
        app = app_holder.get("app")
        if app is not None:
            app.quit()

    app = AssistantApplication(
        ptt_event=ptt_event,
        on_window_ready=on_window_ready,
        on_quit=on_quit,
    )
    app_holder["app"] = app

    def _sig_handler(signum, _frame) -> None:
        logger.info("Received signal %s; stopping.", signum)
        daemon.stop()
        app.quit()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    rc = app.run([])
    daemon.stop()
    if daemon_thread.is_alive():
        daemon_thread.join(timeout=5)
    return rc


if __name__ == "__main__":
    sys.exit(main())
