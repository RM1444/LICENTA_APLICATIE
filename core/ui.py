"""Runtime status window for the assistant daemon.

A small GTK4 + libadwaita window that runs on the main thread while the
audio pipeline and command dispatcher run in a worker thread. Surfaces:

    * Assistant state (Idle, Listening, Processing, Speaking, ...).
    * Last transcript and last executed action.
    * A Hold-to-Talk button (also bound to the Space key).
    * An LLM provider selector dropdown (switch at runtime without restart).

The daemon pushes updates via the `push_*` methods, which marshal onto
the GTK main loop with `GLib.idle_add`.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, GLib, Gtk  # noqa: E402

from core.provider_manager import SUPPORTED_PROVIDERS, provider_mgr

logger = logging.getLogger(__name__)

APP_ID = "io.fedora.voiceassistant.Assistant"


class AssistantWindow(Adw.ApplicationWindow):
    def __init__(
        self,
        app: Adw.Application,
        *,
        ptt_event: threading.Event,
        on_quit: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(application=app)
        self.set_title("Fedora Voice Assistant")
        self.set_default_size(520, 420)

        self._ptt_event = ptt_event
        self._on_quit = on_quit

        body = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=14,
            margin_start=24, margin_end=24, margin_top=16, margin_bottom=24,
        )

        self._status_label = Gtk.Label(xalign=0.5)
        self._status_label.add_css_class("title-2")
        self._status_label.set_text("Starting...")
        body.append(self._status_label)

        self._detail_label = Gtk.Label(xalign=0.5, wrap=True)
        self._detail_label.add_css_class("dim-label")
        self._detail_label.set_text("Loading...")
        body.append(self._detail_label)

        group = Adw.PreferencesGroup()
        self._transcript_row = Adw.ActionRow(title="Last transcript", subtitle="---")
        self._action_row = Adw.ActionRow(title="Last action", subtitle="---")
        group.add(self._transcript_row)
        group.add(self._action_row)
        body.append(group)

        # --- LLM Provider Selector ---
        provider_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=10,
            halign=Gtk.Align.CENTER,
        )
        provider_label = Gtk.Label(label="LLM Provider:")
        provider_label.add_css_class("dim-label")
        provider_box.append(provider_label)

        self._provider_dropdown = Gtk.DropDown()
        provider_names = list(SUPPORTED_PROVIDERS)
        string_list = Gtk.StringList()
        for name in provider_names:
            string_list.append(name.capitalize())
        self._provider_dropdown.set_model(string_list)
        self._provider_names = provider_names

        try:
            active_idx = provider_names.index(provider_mgr.active_provider)
        except ValueError:
            active_idx = 0
        self._provider_dropdown.set_selected(active_idx)
        self._provider_dropdown.connect("notify::selected", self._on_provider_changed)
        provider_box.append(self._provider_dropdown)

        self._provider_status = Gtk.Label(label="")
        self._provider_status.add_css_class("dim-label")
        provider_box.append(self._provider_status)
        body.append(provider_box)

        # --- Hold to Talk ---
        self._ptt_button = Gtk.Button(label="Hold to Talk   (Space)")
        self._ptt_button.add_css_class("suggested-action")
        self._ptt_button.add_css_class("pill")
        self._ptt_button.set_size_request(-1, 56)
        body.append(self._ptt_button)

        gesture = Gtk.GestureClick()
        gesture.set_button(0)
        gesture.connect("pressed", lambda *_: self._ptt_begin())
        gesture.connect("released", lambda *_: self._ptt_end())
        gesture.connect("stopped", lambda *_: self._ptt_end())
        self._ptt_button.add_controller(gesture)

        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect("key-pressed", self._on_key_pressed)
        key_ctrl.connect("key-released", self._on_key_released)
        self.add_controller(key_ctrl)

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())
        toolbar.set_content(body)
        self.set_content(toolbar)

        self.connect("close-request", self._on_close_request)

    # ------------------------------------------------------------------
    # Thread-safe setters
    # ------------------------------------------------------------------

    def push_status(self, status: str, detail: Optional[str] = None) -> None:
        GLib.idle_add(self._apply_status, status, detail)

    def push_transcript(self, text: str) -> None:
        GLib.idle_add(self._apply_transcript, text)

    def push_action(self, text: str) -> None:
        GLib.idle_add(self._apply_action, text)

    # ------------------------------------------------------------------
    # Main-thread handlers
    # ------------------------------------------------------------------

    def _apply_status(self, status: str, detail: Optional[str]) -> bool:
        self._status_label.set_text(status)
        if detail is not None:
            self._detail_label.set_text(detail)
        return False

    def _apply_transcript(self, text: str) -> bool:
        self._transcript_row.set_subtitle(text or "---")
        return False

    def _apply_action(self, text: str) -> bool:
        self._action_row.set_subtitle(text or "---")
        return False

    # ------------------------------------------------------------------
    # PTT
    # ------------------------------------------------------------------

    def _ptt_begin(self) -> None:
        if self._ptt_event.is_set():
            return
        self._ptt_event.set()
        self._ptt_button.set_label("Listening...")

    def _ptt_end(self) -> None:
        if not self._ptt_event.is_set():
            return
        self._ptt_event.clear()
        self._ptt_button.set_label("Hold to Talk   (Space)")

    def _on_key_pressed(self, _ctrl, keyval, _keycode, _state) -> bool:
        if keyval == Gdk.KEY_space:
            self._ptt_begin()
            return True
        return False

    def _on_key_released(self, _ctrl, keyval, _keycode, _state) -> bool:
        if keyval == Gdk.KEY_space:
            self._ptt_end()
            return True
        return False

    def _on_close_request(self, *_args) -> bool:
        if self._on_quit is not None:
            self._on_quit()
        return False

    def _on_provider_changed(self, dropdown, _pspec) -> None:
        idx = dropdown.get_selected()
        if idx < 0 or idx >= len(self._provider_names):
            return
        new_provider = self._provider_names[idx]
        success = provider_mgr.set_provider(new_provider)
        if success:
            self._provider_status.set_text("Active")
            logger.info("UI: switched LLM provider to %s", new_provider)
        else:
            self._provider_status.set_text("Unavailable")
            logger.warning("UI: failed to switch to %s (missing API key?)", new_provider)
            try:
                actual_idx = self._provider_names.index(provider_mgr.active_provider)
                dropdown.set_selected(actual_idx)
            except ValueError:
                pass


class AssistantApplication(Adw.Application):
    def __init__(
        self,
        *,
        ptt_event: threading.Event,
        on_window_ready: Callable[[AssistantWindow], None],
        on_quit: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(application_id=APP_ID)
        self._ptt_event = ptt_event
        self._on_window_ready = on_window_ready
        self._on_quit = on_quit
        self.connect("activate", self._do_activate)

    def _do_activate(self, _app: Adw.Application) -> None:
        win = AssistantWindow(
            self,
            ptt_event=self._ptt_event,
            on_quit=self._on_quit,
        )
        win.present()
        self._on_window_ready(win)
