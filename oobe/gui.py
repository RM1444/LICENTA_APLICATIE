"""GTK4 + libadwaita OOBE.

Three screens driven by an Adw.Carousel:
  1. Welcome   - project introduction + security disclaimer.
  2. System check - live results from oobe.system_check; blocks progression.
  3. Profile + enrolment - user name, wake word, TTS voice, then 5 recordings.

Enrolment recordings run on a background thread so the UI stays responsive.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, Gtk  # noqa: E402

from core import db_manager, tts_engine
from core.config import get_config
from oobe import enrollment, system_check

logger = logging.getLogger(__name__)


APP_ID = "io.fedora.voiceassistant.Oobe"


class OobeWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app)
        self.set_title("Fedora Voice Assistant - Setup")
        self.set_default_size(720, 600)

        self._samples: list[enrollment.EnrollmentSample] = []
        self._sentences = enrollment.load_enrollment_sentences()
        self._sentence_index = 0
        self._owner_name = ""

        self._carousel = Adw.Carousel()
        self._carousel.set_allow_scroll_wheel(False)
        self._carousel.set_allow_mouse_drag(False)
        self._carousel.set_allow_long_swipes(False)
        self._carousel.set_hexpand(True)
        self._carousel.set_vexpand(True)

        self._carousel.append(self._build_welcome_page())
        self._carousel.append(self._build_checks_page())
        self._carousel.append(self._build_profile_page())
        self._carousel.append(self._build_enrolment_page())
        self._carousel.append(self._build_finish_page())

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())
        toolbar.set_content(self._carousel)
        self.set_content(toolbar)

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _build_welcome_page(self) -> Gtk.Widget:
        page = Adw.StatusPage(
            title="Welcome",
            description=(
                "A local, biometrically gated voice assistant for Fedora.\n\n"
                "Voice authentication provides convenience-level security, "
                "not cryptographic-level security."
            ),
            icon_name="audio-input-microphone-symbolic",
        )
        btn = Gtk.Button(label="Get started")
        btn.add_css_class("suggested-action")
        btn.add_css_class("pill")
        btn.connect("clicked", lambda *_: self._goto_page(1))
        page.set_child(btn)
        return page

    def _build_checks_page(self) -> Gtk.Widget:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_start=24,
                       margin_end=24, margin_top=24, margin_bottom=24)
        title = Gtk.Label(xalign=0)
        title.set_markup("<span size='x-large' weight='bold'>System checks</span>")
        outer.append(title)

        self._checks_list = Gtk.ListBox()
        self._checks_list.add_css_class("boxed-list")
        outer.append(self._checks_list)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        refresh_btn = Gtk.Button(label="Re-run checks")
        refresh_btn.connect("clicked", lambda *_: self._refresh_checks())
        controls.append(refresh_btn)

        self._continue_checks_btn = Gtk.Button(label="Continue")
        self._continue_checks_btn.add_css_class("suggested-action")
        self._continue_checks_btn.set_sensitive(False)
        self._continue_checks_btn.connect("clicked", lambda *_: self._goto_page(2))
        controls.append(self._continue_checks_btn)

        outer.append(controls)
        self._refresh_checks()
        return outer

    def _build_profile_page(self) -> Gtk.Widget:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_start=24,
                       margin_end=24, margin_top=24, margin_bottom=24)
        title = Gtk.Label(xalign=0)
        title.set_markup("<span size='x-large' weight='bold'>Create your profile</span>")
        outer.append(title)

        prefs = Adw.PreferencesGroup()
        self._name_row = Adw.EntryRow(title="Your name")
        prefs.add(self._name_row)

        cfg = get_config()
        wake_row = Adw.ComboRow(title="Wake word")
        wake_model = Gtk.StringList.new(["hey_jarvis", "alexa", "hey_mycroft"])
        wake_row.set_model(wake_model)
        self._wake_row = wake_row
        prefs.add(wake_row)

        tts_row = Adw.ComboRow(title="TTS voice")
        tts_model = Gtk.StringList.new([
            cfg["tts"].get("voice_model", "en_US-lessac-medium"),
            "en_US-amy-medium",
            "en_GB-alan-medium",
        ])
        tts_row.set_model(tts_model)
        self._tts_row = tts_row
        prefs.add(tts_row)
        outer.append(prefs)

        next_btn = Gtk.Button(label="Start voice enrolment")
        next_btn.add_css_class("suggested-action")
        next_btn.add_css_class("pill")
        next_btn.connect("clicked", self._on_profile_next)
        outer.append(next_btn)
        return outer

    def _build_enrolment_page(self) -> Gtk.Widget:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16, margin_start=24,
                       margin_end=24, margin_top=24, margin_bottom=24)
        title = Gtk.Label(xalign=0)
        title.set_markup("<span size='x-large' weight='bold'>Voice enrolment</span>")
        outer.append(title)

        self._progress = Gtk.ProgressBar(show_text=True)
        self._progress.set_fraction(0.0)
        outer.append(self._progress)

        self._sentence_label = Gtk.Label(wrap=True, xalign=0)
        self._sentence_label.add_css_class("title-3")
        outer.append(self._sentence_label)

        self._status_label = Gtk.Label(xalign=0)
        self._status_label.add_css_class("dim-label")
        outer.append(self._status_label)

        self._record_btn = Gtk.Button(label="Record")
        self._record_btn.add_css_class("suggested-action")
        self._record_btn.add_css_class("pill")
        self._record_btn.connect("clicked", lambda *_: self._start_recording())
        outer.append(self._record_btn)
        return outer

    def _build_finish_page(self) -> Gtk.Widget:
        self._finish_page = Adw.StatusPage(
            title="You are registered",
            description="Your voice profile has been saved.\nClose this window and start the daemon.",
            icon_name="emblem-ok-symbolic",
        )
        close_btn = Gtk.Button(label="Close")
        close_btn.add_css_class("pill")
        close_btn.connect("clicked", lambda *_: self.close())
        self._finish_page.set_child(close_btn)
        return self._finish_page

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _goto_page(self, idx: int) -> None:
        page = self._carousel.get_nth_page(idx)
        if page is not None:
            self._carousel.scroll_to(page, True)

    def _refresh_checks(self) -> None:
        child = self._checks_list.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._checks_list.remove(child)
            child = nxt

        results = system_check.run_all_checks()
        for r in results:
            row = Adw.ActionRow(title=r.name, subtitle=r.detail)
            icon = "emblem-ok-symbolic" if r.ok else ("dialog-error-symbolic" if r.required else "dialog-warning-symbolic")
            row.add_suffix(Gtk.Image.new_from_icon_name(icon))
            self._checks_list.append(row)

        self._continue_checks_btn.set_sensitive(system_check.all_required_passed(results))

    def _on_profile_next(self, _btn: Gtk.Button) -> None:
        name = self._name_row.get_text().strip()
        if not name:
            self._toast("Enter a name to continue.")
            return
        self._owner_name = name

        # Persist preferences to the settings table (DB is initialised on demand).
        db_manager.initialize_database()
        db_manager.set_setting("owner_name", name)
        db_manager.set_setting("wake_word", self._combo_value(self._wake_row))
        db_manager.set_setting("tts_voice", self._combo_value(self._tts_row))

        self._sentence_index = 0
        self._samples.clear()
        self._update_enrolment_ui()
        self._goto_page(3)

    def _combo_value(self, row: Adw.ComboRow) -> str:
        idx = row.get_selected()
        model = row.get_model()
        if isinstance(model, Gtk.StringList):
            return model.get_string(idx) or ""
        return ""

    def _update_enrolment_ui(self) -> None:
        total = len(self._sentences)
        current = self._sentence_index
        if current >= total:
            self._record_btn.set_sensitive(False)
            return
        self._sentence_label.set_text(self._sentences[current])
        self._progress.set_fraction(current / total)
        self._progress.set_text(f"Sentence {current + 1} of {total}")
        self._status_label.set_text("Click Record and read the sentence aloud.")
        self._record_btn.set_sensitive(True)

    def _start_recording(self) -> None:
        self._record_btn.set_sensitive(False)
        self._status_label.set_text("Listening... read the sentence now.")

        sentence = self._sentences[self._sentence_index]

        def worker() -> None:
            try:
                sample = enrollment.record_sample(sentence)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Recording failed")
                GLib.idle_add(self._on_sample_failed, str(exc))
                return
            GLib.idle_add(self._on_sample_ready, sample)

        threading.Thread(target=worker, daemon=True).start()

    def _on_sample_ready(self, sample: enrollment.EnrollmentSample) -> bool:
        self._samples.append(sample)
        self._sentence_index += 1
        if self._sentence_index >= len(self._sentences):
            self._status_label.set_text("Building master profile...")
            threading.Thread(target=self._finalise_enrolment, daemon=True).start()
        else:
            self._update_enrolment_ui()
        return False  # don't repeat idle_add

    def _on_sample_failed(self, err: str) -> bool:
        self._status_label.set_text(f"Recording failed: {err}. Click Record to retry.")
        self._record_btn.set_sensitive(True)
        return False

    def _finalise_enrolment(self) -> None:
        try:
            enrollment.enrol_owner(name=self._owner_name, samples=self._samples)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Enrolment persistence failed")
            GLib.idle_add(self._on_sample_failed, str(exc))
            return

        try:
            tts_engine.speak("Voice profile saved. You are now registered as the system owner.")
        except Exception:  # noqa: BLE001
            logger.exception("TTS confirmation failed (ignored)")

        GLib.idle_add(self._goto_page, 4)

    def _toast(self, text: str) -> None:
        dialog = Adw.MessageDialog.new(self, "Notice", text)
        dialog.add_response("ok", "OK")
        dialog.present()


class OobeApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID)
        self.connect("activate", self._on_activate)

    def _on_activate(self, app: Adw.Application) -> None:
        win = OobeWindow(app)
        win.present()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    app = OobeApplication()
    return app.run(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    sys.exit(main())
