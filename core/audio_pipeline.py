"""Audio pipeline: record -> transcribe -> match hot word -> record command.

Two-phase architecture modeled on linux-voice-control
(https://github.com/omegaui/linux-voice-control):

Phase 1 -- Hot word detection:
    Record a short fixed-duration clip (default 2 s) -> check amplitude ->
    transcribe with local faster-whisper -> text-match against hot words.
    Loops until a hot word is found.

Phase 2 -- Command recording:
    Fire the ``on_wake`` callback (e.g. TTS "listening") -> flush any audio
    that accumulated during TTS playback -> record a longer fixed-duration
    clip (default 3 s) -> check amplitude -> yield the audio to the caller
    for full STT and command matching.

No neural wake-word model (openWakeWord) is used.  Hot word detection is done
by transcribing short audio clips with local Whisper and matching the text,
which eliminates false-positive and partial-trigger problems.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from core.config import get_config, resolve_path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2          # bytes per sample (16-bit PCM)
CHANNELS = 1


# ------------------------------------------------------------------
# Public data class
# ------------------------------------------------------------------

@dataclass
class CapturedUtterance:
    """A single utterance captured after hot-word detection."""
    pcm16: bytes
    audio_f32: np.ndarray
    sample_rate: int
    duration_s: float
    source: str = "wake"   # "wake" | "ptt"


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class AudioPipeline:
    """Whisper-based hot word detection + fixed-duration command recording."""

    def __init__(self) -> None:
        cfg = get_config()
        audio = cfg["audio"]
        wake = cfg["wakeword"]

        self._chunk_size: int = int(audio.get("chunk_size", 1024))
        self._hot_word_dur: float = float(audio.get("hot_word_duration_s", 2.0))
        self._command_dur: float = float(audio.get("record_duration_s", 3.0))
        self._speech_thr: int = int(audio.get("speech_threshold", 4000))
        self._max_ptt_dur: float = float(audio.get("max_command_duration_s", 10))

        # Build the hot-word set (cleaned, lower-case, alpha+digit+space)
        raw = wake.get("hot_words", ["hey jarvis", "jarvis"])
        if isinstance(raw, str):
            raw = [raw]
        self._hot_words: set[str] = set()
        for hw in raw:
            c = _clean(hw)
            if c:
                self._hot_words.add(c)

        # Runtime state -- set by start() / stop()
        self._pyaudio = None
        self._stream = None
        self._whisper = None
        self._stop_event = threading.Event()

    # ---- lifecycle ------------------------------------------------

    def start(self) -> None:
        import pyaudio  # type: ignore

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self._chunk_size,
        )
        self._whisper = _load_whisper()
        logger.info(
            "AudioPipeline started  sr=%d  chunk=%d  hot_words=%s  threshold=%d",
            SAMPLE_RATE, self._chunk_size, self._hot_words, self._speech_thr,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing PyAudio stream")
            self._stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:  # noqa: BLE001
                logger.exception("Error terminating PyAudio")
            self._pyaudio = None
        logger.info("AudioPipeline stopped.")

    def __enter__(self) -> "AudioPipeline":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ---- main generator -------------------------------------------

    def listen_for_utterances(
        self,
        ptt_event: Optional[threading.Event] = None,
        on_wake: Optional[Callable[[], None]] = None,
    ) -> Iterator[CapturedUtterance]:
        """Blocking generator yielding one ``CapturedUtterance`` per command.

        Parameters
        ----------
        ptt_event:
            Push-to-Talk flag.  While set, audio is captured directly and
            yielded without hot-word gating.
        on_wake:
            Called immediately after a hot word is recognised, *before* the
            command is recorded.  Use this to play a "listening" TTS prompt
            so the user knows the assistant is ready for their command.
        """
        assert self._stream is not None, "Call start() first."

        while not self._stop_event.is_set():
            # ---- PTT shortcut (bypasses hot word gate) ----
            if ptt_event is not None and ptt_event.is_set():
                utt = self._capture_ptt(ptt_event)
                if utt is not None:
                    yield utt
                continue

            # ---- Phase 1: hot word detection ----
            frames = self._record_fixed(self._hot_word_dur)
            if self._stop_event.is_set():
                break
            if not _has_speech(frames, self._speech_thr):
                continue

            text = self._transcribe_local(frames)
            if not text:
                continue

            cleaned = _clean(text)
            if not self._matches_hot_word(cleaned):
                logger.debug("No hot word in: %r", cleaned)
                continue

            logger.info("Wake word detected: %r", cleaned)

            # ---- Callback (TTS "listening") ----
            if on_wake is not None:
                try:
                    on_wake()
                except Exception:  # noqa: BLE001
                    logger.exception("on_wake callback failed")

            # Flush any audio that accumulated during the TTS callback
            # (prevents the TTS echo from contaminating the command clip)
            self._flush()

            # ---- Phase 2: record command ----
            cmd_frames = self._record_fixed(self._command_dur)
            if self._stop_event.is_set():
                break
            if not _has_speech(cmd_frames, self._speech_thr):
                logger.info("Command clip silent -- back to listening.")
                continue

            pcm = b"".join(cmd_frames)
            f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(pcm) / float(SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
            logger.info("Captured command: %.2fs", dur)

            yield CapturedUtterance(
                pcm16=pcm, audio_f32=f32,
                sample_rate=SAMPLE_RATE, duration_s=dur,
                source="wake",
            )

    # ---- single-shot capture (enrolment, approval) ----------------

    def capture_once(
        self,
        *,
        timeout_s: Optional[float] = None,
        require_speech: bool = False,
        min_speech_frames: int = 0,           # kept for API compat
    ) -> Optional[CapturedUtterance]:
        """Record a single fixed-duration clip, bypassing hot-word detection.

        Used for enrollment recordings, owner-approval windows, and any
        other context where the caller drives capture manually.
        """
        assert self._stream is not None, "Call start() first."
        dur = timeout_s if timeout_s is not None else self._command_dur
        frames = self._record_fixed(dur)

        if require_speech and not _has_speech(frames, self._speech_thr):
            return None
        if not frames:
            return None

        pcm = b"".join(frames)
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        d = len(pcm) / float(SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        return CapturedUtterance(
            pcm16=pcm, audio_f32=f32,
            sample_rate=SAMPLE_RATE, duration_s=d,
        )

    # ---- internal helpers -----------------------------------------

    def _record_fixed(self, duration_s: float) -> list[bytes]:
        """Record *duration_s* seconds of audio and return raw PCM frames.

        This mirrors linux-voice-control's fixed-duration recording loop::

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        """
        assert self._stream is not None
        n_chunks = int(SAMPLE_RATE / self._chunk_size * duration_s)
        frames: list[bytes] = []
        for _ in range(n_chunks):
            if self._stop_event.is_set():
                break
            try:
                frames.append(
                    self._stream.read(self._chunk_size,
                                      exception_on_overflow=False),
                )
            except Exception:  # noqa: BLE001
                logger.exception("PyAudio read failed")
                time.sleep(0.02)
        return frames

    def _flush(self) -> None:
        """Discard any frames sitting in the PyAudio input buffer.

        Called after TTS playback to prevent the echo from being captured
        as part of the next recording.
        """
        if self._stream is None:
            return
        try:
            avail = self._stream.get_read_available()
            if avail > 0:
                self._stream.read(avail, exception_on_overflow=False)
        except Exception:  # noqa: BLE001
            pass

    def _transcribe_local(self, frames: list[bytes]) -> str:
        """Transcribe *frames* with the local faster-whisper model.

        Used exclusively for hot word detection.  The command clip is
        transcribed later by the caller using the configured STT provider
        (which might be Groq cloud or local Whisper).
        """
        if self._whisper is None:
            return ""
        pcm = b"".join(frames)
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segs, _ = self._whisper.transcribe(
                f32, language="en", beam_size=5, vad_filter=True,
            )
            text = "".join(s.text for s in segs).strip()
            if text:
                logger.debug("Hot word clip transcription: %r", text)
            return text
        except Exception:  # noqa: BLE001
            logger.exception("Local Whisper transcription failed")
            return ""

    def _matches_hot_word(self, cleaned: str) -> bool:
        """Return True if *cleaned* text matches any configured hot word.

        Checks both exact match and substring containment so that
        "hey jarvis how are you" still triggers on "hey jarvis".
        """
        if cleaned in self._hot_words:
            return True
        return any(hw in cleaned for hw in self._hot_words)

    def _capture_ptt(
        self, ptt_event: threading.Event,
    ) -> Optional[CapturedUtterance]:
        """Record audio while the Push-to-Talk flag is held.

        Stops when the user releases the button or the hard duration cap
        is reached.  Clips shorter than 150 ms (accidental taps) are
        discarded.
        """
        assert self._stream is not None
        frames: list[bytes] = []
        t0 = time.monotonic()
        while ptt_event.is_set() and not self._stop_event.is_set():
            try:
                frames.append(
                    self._stream.read(self._chunk_size,
                                      exception_on_overflow=False),
                )
            except Exception:  # noqa: BLE001
                logger.exception("PyAudio read failed during PTT")
                time.sleep(0.02)
            if (time.monotonic() - t0) >= self._max_ptt_dur:
                break
        if not frames:
            return None
        pcm = b"".join(frames)
        dur = len(pcm) / float(SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        if dur < 0.15:
            logger.info("PTT tap too short (%.2fs), ignoring.", dur)
            return None
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        logger.info("Captured PTT utterance: %.2fs", dur)
        return CapturedUtterance(
            pcm16=pcm, audio_f32=f32,
            sample_rate=SAMPLE_RATE, duration_s=dur, source="ptt",
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _clean(text: str) -> str:
    """Lower-case, keep only alpha / digit / space."""
    return "".join(
        ch for ch in text if ch.isalpha() or ch.isdigit() or ch == " "
    ).lower().strip()


def _has_speech(frames: list[bytes], threshold: int) -> bool:
    """Return True if the clip's peak amplitude exceeds *threshold*.

    This mirrors linux-voice-control's silence gate::

        chunk_array = trim(chunk_array)
        if len(chunk_array) == 0:
            continue
        elif max(chunk_array) < SPEECH_THRESHOLD:
            continue
    """
    if not frames:
        return False
    pcm = b"".join(frames)
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return False
    peak = int(np.max(np.abs(samples)))
    if peak < threshold:
        logger.debug("No speech (peak=%d < threshold=%d)", peak, threshold)
        return False
    return True


def _load_whisper():
    """Return the local faster-whisper model, sharing it with core.brain.

    Tries to reuse the model already loaded by ``core.brain`` to avoid
    doubling memory usage.  Falls back to loading independently if
    brain's model is not yet available.
    """
    try:
        from core.brain import get_local_whisper_model
        model = get_local_whisper_model()
        if model is not None:
            logger.info("Sharing Whisper model with core.brain.")
            return model
    except ImportError:
        logger.debug("core.brain not available; loading Whisper independently.")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to get Whisper model from core.brain.")

    # Standalone fallback -- load our own instance
    try:
        from faster_whisper import WhisperModel  # type: ignore

        cfg = get_config()["stt"]
        model_size = cfg.get("model_size", "base.en")
        device = cfg.get("device", "cpu")
        compute_type = cfg.get("compute_type", "int8")
        model_path = str(resolve_path(
            cfg.get("model_path", "assets/models/whisper"),
        ))
        logger.info("Loading Whisper %s independently for hot word detection.",
                     model_size)
        return WhisperModel(
            model_size, device=device, compute_type=compute_type,
            download_root=model_path,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Cannot load any Whisper model -- hot word detection disabled. "
            "Push-to-Talk still works."
        )
        return None
