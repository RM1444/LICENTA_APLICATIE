"""Piper-based local neural text-to-speech.

Provides a single entry point, `speak(text)`, that synthesises text with the
configured Piper voice and plays the resulting PCM through the default audio
device. Falls back to `espeak-ng` if Piper or its voice model is missing so
the assistant can still surface error messages during a partial install.
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from core import groq_cooldown
from core.config import get_config, resolve_path

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_voice = None  # Piper voice instance
_voice_probed = False  # becomes True once we've tried (and possibly failed) to load


def _load_voice():
    """Lazy-load the Piper voice model. Missing model -> None, caller falls back.

    The probe result is cached so a missing model only logs one warning,
    not one per TTS call.
    """
    global _voice, _voice_probed
    if _voice is not None:
        return _voice
    if _voice_probed:
        return None
    with _engine_lock:
        if _voice is not None:
            return _voice
        if _voice_probed:
            return None

        cfg = get_config()["tts"]
        if cfg.get("engine") != "piper":
            _voice_probed = True
            return None

        model_path = resolve_path(cfg["voice_model"])
        if not model_path.exists():
            logger.warning("Piper voice model not found at %s; espeak-ng fallback.", model_path)
            _voice_probed = True
            return None

        try:
            from piper.voice import PiperVoice  # type: ignore
        except ImportError:
            logger.warning("piper-tts not installed; espeak-ng fallback.")
            _voice_probed = True
            return None

        logger.info("Loading Piper voice from %s", model_path)
        _voice = PiperVoice.load(str(model_path))
        _voice_probed = True
        return _voice


def synthesise(text: str) -> tuple[bytes, int]:
    """Return (pcm_s16_bytes, sample_rate) for the given utterance."""
    cfg = get_config()["tts"]
    provider = str(cfg.get("provider", "local")).lower()

    if provider == "groq":
        result = _synthesise_groq(text, cfg)
        if result is not None:
            return result
        logger.warning("Groq TTS unavailable; falling back to local Piper/espeak-ng.")

    voice = _load_voice()
    if voice is None:
        return _synthesise_espeak(text)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(text, wf)
    buf.seek(0)

    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return frames, sr


def _synthesise_groq(text: str, cfg: dict) -> Optional[tuple[bytes, int]]:
    """Call Groq's PlayAI TTS. Returns (pcm_s16_bytes, sample_rate) or None on failure."""
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY unset; Groq TTS disabled.")
        return None

    try:
        from groq import Groq  # type: ignore
    except ImportError:
        logger.warning("groq SDK not installed; Groq TTS disabled.")
        return None

    model = cfg.get("groq_model", "playai-tts")
    if groq_cooldown.should_skip(model):
        logger.debug("Groq TTS skipped: model %s in cooldown (%.0fs left).",
                     model, groq_cooldown.seconds_remaining(model))
        return None

    try:
        client = Groq(api_key=api_key)
        response = client.audio.speech.create(
            model=model,
            voice=cfg.get("groq_voice", "Celeste-PlayAI"),
            response_format="wav",
            input=text,
        )
        wav_bytes = response.read()
    except Exception as exc:  # noqa: BLE001 - we must never let TTS crash the daemon
        if _is_rate_limit(exc):
            groq_cooldown.record_rate_limit(model, exc)
        else:
            logger.exception("Groq TTS request failed.")
        return None

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return frames, sr


def _is_rate_limit(exc: BaseException) -> bool:
    """Groq SDK raises RateLimitError on 429. Detect without importing groq
    at module scope (keeps the module importable in environments without it)."""
    try:
        from groq import RateLimitError  # type: ignore
    except ImportError:
        return False
    return isinstance(exc, RateLimitError)


def _synthesise_espeak(text: str) -> tuple[bytes, int]:
    """Fallback synthesis via `espeak-ng`, which is a hard dep from bootstrap."""
    if shutil.which("espeak-ng") is None:
        raise RuntimeError("Neither Piper nor espeak-ng is available for TTS.")

    sr = 22_050
    proc = subprocess.run(
        ["espeak-ng", "--stdout", text],
        check=True,
        capture_output=True,
    )
    # espeak-ng --stdout emits a RIFF WAV, strip the header:
    buf = io.BytesIO(proc.stdout)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return frames, sr


def speak(text: str) -> None:
    """Synthesise and blockingly play `text` through the default audio device."""
    if not text:
        return
    logger.info("TTS: %s", text)
    try:
        pcm, sr = synthesise(text)
    except Exception:  # noqa: BLE001 - we must never let TTS crash the daemon
        logger.exception("TTS synthesis failed; text lost: %s", text)
        return

    try:
        _play_pcm(pcm, sr)
    except Exception:  # noqa: BLE001
        logger.exception("TTS playback failed; text lost: %s", text)


def _play_pcm(pcm: bytes, sample_rate: int) -> None:
    """Play raw 16-bit signed PCM through PyAudio. Falls back to aplay."""
    try:
        import pyaudio  # type: ignore
    except ImportError:
        _play_pcm_aplay(pcm, sample_rate)
        return

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
        )
        try:
            stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()


def _play_pcm_aplay(pcm: bytes, sample_rate: int) -> None:
    if shutil.which("aplay") is None:
        raise RuntimeError("No PyAudio and no aplay available to play TTS audio.")
    with subprocess.Popen(
        ["aplay", "-q", "-f", "S16_LE", "-c", "1", "-r", str(sample_rate)],
        stdin=subprocess.PIPE,
    ) as proc:
        assert proc.stdin is not None
        proc.stdin.write(pcm)
        proc.stdin.close()
        proc.wait()
