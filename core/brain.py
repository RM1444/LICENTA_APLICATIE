"""The Brain: faster-whisper STT + multi-provider LLM intent parsing.

Two responsibilities:
1. `transcribe(audio)` - local speech-to-text via faster-whisper.
2. `parse_intent(text)` - extract a strict JSON intent via an LLM (Claude,
   Gemini, Groq, or Ollama) with automatic fallback.

Intent output conforms to the schema in resources/references/intent_system_prompt.txt.

Message routing:
    - **Default messages** are resolved by hardcoded patterns in
      `core/default_responses` without calling any LLM at all.
    - **Advanced messages** go through the active LLM provider (switchable at
      runtime via `core/provider_manager`).

A fallback heuristic parser exists for cases where all LLMs are unavailable,
so the assistant still returns structured intents during development.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core import groq_cooldown
from core.config import PROJECT_ROOT, get_config, resolve_path
from core.default_responses import try_default_chat, try_default_intent
from core.provider_manager import provider_mgr

logger = logging.getLogger(__name__)


def _is_groq_rate_limit(exc: BaseException) -> bool:
    """Detect a Groq SDK RateLimitError without importing groq at module load."""
    try:
        from groq import RateLimitError  # type: ignore
    except ImportError:
        return False
    return isinstance(exc, RateLimitError)


VALID_ACTIONS = frozenset({
    "open_app", "close_app",
    "install_package", "remove_package",
    "search_web",
    "get_time", "set_alarm",
    "get_calendar", "add_calendar_event",
    "get_weather",
    "system_info",
    "modify_file", "run_command",
    "volume_control", "brightness_control",
    "screenshot", "lock_screen",
    "shutdown", "restart",
    "switch_llm",
    "unknown",
})

SUDO_ACTIONS = frozenset({
    "install_package", "remove_package",
    "modify_file", "run_command",
    "shutdown", "restart",
})


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transcription:
    text: str
    confidence: float
    language: str


@dataclass(frozen=True)
class Intent:
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_sudo: bool = False
    confidence: float = 0.0

    @property
    def is_unknown(self) -> bool:
        return self.action == "unknown"


# ---------------------------------------------------------------------------
# Conversational fallback via Claude
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_PROMPT = (
    "You are a helpful Fedora Linux voice assistant. Reply in one or two short "
    "sentences suitable for text-to-speech playback. No markdown, no lists, "
    "no code fences - just plain spoken prose. If you don't know, say so briefly."
)


def chat_response(text: str) -> str:
    """Conversational reply for queries that don't map to a tool action.

    Routing order:
        1. Default (hardcoded) responses — zero latency, no API cost.
        2. Active LLM provider (runtime-switchable via provider_manager).
        3. Fallback through remaining providers.
        4. Canned "unreachable" message (daemon never goes silent).
    """
    text = (text or "").strip()
    if not text:
        return "I did not hear anything."

    # --- Layer 1: Default responses (no AI needed) ---
    default = try_default_chat(text)
    if default is not None:
        return default

    # --- Layer 2: Active LLM provider ---
    cfg = get_config().get("brain", {})
    active = provider_mgr.active_provider

    # Try the active provider first
    reply = _chat_by_provider(active, text, cfg)
    if reply is not None:
        return reply

    # --- Layer 3: Fallback through other providers ---
    fallback_order = [p for p in ("anthropic", "gemini", "groq", "ollama") if p != active]
    for provider in fallback_order:
        reply = _chat_by_provider(provider, text, cfg)
        if reply is not None:
            return reply

    return "I could not reach the language model just now."


def _chat_by_provider(provider: str, text: str, cfg: dict[str, Any]) -> Optional[str]:
    """Dispatch a chat request to the specified provider."""
    if provider == "anthropic":
        return _chat_anthropic(text, cfg)
    elif provider == "gemini":
        return _chat_gemini(text, cfg)
    elif provider == "groq":
        return _chat_groq(text, cfg)
    elif provider == "ollama":
        return _chat_ollama(text, cfg)
    return None


def _chat_anthropic(text: str, cfg: dict[str, Any]) -> Optional[str]:
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError:
        return None

    model_name = cfg.get("anthropic_model", "claude-haiku-4-5-20251001")
    try:
        start = time.monotonic()
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=256,
            system=_CHAT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            temperature=0.3,
        )
        logger.info("Claude chat completed in %.2fs", time.monotonic() - start)
    except Exception:  # noqa: BLE001
        logger.exception("Claude chat failed.")
        return None

    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        text_part = getattr(block, "text", None)
        if text_part:
            parts.append(text_part)
    reply = " ".join(parts).strip()
    return reply or None


def _chat_groq(text: str, cfg: dict[str, Any]) -> Optional[str]:
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        return None

    model_name = cfg.get("groq_model", "llama-3.1-8b-instant")
    if groq_cooldown.should_skip(model_name):
        return None
    try:
        start = time.monotonic()
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        logger.info("Groq chat completed in %.2fs", time.monotonic() - start)
    except Exception as exc:  # noqa: BLE001
        if _is_groq_rate_limit(exc):
            groq_cooldown.record_rate_limit(model_name, exc)
        else:
            logger.exception("Groq chat failed.")
        return None

    if not response.choices:
        return None
    content = response.choices[0].message.content or ""
    return content.strip() or None


def _chat_gemini(text: str, cfg: dict[str, Any]) -> Optional[str]:
    """Chat via Google Gemini (needs GEMINI_API_KEY)."""
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning("google-generativeai SDK not installed; Gemini chat disabled.")
        return None

    model_name = cfg.get("gemini_model", "gemini-2.0-flash")
    try:
        start = time.monotonic()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_CHAT_SYSTEM_PROMPT,
        )
        response = model.generate_content(
            text,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=256,
                temperature=0.3,
            ),
        )
        logger.info("Gemini chat completed in %.2fs", time.monotonic() - start)
    except Exception:  # noqa: BLE001
        logger.exception("Gemini chat failed.")
        return None

    reply = getattr(response, "text", None) or ""
    return reply.strip() or None


def _chat_ollama(text: str, cfg: dict[str, Any]) -> Optional[str]:
    """Chat via local Ollama server."""
    model_name = cfg.get("ollama_model", "llama3:8b")
    try:
        import ollama  # type: ignore
    except ImportError:
        return None

    client = ollama.Client(host=cfg.get("ollama_base_url", "http://127.0.0.1:11434"))
    try:
        start = time.monotonic()
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            options={"temperature": 0.3},
        )
        logger.info("Ollama chat completed in %.2fs", time.monotonic() - start)
    except Exception:  # noqa: BLE001
        logger.exception("Ollama chat failed.")
        return None

    content = ""
    if isinstance(response, dict):
        content = response.get("message", {}).get("content", "")
    return content.strip() or None


# ---------------------------------------------------------------------------
# STT - faster-whisper
# ---------------------------------------------------------------------------

_whisper_lock = threading.Lock()
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        cfg = get_config()["stt"]
        from faster_whisper import WhisperModel  # type: ignore

        logger.info("Loading faster-whisper model %s (%s/%s)",
                    cfg["model_size"], cfg["device"], cfg["compute_type"])
        _whisper_model = WhisperModel(
            cfg["model_size"],
            device=cfg["device"],
            compute_type=cfg["compute_type"],
            download_root=str(resolve_path(cfg["model_path"])),
        )
        return _whisper_model


def get_local_whisper_model():
    """Return the shared local faster-whisper model instance (lazy-loaded).

    Used by ``core.audio_pipeline`` for hot word detection so both modules
    share a single model and avoid doubling memory usage.
    """
    return _load_whisper()


def transcribe(
    audio: np.ndarray | bytes,
    sample_rate: int = 16_000,
    *,
    prompt: Optional[str] = None,
) -> Transcription:
    """Transcribe a mono 16 kHz audio clip. Provider-agnostic entry point.

    `prompt` biases the decoder toward expected vocabulary - used in the
    approval window so short utterances like "approve" or "deny" don't
    get hallucinated into "Thank you." by Whisper.
    """
    cfg = get_config()["stt"]

    if isinstance(audio, (bytes, bytearray, memoryview)):
        samples = np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.asarray(audio, dtype=np.float32)

    provider = str(cfg.get("provider", "local")).lower()
    if provider == "groq":
        result = _transcribe_groq(samples, sample_rate, cfg, prompt=prompt)
        if result is not None:
            return result
        logger.warning("Groq STT unavailable; falling back to local faster-whisper.")

    return _transcribe_local(samples, cfg, prompt=prompt)


def _transcribe_local(samples: np.ndarray, cfg: dict[str, Any], *, prompt: Optional[str] = None) -> Transcription:
    model = _load_whisper()
    segments, info = model.transcribe(
        samples,
        language=cfg.get("language", "en"),
        beam_size=int(cfg.get("beam_size", 5)),
        vad_filter=bool(cfg.get("vad_filter", True)),
        initial_prompt=prompt,
    )
    segments = list(segments)
    text = "".join(seg.text for seg in segments).strip()

    if segments:
        avg_logprob = float(np.mean([
            getattr(s, "avg_logprob", -1.0) for s in segments
        ]))
        confidence = float(np.exp(avg_logprob))
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.0

    logger.info("STT[local]: %r (confidence=%.3f lang=%s)", text, confidence, info.language)
    return Transcription(text=text, confidence=confidence, language=info.language)


def _transcribe_groq(samples: np.ndarray, sample_rate: int, cfg: dict[str, Any], *, prompt: Optional[str] = None) -> Optional[Transcription]:
    """Send the clip to Groq's Whisper endpoint. Returns None on any error so
    the caller can fall back to the local model."""
    import io
    import os
    import wave

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY unset; Groq STT disabled.")
        return None

    try:
        from groq import Groq  # type: ignore
    except ImportError:
        logger.warning("groq SDK not installed; Groq STT disabled.")
        return None

    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    buf.seek(0)

    model_name = cfg.get("groq_model", "whisper-large-v3-turbo")
    if groq_cooldown.should_skip(model_name):
        logger.debug("Groq STT skipped: model %s in cooldown (%.0fs left).",
                     model_name, groq_cooldown.seconds_remaining(model_name))
        return None
    try:
        start = time.monotonic()
        client = Groq(api_key=api_key)
        stt_kwargs = dict(
            file=("utterance.wav", buf.read()),
            model=model_name,
            response_format="verbose_json",
            language=cfg.get("language", "en"),
            temperature=0.0,
        )
        if prompt:
            stt_kwargs["prompt"] = prompt
        response = client.audio.transcriptions.create(**stt_kwargs)
        logger.info("Groq STT completed in %.2fs", time.monotonic() - start)
    except Exception as exc:  # noqa: BLE001
        if _is_groq_rate_limit(exc):
            groq_cooldown.record_rate_limit(model_name, exc)
        else:
            logger.exception("Groq STT request failed.")
        return None

    text = str(getattr(response, "text", "") or "").strip()
    language = str(getattr(response, "language", cfg.get("language", "en")) or "en")

    # Groq returns per-segment avg_logprob when response_format="verbose_json".
    segments = getattr(response, "segments", None) or []
    if segments:
        try:
            logprobs = [float(s.get("avg_logprob", -1.0)) for s in segments]
            confidence = float(np.exp(float(np.mean(logprobs))))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.85 if text else 0.0
    else:
        confidence = 0.85 if text else 0.0

    logger.info("STT[groq]: %r (confidence=%.3f lang=%s)", text, confidence, language)
    return Transcription(text=text, confidence=confidence, language=language)


# ---------------------------------------------------------------------------
# Intent parsing - Ollama + fallback heuristic
# ---------------------------------------------------------------------------

_WAKE_STRIP_RE = re.compile(
    r"^(hey|ok|hi|yo)[\s,]+(jarvis|mycroft|rhasspy|assistant)[\s,.:;!?-]*",
    re.I,
)


def _strip_wake_phrase(text: str) -> str:
    return _WAKE_STRIP_RE.sub("", text, count=1).strip()


def parse_intent(text: str) -> Intent:
    """Return a single structured Intent (first action from a compound command).

    Convenience wrapper around `parse_intents()` for callers that only
    need one intent.
    """
    intents = parse_intents(text)
    return intents[0]


def parse_intents(text: str) -> list[Intent]:
    """Return one or more structured Intents for the transcript.

    Compound commands like "open Firefox and launch YouTube" produce
    multiple intents that the daemon executes in sequence.

    Routing:
        1. **Default intents** — resolved by hardcoded patterns with zero LLM
           cost (time, screenshot, open/close app, etc.).
        2. **Active LLM provider** — runtime-switchable via provider_manager.
        3. **Fallback** — tries remaining providers in order, then heuristic.

    On any LLM failure we fall through to the next available provider, and
    finally to the regex heuristic so the assistant still returns a best-effort
    structured intent even when every cloud backend is down.
    """
    text = (text or "").strip()
    if not text:
        return [Intent(action="unknown", confidence=0.0)]

    cleaned = _strip_wake_phrase(text)

    # --- Layer 1: Default intent shortcuts (no AI cost) ---
    default_intent = try_default_intent(cleaned)
    if default_intent is not None:
        return [default_intent]

    # --- Layer 2: Active LLM provider (runtime-switchable) ---
    cfg = get_config().get("brain", {})
    active = provider_mgr.active_provider

    intents = _parse_intent_by_provider(active, cleaned, cfg)
    if intents is not None:
        return intents
    logger.warning("%s intent parser unavailable; trying fallbacks.", active)

    # --- Layer 3: Fallback through other providers ---
    fallback_order = [p for p in ("anthropic", "gemini", "groq", "ollama") if p != active]
    for provider in fallback_order:
        intents = _parse_intent_by_provider(provider, cleaned, cfg)
        if intents is not None:
            return intents

    # --- Layer 4: Regex heuristic (always works) ---
    return [_parse_intent_heuristic(cleaned)]


def _parse_intent_by_provider(provider: str, text: str, cfg: dict[str, Any]) -> Optional[list[Intent]]:
    """Dispatch intent parsing to the specified provider."""
    if provider == "anthropic":
        return _parse_intent_anthropic(text, cfg)
    elif provider == "gemini":
        return _parse_intent_gemini(text, cfg)
    elif provider == "groq":
        return _parse_intent_groq(text, cfg)
    elif provider == "ollama":
        return _parse_intent_ollama(text)
    return None


def _parse_intent_anthropic(text: str, cfg: dict[str, Any]) -> Optional[list[Intent]]:
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY unset; Claude intent parser disabled.")
        return None
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError:
        logger.warning("anthropic SDK not installed; Claude intent parser disabled.")
        return None

    model_name = cfg.get("anthropic_model", "claude-haiku-4-5-20251001")
    try:
        start = time.monotonic()
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=256,
            system=_system_prompt(),
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
        )
        logger.info("Claude intent completed in %.2fs", time.monotonic() - start)
    except Exception:  # noqa: BLE001
        logger.exception("Claude intent request failed.")
        return None

    content_blocks = getattr(response, "content", None) or []
    raw = ""
    for block in content_blocks:
        text_part = getattr(block, "text", None)
        if text_part:
            raw += text_part
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("Claude returned non-JSON content: %r", raw[:200])
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Claude returned malformed JSON: %r", raw[:200])
            return None

    return _normalise_intent_payload(payload)


def _parse_intent_gemini(text: str, cfg: dict[str, Any]) -> Optional[list[Intent]]:
    """Parse intent via Google Gemini (needs GEMINI_API_KEY)."""
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY unset; Gemini intent parser disabled.")
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning("google-generativeai SDK not installed; Gemini intent parser disabled.")
        return None

    model_name = cfg.get("gemini_model", "gemini-2.0-flash")
    try:
        start = time.monotonic()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_system_prompt(),
        )
        response = model.generate_content(
            text,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=256,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        logger.info("Gemini intent completed in %.2fs", time.monotonic() - start)
    except Exception:  # noqa: BLE001
        logger.exception("Gemini intent request failed.")
        return None

    raw = getattr(response, "text", None) or ""
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("Gemini returned non-JSON content: %r", raw[:200])
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Gemini returned malformed JSON: %r", raw[:200])
            return None

    return _normalise_intent_payload(payload)


def _parse_intent_groq(text: str, cfg: dict[str, Any]) -> Optional[list[Intent]]:
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY unset; Groq intent parser disabled.")
        return None
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        logger.warning("groq SDK not installed; Groq intent parser disabled.")
        return None

    model_name = cfg.get("groq_model", "llama-3.1-8b-instant")
    if groq_cooldown.should_skip(model_name):
        logger.debug("Groq intent skipped: model %s in cooldown (%.0fs left).",
                     model_name, groq_cooldown.seconds_remaining(model_name))
        return None
    try:
        start = time.monotonic()
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        logger.info("Groq intent chat completed in %.2fs", time.monotonic() - start)
    except Exception as exc:  # noqa: BLE001
        if _is_groq_rate_limit(exc):
            groq_cooldown.record_rate_limit(model_name, exc)
        else:
            logger.exception("Groq intent request failed.")
        return None

    content = response.choices[0].message.content if response.choices else ""
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return _normalise_intent_payload(payload)


def _system_prompt() -> str:
    path = PROJECT_ROOT / "resources" / "references" / "intent_system_prompt.txt"
    return path.read_text(encoding="utf-8")


def _parse_intent_ollama(text: str) -> Optional[list[Intent]]:
    cfg = get_config()["brain"]
    model_name = cfg.get("ollama_model", "llama3:8b")
    try:
        import ollama  # type: ignore
    except ImportError:
        logger.warning("ollama SDK not installed; using heuristic parser.")
        return None

    client = ollama.Client(host=cfg.get("ollama_base_url", "http://127.0.0.1:11434"))
    try:
        start = time.monotonic()
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": text},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        logger.info("Ollama chat completed in %.2fs", time.monotonic() - start)
    except Exception:  # noqa: BLE001
        logger.exception("Ollama request failed; using heuristic parser.")
        return None

    raw_content = response.get("message", {}).get("content", "") if isinstance(response, dict) else ""
    if not raw_content:
        return None
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        # Attempt to salvage JSON from a wrapped response.
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            logger.warning("Ollama returned non-JSON content: %r", raw_content)
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Ollama returned malformed JSON: %r", raw_content)
            return None

    return _normalise_intent_payload(payload)


def _normalise_intent_payload(payload: dict[str, Any] | list) -> list[Intent]:
    """Normalise raw LLM JSON into a list of Intent objects.

    Handles both single-action dicts and multi-action lists (compound
    commands like "open Firefox and launch YouTube").
    """
    if isinstance(payload, list):
        items = [p for p in payload if isinstance(p, dict)]
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return [Intent(action="unknown", confidence=0.0)]

    if not items:
        return [Intent(action="unknown", confidence=0.0)]

    intents: list[Intent] = []
    for item in items:
        action = str(item.get("action", "unknown")).strip().lower()
        if action not in VALID_ACTIONS:
            action = "unknown"

        parameters = item.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}

        requires_sudo_raw = item.get("requires_sudo", action in SUDO_ACTIONS)
        if isinstance(requires_sudo_raw, str):
            requires_sudo = requires_sudo_raw.lower() in {"true", "yes", "1"}
        else:
            requires_sudo = bool(requires_sudo_raw)

        # Enforce the canonical sudo mapping.
        requires_sudo = requires_sudo or action in SUDO_ACTIONS

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        intents.append(Intent(
            action=action,
            parameters=parameters,
            requires_sudo=requires_sudo,
            confidence=confidence,
        ))
    return intents


# ---------------------------------------------------------------------------
# Heuristic fallback - keeps the daemon useful even when Ollama is missing.
# ---------------------------------------------------------------------------

_HEURISTIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(install|add)\b.*\b(app|package|program)?\b", re.I), "install_package"),
    (re.compile(r"\b(remove|uninstall|delete)\b.*\b(app|package|program)?\b", re.I), "remove_package"),
    (re.compile(r"\b(open|launch|start)\b", re.I), "open_app"),
    (re.compile(r"\b(close|kill|quit)\b", re.I), "close_app"),
    (re.compile(r"\bshutdown\b|\bpower off\b|\bturn off\b", re.I), "shutdown"),
    (re.compile(r"\brestart\b|\breboot\b", re.I), "restart"),
    (re.compile(r"\block (the )?screen\b", re.I), "lock_screen"),
    (re.compile(r"\b(take (a )?screenshot|capture screen)\b", re.I), "screenshot"),
    (re.compile(r"\b(what('| i)?s )?the time\b|\bwhat time is it\b", re.I), "get_time"),
    (re.compile(r"\bweather\b", re.I), "get_weather"),
    (re.compile(r"\b(volume|mute|louder|quieter)\b", re.I), "volume_control"),
    (re.compile(r"\b(brightness|dim|brighter)\b", re.I), "brightness_control"),
    (re.compile(r"\bset (an )?alarm\b", re.I), "set_alarm"),
    (re.compile(r"\bcalendar\b", re.I), "get_calendar"),
    (re.compile(r"\bsearch (the web|online|for)\b", re.I), "search_web"),
)


def _parse_intent_heuristic(text: str) -> Intent:
    for pattern, action in _HEURISTIC_PATTERNS:
        if pattern.search(text):
            parameters = _extract_heuristic_parameters(action, text)
            return Intent(
                action=action,
                parameters=parameters,
                requires_sudo=action in SUDO_ACTIONS,
                confidence=0.55,  # heuristic confidence is intentionally modest
            )
    return Intent(action="unknown", parameters={"raw": text}, confidence=0.0)


def _extract_heuristic_parameters(action: str, text: str) -> dict[str, Any]:
    def _clean(token: str) -> str:
        # Whisper leaves trailing punctuation on the last word ("settings.").
        return token.strip(".,!?;:'\"").lower()

    if action in {"install_package", "remove_package"}:
        # crude: last noun-like token
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9._+-]{1,}", text)
        return {"package": _clean(tokens[-1])} if tokens else {}
    if action in {"open_app", "close_app"}:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9._+-]{1,}", text)
        # Drop the verb itself
        tokens = [t for t in tokens if t.lower() not in {"open", "launch", "start", "close", "kill", "quit"}]
        name = " ".join(_clean(t) for t in tokens).strip()
        return {"app_name": name} if name else {}
    if action == "volume_control":
        lower = text.lower()
        if "mute" in lower:
            return {"direction": "mute"}
        if "up" in lower or "louder" in lower:
            return {"direction": "up"}
        if "down" in lower or "quieter" in lower:
            return {"direction": "down"}
        m = re.search(r"\b(\d{1,3})\b", text)
        if m:
            val = max(0, min(100, int(m.group(1))))
            return {"direction": "set", "value": val}
        return {"direction": "up"}
    if action == "brightness_control":
        lower = text.lower()
        m = re.search(r"\b(\d{1,3})\b", text)
        if m:
            val = max(0, min(100, int(m.group(1))))
            return {"direction": "set", "value": val}
        if "dim" in lower or "down" in lower:
            return {"direction": "down"}
        return {"direction": "up"}
    return {}


# ---------------------------------------------------------------------------
# Fallback response for unknown intents
# ---------------------------------------------------------------------------

def fallback_response(text: str) -> str:
    return (
        "I did not recognise that command. "
        "Try saying: open firefox, install vscode, lock the screen, or what time is it."
    )
