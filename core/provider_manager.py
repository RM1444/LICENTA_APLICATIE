"""Runtime LLM provider manager.

Allows switching the active LLM provider (for both intent parsing and chat)
at runtime without restarting the daemon. Thread-safe.

Supported providers:
    - "anthropic" (Claude) — needs ANTHROPIC_API_KEY
    - "gemini"   (Google)  — needs GEMINI_API_KEY
    - "groq"     (Groq)    — needs GROQ_API_KEY
    - "ollama"   (Local)   — needs local Ollama server running

Usage:
    from core.provider_manager import provider_mgr

    # Get current provider
    current = provider_mgr.active_provider

    # Switch provider at runtime
    provider_mgr.set_provider("gemini")

    # List available (configured) providers
    available = provider_mgr.available_providers
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from core.config import get_config

logger = logging.getLogger(__name__)

# All supported LLM providers
SUPPORTED_PROVIDERS = ("anthropic", "gemini", "groq", "ollama")


class ProviderManager:
    """Thread-safe runtime provider selector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[str] = None
        self._listeners: list[callable] = []

    def initialize(self) -> None:
        """Load the initial provider from config.yaml."""
        cfg = get_config().get("brain", {})
        initial = str(cfg.get("provider", "anthropic")).lower()
        if initial not in SUPPORTED_PROVIDERS:
            logger.warning("Unknown provider %r in config; defaulting to anthropic.", initial)
            initial = "anthropic"
        with self._lock:
            self._active = initial
        logger.info("Provider manager initialized: active=%s", initial)

    @property
    def active_provider(self) -> str:
        """Return the currently active provider name."""
        with self._lock:
            if self._active is None:
                self.initialize()
            return self._active  # type: ignore[return-value]

    def set_provider(self, provider: str) -> bool:
        """Switch the active LLM provider at runtime.

        Returns True if the switch was successful, False if the provider is
        unsupported or not configured (missing API key / server).
        """
        provider = provider.lower().strip()
        if provider not in SUPPORTED_PROVIDERS:
            logger.warning("Cannot switch to unsupported provider: %s", provider)
            return False

        if not self._is_provider_available(provider):
            logger.warning("Provider %s is not available (missing key or server).", provider)
            return False

        with self._lock:
            old = self._active
            self._active = provider

        logger.info("LLM provider switched: %s -> %s", old, provider)
        self._notify_listeners(provider)
        return True

    @property
    def available_providers(self) -> list[str]:
        """Return list of providers that are currently usable (keys set, etc)."""
        return [p for p in SUPPORTED_PROVIDERS if self._is_provider_available(p)]

    def add_listener(self, callback: callable) -> None:
        """Register a callback invoked on provider change: callback(new_provider)."""
        self._listeners.append(callback)

    def _notify_listeners(self, new_provider: str) -> None:
        for cb in self._listeners:
            try:
                cb(new_provider)
            except Exception:  # noqa: BLE001
                logger.exception("Provider change listener failed")

    @staticmethod
    def _is_provider_available(provider: str) -> bool:
        """Check if a provider can be used right now."""
        if provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if provider == "gemini":
            return bool(os.environ.get("GEMINI_API_KEY"))
        if provider == "groq":
            return bool(os.environ.get("GROQ_API_KEY"))
        if provider == "ollama":
            # Ollama is local — assume available if the package exists.
            try:
                import ollama  # type: ignore  # noqa: F401
                return True
            except ImportError:
                return False
        return False


# Module-level singleton
provider_mgr = ProviderManager()
