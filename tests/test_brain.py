"""Intent parsing tests - focus on the heuristic fallback + payload normalisation.

The Ollama path is mocked so these tests stay hermetic.
"""
from __future__ import annotations

import pytest

from core import brain


def test_empty_text_returns_unknown() -> None:
    intent = brain.parse_intent("")
    assert intent.action == "unknown"
    assert intent.confidence == 0.0


def test_normalise_payload_drops_unknown_action() -> None:
    intent = brain._normalise_intent_payload(
        {"action": "hack_the_planet", "parameters": {}, "requires_sudo": False, "confidence": 0.9}
    )
    assert intent.action == "unknown"


def test_normalise_payload_forces_sudo_for_privileged_actions() -> None:
    intent = brain._normalise_intent_payload(
        {"action": "install_package", "parameters": {"package": "vim"}, "requires_sudo": False, "confidence": 0.9}
    )
    assert intent.requires_sudo is True
    assert intent.action == "install_package"
    assert intent.parameters == {"package": "vim"}


def test_normalise_payload_clamps_confidence() -> None:
    high = brain._normalise_intent_payload({"action": "get_time", "confidence": 2.5})
    low = brain._normalise_intent_payload({"action": "get_time", "confidence": -0.5})
    assert high.confidence == 1.0
    assert low.confidence == 0.0


def test_heuristic_install_package() -> None:
    intent = brain._parse_intent_heuristic("install firefox")
    assert intent.action == "install_package"
    assert intent.parameters["package"] == "firefox"
    assert intent.requires_sudo is True


def test_heuristic_open_app_strips_verb() -> None:
    intent = brain._parse_intent_heuristic("open firefox browser")
    assert intent.action == "open_app"
    assert "firefox" in intent.parameters["app_name"]
    assert "open" not in intent.parameters["app_name"]


def test_heuristic_lock_screen() -> None:
    assert brain._parse_intent_heuristic("lock the screen").action == "lock_screen"


def test_heuristic_volume_up() -> None:
    intent = brain._parse_intent_heuristic("turn the volume up")
    assert intent.action == "volume_control"
    assert intent.parameters["direction"] == "up"


def test_heuristic_volume_set_percentage() -> None:
    intent = brain._parse_intent_heuristic("set volume to 45")
    assert intent.action == "volume_control"
    assert intent.parameters["direction"] == "set"
    assert intent.parameters["value"] == 45


def test_heuristic_unknown_preserves_raw() -> None:
    intent = brain._parse_intent_heuristic("banana phone disco")
    assert intent.action == "unknown"
    assert intent.parameters["raw"] == "banana phone disco"


def test_parse_intent_uses_heuristic_when_ollama_missing(monkeypatch) -> None:
    """When the Ollama path yields None, we must fall through to the heuristic."""
    monkeypatch.setattr(brain, "_parse_intent_ollama", lambda _text: None)
    intent = brain.parse_intent("shutdown")
    assert intent.action == "shutdown"
    assert intent.requires_sudo is True
