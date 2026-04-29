# Fedora Voice Assistant

A biometrically secured, locally-hosted voice assistant for Fedora 43 (GNOME).

Every command is gated by voice-fingerprint verification (SpeechBrain ECAPA-TDNN).
Privileged actions (package installation, system config, file modification) execute
with `sudo` only when the speaker's voice matches the registered system owner.

## Architecture

Four phases, each isolated in its own module:

1. **OOBE (`oobe/`)** — GTK4 + libadwaita onboarding, 5-sentence voice enrollment.
2. **Bouncer (`core/audio_pipeline.py`, `core/biometrics.py`)** — Wake word detection,
   VAD, ECAPA-TDNN cosine similarity verification.
3. **Brain (`core/brain.py`)** — faster-whisper STT + Ollama/LangChain intent parsing
   with strict JSON schema output.
4. **Hands (`system/executor.py`, `system/sudo_executor.py`, `system/package_manager.py`)** —
   D-Bus + subprocess execution, voice-tied NOPASSWD sudo for whitelisted operations.

## Install

```bash
sudo bash bootstrap_fedora.sh
python3 -m oobe.gui              # Run OOBE to enroll as owner
sudo cp resources/systemd/fedora-voice-assistant.service /etc/systemd/user/
systemctl --user enable --now fedora-voice-assistant.service
```

## Tech Stack

| Component | Library |
|---|---|
| Wake word | `openWakeWord` |
| STT | `faster-whisper` |
| TTS | `Piper` |
| Biometrics | `SpeechBrain` (ECAPA-TDNN) |
| LLM | `Ollama` + `LangChain` |
| UI | PyGObject (GTK4 + libadwaita) |
| Storage | SQLite |

All inference runs 100% locally. No cloud fallback.

## Security

Voice biometrics provide **convenience-level security**, not cryptographic-level.
The NOPASSWD sudo rule is scoped to a single whitelisted script
([`system/sudo_executor.py`](system/sudo_executor.py)) with argument allow-listing.
