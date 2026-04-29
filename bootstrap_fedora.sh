#!/bin/bash
# bootstrap_fedora.sh - Installs all system-level dependencies for the
# Fedora Voice Assistant on Fedora 43.
#
# Run as: sudo bash bootstrap_fedora.sh

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: This script must be run with sudo." >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-${USER}}"
REAL_HOME="$(getent passwd "${REAL_USER}" | cut -d: -f6)"

echo "=== Fedora Voice Assistant - System Bootstrap ==="
echo "Project root : ${PROJECT_ROOT}"
echo "Installing for user: ${REAL_USER} (home: ${REAL_HOME})"
echo

echo "[1/5] Installing Fedora system packages..."
dnf install -y \
    python3-devel \
    python3-gobject \
    python3-pip \
    gtk4-devel \
    libadwaita-devel \
    cairo-devel \
    portaudio-devel \
    espeak-ng \
    gcc gcc-c++ make cmake git \
    sqlite \
    pipewire-devel \
    alsa-lib-devel \
    dbus-devel \
    curl wget

echo
echo "[2/5] Installing Ollama (if missing)..."
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed: $(ollama --version 2>/dev/null || echo unknown)"
fi

echo
echo "[3/5] Pulling local LLM (llama3:8b)..."
sudo -u "${REAL_USER}" ollama pull llama3:8b || echo "WARN: ollama pull failed; retry manually."

echo
echo "[4/5] Installing Python PIP dependencies..."
sudo -u "${REAL_USER}" pip install --user --upgrade -r "${PROJECT_ROOT}/requirements.txt"

echo
echo "[5/5] Preparing model directories..."
sudo -u "${REAL_USER}" mkdir -p \
    "${PROJECT_ROOT}/assets/models/whisper" \
    "${PROJECT_ROOT}/assets/models/piper" \
    "${PROJECT_ROOT}/assets/models/speechbrain" \
    "${PROJECT_ROOT}/assets/models/wakeword" \
    "${PROJECT_ROOT}/data" \
    "${PROJECT_ROOT}/logs"

echo
echo "=== Bootstrap complete ==="
echo "Next steps:"
echo "  1. Install the sudoers rule:"
echo "     sudo cp ${PROJECT_ROOT}/resources/references/sudoers_template.txt /etc/sudoers.d/fedora-voice-assistant"
echo "     sudo chmod 0440 /etc/sudoers.d/fedora-voice-assistant"
echo "  2. Run OOBE to enrol your voice:"
echo "     python3 -m oobe.gui"
echo "  3. Enable the user systemd service:"
echo "     cp ${PROJECT_ROOT}/resources/systemd/fedora-voice-assistant.service ~/.config/systemd/user/"
echo "     systemctl --user daemon-reload"
echo "     systemctl --user enable --now fedora-voice-assistant.service"
