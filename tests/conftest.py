"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import pytest


@pytest.fixture()
def tmp_config(tmp_path, monkeypatch):
    """Redirect config paths to a temp dir so tests don't touch project state."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  database: {tmp_path / 'test.db'}
  log_file: {tmp_path / 'test.log'}
  models_root: {tmp_path / 'models'}

audio:
  sample_rate: 16000
  channels: 1
  sample_width_bytes: 2
  frames_per_buffer: 512
  silence_timeout_ms: 500
  max_command_duration_s: 5
  vad_aggressiveness: 2

wakeword:
  engine: openwakeword
  keyword: hey_jarvis
  threshold: 0.5
  model_path: {tmp_path / 'wakeword.tflite'}

biometrics:
  model_source: test/model
  savedir: {tmp_path / 'speechbrain'}
  similarity_threshold: 0.75
  enrollment_sample_count: 5

stt:
  model_size: base.en
  model_path: {tmp_path / 'whisper'}
  device: cpu
  compute_type: int8
  beam_size: 5
  language: en
  vad_filter: true

tts:
  engine: piper
  voice_model: {tmp_path / 'voice.onnx'}
  voice_config: {tmp_path / 'voice.onnx.json'}
  sample_rate: 22050

brain:
  llm_provider: ollama
  ollama_base_url: http://127.0.0.1:11434
  ollama_model: llama3:8b
  system_prompt_path: resources/references/intent_system_prompt.txt
  confidence_auto_execute: 0.8
  confidence_confirm: 0.6

security:
  guest_approval_timeout_s: 5
  approval_phrases: [allow it, approve]
  denial_phrases: [deny, block]

logging:
  level: INFO
  max_bytes: 1000000
  backup_count: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FVA_CONFIG", str(config_path))

    from core import config
    config.get_config.cache_clear()
    yield tmp_path
    config.get_config.cache_clear()
