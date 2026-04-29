"""Voice enrolment logic.

Records the 5 phonetically rich sentences listed in
`resources/assets/enrollment_sentences.txt`, extracts an ECAPA-TDNN embedding
per clip, averages them into the Master Owner Profile, and persists the result
through `core.db_manager`.

The GUI layer is intentionally kept in `oobe/gui.py`; this module is pure logic
so it can be driven from a CLI, a test harness, or the Gtk UI.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.config import PROJECT_ROOT, get_config
from core import biometrics, db_manager

logger = logging.getLogger(__name__)

SENTENCES_PATH = PROJECT_ROOT / "resources" / "assets" / "enrollment_sentences.txt"


@dataclass
class EnrollmentSample:
    sentence: str
    audio_f32: np.ndarray
    sample_rate: int


def load_enrollment_sentences() -> list[str]:
    return [
        line.strip()
        for line in SENTENCES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_sample(
    sentence: str,
    *,
    duration_s: Optional[float] = None,
) -> EnrollmentSample:
    """Record one enrolment clip using the core audio pipeline.

    The pipeline uses VAD to stop automatically when the user is done speaking,
    with a max duration cap from config.
    """
    from core.audio_pipeline import AudioPipeline

    cfg = get_config()["audio"]
    max_duration = duration_s if duration_s is not None else float(cfg["max_command_duration_s"])

    with AudioPipeline() as pipeline:
        # Give the user a moment to start speaking after the UI beep.
        time.sleep(0.2)
        utterance = pipeline.capture_once(timeout_s=max_duration)

    if utterance is None:
        raise RuntimeError("No audio captured for this enrolment sentence.")

    return EnrollmentSample(
        sentence=sentence,
        audio_f32=utterance.audio_f32,
        sample_rate=utterance.sample_rate,
    )


def build_master_profile(samples: list[EnrollmentSample]) -> np.ndarray:
    """Generate per-sample embeddings, then average them (L2-normalised)."""
    if len(samples) < 3:
        raise ValueError("At least 3 enrolment samples are required.")

    embeddings: list[np.ndarray] = []
    for sample in samples:
        emb = biometrics.extract_embedding(sample.audio_f32, sample_rate=sample.sample_rate)
        embeddings.append(emb)

    master = biometrics.average_embeddings(embeddings)
    logger.info("Built master profile of dim=%s from %d samples", master.shape, len(samples))
    return master


def enrol_owner(
    name: str,
    samples: list[EnrollmentSample],
) -> int:
    """Persist the owner's master profile and return the new user id."""
    db_manager.initialize_database()
    master = build_master_profile(samples)
    return db_manager.upsert_owner(name=name, embedding=master)


def enrol_from_cli(
    name: str,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """Command-line enrolment helper for scripted / headless testing."""
    sentences = load_enrollment_sentences()
    samples: list[EnrollmentSample] = []
    for idx, sentence in enumerate(sentences, start=1):
        if progress:
            progress(idx, len(sentences), sentence)
        sample = record_sample(sentence)
        samples.append(sample)
    return enrol_owner(name=name, samples=samples)
