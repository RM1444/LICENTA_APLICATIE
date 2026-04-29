"""Voice biometrics: enrolment embedding generation and speaker verification.

Uses SpeechBrain's ECAPA-TDNN pretrained model to produce fixed-length speaker
embeddings from 16 kHz mono PCM audio. Verification is cosine similarity
against a stored `Master Owner Profile` and a configurable threshold.
"""
from __future__ import annotations

import io
import logging
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.config import get_config, resolve_path

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model = None  # SpeechBrain classifier, lazily loaded

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VerificationResult:
    is_owner: bool
    similarity: float
    threshold: float


def _load_model():
    """Lazy-load the ECAPA-TDNN model. Kept out of import-time so tests that
    don't touch biometrics don't pay for torch + speechbrain startup."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        cfg = get_config()["biometrics"]
        source = cfg["model_source"]
        savedir = resolve_path(cfg["savedir"])
        savedir.mkdir(parents=True, exist_ok=True)

        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        logger.info("Loading SpeechBrain encoder from %s (cache=%s)", source, savedir)
        _model = EncoderClassifier.from_hparams(source=source, savedir=str(savedir))
        return _model


def _ensure_float_mono_16k(
    audio: np.ndarray | bytes,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Normalize raw PCM16 bytes or a numpy array into float32 mono at 16 kHz.

    Callers are expected to already produce 16 kHz mono audio; this is a
    defensive fallback to convert obvious PCM16 bytes to float32 [-1, 1].
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
    else:
        arr = np.asarray(audio)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        # Simple linear resample; recording pipeline should already supply 16k,
        # so this is a last-resort safety net.
        ratio = TARGET_SAMPLE_RATE / float(sample_rate)
        new_len = int(round(arr.shape[0] * ratio))
        if new_len <= 0:
            raise ValueError("Cannot resample to zero-length buffer.")
        x_old = np.linspace(0.0, 1.0, num=arr.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype(np.float32)

    return arr


def extract_embedding(audio: np.ndarray | bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Return a single-vector embedding for the given audio clip."""
    samples = _ensure_float_mono_16k(audio, sample_rate=sample_rate)
    if samples.size < TARGET_SAMPLE_RATE // 2:
        raise ValueError(
            "Audio clip is too short for reliable speaker embedding "
            f"({samples.size} samples, minimum 8000)."
        )

    import torch  # Local import - torch is heavy.

    model = _load_model()
    tensor = torch.from_numpy(samples).unsqueeze(0)
    with torch.inference_mode():
        embedding = model.encode_batch(tensor).squeeze().cpu().numpy().astype(np.float32)
    return embedding.reshape(-1)


def average_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """L2-normalize each embedding, average them, then L2-normalize the result.

    Producing a unit-norm master profile makes cosine similarity numerically
    stable and reduces drift between enrolment samples of different loudness.
    """
    if not embeddings:
        raise ValueError("Cannot average zero embeddings.")
    stacked = np.stack([_l2_normalize(np.asarray(e, dtype=np.float32)) for e in embeddings])
    mean = stacked.mean(axis=0)
    return _l2_normalize(mean)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Embedding dim mismatch: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def verify_speaker(
    candidate_audio: np.ndarray | bytes,
    master_embedding: np.ndarray,
    threshold: Optional[float] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> VerificationResult:
    """Compute cosine similarity between the candidate clip and the master
    profile, and decide whether it clears the threshold."""
    if threshold is None:
        threshold = float(get_config()["biometrics"]["similarity_threshold"])

    candidate = extract_embedding(candidate_audio, sample_rate=sample_rate)
    sim = cosine_similarity(candidate, master_embedding)
    result = VerificationResult(
        is_owner=sim > threshold,
        similarity=sim,
        threshold=threshold,
    )
    logger.info(
        "Biometric verification: similarity=%.4f threshold=%.4f -> is_owner=%s",
        sim, threshold, result.is_owner,
    )
    return result


def load_wav_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Utility for tests and enrolment: read a WAV file into (float32, sr)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
        channels = wf.getnchannels()
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported.")
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)
    return arr, sr


def save_wav_file(path: str | Path, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Persist a float32 numpy array to a 16-bit PCM mono WAV file."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
