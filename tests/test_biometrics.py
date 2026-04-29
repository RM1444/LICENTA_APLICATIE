"""Biometrics unit tests - cover the maths without touching SpeechBrain."""
from __future__ import annotations

import numpy as np
import pytest

from core import biometrics


def test_cosine_similarity_identical_is_one() -> None:
    v = np.array([0.5, 0.1, -0.3, 0.9], dtype=np.float32)
    assert biometrics.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_opposite_is_minus_one() -> None:
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert biometrics.cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    zero = np.zeros(4, dtype=np.float32)
    other = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert biometrics.cosine_similarity(zero, other) == 0.0


def test_cosine_similarity_shape_mismatch_raises() -> None:
    a = np.ones(4, dtype=np.float32)
    b = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError):
        biometrics.cosine_similarity(a, b)


def test_average_embeddings_returns_unit_vector() -> None:
    embeddings = [np.random.randn(192).astype(np.float32) for _ in range(5)]
    avg = biometrics.average_embeddings(embeddings)
    assert avg.shape == (192,)
    assert np.linalg.norm(avg) == pytest.approx(1.0, abs=1e-5)


def test_average_embeddings_empty_raises() -> None:
    with pytest.raises(ValueError):
        biometrics.average_embeddings([])


def test_l2_normalize_zero_safe() -> None:
    zero = np.zeros(4, dtype=np.float32)
    result = biometrics._l2_normalize(zero)
    assert result.shape == (4,)
    assert np.all(result == 0.0)


def test_wav_roundtrip(tmp_path) -> None:
    """Writing then reading a WAV file preserves the signal within int16 quantisation."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    path = tmp_path / "tone.wav"
    biometrics.save_wav_file(path, signal, sample_rate=sr)
    loaded, loaded_sr = biometrics.load_wav_file(path)

    assert loaded_sr == sr
    assert loaded.shape == signal.shape
    # Accept modest quantisation noise from int16 roundtrip.
    assert np.max(np.abs(loaded - signal)) < 1e-3


def test_ensure_float_mono_converts_bytes() -> None:
    pcm16 = (np.ones(8000, dtype=np.int16) * 16384).tobytes()
    result = biometrics._ensure_float_mono_16k(pcm16, sample_rate=16000)
    assert result.dtype == np.float32
    assert result.shape == (8000,)
    assert result[0] == pytest.approx(16384 / 32768.0, abs=1e-6)


def test_ensure_float_mono_downmixes_stereo() -> None:
    stereo = np.stack([np.ones(4000), np.full(4000, -1.0)], axis=1).astype(np.float32)
    result = biometrics._ensure_float_mono_16k(stereo, sample_rate=16000)
    assert result.shape == (4000,)
    assert np.allclose(result, 0.0)
