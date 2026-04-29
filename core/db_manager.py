"""SQLite persistence layer for users, settings, and audit logs.

All DB access goes through this module. Other modules must not open raw
connections to `assistant.db`. Voice embeddings are stored as numpy-serialized
BLOBs; the column `embedding_dim` records the source vector length so we can
reshape on read without guessing.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from core.config import PROJECT_ROOT, get_config, resolve_path

logger = logging.getLogger(__name__)

SCHEMA_PATH = PROJECT_ROOT / "resources" / "references" / "db_schema.sql"

_lock = threading.RLock()


@dataclass(frozen=True)
class User:
    id: int
    name: str
    role: str  # 'Owner' | 'Guest'
    embedding: np.ndarray
    created_at: str


@dataclass(frozen=True)
class LogEntry:
    id: Optional[int]
    timestamp: Optional[str]
    speaker_id: Optional[int]
    transcript: Optional[str]
    action: Optional[str]
    result: str  # 'success' | 'denied' | 'error' | 'timeout'
    similarity_score: Optional[float]
    error_message: Optional[str]


def _db_path() -> Path:
    return resolve_path(get_config()["paths"]["database"])


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def initialize_database() -> None:
    """Apply the SQL schema. Idempotent - safe to call on every boot."""
    with _lock, _connect() as conn:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema)
    logger.info("Database initialised at %s", _db_path())


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def _serialize_embedding(embedding: np.ndarray) -> tuple[bytes, int]:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return arr.tobytes(), int(arr.shape[0])


def _deserialize_embedding(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


def create_user(name: str, role: str, embedding: np.ndarray) -> int:
    if role not in ("Owner", "Guest"):
        raise ValueError(f"Invalid role: {role!r}")
    blob, dim = _serialize_embedding(embedding)
    with _lock, _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, role, voice_embedding, embedding_dim) "
            "VALUES (?, ?, ?, ?)",
            (name, role, blob, dim),
        )
        user_id = int(cursor.lastrowid or 0)
    logger.info("Created user id=%s name=%s role=%s dim=%s", user_id, name, role, dim)
    return user_id


def upsert_owner(name: str, embedding: np.ndarray) -> int:
    """Either creates the owner or overwrites the existing owner's profile."""
    blob, dim = _serialize_embedding(embedding)
    with _lock, _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE role = 'Owner' LIMIT 1"
        ).fetchone()
        if existing is None:
            cursor = conn.execute(
                "INSERT INTO users (name, role, voice_embedding, embedding_dim) "
                "VALUES (?, 'Owner', ?, ?)",
                (name, blob, dim),
            )
            user_id = int(cursor.lastrowid or 0)
        else:
            user_id = int(existing["id"])
            conn.execute(
                "UPDATE users SET name = ?, voice_embedding = ?, embedding_dim = ? "
                "WHERE id = ?",
                (name, blob, dim, user_id),
            )
    logger.info("Upserted owner id=%s name=%s dim=%s", user_id, name, dim)
    return user_id


def get_owner() -> Optional[User]:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT id, name, role, voice_embedding, embedding_dim, created_at "
            "FROM users WHERE role = 'Owner' LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return User(
        id=row["id"],
        name=row["name"],
        role=row["role"],
        embedding=_deserialize_embedding(row["voice_embedding"], row["embedding_dim"]),
        created_at=row["created_at"],
    )


def get_all_users() -> list[User]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, role, voice_embedding, embedding_dim, created_at "
            "FROM users ORDER BY id"
        ).fetchall()
    return [
        User(
            id=r["id"],
            name=r["name"],
            role=r["role"],
            embedding=_deserialize_embedding(r["voice_embedding"], r["embedding_dim"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


def delete_user(user_id: int) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = CURRENT_TIMESTAMP",
            (key, value),
        )


def get_similarity_threshold() -> float:
    raw = get_setting("similarity_threshold")
    if raw is None:
        return float(get_config()["biometrics"]["similarity_threshold"])
    return float(raw)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def log_event(
    *,
    speaker_id: Optional[int],
    transcript: Optional[str],
    action: Optional[str],
    result: str,
    similarity_score: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    if result not in ("success", "denied", "error", "timeout"):
        raise ValueError(f"Invalid result: {result!r}")
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO logs (speaker_id, transcript, action, result, "
            "similarity_score, error_message) VALUES (?, ?, ?, ?, ?, ?)",
            (speaker_id, transcript, action, result, similarity_score, error_message),
        )


def recent_logs(limit: int = 50) -> list[LogEntry]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, speaker_id, transcript, action, result, "
            "similarity_score, error_message FROM logs "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        LogEntry(
            id=r["id"],
            timestamp=r["timestamp"],
            speaker_id=r["speaker_id"],
            transcript=r["transcript"],
            action=r["action"],
            result=r["result"],
            similarity_score=r["similarity_score"],
            error_message=r["error_message"],
        )
        for r in rows
    ]
