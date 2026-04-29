-- Fedora Voice Assistant - SQLite schema
-- Applied on first launch by core/db_manager.py::initialize_database.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL,
    role              TEXT NOT NULL CHECK(role IN ('Owner', 'Guest')),
    voice_embedding   BLOB NOT NULL,
    embedding_dim     INTEGER NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_single_owner
    ON users(role) WHERE role = 'Owner';

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP,
    speaker_id        INTEGER,
    transcript        TEXT,
    action            TEXT,
    result            TEXT CHECK(result IN ('success', 'denied', 'error', 'timeout')),
    similarity_score  REAL,
    error_message     TEXT,
    FOREIGN KEY (speaker_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_speaker ON logs(speaker_id);

INSERT OR IGNORE INTO settings (key, value) VALUES ('wake_word', 'hey_jarvis');
INSERT OR IGNORE INTO settings (key, value) VALUES ('similarity_threshold', '0.45');
INSERT OR IGNORE INTO settings (key, value) VALUES ('silence_timeout_ms', '1500');
INSERT OR IGNORE INTO settings (key, value) VALUES ('tts_voice', 'en_US-lessac-medium');
INSERT OR IGNORE INTO settings (key, value) VALUES ('ollama_model', 'llama3:8b');
INSERT OR IGNORE INTO settings (key, value) VALUES ('whisper_model', 'base.en');
INSERT OR IGNORE INTO settings (key, value) VALUES ('log_level', 'INFO');
