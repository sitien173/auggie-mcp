"""SQLite storage foundation for auggie-mcp."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    mtime_ns INTEGER,
    size INTEGER,
    sha256 BLOB
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files ON DELETE CASCADE,
    ordinal INTEGER,
    start_line INTEGER,
    end_line INTEGER,
    text TEXT,
    content_hash BLOB NOT NULL,
    UNIQUE(file_id, ordinal),
    CHECK(start_line <= end_line)
);

CREATE INDEX chunks_hash ON chunks(content_hash);

CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    content_hash BLOB NOT NULL,
    model_key TEXT NOT NULL,
    vector BLOB NOT NULL,
    last_seen INTEGER,
    UNIQUE(content_hash, model_key)
);

PRAGMA user_version = 1;
"""


class StoreError(RuntimeError):
    """Raised when the index database cannot be opened or has an incompatible schema."""


def db_path(repo_root: Path | str) -> Path:
    """Return the path to the SQLite index file for a repository."""
    return Path(repo_root) / ".auggie" / "index-v2.sqlite"


def _check_version(conn: sqlite3.Connection, path: Path) -> int:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    return int(version)


def open_write(path: Path | str, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open a read-write SQLite connection, initializing the schema if absent."""
    db_file = Path(path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")

        version = _check_version(conn, db_file)
        if version == 0:
            has_tables = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
            ).fetchone()
            if has_tables is not None:
                raise StoreError(
                    f"{db_file}: unsupported schema user_version 0 (expected {SCHEMA_VERSION})"
                )
            conn.executescript(_SCHEMA_SQL)
        elif version != SCHEMA_VERSION:
            raise StoreError(
                f"{db_file}: unsupported schema user_version {version} (expected {SCHEMA_VERSION})"
            )
    except Exception:
        conn.close()
        raise
    return conn


def open_read(path: Path | str, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open a read-only SQLite connection without modifying the database."""
    db_file = Path(path)
    if not db_file.exists():
        raise StoreError(f"{db_file}: index database does not exist")
    uri = f"file:{quote(str(db_file.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        conn.execute("PRAGMA query_only = 1")
        version = _check_version(conn, db_file)
        if version != SCHEMA_VERSION:
            raise StoreError(
                f"{db_file}: unsupported schema user_version {version} (expected {SCHEMA_VERSION})"
            )
    except Exception:
        conn.close()
        raise
    return conn


def encode_vector(vec: object) -> bytes:
    """Encode a 1-D vector as a little-endian float32 BLOB."""
    arr = np.asarray(vec, dtype="<f4")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D vector, got shape {arr.shape}")
    return arr.tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    """Decode a little-endian float32 BLOB into a 1-D float32 numpy array."""
    return np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
