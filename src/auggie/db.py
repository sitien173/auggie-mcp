"""
Database module for auggie.

Provides SQLite database connection and schema management for storing
file metadata and chunk metadata. Uses WAL mode for better concurrent
read performance and foreign keys for referential integrity.

Schema:
    files: Tracks indexed files with path, modification time, size, and hash.
    chunks: Stores text chunks and metadata, linked to files via foreign key.
    chunks_fts: FTS5 full-text index on chunk text for hybrid search.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: SQL schema for the auggie database.
#: Uses WAL journal mode for performance and foreign keys for integrity.
CURRENT_SCHEMA_VERSION = "2"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'default',
  mtime_ns INTEGER NOT NULL,
  size INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  UNIQUE(path, branch)
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  text TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  dim INTEGER NOT NULL,
  model TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(file_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_sha256 ON chunks(text_sha256);

-- History table for tracking index operations
-- Used by AI tools to understand what changed (refresh + log workflow)
CREATE TABLE IF NOT EXISTS index_history (
  id INTEGER PRIMARY KEY,
  timestamp TEXT NOT NULL DEFAULT (datetime('now')),
  action TEXT NOT NULL,  -- 'index', 'delete', 'clean', 'refresh', 'reindex'
  files_affected INTEGER NOT NULL DEFAULT 0,
  chunks_affected INTEGER NOT NULL DEFAULT 0,
  details TEXT,  -- JSON with action-specific details
  CONSTRAINT valid_action CHECK (action IN ('index', 'delete', 'clean', 'refresh', 'reindex'))
);

CREATE INDEX IF NOT EXISTS idx_history_timestamp ON index_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_history_action ON index_history(action);

-- Metadata table for index-wide settings
-- Stores key-value pairs like ast_mode, created_at, etc.
CREATE TABLE IF NOT EXISTS index_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
VALUES ('schema_version', '2', datetime('now'));
"""

#: FTS5 schema for full-text search on chunk text.
#: Applied separately since FTS5 may not be available on all SQLite builds.
FTS5_SCHEMA = """
-- FTS5 virtual table for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

-- Triggers to keep FTS index in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class DatabaseError(Exception):
    """Raised when database operations fail with helpful context."""

    pass


def connect(db_path: Path, init_fts: bool = True) -> sqlite3.Connection:
    """
    Connect to the auggie SQLite database, creating it if necessary.

    Creates the parent directory if it doesn't exist, opens the database,
    and initializes the schema if tables don't exist. Also runs migrations
    for existing databases (e.g., adding branch column).

    Args:
        db_path: Path to the SQLite database file.
        init_fts: Whether to initialize FTS5 schema (default: True).
            Set to False to skip FTS5 initialization for faster connections
            when full-text search is not needed.

    Returns:
        An open sqlite3.Connection with the schema initialized.

    Raises:
        DatabaseError: If the database cannot be opened or initialized,
            with a helpful error message explaining the issue.

    Example:
        >>> con = connect(Path(".auggie/index.sqlite"))
        >>> con.execute("SELECT COUNT(*) FROM files").fetchone()
        (0,)
    """
    # Create parent directory
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise DatabaseError(
            f"Cannot create database directory '{db_path.parent}': {e}. "
            f"Check that you have write permissions to this location."
        ) from e

    # Open database
    try:
        con = sqlite3.connect(str(db_path))
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        if "unable to open" in error_msg:
            # Check if file exists but is not readable
            if db_path.exists():
                raise DatabaseError(
                    f"Cannot open database '{db_path}': file exists but is not accessible. "
                    f"Check file permissions or remove the corrupted file with: rm '{db_path}'"
                ) from e
            else:
                raise DatabaseError(
                    f"Cannot create database '{db_path}': {e}. "
                    f"Check that you have write permissions to '{db_path.parent}'."
                ) from e
        raise DatabaseError(f"Database error opening '{db_path}': {e}") from e

    # Initialize schema
    try:
        con.executescript(SCHEMA)
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        con.close()
        error_msg = str(e).lower()
        if "database is locked" in error_msg:
            raise DatabaseError(
                f"Database '{db_path}' is locked by another process. "
                f"Wait for other auggie commands to finish, or remove stale locks."
            ) from e
        if "disk i/o error" in error_msg or "readonly" in error_msg:
            raise DatabaseError(
                f"Cannot write to database '{db_path}': {e}. "
                f"Check disk space and write permissions."
            ) from e
        if "malformed" in error_msg or "corrupt" in error_msg or "not a database" in error_msg:
            raise DatabaseError(
                f"Database '{db_path}' is corrupted or not a valid SQLite database. "
                f"Remove it and reindex: rm '{db_path}' && auggie index ."
            ) from e
        raise DatabaseError(f"Failed to initialize database '{db_path}': {e}") from e

    # Run migrations for existing databases
    _migrate_branch_column(con)
    _validate_chunks_schema(con)

    if init_fts:
        try:
            con.executescript(FTS5_SCHEMA)
        except sqlite3.OperationalError:
            # FTS5 not available in this SQLite build - silently continue
            pass

    return con


def has_fts5(con: sqlite3.Connection) -> bool:
    """
    Check if FTS5 index is available in the database.

    Args:
        con: Open database connection.

    Returns:
        True if chunks_fts table exists and is functional.
    """
    try:
        # Check if the table exists
        result = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        return result is not None
    except sqlite3.OperationalError:
        return False


def log_history(
    con: sqlite3.Connection,
    action: str,
    files_affected: int = 0,
    chunks_affected: int = 0,
    details: dict | None = None,
) -> int:
    """
    Log an index operation to history.

    AI TOOL HINT: Use `auggie log --json` to query this history.
    Combined with `auggie index --refresh`, this lets you track what
    changed since your last check.

    Args:
        con: Open database connection.
        action: Operation type ('index', 'delete', 'clean', 'refresh', 'reindex').
        files_affected: Number of files affected.
        chunks_affected: Number of chunks affected.
        details: Optional dict with action-specific details (stored as JSON).

    Returns:
        The history entry ID.
    """
    import json as json_module

    details_json = json_module.dumps(details) if details else None
    cur = con.execute(
        """
        INSERT INTO index_history (action, files_affected, chunks_affected, details)
        VALUES (?, ?, ?, ?)
        """,
        (action, files_affected, chunks_affected, details_json),
    )
    con.commit()
    return cur.lastrowid


def rebuild_fts5(con: sqlite3.Connection) -> int:
    """
    Rebuild the FTS5 index from existing chunks.

    Use this to populate FTS5 after upgrading an existing database,
    or to repair a corrupted FTS5 index.

    Args:
        con: Open database connection.

    Returns:
        Number of chunks indexed.

    Raises:
        sqlite3.OperationalError: If FTS5 is not available.
    """
    # Drop and recreate FTS5 schema to ensure clean state
    # This handles cases where the table exists but is corrupted
    con.execute("DROP TABLE IF EXISTS chunks_fts")
    con.execute("DROP TRIGGER IF EXISTS chunks_fts_ai")
    con.execute("DROP TRIGGER IF EXISTS chunks_fts_ad")
    con.execute("DROP TRIGGER IF EXISTS chunks_fts_au")
    con.executescript(FTS5_SCHEMA)

    # Rebuild from chunks table
    con.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
    con.commit()

    # Return count
    return con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


def get_metadata(con: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    """
    Get a metadata value from the index.

    Args:
        con: Open database connection.
        key: Metadata key to retrieve.
        default: Default value if key not found.

    Returns:
        The metadata value, or default if not found.
    """
    row = con.execute("SELECT value FROM index_metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_metadata(con: sqlite3.Connection, key: str, value: str) -> None:
    """
    Set a metadata value in the index.

    Uses INSERT OR REPLACE to upsert the value.

    Args:
        con: Open database connection.
        key: Metadata key to set.
        value: Value to store.
    """
    con.execute(
        """
        INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        """,
        (key, value),
    )
    con.commit()


def get_all_metadata(con: sqlite3.Connection) -> dict[str, str]:
    """
    Get all metadata from the index.

    Returns:
        Dict of key-value pairs.
    """
    rows = con.execute("SELECT key, value FROM index_metadata").fetchall()
    return dict(rows)


def _migrate_branch_column(con: sqlite3.Connection) -> bool:
    """
    Migrate existing database to add branch column if missing.

    This handles databases created before branch-aware indexing was added.
    Existing files get branch='default'.

    SQLite doesn't allow modifying UNIQUE constraints in-place, so we use
    the table recreation approach:
    1. Create new table with correct schema
    2. Copy data from old table
    3. Drop old table
    4. Rename new table

    Args:
        con: Open database connection.

    Returns:
        True if migration was performed, False if already migrated.
    """
    # Check if branch column exists
    columns = con.execute("PRAGMA table_info(files)").fetchall()
    column_names = {col[1] for col in columns}

    if "branch" in column_names:
        return False  # Already migrated

    # SQLite doesn't allow dropping auto-generated unique indexes,
    # so we need to recreate the table with the new schema

    # Step 1: Create new table with branch column and updated unique constraint
    con.execute("""
        CREATE TABLE IF NOT EXISTS files_new (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            branch TEXT NOT NULL DEFAULT 'default',
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            UNIQUE(path, branch)
        )
    """)

    # Step 2: Copy data from old table (all existing files get branch='default')
    con.execute("""
        INSERT INTO files_new (id, path, branch, mtime_ns, size, sha256)
        SELECT id, path, 'default', mtime_ns, size, sha256 FROM files
    """)

    # Step 3: Drop old table
    con.execute("DROP TABLE files")

    # Step 4: Rename new table
    con.execute("ALTER TABLE files_new RENAME TO files")

    con.commit()
    return True


def _validate_chunks_schema(con: sqlite3.Connection) -> None:
    """
    Validate the Phase 2 chunks schema.

    Phase 2 is a rebuild-from-scratch migration. Old indexes that still
    store embeddings in SQLite must be reset and rebuilt.
    """
    columns = con.execute("PRAGMA table_info(chunks)").fetchall()
    if not columns:
        return

    column_names = {col[1] for col in columns}
    if "embedding" in column_names:
        raise DatabaseError(
            "Old SQLite vector schema detected. "
            "Run 'auggie reset -f' and then 'auggie index .' to rebuild with LanceDB storage."
        )

    required = {
        "id",
        "file_id",
        "chunk_index",
        "start_line",
        "end_line",
        "text",
        "text_sha256",
        "dim",
        "model",
        "created_at",
    }
    if not required.issubset(column_names):
        raise DatabaseError(
            "Unsupported chunks schema detected. "
            "Run 'auggie reset -f' and then 'auggie index .' to rebuild the index."
        )

    con.execute(
        """
        INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
        VALUES ('schema_version', ?, datetime('now'))
        """,
        (CURRENT_SCHEMA_VERSION,),
    )
    con.commit()


def get_indexed_branches(con: sqlite3.Connection) -> set[str]:
    """
    Get all branches that have indexed files.

    Returns:
        Set of branch names.
    """
    rows = con.execute("SELECT DISTINCT branch FROM files").fetchall()
    return {row[0] for row in rows}


def get_branch_file_counts(con: sqlite3.Connection) -> dict[str, int]:
    """
    Get file counts per branch.

    Returns:
        Dict mapping branch name to file count.
    """
    rows = con.execute(
        "SELECT branch, COUNT(*) FROM files GROUP BY branch ORDER BY COUNT(*) DESC"
    ).fetchall()
    return dict(rows)


def get_file_states(con: sqlite3.Connection, branch: str) -> dict[str, tuple[int, int, int, str]]:
    """
    Get indexed file state for a branch.

    Returns:
        Dict mapping path -> (file_id, mtime_ns, size, sha256).
    """
    rows = con.execute(
        "SELECT path, id, mtime_ns, size, sha256 FROM files WHERE branch = ?",
        (branch,),
    ).fetchall()
    return {
        str(path): (int(file_id), int(mtime_ns), int(size), str(sha256))
        for path, file_id, mtime_ns, size, sha256 in rows
    }


def get_file_chunks(
    con: sqlite3.Connection, file_id: int
) -> list[tuple[int, int, int, int, str, str, int, str]]:
    """
    Get chunk metadata for a file.

    Returns:
        List of (chunk_id, chunk_index, start_line, end_line, text, text_sha256, dim, model).
    """
    rows = con.execute(
        """
        SELECT id, chunk_index, start_line, end_line, text, text_sha256, dim, model
        FROM chunks
        WHERE file_id = ?
        ORDER BY chunk_index ASC, id ASC
        """,
        (file_id,),
    ).fetchall()
    return [
        (
            int(chunk_id),
            int(chunk_index),
            int(start_line),
            int(end_line),
            str(text),
            str(text_sha256),
            int(dim),
            str(model),
        )
        for chunk_id, chunk_index, start_line, end_line, text, text_sha256, dim, model in rows
    ]


def touch_file_state(con: sqlite3.Connection, file_id: int, mtime_ns: int, size: int) -> None:
    """
    Update file stat fields without changing content hash.
    """
    con.execute(
        "UPDATE files SET mtime_ns = ?, size = ? WHERE id = ?",
        (mtime_ns, size, file_id),
    )


def delete_file_record(con: sqlite3.Connection, file_id: int) -> None:
    """
    Delete a file row. Chunk rows are removed via CASCADE.
    """
    con.execute("DELETE FROM files WHERE id = ?", (file_id,))


def delete_branch_files(con: sqlite3.Connection, branch: str) -> int:
    """
    Delete all file entries for a specific branch.

    Note: Chunks are deleted via CASCADE, but shared embeddings
    (same text_sha256) used by other branches are preserved.

    Args:
        con: Open database connection.
        branch: Branch name to delete.

    Returns:
        Number of files deleted.
    """
    cur = con.execute("DELETE FROM files WHERE branch = ?", (branch,))
    con.commit()
    return cur.rowcount
