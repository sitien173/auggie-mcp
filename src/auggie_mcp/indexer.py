"""Incremental repository indexer and status inspector."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auggie_mcp.chunking import CHUNKER_VERSION, Chunk, chunk_file
from auggie_mcp.config import Config
from auggie_mcp.config import load as load_config
from auggie_mcp.embed import Embedder, VoyageEmbedder
from auggie_mcp.locking import index_lock
from auggie_mcp.store import db_path, encode_vector, open_read, open_write
from auggie_mcp.walk import _is_binary, walk_repo

_ORPHAN_RETENTION_SECONDS = 30 * 86400
_SQLITE_IN_CHUNK = 900
_V2_INDEX_ENTRIES = frozenset(
    {"index-v2.sqlite", "index-v2.sqlite-wal", "index-v2.sqlite-shm", "index.lock"}
)


class IndexerError(RuntimeError):
    """Base error for indexing operations."""


class ModelMismatchError(IndexerError):
    """Raised when the configured model_key differs from the existing index without rebuild."""


@dataclass(frozen=True)
class _PreparedFile:
    path: str
    mtime_ns: int
    size: int
    sha256: bytes
    chunks: list[tuple[Chunk, bytes]]


def _read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(k): str(v) for k, v in rows if v is not None}


def _existing_embedding_hashes(
    conn: sqlite3.Connection, model_key: str, hashes: list[bytes]
) -> set[bytes]:
    found: set[bytes] = set()
    for i in range(0, len(hashes), _SQLITE_IN_CHUNK):
        slice_hashes = hashes[i : i + _SQLITE_IN_CHUNK]
        placeholders = ",".join("?" for _ in slice_hashes)
        rows = conn.execute(
            f"SELECT content_hash FROM embeddings "
            f"WHERE model_key = ? AND content_hash IN ({placeholders})",
            [model_key, *slice_hashes],
        ).fetchall()
        found.update(bytes(r[0]) for r in rows)
    return found


def _clear_for_rebuild(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")
        conn.execute(
            "DELETE FROM meta WHERE key IN ('model_key', 'chunker_version')"
        )
        _increment_generation(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _increment_generation(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('generation', '1') "
        "ON CONFLICT(key) DO UPDATE SET "
        "value = CAST(CAST(COALESCE(meta.value, '0') AS INTEGER) + 1 AS TEXT)"
    )


def _upsert_meta(conn: sqlite3.Connection, items: dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        list(items.items()),
    )


def _refresh_chunks_last_seen_for_file(
    conn: sqlite3.Connection, file_id: int, model_key: str, now_ts: int
) -> None:
    conn.execute(
        "UPDATE embeddings SET last_seen = ? "
        "WHERE model_key = ? AND content_hash IN ("
        "SELECT content_hash FROM chunks WHERE file_id = ?)",
        (now_ts, model_key, file_id),
    )


def _refresh_chunks_last_seen_for_paths(
    conn: sqlite3.Connection, paths: list[str], model_key: str, now_ts: int
) -> None:
    for i in range(0, len(paths), _SQLITE_IN_CHUNK):
        slice_paths = paths[i : i + _SQLITE_IN_CHUNK]
        placeholders = ",".join("?" for _ in slice_paths)
        conn.execute(
            f"UPDATE embeddings SET last_seen = ? "
            f"WHERE model_key = ? AND content_hash IN ("
            f"SELECT chunks.content_hash FROM chunks "
            f"JOIN files ON files.id = chunks.file_id "
            f"WHERE files.path IN ({placeholders}))",
            [now_ts, model_key, *slice_paths],
        )


def _apply_file_batch(
    conn: sqlite3.Connection,
    batch: list[_PreparedFile],
    *,
    model_key: str,
    new_vectors: dict[bytes, bytes],
    now_ts: int,
) -> None:
    for item in batch:
        conn.execute(
            "INSERT INTO files(path, mtime_ns, size, sha256) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "mtime_ns = excluded.mtime_ns, "
            "size = excluded.size, "
            "sha256 = excluded.sha256",
            (item.path, item.mtime_ns, item.size, item.sha256),
        )
        file_id = conn.execute(
            "SELECT id FROM files WHERE path = ?", (item.path,)
        ).fetchone()[0]
        _refresh_chunks_last_seen_for_file(conn, file_id, model_key, now_ts)
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

        for chunk, content_hash in item.chunks:
            if content_hash in new_vectors:
                conn.execute(
                    "INSERT INTO embeddings(content_hash, model_key, vector, last_seen) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(content_hash, model_key) DO UPDATE SET "
                    "vector = excluded.vector, last_seen = excluded.last_seen",
                    (content_hash, model_key, new_vectors[content_hash], now_ts),
                )
            else:
                conn.execute(
                    "UPDATE embeddings SET last_seen = ? "
                    "WHERE content_hash = ? AND model_key = ?",
                    (now_ts, content_hash, model_key),
                )
            conn.execute(
                "INSERT INTO chunks(file_id, ordinal, start_line, end_line, text, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    chunk.ordinal,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    content_hash,
                ),
            )
    _upsert_meta(conn, {"model_key": model_key})
    _increment_generation(conn)


def _apply_final_housekeeping(
    conn: sqlite3.Connection,
    *,
    mtime_only_updates: list[tuple[int, int, str]],
    deleted_paths: list[str],
    model_key: str,
    now_ts: int,
    now_iso: str,
    meta_needs_update: bool,
    write_chunker_version: bool,
    increment_generation: bool,
) -> None:
    if mtime_only_updates:
        conn.executemany(
            "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?",
            mtime_only_updates,
        )
    if deleted_paths:
        _refresh_chunks_last_seen_for_paths(conn, deleted_paths, model_key, now_ts)
        conn.executemany(
            "DELETE FROM files WHERE path = ?",
            [(p,) for p in deleted_paths],
        )
    cutoff = now_ts - _ORPHAN_RETENTION_SECONDS
    cur = conn.execute(
        "DELETE FROM embeddings WHERE last_seen < ? "
        "AND NOT EXISTS (SELECT 1 FROM chunks WHERE chunks.content_hash = embeddings.content_hash)",
        (cutoff,),
    )
    orphans_deleted = cur.rowcount > 0
    changed = bool(mtime_only_updates or deleted_paths or orphans_deleted or meta_needs_update)
    if changed:
        meta_updates = {
            "model_key": model_key,
            "last_indexed": now_iso,
        }
        if write_chunker_version:
            meta_updates["chunker_version"] = str(CHUNKER_VERSION)
        _upsert_meta(conn, meta_updates)
        if increment_generation:
            _increment_generation(conn)


def index_update(
    repo_root: Path | str,
    *,
    config: Config | None = None,
    embedder: Embedder | None = None,
    rebuild: bool = False,
    file_batch_size: int = 50,
    now_fn: Callable[[], float] | None = None,
    after_batch_hook: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Incrementally index a repository into ``<repo>/.auggie/index-v2.sqlite``."""
    started = time.monotonic()
    repo = Path(repo_root)
    cfg = config or load_config()
    active_embedder = embedder or VoyageEmbedder(cfg)
    target_model_key = active_embedder.model_key
    clock = now_fn or time.time
    batch_limit = max(1, file_batch_size)

    with index_lock(repo):
        conn = open_write(db_path(repo))
        try:
            meta = _read_meta(conn)
            if rebuild:
                _clear_for_rebuild(conn)
                meta = _read_meta(conn)
            elif "model_key" in meta and meta["model_key"] != target_model_key:
                raise ModelMismatchError(
                    f"Index model_key {meta['model_key']!r} does not match "
                    f"configured {target_model_key!r}; pass --rebuild to re-index"
                )

            existing_chunker = meta.get("chunker_version")
            # An absent version with committed files means a partial first run,
            # whose chunker is unknown, so those files are rechunked too.
            force_rechunk = existing_chunker != str(CHUNKER_VERSION)

            existing_files: dict[str, tuple[int, int, bytes]] = {
                row[0]: (int(row[1]), int(row[2]), bytes(row[3]))
                for row in conn.execute(
                    "SELECT path, mtime_ns, size, sha256 FROM files"
                ).fetchall()
            }

            files_scanned = 0
            seen_paths: set[str] = set()
            mtime_only_updates: list[tuple[int, int, str]] = []
            changed_files: list[_PreparedFile] = []
            rechunk_incomplete = False

            for rel_path in walk_repo(repo):
                files_scanned += 1
                full_path = repo / rel_path
                try:
                    st = full_path.stat()
                except FileNotFoundError:
                    continue
                except OSError:
                    if rel_path in existing_files and full_path.exists():
                        seen_paths.add(rel_path)
                        if force_rechunk:
                            rechunk_incomplete = True
                    continue

                mtime_ns = st.st_mtime_ns
                size = st.st_size

                existing = existing_files.get(rel_path)
                if (
                    existing is not None
                    and not force_rechunk
                    and existing[0] == mtime_ns
                    and existing[1] == size
                ):
                    seen_paths.add(rel_path)
                    continue

                try:
                    raw_bytes = full_path.read_bytes()
                except FileNotFoundError:
                    continue
                except OSError:
                    if rel_path in existing_files and full_path.exists():
                        seen_paths.add(rel_path)
                        if force_rechunk:
                            rechunk_incomplete = True
                    continue

                seen_paths.add(rel_path)
                sha256 = hashlib.sha256(raw_bytes).digest()
                if (
                    existing is not None
                    and not force_rechunk
                    and existing[2] == sha256
                ):
                    mtime_only_updates.append((mtime_ns, size, rel_path))
                    continue

                text = raw_bytes.decode("utf-8", errors="replace")
                chunks = chunk_file(
                    rel_path,
                    text,
                    max_lines=cfg.chunking.lines,
                    overlap_lines=cfg.chunking.overlap_lines,
                )
                chunk_pairs = [
                    (ch, hashlib.sha256(ch.text.encode("utf-8")).digest())
                    for ch in chunks
                ]
                changed_files.append(
                    _PreparedFile(
                        path=rel_path,
                        mtime_ns=mtime_ns,
                        size=size,
                        sha256=sha256,
                        chunks=chunk_pairs,
                    )
                )

            deleted_paths = sorted(set(existing_files) - seen_paths)

            unique_texts: dict[bytes, str] = {}
            total_chunks = 0
            for item in changed_files:
                for ch, content_hash in item.chunks:
                    total_chunks += 1
                    unique_texts.setdefault(content_hash, ch.text)

            existing_hashes = (
                _existing_embedding_hashes(
                    conn, target_model_key, list(unique_texts.keys())
                )
                if unique_texts
                else set()
            )
            missing_items = [
                (h, txt)
                for h, txt in unique_texts.items()
                if h not in existing_hashes
            ]
            chunks_embedded = len(missing_items)
            chunks_reused = sum(
                1
                for item in changed_files
                for _, h in item.chunks
                if h in existing_hashes
            )

            # Embed outside any SQLite transaction.
            new_vectors: dict[bytes, bytes] = {}
            if missing_items:
                vectors = active_embedder.embed([txt for _, txt in missing_items])
                if len(vectors) != len(missing_items):
                    raise IndexerError(
                        f"Embedder returned {len(vectors)} vectors for {len(missing_items)} texts"
                    )
                new_vectors = {
                    h: encode_vector(vec)
                    for (h, _), vec in zip(missing_items, vectors)
                }

            now_epoch = clock()
            now_ts = int(now_epoch)
            now_iso = datetime.fromtimestamp(now_epoch, tz=UTC).isoformat()

            file_batches = [
                changed_files[i : i + batch_limit]
                for i in range(0, len(changed_files), batch_limit)
            ]

            meta_needs_update = (
                meta.get("model_key") != target_model_key
                or (
                    not rechunk_incomplete
                    and meta.get("chunker_version") != str(CHUNKER_VERSION)
                )
            )
            write_chunker_version = not rechunk_incomplete

            for batch_idx, batch in enumerate(file_batches):
                is_last_batch = batch_idx == len(file_batches) - 1
                conn.execute("BEGIN IMMEDIATE")
                try:
                    _apply_file_batch(
                        conn,
                        batch,
                        model_key=target_model_key,
                        new_vectors=new_vectors,
                        now_ts=now_ts,
                    )
                    if is_last_batch:
                        _apply_final_housekeeping(
                            conn,
                            mtime_only_updates=mtime_only_updates,
                            deleted_paths=deleted_paths,
                            model_key=target_model_key,
                            now_ts=now_ts,
                            now_iso=now_iso,
                            meta_needs_update=True,
                            write_chunker_version=write_chunker_version,
                            increment_generation=False,
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                if after_batch_hook is not None:
                    after_batch_hook(batch_idx)

            if not file_batches:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    _apply_final_housekeeping(
                        conn,
                        mtime_only_updates=mtime_only_updates,
                        deleted_paths=deleted_paths,
                        model_key=target_model_key,
                        now_ts=now_ts,
                        now_iso=now_iso,
                        meta_needs_update=meta_needs_update,
                        write_chunker_version=write_chunker_version,
                        increment_generation=True,
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "files_scanned": files_scanned,
                "files_changed": len(changed_files),
                "files_deleted": len(deleted_paths),
                "chunks_embedded": chunks_embedded,
                "chunks_reused": chunks_reused,
                "duration_ms": duration_ms,
            }
        finally:
            conn.close()


def _entry_bytes(entry: Path) -> int:
    try:
        st = entry.lstat()
    except OSError:
        return 0
    if not entry.is_dir() or entry.is_symlink():
        return st.st_size
    total = 0
    for child in entry.rglob("*"):
        try:
            cst = child.lstat()
            if not child.is_dir() or child.is_symlink():
                total += cst.st_size
        except OSError:
            continue
    return total


def _list_legacy_files(repo: Path) -> list[dict[str, Any]]:
    auggie_dir = repo / ".auggie"
    if not auggie_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for entry in sorted(auggie_dir.iterdir(), key=lambda p: p.name):
        if entry.name in _V2_INDEX_ENTRIES:
            continue
        entries.append(
            {
                "path": entry.relative_to(repo).as_posix(),
                "bytes": _entry_bytes(entry),
            }
        )
    return entries


def index_status(
    repo_root: Path | str,
    *,
    include_legacy: bool = False,
) -> dict[str, Any]:
    """Report index state using stat-only staleness detection."""
    repo = Path(repo_root)
    db_file = db_path(repo)

    if not db_file.exists():
        added = sum(1 for _ in walk_repo(repo, check_binary=False))
        result: dict[str, Any] = {
            "exists": False,
            "model_key": None,
            "files": 0,
            "chunks": 0,
            "embeddings": 0,
            "stale": {"modified": 0, "deleted": 0, "added": added},
            "next_step": "index_update",
        }
        if include_legacy:
            result["legacy_files"] = _list_legacy_files(repo)
        return result

    conn = open_read(db_file)
    try:
        meta = _read_meta(conn)
        file_count = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        embedding_count = int(
            conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        )
        indexed: dict[str, tuple[int, int]] = {
            row[0]: (int(row[1]), int(row[2]))
            for row in conn.execute("SELECT path, mtime_ns, size FROM files").fetchall()
        }
    finally:
        conn.close()

    seen: set[str] = set()
    modified = 0
    added = 0
    for rel_path in walk_repo(repo, check_binary=False):
        full_path = repo / rel_path
        try:
            st = full_path.stat()
        except FileNotFoundError:
            continue
        except OSError:
            if rel_path in indexed:
                seen.add(rel_path)
            continue
        if rel_path not in indexed:
            if _is_binary(full_path):
                continue
            seen.add(rel_path)
            added += 1
            continue
        seen.add(rel_path)
        if (st.st_mtime_ns, st.st_size) != indexed[rel_path]:
            modified += 1

    deleted = len(set(indexed) - seen)
    next_step = "index_update" if (modified or deleted or added) else None

    result = {
        "exists": True,
        "model_key": meta.get("model_key"),
        "files": file_count,
        "chunks": chunk_count,
        "embeddings": embedding_count,
        "stale": {"modified": modified, "deleted": deleted, "added": added},
        "next_step": next_step,
    }
    if include_legacy:
        result["legacy_files"] = _list_legacy_files(repo)
    return result
