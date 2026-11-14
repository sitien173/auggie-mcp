"""Semantic search over the SQLite index."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from auggie_mcp.config import Config
from auggie_mcp.config import load as load_config
from auggie_mcp.embed import Embedder, VoyageEmbedder
from auggie_mcp.output_limits import MAX_OUTPUT_BYTES, serialize_json, truncate_output
from auggie_mcp.store import StoreError, db_path, decode_vector, open_read
from auggie_mcp.vectors import normalize_rows, normalize_vector


class IndexMissingError(RuntimeError):
    """Raised when a repository has no populated semantic index."""


class ModelMismatchError(RuntimeError):
    """Raised when the index and active embedder use different models."""


@dataclass
class _Index:
    generation: int
    last_indexed: str | None
    rows: list[tuple[str, int, int, str, int | None, int | None]]
    matrix: np.ndarray


_CACHE: OrderedDict[str, _Index] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 2


def clear_cache() -> None:
    """Clear the process-local semantic matrix cache."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _read_index(database: Path, model_key: str) -> _Index:
    try:
        conn = open_read(database)
    except StoreError as exc:
        raise IndexMissingError(f"No semantic index at {database}") from exc
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        stored_model = meta.get("model_key")
        if stored_model != model_key:
            raise ModelMismatchError(
                f"Index model {stored_model!r} does not match configured model {model_key!r}"
            )
        generation = int(meta.get("generation", "0"))
        records = conn.execute(
            "SELECT files.path, chunks.start_line, chunks.end_line, chunks.text, "
            "files.mtime_ns, files.size, embeddings.vector "
            "FROM chunks JOIN files ON files.id = chunks.file_id "
            "JOIN embeddings ON embeddings.content_hash = chunks.content_hash "
            "AND embeddings.model_key = ? ORDER BY files.path, chunks.ordinal",
            (model_key,),
        ).fetchall()
        if not records:
            raise IndexMissingError(f"Semantic index at {database} is empty")
        rows = [(str(p), int(s), int(e), str(t or ""), m, z) for p, s, e, t, m, z, _ in records]
        matrix = normalize_rows(np.vstack([decode_vector(bytes(row[6])) for row in records]))
        return _Index(generation, _valid_timestamp(meta.get("last_indexed")), rows, matrix)
    finally:
        conn.close()


def _valid_timestamp(value: str | None) -> str | None:
    """Drop a malformed last_indexed value so metadata stays bounded."""
    if value is None or len(value) > 64:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def _get_index(repo: Path, model_key: str) -> _Index:
    key = str(repo)
    with _CACHE_LOCK:
        database = db_path(repo)
        try:
            conn = open_read(database)
        except StoreError as exc:
            raise IndexMissingError(f"No semantic index at {database}") from exc
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            if meta.get("model_key") != model_key:
                raise ModelMismatchError(
                    f"Index model {meta.get('model_key')!r} does not match configured model {model_key!r}"
                )
            generation = int(meta.get("generation", "0"))
        finally:
            conn.close()
        cached = _CACHE.get(key)
        if cached is None or cached.generation != generation:
            cached = _read_index(database, model_key)
            _CACHE[key] = cached
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
        return cached


def _resolve_filter(repo: Path, path: str | Path) -> tuple[str, Path] | None:
    target = Path(path)
    if not target.is_absolute():
        target = repo / target
    target = target.resolve()
    if target != repo and repo not in target.parents:
        return None
    relative = target.relative_to(repo).as_posix()
    return ("" if relative == "." else relative.rstrip("/") + "/", target)


def semantic_search(
    repository_path: Path | str,
    query: str,
    *,
    path: str | Path = ".",
    top_k: int = 10,
    context_lines: int = 0,
    config: Config | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Return cosine-ranked, repository-relative semantic hits from SQLite."""
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    if not 0 <= context_lines <= 20:
        raise ValueError("context_lines must be between 0 and 20")
    repo = Path(repository_path).resolve()
    path_filter = _resolve_filter(repo, path)
    if path_filter is None:
        return {
            "hits": [],
            "index": {"generation": 0, "last_indexed": None},
            "truncated": False,
            "archive_path": None,
            "hits_omitted": 0,
        }
    cfg = config or load_config()
    active_embedder = embedder or VoyageEmbedder(cfg)
    index = _get_index(repo, active_embedder.model_key)
    query_method = getattr(active_embedder, "embed_query", active_embedder.embed)
    query_vector = normalize_vector(query_method([query])[0])
    scores = index.matrix @ query_vector
    prefix, _ = path_filter
    candidates = []
    for row, score in zip(index.rows, scores, strict=True):
        rel_path, start, end, chunk_text, indexed_mtime, indexed_size = row
        if prefix and not (rel_path == prefix[:-1] or rel_path.startswith(prefix)):
            continue
        candidates.append((float(score), rel_path, start, end, chunk_text, indexed_mtime, indexed_size))
    candidates.sort(key=lambda item: item[0], reverse=True)
    hits = []
    for score, rel_path, start, end, chunk_text, indexed_mtime, indexed_size in candidates[:top_k]:
        file_path = repo / rel_path
        try:
            stat = file_path.stat()
            stale = stat.st_mtime_ns != indexed_mtime or stat.st_size != indexed_size
        except OSError:
            stale = True
        text = ""
        if not stale:
            if context_lines:
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    text = "\n".join(lines[max(0, start - 1 - context_lines): min(len(lines), end + context_lines)])
                except OSError:
                    stale = True
            if not context_lines:
                text = chunk_text.removeprefix(f"{rel_path}\n")
        if stale:
            text = ""
        hits.append({
            "path": rel_path,
            "start_line": start,
            "end_line": end,
            "score": float(score),
            "stale": stale,
            "text": text,
        })
    result = {
        "hits": hits,
        "index": {"generation": index.generation, "last_indexed": index.last_indexed},
        "truncated": False,
        "archive_path": None,
        "hits_omitted": 0,
    }
    full_text = serialize_json(result)
    if len(full_text.encode("utf-8")) < MAX_OUTPUT_BYTES:
        return result

    _, archive_path = truncate_output(full_text, repo)
    result["truncated"] = True
    result["archive_path"] = archive_path
    for hit in reversed(hits):
        lines = cast(str, hit["text"]).splitlines(keepends=True)
        while lines:
            lines.pop()
            hit["text"] = "".join(lines)
            serialized = serialize_json(result)
            if len(serialized.encode("utf-8")) < MAX_OUTPUT_BYTES:
                return result
    while hits:
        hits.pop()
        result["hits_omitted"] += 1
        serialized = serialize_json(result)
        if len(serialized.encode("utf-8")) < MAX_OUTPUT_BYTES:
            return result
    return result
