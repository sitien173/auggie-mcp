"""MCP tool functions for auggie."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import db
from .commands._common import find_git_root, get_current_branch, resolve_db_path
from .commands.context import _expand_results
from .commands.status import _format_size
from .lance_store import LanceStore, get_lance_dir
from .search import query as query_db

def _resolve_root(repo_path: str = ".") -> Path:
    path = Path(repo_path).expanduser().resolve()
    return find_git_root(path) or path


def _resolve_db(root: Path) -> Path:
    return resolve_db_path(None, None, False, root)


def _missing_index_error(repo_path: str, db_path: Path) -> dict[str, Any]:
    return {
        "error": f"No auggie index found at {db_path}",
        "hint": f"run auggie index {repo_path}",
    }


def _missing_embedding_error() -> dict[str, str]:
    return {
        "error": "VOYAGE_API_KEY is not set",
        "hint": "Set VOYAGE_API_KEY for Voyage embeddings.",
    }


def _bad_repo_path_error(repo_path: str) -> dict[str, str]:
    return {"error": f"Invalid repo_path: {repo_path}"}


def _search_exception_error(exc: Exception) -> dict[str, str]:
    message = str(exc)
    if "VOYAGE_API_KEY" in message or "api key" in message.lower():
        return _missing_embedding_error()
    return {"error": f"Search failed: {message}"}


def _search_inputs(
    repo_path: str,
) -> tuple[Path, Path, str | None, int | None, dict[str, Any] | None]:
    try:
        root = _resolve_root(repo_path)
    except (OSError, RuntimeError, ValueError):
        return Path(), Path(), None, None, _bad_repo_path_error(repo_path)

    db_path = _resolve_db(root)
    if not db_path.exists():
        return root, db_path, None, None, _missing_index_error(repo_path, db_path)

    try:
        con = db.connect(db_path, init_fts=False)
        try:
            model, dim = _indexed_model(con)
        finally:
            con.close()
    except Exception:
        return root, db_path, None, None, {"error": f"Cannot read auggie index at {db_path}"}

    if not os.environ.get("VOYAGE_API_KEY"):
        return root, db_path, model, dim, _missing_embedding_error()

    return root, db_path, model, dim, None


def _indexed_model(con) -> tuple[str | None, int | None]:
    row = con.execute("SELECT model, dim FROM chunks LIMIT 1").fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


def _hit_to_dict(hit, max_text: int) -> dict[str, Any]:
    confidence_details = hit.confidence_details
    if confidence_details is not None and hasattr(confidence_details, "to_dict"):
        confidence_details = confidence_details.to_dict()

    return {
        "score": round(float(hit.score), 6),
        "path": hit.path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "text": hit.text[:max_text],
        "chunk_id": hit.chunk_id,
        "chunk_index": hit.chunk_index,
        "confidence": hit.confidence,
        "confidence_details": confidence_details,
    }


def auggie_query(
    query: str,
    repo_path: str = ".",
    top_k: int = 10,
    mode: str | None = None,
) -> dict[str, Any]:
    """Use for ranked semantic code search hits without surrounding context."""
    root, db_path, model, dim, error = _search_inputs(repo_path)
    if error is not None:
        return error

    try:
        hits, fts_available = query_db(
            db_path=db_path,
            q=query,
            top_k=top_k,
            model=model,
            dimensions=dim,
            mode=mode,
            no_cache=True,
            branch=get_current_branch(root),
        )
    except Exception as exc:
        return _search_exception_error(exc)

    return {
        "hits": [_hit_to_dict(hit, 4000) for hit in hits],
        "fts_available": fts_available,
    }


def auggie_context(
    query: str,
    repo_path: str = ".",
    top_k: int = 10,
    context_lines: int = 2,
) -> dict[str, Any]:
    """Use for semantic code hits with surrounding source lines for review."""
    root, db_path, model, dim, error = _search_inputs(repo_path)
    if error is not None:
        return error

    try:
        hits, fts_available = query_db(
            db_path=db_path,
            q=query,
            top_k=top_k,
            model=model,
            dimensions=dim,
            mode="semantic",
            no_cache=True,
            branch=get_current_branch(root),
        )
    except Exception as exc:
        return _search_exception_error(exc)

    expanded = _expand_results(hits[:top_k], context_lines)
    payload_hits = []
    for hit, context in zip(hits[:top_k], expanded):
        payload = _hit_to_dict(hit, 8000)
        payload.update(
            {
                "start_line": context["start_line"],
                "end_line": context["end_line"],
                "text": str(context["text"])[:8000],
                "match_source": context["match_source"],
            }
        )
        payload_hits.append(payload)

    return {
        "hits": payload_hits,
        "fts_available": fts_available,
    }


def auggie_status(repo_path: str = ".") -> dict[str, Any]:
    """Return status for the auggie index in a repository."""
    root = _resolve_root(repo_path)
    db_path = _resolve_db(root)

    if not db_path.exists():
        return _missing_index_error(repo_path, db_path)

    con = db.connect(db_path, init_fts=False)
    try:
        branches = db.get_branch_file_counts(con)
        total_files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        row = con.execute("SELECT model, dim FROM chunks LIMIT 1").fetchone()
    finally:
        con.close()

    model = row[0] if row else None
    dim = row[1] if row else None
    size_bytes = db_path.stat().st_size
    lance_dir = get_lance_dir(db_path)
    lance_chunks = LanceStore(lance_dir).count() if lance_dir.exists() else 0

    return {
        "exists": True,
        "db_path": str(db_path),
        "total_files": total_files,
        "chunks": chunks,
        "model": model,
        "dim": dim,
        "branches": branches,
        "size_bytes": size_bytes,
        "size_human": _format_size(size_bytes),
        "lance_path": str(lance_dir),
        "lance_chunks": lance_chunks,
    }


def register_tools(server) -> None:
    """Register the semantic index tools on an MCP server."""
    for fn in (auggie_query, auggie_context, auggie_status):
        server.tool()(fn)
