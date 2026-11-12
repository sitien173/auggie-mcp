"""Semantic context command for auggie."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from ..search import query as query_db
from ._common import get_current_branch, require_embedding_config, resolve_db_path
from .chunk import expand_chunk_text_context


def _expand_results(results, context_lines: int) -> list[dict[str, Any]]:
    """Expand line context for final output."""
    output: list[dict[str, Any]] = []
    for item in results:
        start_line = item.start_line
        end_line = item.end_line
        text = item.text
        if context_lines > 0:
            try:
                start_line, end_line, text = expand_chunk_text_context(
                    item.path,
                    item.start_line,
                    item.end_line,
                    context_lines,
                )
            except OSError:
                pass

        output.append(
            {
                "chunk_id": item.chunk_id,
                "path": item.path,
                "start_line": start_line,
                "end_line": end_line,
                "score": round(float(item.score), 6),
                "match_source": "semantic",
                "text": text,
            }
        )
    return output


def _format_text_results(results: list[dict[str, Any]]) -> None:
    """Emit human-readable output."""
    for item in results:
        print(
            f"{item['path']}:{item['start_line']}-{item['end_line']}  "
            f"score={item['score']:.6f}  {item['match_source']}"
        )
        snippet = str(item["text"]).strip().replace("\n", "\\n")
        print(f"  {snippet}")


def cmd_context(args: argparse.Namespace) -> int:
    """Run semantic context search."""
    use_json = getattr(args, "json", False)
    if not require_embedding_config():
        if use_json:
            print(json.dumps({"error": "VOYAGE_API_KEY is not set"}))
        return 1

    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    if not db.exists():
        if use_json:
            print(json.dumps({"error": f"Database not found at {db}"}))
        else:
            print(f"Error: Database not found at {db}", file=sys.stderr)
        return 1

    branch = get_current_branch(repo_root)
    # Resolve the embedding model/dimensions from the index itself, not CLI
    # defaults, so the query embedding matches the indexed vectors (and so the
    # query-embedding cache row has a non-null model).
    index_model, index_dim = args.model, args.dimensions
    try:
        row = sqlite3.connect(str(db)).execute(
            "SELECT model, dim FROM chunks LIMIT 1"
        ).fetchone()
        if row:
            index_model, index_dim = row[0], row[1]
    except sqlite3.Error:
        pass
    semantic_hits, _fts_available = query_db(
        db_path=db,
        q=args.query,
        top_k=args.top,
        model=index_model,
        dimensions=index_dim,
        mode="semantic",
        branch=branch,
    )
    results = _expand_results(semantic_hits[: args.top], getattr(args, "context", 0))

    if use_json:
        payload: dict[str, Any] = {
            "query": args.query,
            "branch": branch,
            "results": results,
        }
        print(json.dumps(payload, indent=2))
    else:
        _format_text_results(results)

    return 0
