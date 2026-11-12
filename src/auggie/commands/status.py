"""
Status command for auggie.

Displays index statistics including file count, chunk count,
embedding model info, and database size.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ..db import connect, get_branch_file_counts
from ..lance_store import LanceStore, get_lance_dir
from ._common import get_current_branch, resolve_db_path


def _format_size(size_bytes: int) -> str:
    """
    Format byte size into human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "1.5 MB", "256 KB", "512 bytes").
    """
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} bytes"


def cmd_status(args: argparse.Namespace) -> int:
    """
    Show index status and statistics.

    Displays information about the current index including:
        - Database location
        - Number of indexed files
        - Number of chunks
        - Embedding model and dimensions
        - Database file size

    Args:
        args: Parsed command-line arguments containing:
            - db, profile, global_cache, repo_root: Scope options
            - json: Whether to output as JSON

    Returns:
        Exit code (0 for success).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    use_json = getattr(args, "json", False)

    if not db.exists():
        if use_json:
            print(
                json.dumps(
                    {
                        "database": str(db),
                        "status": "not_indexed",
                        "indexed": False,
                    }
                )
            )
        else:
            print(f"Database: {db}")
            print("Status: Not indexed")
        return 0

    # Detect current branch
    current_branch = get_current_branch(repo_root)

    con = connect(db, init_fts=False)  # Runs migrations, skip FTS for status
    cur = con.cursor()

    # Get per-branch file counts
    branch_counts = get_branch_file_counts(con)
    current_branch_files = branch_counts.get(current_branch, 0)

    cur.execute("SELECT COUNT(*) FROM files")
    file_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]

    cur.execute("SELECT model, dim FROM chunks LIMIT 1")
    row = cur.fetchone()
    model = row[0] if row else None
    dim = row[1] if row else None

    # Check for AST mode metadata
    ast_mode = None
    try:
        cur.execute("SELECT value FROM index_metadata WHERE key = 'ast_mode'")
        ast_row = cur.fetchone()
        if ast_row:
            ast_mode = ast_row[0] == "true"
    except sqlite3.OperationalError:
        # index_metadata table doesn't exist (older index)
        pass

    size_bytes = db.stat().st_size
    size_str = _format_size(size_bytes)
    lance_dir = get_lance_dir(db)
    lance_chunks = LanceStore(lance_dir).count() if lance_dir.exists() else 0

    con.close()

    if use_json:
        result = {
            "database": str(db),
            "status": "indexed",
            "indexed": True,
            "branch": current_branch,
            "branch_files": current_branch_files,
            "total_files": file_count,
            "branches": branch_counts,
            "chunks": chunk_count,
            "model": model,
            "dimensions": dim,
            "size_bytes": size_bytes,
            "size_human": size_str,
            "lance_path": str(lance_dir),
            "lance_chunks": lance_chunks,
        }
        if ast_mode is not None:
            result["ast_mode"] = ast_mode
        print(json.dumps(result))
    else:
        print(f"Database: {db}")
        print("Status: Indexed")
        print(f"Branch: {current_branch}")
        if len(branch_counts) > 1:
            # Show per-branch breakdown when multiple branches exist
            print(f"Files: {current_branch_files} (current branch), {file_count} total")
            print("  Branches:")
            for branch_name, count in sorted(branch_counts.items()):
                marker = " *" if branch_name == current_branch else ""
                print(f"    {branch_name}: {count} files{marker}")
        else:
            print(f"Files: {file_count}")
        print(f"Chunks: {chunk_count}")
        print(f"Model: {model}")
        print(f"Dimensions: {dim}")
        if ast_mode is not None:
            print(f"AST Mode: {'enabled' if ast_mode else 'disabled'}")
        print(f"Size: {size_str}")
        print(f"Lance: {lance_dir}")
        print(f"Lance Chunks: {lance_chunks}")

    return 0
