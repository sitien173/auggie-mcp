"""
Chunk command for auggie.

Retrieves chunks by reference (path:index) or ID, with optional
context (neighboring chunks). Useful for expanding context after
a query finds something interesting.

Examples:
    auggie chunk "src/auth.py:2"              # Get chunk by ref
    auggie chunk "src/auth.py:2" --before 1   # + 1 chunk before
    auggie chunk "src/auth.py:2" --after 1    # + 1 chunk after
    auggie chunk "src/auth.py:2" --context 1  # + 1 before AND after
    auggie chunk 42                           # Also works with raw ID
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from ._common import detect_language, resolve_db_path


def _parse_chunk_ref(ref: str) -> tuple[str | None, int | None, int | None]:
    """
    Parse a chunk reference into its components.

    Supports two formats:
    - "path/to/file.py:2" - relative path with chunk index
    - "42" - raw chunk ID

    Args:
        ref: Chunk reference string.

    Returns:
        Tuple of (relative_path, chunk_index, chunk_id).
        One of (path, index) or (None, None, id) will be set.
    """
    # Try raw ID first (just a number)
    if ref.isdigit():
        return None, None, int(ref)

    # Try path:index format
    match = re.match(r"^(.+):(\d+)$", ref)
    if match:
        return match.group(1), int(match.group(2)), None

    # Fallback: might be a path without index
    return ref, None, None


def _chunk_to_dict(row: tuple, repo_root: Path) -> dict:
    """Convert a database row to a chunk dictionary."""
    chunk_id, chunk_index, path, start_line, end_line, text = row

    # Calculate relative path
    try:
        rel_path = str(Path(path).relative_to(repo_root))
    except ValueError:
        rel_path = path

    return {
        "chunk_ref": f"{rel_path}:{chunk_index}",
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "path": path,
        "relative_path": rel_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": detect_language(path),
        "text": text,
    }


def expand_chunk_text_context(
    path: str,
    start_line: int,
    end_line: int,
    context_lines: int,
) -> tuple[int, int, str]:
    """
    Expand a line span with file context lines.

    Returns the expanded start/end lines and the extracted text.
    """
    if context_lines <= 0:
        context_lines = 0

    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_idx = max(0, start_line - 1 - context_lines)
    end_idx = min(len(lines), end_line + context_lines)
    text = "\n".join(lines[start_idx:end_idx])
    return start_idx + 1, end_idx, text


def cmd_chunk(args: argparse.Namespace) -> int:
    """
    Retrieve a chunk by reference or ID with optional context.

    Args:
        args: Parsed command-line arguments containing:
            - ref: Chunk reference (path:index) or raw ID
            - before: Number of chunks to include before
            - after: Number of chunks to include after
            - context: Shorthand for --before N --after N
            - db, profile, global_cache, repo_root: Scope options

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)

    if not db.exists():
        print(json.dumps({"error": f"Database not found at {db}"}))
        return 1

    # Parse the reference
    rel_path, chunk_index, chunk_id = _parse_chunk_ref(args.ref)

    # Calculate before/after counts
    before_count = args.before or 0
    after_count = args.after or 0
    if args.context:
        before_count = max(before_count, args.context)
        after_count = max(after_count, args.context)

    con = sqlite3.connect(str(db))

    try:
        # Find the requested chunk
        if chunk_id is not None:
            # Lookup by raw ID
            row = con.execute(
                """SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
                   FROM chunks c
                   JOIN files f ON f.id = c.file_id
                   WHERE c.id = ?""",
                (chunk_id,),
            ).fetchone()
        elif rel_path and chunk_index is not None:
            # Lookup by path:index - need to match relative or absolute path
            row = con.execute(
                """SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
                   FROM chunks c
                   JOIN files f ON f.id = c.file_id
                   WHERE c.chunk_index = ? AND (f.path LIKE ? OR f.path = ?)""",
                (chunk_index, f"%/{rel_path}", str(repo_root / rel_path)),
            ).fetchone()
        else:
            print(
                json.dumps(
                    {
                        "error": f"Invalid chunk reference: {args.ref}. "
                        "Use 'path/file.py:N' or raw chunk ID."
                    }
                )
            )
            return 1

        if row is None:
            print(json.dumps({"error": f"Chunk not found: {args.ref}"}))
            return 1

        # Convert to dict and get file info for context
        requested = _chunk_to_dict(row, repo_root)
        file_path = row[2]  # path from the row
        file_chunk_index = row[1]  # chunk_index

        # Get neighboring chunks
        before_chunks = []
        after_chunks = []

        if before_count > 0:
            before_rows = con.execute(
                """SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
                   FROM chunks c
                   JOIN files f ON f.id = c.file_id
                   WHERE f.path = ? AND c.chunk_index < ?
                   ORDER BY c.chunk_index DESC
                   LIMIT ?""",
                (file_path, file_chunk_index, before_count),
            ).fetchall()
            # Reverse to get ascending order
            before_chunks = [_chunk_to_dict(r, repo_root) for r in reversed(before_rows)]

        if after_count > 0:
            after_rows = con.execute(
                """SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
                   FROM chunks c
                   JOIN files f ON f.id = c.file_id
                   WHERE f.path = ? AND c.chunk_index > ?
                   ORDER BY c.chunk_index ASC
                   LIMIT ?""",
                (file_path, file_chunk_index, after_count),
            ).fetchall()
            after_chunks = [_chunk_to_dict(r, repo_root) for r in after_rows]

    finally:
        con.close()

    # Build output
    output = {
        "requested": requested,
        "before": before_chunks,
        "after": after_chunks,
    }

    print(json.dumps(output, indent=2))
    return 0
