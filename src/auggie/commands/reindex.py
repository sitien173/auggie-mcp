"""
Reindex command for auggie.

Force rebuilds the index from scratch by removing the existing
database and re-indexing the entire directory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from ..index_lock import IndexLock, IndexLockError
from ..indexer import index_path
from ..models import get_optimal_chunk_lines, get_optimal_overlap
from ._common import find_git_root, require_embedding_config, resolve_db_path


def cmd_reindex(args: argparse.Namespace) -> int:
    """
    Force rebuild the index from scratch.

    Removes any existing index database and performs a fresh index
    of the target directory. Useful when changing embedding models
    or chunk sizes, or when the index becomes corrupted.

    Args:
        args: Parsed command-line arguments containing:
            - path: Directory to index (default: current directory)
            - db, profile, global_cache, repo_root: Scope options
            - model: Voyage embedding model name
            - dimensions: Embedding dimensions
            - chunk_lines: Lines per chunk (None = model-specific default)
            - overlap: Overlapping lines between chunks (None = model-specific default)
            - max_bytes: Maximum file size to index
            - exclude: Additional glob patterns to exclude
            - include: Glob patterns to include (override excludes)
            - json: Whether to output as JSON

    Returns:
        Exit code (0 for success, 1 for configuration error).
    """
    use_json = getattr(args, "json", False)

    if not require_embedding_config():
        if use_json:
            print(
                json.dumps(
                    {
                        "error": "VOYAGE_API_KEY is not set"
                    }
                )
            )
        return 1

    root = Path(args.path).resolve()
    # If root is a file, use its parent directory for searching
    search_path = root.parent if root.is_file() else root

    # Determine repo_root for .auggie location
    if args.repo_root:
        # Explicit --repo-root takes priority
        repo_root = args.repo_root.resolve()
    else:
        # Try to find git repository root
        git_root = find_git_root(search_path)
        if git_root:
            repo_root = git_root
        else:
            # Not in a git repo - use the indexed path
            repo_root = search_path

    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)

    # Use model-specific optimal settings if not explicitly specified
    chunk_lines = args.chunk_lines
    if chunk_lines is None:
        chunk_lines = get_optimal_chunk_lines(args.model)

    overlap = args.overlap
    if overlap is None:
        overlap = get_optimal_overlap(args.model)

    # Hold the shared index lock before deleting anything so a busy index is
    # never removed.
    lock = IndexLock(db.parent / "index.lock")
    try:
        lock.acquire()
    except IndexLockError as e:
        if use_json:
            print(json.dumps({"error": str(e), "error_code": "INDEX_BUSY"}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        # Remove existing database and cache
        removed_existing = False
        cache_removed = False
        if db.exists():
            db.unlink()
            removed_existing = True

            # Also delete cache file if it exists
            from ..cache import get_cache_path

            cache_path = get_cache_path(db)
            if cache_path.exists():
                cache_path.unlink()
                cache_removed = True

            if not use_json:
                print(f"Removed existing index at {db}")
                if cache_removed:
                    print(f"Removed cache at {cache_path}")

        # Also wipe the Lance vector store so it is rebuilt from scratch; otherwise
        # stale vectors with old chunk_ids remain orphaned after the SQLite reset.
        import shutil

        from ..lance_store import get_lance_dir

        lance_dir = get_lance_dir(db)
        if lance_dir.exists():
            shutil.rmtree(lance_dir, ignore_errors=True)

        # Reindex
        try:
            stats = index_path(
                root=root,
                db_path=db,
                model=args.model,
                dimensions=args.dimensions,
                chunk_lines=chunk_lines,
                overlap=overlap,
                max_bytes=args.max_bytes,
                exclude=args.exclude,
                include=args.include,
                ast=getattr(args, "ast", False),
            )
        except KeyboardInterrupt:
            if use_json:
                print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
            else:
                print("\n\nInterrupted by user (Ctrl-C).")
                print("Partial progress may have been saved to the index.")
                print("Run 'auggie reindex .' again to rebuild from scratch.")
            return 130  # Standard SIGINT exit code (128 + 2)

        # Log reindex action to history (overrides the "index" entry from index_path)
        # We update the most recent entry to be "reindex" for clarity
        # Note: db_version is already incremented by index_path, no need to increment again
        con = sqlite3.connect(str(db))
        try:
            con.execute(
                "UPDATE index_history SET action = 'reindex' WHERE id = (SELECT MAX(id) FROM index_history)"
            )
            con.commit()
        except sqlite3.OperationalError:
            pass  # History table might not exist in older databases
        finally:
            con.close()
    finally:
        lock.release()

    if use_json:
        result = {
            "database": str(db),
            "removed_existing": removed_existing,
            "cache_removed": cache_removed,
            "model": stats.model,
            "files_indexed": stats.files_indexed,
            "files_skipped": stats.files_skipped,
            "files_scanned": stats.files_scanned,
            "chunks_total": stats.chunks_total,
            "chunks_embedded": stats.chunks_embedded,
            "chunks_reused": stats.chunks_reused,
            "chunks_reused_global": stats.chunks_reused_global,
            "chunks_reused_local": stats.chunks_reused_local,
            "tokens_saved_estimate": stats.tokens_saved_estimate,
            "dedup_ratio": stats.dedup_ratio,
            "ast_mode": stats.ast_mode,
        }
        if stats.ast_hint:
            result["ast_hint"] = stats.ast_hint
        print(json.dumps(result))
    else:
        print(f"Reindexed into {db}")
        print(f"  Model: {stats.model}")
        print(f"  Files: {stats.files_indexed} indexed, {stats.files_skipped} skipped")
        if stats.chunks_total > 0:
            msg = f"  Chunks: {stats.chunks_total} total"
            if stats.chunks_reused > 0:
                msg += (
                    f" ({stats.chunks_reused} reused, ~{stats.tokens_saved_estimate} tokens saved)"
                )
            else:
                msg += f" ({stats.chunks_embedded} embedded)"
            print(msg)

    return 0
