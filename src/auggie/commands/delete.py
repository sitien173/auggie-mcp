"""
Delete command for auggie.

Removes specific files from the index by path or glob pattern.
Optionally adds deleted paths to .auggieignore to prevent re-indexing.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sqlite3
from pathlib import Path

from ..db import log_history
from ._common import resolve_db_path


def _add_to_auggieignore(root: Path, paths: list[str]) -> tuple[bool, str]:
    """
    Add paths to .auggieignore file.

    Args:
        root: Repository root directory.
        paths: List of relative paths to add.

    Returns:
        Tuple of (success, ignore_file_path).
    """
    ignore_file = root / ".auggieignore"

    # Read existing patterns
    existing = set()
    if ignore_file.exists():
        try:
            existing = {
                line.strip()
                for line in ignore_file.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
        except Exception:
            pass

    # Add new paths (only if not already present)
    new_paths = [p for p in paths if p not in existing]
    if not new_paths:
        return True, str(ignore_file)

    # Append to file
    try:
        with open(ignore_file, "a") as f:
            # Add newline if file doesn't end with one
            if ignore_file.exists():
                content = ignore_file.read_text()
                if content and not content.endswith("\n"):
                    f.write("\n")
            # Add comment header if this is a new section
            if not existing:
                f.write("# auggie delete --save additions\n")
            for p in new_paths:
                f.write(f"{p}\n")
        return True, str(ignore_file)
    except Exception as e:
        return False, str(e)


def _match_paths(
    con: sqlite3.Connection,
    patterns: list[str],
    root: Path,
) -> list[tuple[int, str, str]]:
    """
    Find files in index matching the given patterns.

    Args:
        con: Database connection.
        patterns: List of path patterns (exact paths or globs).
        root: Root directory for relative path resolution.

    Returns:
        List of (file_id, db_path, display_path) tuples.
    """
    cur = con.cursor()
    cur.execute("SELECT id, path FROM files")
    rows = cur.fetchall()

    matches = []
    for file_id, db_path in rows:
        # Try to make path relative for display and matching
        try:
            rel_path = str(Path(db_path).relative_to(root))
        except ValueError:
            rel_path = db_path

        for pattern in patterns:
            # Normalize pattern
            norm_pattern = pattern.rstrip("/")

            # Match against relative path, absolute path, or just filename
            if (
                fnmatch.fnmatch(rel_path, norm_pattern)
                or fnmatch.fnmatch(db_path, norm_pattern)
                or fnmatch.fnmatch(Path(db_path).name, norm_pattern)
                or rel_path == norm_pattern
                or db_path == norm_pattern
            ):
                matches.append((file_id, db_path, rel_path))
                break  # Don't add same file multiple times

    return matches


def cmd_delete(args: argparse.Namespace) -> int:
    """
    Delete specific files from the index.

    Removes files matching the given paths or glob patterns from the index.
    Supports exact paths, relative paths, and glob patterns like '*.log'.

    Args:
        args: Parsed command-line arguments containing:
            - paths: One or more paths or glob patterns to delete
            - dry_run: Preview without actually deleting
            - save: Add deleted paths to .auggieignore
            - json: Output as JSON
            - db, profile, global_cache, repo_root: Scope options

    Returns:
        Exit code (0 for success).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    use_json = getattr(args, "json", False)
    dry_run = getattr(args, "dry_run", False)
    save = getattr(args, "save", False)

    if not db.exists():
        if use_json:
            print(json.dumps({"error": f"No database found at {db}"}))
        else:
            print(f"No database found at {db}")
        return 1

    con = sqlite3.connect(str(db))

    try:
        # Find matching files
        matches = _match_paths(con, args.paths, repo_root)

        if not matches:
            if use_json:
                print(
                    json.dumps(
                        {
                            "database": str(db),
                            "patterns": args.paths,
                            "matched": 0,
                            "deleted": 0,
                            "dry_run": dry_run,
                        }
                    )
                )
            else:
                print(f"No files matched: {', '.join(args.paths)}")
            con.close()
            return 0

        # Get chunk counts for matched files
        file_ids = [m[0] for m in matches]
        placeholders = ",".join("?" * len(file_ids))
        chunk_counts = {}
        for row in con.execute(
            f"SELECT file_id, COUNT(*) FROM chunks WHERE file_id IN ({placeholders}) GROUP BY file_id",
            file_ids,
        ):
            chunk_counts[row[0]] = row[1]

        total_chunks = sum(chunk_counts.values())

        if dry_run:
            deleted_paths = [rel for _, _, rel in matches]
            if use_json:
                output = {
                    "database": str(db),
                    "patterns": args.paths,
                    "matched": len(matches),
                    "deleted": 0,
                    "dry_run": True,
                    "would_delete": [
                        {"path": rel, "chunks": chunk_counts.get(fid, 0)} for fid, _, rel in matches
                    ],
                    "total_chunks": total_chunks,
                    "save": save,
                }
                if save:
                    output["would_add_to_auggieignore"] = deleted_paths
                print(json.dumps(output))
            else:
                print(f"Would delete {len(matches)} file(s) ({total_chunks} chunks):")
                for fid, _, rel in matches:
                    chunks = chunk_counts.get(fid, 0)
                    print(f"  {rel} ({chunks} chunks)")
                if save:
                    print("\nWould add to .auggieignore:")
                    for p in deleted_paths:
                        print(f"  {p}")
                else:
                    print("\nTip: Use --save to add these to .auggieignore and prevent re-indexing")
            con.close()
            return 0

        # Actually delete
        cur = con.cursor()
        for file_id, _, _ in matches:
            cur.execute("DELETE FROM files WHERE id = ?", (file_id,))
        con.commit()

        deleted_paths = [rel for _, _, rel in matches]

        # Log to history (AI tool integration)
        log_history(
            con,
            action="delete",
            files_affected=len(matches),
            chunks_affected=total_chunks,
            details={
                "patterns": args.paths,
                "deleted_files": deleted_paths,
                "save_to_auggieignore": save,
            },
        )

        # Increment db_version to invalidate L2 cache
        from ..cache import increment_db_version

        increment_db_version(con)

        # Handle --save: add to .auggieignore
        saved_to_ignore = False
        ignore_file_path = None
        if save:
            saved_to_ignore, ignore_file_path = _add_to_auggieignore(repo_root, deleted_paths)

        if use_json:
            output = {
                "database": str(db),
                "patterns": args.paths,
                "matched": len(matches),
                "deleted": len(matches),
                "dry_run": False,
                "deleted_files": deleted_paths,
                "total_chunks": total_chunks,
                "save": save,
            }
            if save:
                output["saved_to_auggieignore"] = saved_to_ignore
                output["auggieignore_path"] = ignore_file_path
                output["added_to_auggieignore"] = deleted_paths if saved_to_ignore else []
            print(json.dumps(output))
        else:
            print(f"Deleted {len(matches)} file(s) ({total_chunks} chunks):")
            for p in deleted_paths:
                print(f"  {p}")
            if save:
                if saved_to_ignore:
                    print("\nAdded to .auggieignore (will be excluded from future indexing):")
                    for p in deleted_paths:
                        print(f"  {p}")
                else:
                    print(f"\nWarning: Failed to update .auggieignore: {ignore_file_path}")
            else:
                print("\nTip: Use --save to add these to .auggieignore and prevent re-indexing")

    except KeyboardInterrupt:
        con.close()
        if use_json:
            print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
        else:
            print("\n\nInterrupted by user (Ctrl-C).")
        return 130

    con.close()
    return 0
