"""
Clean command for auggie.

Removes stale entries from the index where the source files
no longer exist on disk, prunes orphaned branch data, and
optionally compacts the database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ..db import delete_branch_files, get_indexed_branches, log_history
from ..lance_store import LanceStore, get_lance_dir
from ._common import get_git_branches, resolve_db_path


def cmd_clean(args: argparse.Namespace) -> int:
    """
    Remove stale entries, prune deleted branches, and optionally vacuum.

    Scans the index for files that no longer exist on disk and
    removes their entries. Also removes data for git branches that
    no longer exist. This keeps the index lean and prevents stale
    results from appearing in searches.

    Args:
        args: Parsed command-line arguments containing:
            - vacuum: Whether to compact the database after cleaning
            - json: Whether to output as JSON
            - db, profile, global_cache, repo_root: Scope options

    Returns:
        Exit code (0 for success).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    use_json = getattr(args, "json", False)

    if not db.exists():
        if use_json:
            print(json.dumps({"database": str(db), "exists": False, "removed": 0}))
        else:
            print(f"No database found at {db}")
        return 0

    con = sqlite3.connect(str(db))
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()
    lance_store = LanceStore(get_lance_dir(db))

    try:
        # Find files that no longer exist
        cur.execute("SELECT id, path FROM files")
        rows = cur.fetchall()

        removed = 0
        removed_paths = []
        for file_id, path in rows:
            if not Path(path).exists():
                chunk_ids = [
                    int(row[0])
                    for row in cur.execute(
                        "SELECT id FROM chunks WHERE file_id = ?",
                        (file_id,),
                    ).fetchall()
                ]
                lance_store.delete_chunk_ids(chunk_ids)
                cur.execute("DELETE FROM files WHERE id = ?", (file_id,))
                removed += 1
                removed_paths.append(path)

        con.commit()

        # Prune orphaned branches (branches that no longer exist in git)
        indexed_branches = get_indexed_branches(con)
        git_branches = get_git_branches(repo_root)
        orphaned_branches = indexed_branches - git_branches
        pruned_branches = []
        pruned_files = 0

        for branch in orphaned_branches:
            chunk_ids = [
                int(row[0])
                for row in con.execute(
                    """
                    SELECT c.id
                    FROM chunks c
                    JOIN files f ON f.id = c.file_id
                    WHERE f.branch = ?
                    """,
                    (branch,),
                ).fetchall()
            ]
            lance_store.delete_chunk_ids(chunk_ids)
            files_deleted = delete_branch_files(con, branch)
            if files_deleted > 0:
                pruned_branches.append(branch)
                pruned_files += files_deleted

        # Log to history if anything was cleaned (AI tool integration)
        if removed > 0 or pruned_files > 0:
            log_history(
                con,
                action="clean",
                files_affected=removed + pruned_files,
                chunks_affected=0,  # Chunks are cascade-deleted
                details={
                    "removed_paths": removed_paths,
                    "pruned_branches": pruned_branches,
                    "pruned_files": pruned_files,
                    "vacuumed": args.vacuum,
                },
            )

            # Increment db_version to invalidate L2 cache
            from ..cache import increment_db_version

            increment_db_version(con)

        vacuumed = False
        if args.vacuum:
            if not use_json:
                print("Running VACUUM...")
            con.execute("VACUUM")
            vacuumed = True
            if not use_json:
                print("Database compacted")

        if use_json:
            print(
                json.dumps(
                    {
                        "database": str(db),
                        "exists": True,
                        "removed": removed,
                        "removed_paths": removed_paths,
                        "pruned_branches": pruned_branches,
                        "pruned_files": pruned_files,
                        "vacuumed": vacuumed,
                    }
                )
            )
        else:
            print(f"Removed {removed} stale file entries")
            if pruned_branches:
                print(
                    f"Pruned {pruned_files} files from deleted branches: {', '.join(pruned_branches)}"
                )

    except KeyboardInterrupt:
        con.close()
        if use_json:
            print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
        else:
            print("\n\nInterrupted by user (Ctrl-C).")
            print("Clean cancelled. Partial changes may have been saved.")
        return 130  # Standard SIGINT exit code (128 + 2)

    con.close()
    return 0
