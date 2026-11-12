"""
Reset command for auggie.

Removes indexed data for the current branch (default) or the entire
database (with --all flag).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..lance_store import LanceStore, get_lance_dir
from ._common import get_current_branch, resolve_db_path


def cmd_reset(args: argparse.Namespace) -> int:
    """
    Remove indexed data for the current branch or entire database.

    By default, only removes files indexed on the current git branch,
    preserving other branches' data. Use --all to delete the entire
    database.

    Args:
        args: Parsed command-line arguments containing:
            - force: Skip confirmation prompt
            - all: Delete entire database instead of just current branch
            - json: Whether to output as JSON
            - db, profile, global_cache, repo_root: Scope options

    Returns:
        Exit code (0 for success, 1 if aborted).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    use_json = getattr(args, "json", False)
    reset_all = getattr(args, "all", False)

    if not db.exists():
        if use_json:
            print(json.dumps({"database": str(db), "existed": False, "removed": False}))
        else:
            print(f"No database found at {db}")
        return 0

    # Get current branch for branch-scoped reset
    branch = get_current_branch(repo_root)

    # Determine what we're resetting
    if reset_all:
        action_desc = f"Delete entire database {db}"
    else:
        action_desc = f"Clear branch '{branch}' from {db}"

    if not args.force:
        import sys

        if not sys.stdin.isatty():
            if use_json:
                print(
                    json.dumps(
                        {
                            "error": "Non-interactive mode requires --force (-f) flag",
                            "database": str(db),
                        }
                    )
                )
            else:
                print("Non-interactive mode requires --force (-f) flag.")
            return 1
        confirm = input(f"{action_desc}? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            if use_json:
                print(
                    json.dumps(
                        {
                            "database": str(db),
                            "existed": True,
                            "removed": False,
                            "aborted": True,
                        }
                    )
                )
            else:
                print("Aborted.")
            return 1

    if reset_all:
        # Delete entire database (original behavior)
        db.unlink()
        lance_dir = get_lance_dir(db)
        lance_removed = False
        if lance_dir.exists():
            shutil.rmtree(lance_dir)
            lance_removed = True

        # Also delete cache file if it exists
        from ..cache import get_cache_path

        cache_path = get_cache_path(db)
        cache_removed = False
        if cache_path.exists():
            cache_path.unlink()
            cache_removed = True

        # Clean up empty parent directories
        parent = db.parent
        parent_removed = False
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent_removed = True
        except OSError:
            pass

        if use_json:
            print(
                json.dumps(
                    {
                        "database": str(db),
                        "existed": True,
                        "removed": True,
                        "reset_all": True,
                        "lance_removed": lance_removed,
                        "cache_removed": cache_removed,
                        "parent_removed": parent_removed,
                    }
                )
            )
        else:
            print(f"Removed {db}")
            if lance_removed:
                print(f"Removed Lance store at {lance_dir}")
            if cache_removed:
                print(f"Removed cache at {cache_path}")
    else:
        # Branch-scoped reset: only delete files for current branch
        from ..db import connect, delete_branch_files

        con = connect(db)
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
        files_deleted = delete_branch_files(con, branch)
        con.close()
        LanceStore(get_lance_dir(db)).delete_chunk_ids(chunk_ids)

        if use_json:
            print(
                json.dumps(
                    {
                        "database": str(db),
                        "branch": branch,
                        "files_deleted": files_deleted,
                        "lance_chunks_deleted": len(chunk_ids),
                        "reset_all": False,
                    }
                )
            )
        else:
            print(f"Cleared branch '{branch}' ({files_deleted} files)")
            if chunk_ids:
                print(f"Removed {len(chunk_ids)} Lance vectors")
            print("Use --all to delete entire database")

    return 0
