"""
Log command for auggie.

Shows change history for the index database. Useful for AI tools
to understand what changed after running refresh operations.

AI TOOL HINT: Use `auggie log --json` after `auggie query --refresh` to
see what files were indexed/deleted. This helps you understand what
changed in the codebase since your last check.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from ._common import resolve_db_path


def _parse_datetime(value: str) -> datetime | None:
    """
    Parse datetime from ISO8601 format.

    Supports:
        - Full ISO8601: 2024-01-15T10:30:00
        - Date only: 2024-01-15
        - Date + hour: 2024-01-15T10
        - With timezone: 2024-01-15T10:30:00Z or +00:00

    Args:
        value: Datetime string in ISO8601 format.

    Returns:
        Parsed datetime or None if invalid.
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]

    # Handle trailing Z (UTC indicator)
    if value.endswith("Z"):
        value = value[:-1]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def _format_timestamp(ts: str) -> str:
    """Format timestamp for human display."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def _format_action(action: str) -> str:
    """Format action with emoji for terminal display."""
    icons = {
        "index": "+",
        "delete": "-",
        "clean": "~",
        "refresh": "r",
        "reindex": "R",
    }
    return f"[{icons.get(action, '?')}] {action}"


def cmd_log(args: argparse.Namespace) -> int:
    """
    Show index change history.

    AI TOOL HINT: Run this command after `auggie query --refresh` or
    `auggie index .` to see what changed. Use `--since` to filter
    to recent changes only.

    Example workflow for AI tools:
        1. auggie query "search" --refresh --json  # Search + refresh
        2. auggie log --limit 5 --json             # See what changed

    Args:
        args: Parsed command-line arguments containing:
            - since: ISO8601 datetime filter (show entries after this)
            - until: ISO8601 datetime filter (show entries before this)
            - action: Filter by action type
            - limit: Max entries to return (default: 50)
            - offset: Skip first N entries for pagination
            - json: Output as JSON (default: True for this command)

    Returns:
        Exit code (0 for success).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    use_json = getattr(args, "json", True)  # Default to JSON
    limit = getattr(args, "limit", 50)
    offset = getattr(args, "offset", 0)
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    action_filter = getattr(args, "action", None)

    if not db.exists():
        if use_json:
            print(json.dumps({"error": f"No database found at {db}"}))
        else:
            print(f"No database found at {db}")
        return 1

    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row

    try:
        # Check if history table exists (for older databases)
        table_check = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='index_history'"
        ).fetchone()

        if not table_check:
            if use_json:
                print(
                    json.dumps(
                        {
                            "error": "History table not found. Run 'auggie reindex .' to upgrade database.",
                            "hint": "AI TOOL: The database schema is outdated. Reindex to enable history tracking.",
                        }
                    )
                )
            else:
                print("History table not found.")
                print(
                    "Run 'auggie reindex .' to upgrade database schema and enable history tracking."
                )
            con.close()
            return 1

        # Build query with filters
        query = "SELECT * FROM index_history WHERE 1=1"
        params: list = []

        if since:
            dt = _parse_datetime(since)
            if dt:
                query += " AND timestamp >= ?"
                params.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                if use_json:
                    print(
                        json.dumps(
                            {
                                "error": f"Invalid datetime format: {since}",
                                "expected": "ISO8601 (e.g., 2024-01-15T10:30:00)",
                            }
                        )
                    )
                else:
                    print(f"Invalid datetime format: {since}")
                    print("Expected ISO8601 format, e.g.: 2024-01-15T10:30:00")
                con.close()
                return 1

        if until:
            dt = _parse_datetime(until)
            if dt:
                query += " AND timestamp <= ?"
                params.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                if use_json:
                    print(
                        json.dumps(
                            {
                                "error": f"Invalid datetime format: {until}",
                                "expected": "ISO8601 (e.g., 2024-01-15T10:30:00)",
                            }
                        )
                    )
                else:
                    print(f"Invalid datetime format: {until}")
                con.close()
                return 1

        if action_filter:
            query += " AND action = ?"
            params.append(action_filter)

        # Get total count for pagination
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = con.execute(count_query, params).fetchone()[0]

        # Add ordering and pagination
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = con.execute(query, params).fetchall()

        entries = []
        for row in rows:
            details = None
            if row["details"]:
                try:
                    details = json.loads(row["details"])
                except json.JSONDecodeError:
                    details = row["details"]

            entries.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "action": row["action"],
                    "files_affected": row["files_affected"],
                    "chunks_affected": row["chunks_affected"],
                    "details": details,
                }
            )

        # Calculate pagination info
        has_more = (offset + len(entries)) < total
        has_prev = offset > 0

        if use_json:
            output = {
                "database": str(db),
                "entries": entries,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "returned": len(entries),
                    "has_more": has_more,
                    "has_prev": has_prev,
                },
                "filters": {
                    "since": since,
                    "until": until,
                    "action": action_filter,
                },
                "hint": "AI TOOL: Use --since with ISO8601 datetime to filter recent changes. "
                "Combine with 'auggie query --refresh' to track what changed.",
            }
            print(json.dumps(output))
        else:
            if not entries:
                print("No history entries found.")
                if since or until or action_filter:
                    print("(filters may have excluded all entries)")
            else:
                # Header
                print(f"Index History ({len(entries)} of {total} entries)")
                if since or until or action_filter:
                    filters = []
                    if since:
                        filters.append(f"since={since}")
                    if until:
                        filters.append(f"until={until}")
                    if action_filter:
                        filters.append(f"action={action_filter}")
                    print(f"Filters: {', '.join(filters)}")
                print()

                # Table
                print(
                    f"{'ID':>5}  {'Timestamp':<20}  {'Action':<12}  {'Files':>6}  {'Chunks':>7}  Details"
                )
                print(f"{'-' * 5}  {'-' * 20}  {'-' * 12}  {'-' * 6}  {'-' * 7}  {'-' * 30}")

                for entry in entries:
                    details_str = ""
                    if entry["details"]:
                        if isinstance(entry["details"], dict):
                            # Show key summary
                            keys = list(entry["details"].keys())[:3]
                            details_str = ", ".join(keys)
                            if len(entry["details"]) > 3:
                                details_str += f"... (+{len(entry['details']) - 3})"
                        else:
                            details_str = str(entry["details"])[:30]

                    print(
                        f"{entry['id']:>5}  "
                        f"{_format_timestamp(entry['timestamp']):<20}  "
                        f"{_format_action(entry['action']):<12}  "
                        f"{entry['files_affected']:>6}  "
                        f"{entry['chunks_affected']:>7}  "
                        f"{details_str}"
                    )

                # Pagination hint
                print()
                if has_more:
                    next_offset = offset + limit
                    print(f"More entries available. Use --offset {next_offset} to see next page.")
                if has_prev:
                    prev_offset = max(0, offset - limit)
                    print(f"Previous page: --offset {prev_offset}")

                # AI tool hint in non-JSON mode too
                print()
                print("Tip: Use 'auggie log --json' for structured output (AI tools).")
                print("     Use 'auggie query --refresh' + 'auggie log --limit 5' to track changes.")

    except KeyboardInterrupt:
        con.close()
        if use_json:
            print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
        else:
            print("\n\nInterrupted by user (Ctrl-C).")
        return 130

    con.close()
    return 0
