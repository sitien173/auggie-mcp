"""
Command-line interface for auggie.

Semantic grep for codebases — search by meaning, not just keywords.
Supports hybrid (semantic + keyword), pure semantic, or FTS5 fulltext modes.

All commands output JSON by default for AI/machine use.
Use --no-json for human-readable text output.

Usage:
    auggie index .                           # Index current directory
    auggie query "search text"               # Search (hybrid mode)
    auggie query "text" --mode semantic      # Pure semantic search
    auggie query "text" --mode fulltext      # Keyword search (FTS5)
    auggie query "text" --no-json            # Human-readable output
    auggie chunk "path:N" -C 1               # Get chunk with context
    auggie status                            # Show index statistics
    auggie reset --force                     # Delete index
    auggie reindex .                         # Rebuild (enables FTS5)
    auggie clean --vacuum                    # Remove stale entries
    auggie tune .                            # Auto-tune chunk size

Search Modes:
    hybrid   - Combines semantic + keyword (default, best for most queries)
    semantic - Embeddings only (conceptual questions)
    fulltext - FTS5 keywords (exact identifiers)

Environment Variables:
    VOYAGE_API_KEY: Required for Voyage embeddings.
    AUGGIE_MODEL: Default embedding model.
    AUGGIE_SEARCH_MODE: Default search mode (hybrid/semantic/fulltext).
    AUGGIE_HYBRID_ALPHA: Semantic weight in hybrid mode (0.0-1.0).
"""

from __future__ import annotations

import argparse
import sys

from .commands import (
    cmd_cache_report,
    cmd_chunk,
    cmd_clean,
    cmd_context,
    cmd_delete,
    cmd_health,
    cmd_index,
    cmd_log,
    cmd_query,
    cmd_reindex,
    cmd_reset,
    cmd_status,
    cmd_tune,
    cmd_watch,
)
from .commands._arg_builders import add_indexing_args, add_model_args
from .commands._common import add_scope_args

__version__ = "0.12.0"


def _add_index_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'index' subcommand."""
    p = sub.add_parser(
        "index",
        help="Index a directory for semantic search",
        description="Scan files, generate embeddings, and store in local SQLite database.",
    )
    p.add_argument("path", nargs="?", default=".", help="Root path (default: .)")
    add_scope_args(p)
    add_model_args(p)
    add_indexing_args(p)
    p.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List files that would be indexed (sorted by extension, biggest last). "
        "Does not actually index.",
    )
    p.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable file type detection (use fast null-byte check only). "
        "By default, uses 'file' command for accurate MIME type detection.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show files being indexed (useful for tracking new files)",
    )
    p.set_defaults(func=cmd_index)


def _add_query_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'query' subcommand."""
    p = sub.add_parser(
        "query",
        help="Semantic search against the index",
        description="Search indexed code by meaning using natural language queries.",
    )
    p.add_argument("query", help="Natural language search query")
    add_scope_args(p)
    p.add_argument(
        "--top",
        "-n",
        type=int,
        default=10,
        help="Number of results (default: 10)",
    )
    p.add_argument(
        "--refresh",
        "-r",
        action="store_true",
        help="Check for changed files and reindex before querying. "
        "Recommended for AI tools to ensure results reflect current code.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.add_argument(
        "--mode",
        "-M",
        choices=["semantic", "fulltext", "hybrid"],
        default=None,
        help="Search mode: semantic (embeddings only), fulltext (FTS5 keywords), "
        "hybrid (combined, default). Uses AUGGIE_SEARCH_MODE env var if not specified.",
    )
    p.add_argument(
        "--branch",
        "-b",
        default=None,
        help="Git branch to search (default: auto-detect current branch). "
        "Allows querying a different branch's index without switching.",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        help="Enable Voyage reranking for improved result ordering.",
    )
    p.add_argument(
        "--rerank-top",
        type=int,
        default=None,
        metavar="N",
        help="Number of candidates to rerank (default: 50, via AUGGIE_RERANK_TOPN). "
        "Implies --rerank.",
    )
    p.add_argument(
        "--rerank-model",
        metavar="MODEL",
        default=None,
        help="Reranking model: voyage, voyage:lite, voyage:2. "
        "Voyage reranking requires VOYAGE_API_KEY. "
        "Use AUGGIE_RERANK_MODEL for a persistent setting. Implies --rerank.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable query result caching (L1/L2/L3). Useful for debugging or benchmarking.",
    )
    # Path filtering (Phase 3 of adoption improvements)
    p.add_argument(
        "--glob",
        "-g",
        action="append",
        metavar="PATTERN",
        help="Filter results to files matching glob pattern (e.g., '*.py', 'src/*.php'). "
        "Can be specified multiple times. Supports ** for recursive matching.",
    )
    p.add_argument(
        "--exclude",
        "-x",
        action="append",
        metavar="PATTERN",
        help="Exclude files matching glob pattern from results (e.g., 'tests/*'). "
        "Can be specified multiple times. Supports ** for recursive matching.",
    )
    # Summary mode (Phase 4 of adoption improvements)
    p.add_argument(
        "--summarize",
        "-S",
        action="store_true",
        help="Return file-level summary instead of full chunk text. "
        "Shows which files match, how many chunks per file, and line ranges. "
        "Token-efficient for AI scanning. Use 'auggie chunk' to expand specific files.",
    )
    add_model_args(p, for_query=True)
    p.set_defaults(func=cmd_query)


def _add_chunk_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'chunk' subcommand."""
    p = sub.add_parser(
        "chunk",
        help="Get a chunk by reference with optional context",
        description="Retrieve chunks by path:index reference or raw ID. "
        "Useful for expanding context after query finds something interesting.",
    )
    p.add_argument(
        "ref",
        help="Chunk reference: 'path/file.py:N' (path:chunk_index) or raw chunk ID",
    )
    add_scope_args(p)
    p.add_argument(
        "--before",
        "-B",
        type=int,
        default=0,
        metavar="N",
        help="Include N chunks before the requested chunk",
    )
    p.add_argument(
        "--after",
        "-A",
        type=int,
        default=0,
        metavar="N",
        help="Include N chunks after the requested chunk",
    )
    p.add_argument(
        "--context",
        "-C",
        type=int,
        default=0,
        metavar="N",
        help="Include N chunks before AND after (shorthand for -B N -A N)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_chunk)

def _add_reset_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'reset' subcommand."""
    p = sub.add_parser(
        "reset",
        help="Clear indexed data (current branch by default)",
        description="Clear indexed data for the current branch. Use --all to delete the entire database.",
    )
    add_scope_args(p)
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    p.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Delete entire database instead of just current branch",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_reset)


def _add_reindex_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'reindex' subcommand."""
    p = sub.add_parser(
        "reindex",
        help="Force rebuild index from scratch",
        description="Remove existing index and rebuild completely.",
    )
    p.add_argument("path", nargs="?", default=".", help="Root path (default: .)")
    add_scope_args(p)
    add_model_args(p)
    add_indexing_args(p)
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_reindex)


def _add_clean_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'clean' subcommand."""
    p = sub.add_parser(
        "clean",
        help="Remove stale entries from index",
        description="Remove entries for files that no longer exist.",
    )
    add_scope_args(p)
    p.add_argument(
        "--vacuum",
        action="store_true",
        help="Compact database after cleaning",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_clean)


def _add_delete_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'delete' subcommand."""
    p = sub.add_parser(
        "delete",
        help="Remove specific files from the index",
        description="Delete files from the index by path or glob pattern. "
        "Supports exact paths, relative paths, and glob patterns like '*.log' or 'jj'.",
    )
    p.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Paths or glob patterns to delete (supports spaces in names)",
    )
    add_scope_args(p)
    p.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    p.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Add deleted paths to .auggieignore to prevent re-indexing",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_delete)


def _add_log_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'log' subcommand."""
    p = sub.add_parser(
        "log",
        help="Show index change history (AI tool integration)",
        description="Display history of index operations (index, delete, clean, refresh). "
        "AI TOOL HINT: Use after 'auggie query --refresh' to see what changed.",
    )
    add_scope_args(p)
    p.add_argument(
        "--since",
        metavar="DATETIME",
        help="Show entries after this datetime (ISO8601: 2024-01-15T10:30:00 or 2024-01-15)",
    )
    p.add_argument(
        "--until",
        metavar="DATETIME",
        help="Show entries before this datetime (ISO8601 format)",
    )
    p.add_argument(
        "--action",
        choices=["index", "delete", "clean", "refresh", "reindex"],
        help="Filter by action type",
    )
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=50,
        help="Maximum entries to return (default: 50)",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N entries (for pagination)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default: True for AI tool integration)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_log)


def _add_status_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'status' subcommand."""
    p = sub.add_parser(
        "status",
        help="Show index status and statistics",
        description="Display index location, file count, chunk count, and model info.",
    )
    add_scope_args(p)
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_status)


def _add_health_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'health' subcommand."""
    p = sub.add_parser(
        "health",
        help="Show database health and diagnostics",
        description="Display comprehensive database diagnostics including table sizes, "
        "indexes, SQLite info, FTS5 stats, and integrity checks. "
        "Supports repair operations via flags.",
    )
    add_scope_args(p)
    p.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM to reclaim space and defragment database",
    )
    p.add_argument(
        "--rebuild-fts",
        action="store_true",
        help="Drop and rebuild FTS5 index from chunks table",
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="Show reindex command (does not run automatically - requires re-embedding)",
    )
    p.add_argument(
        "--integrity",
        action="store_true",
        help="Run full PRAGMA integrity_check (slow on large databases)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Run vacuum + rebuild-fts + integrity (not reindex)",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Delete Lance-only orphan vectors and report SQLite chunks missing from Lance",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_health)


def _add_tune_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'tune' subcommand."""
    p = sub.add_parser(
        "tune",
        help="Auto-tune chunk size for optimal relevance",
        description="Test different chunk sizes and recommend optimal settings.",
    )
    p.add_argument("path", nargs="?", default=".", help="Root path (default: .)")
    add_scope_args(p)
    add_model_args(p)
    p.add_argument(
        "--samples",
        "-s",
        type=int,
        default=5,
        help="Number of code patterns to test (default: 5)",
    )
    p.add_argument(
        "--apply",
        "-a",
        action="store_true",
        help="Apply optimal settings and reindex",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save optimal chunk size to .env file as AUGGIE_CHUNK_LINES",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_tune)


def _add_watch_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'watch' subcommand."""
    p = sub.add_parser(
        "watch",
        help="Watch a directory and keep the index fresh",
        description="Run a foreground watcher that debounces file events and runs incremental refresh passes.",
    )
    p.add_argument("path", nargs="?", default=".", help="Root path (default: .)")
    add_scope_args(p)
    add_model_args(p)
    add_indexing_args(p)
    p.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable file type detection during incremental passes.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-pass stderr logs",
    )
    p.set_defaults(func=cmd_watch)


def _add_cache_report_command(sub: argparse._SubParsersAction) -> None:
    """Add the 'cache-report' subcommand."""
    p = sub.add_parser(
        "cache-report",
        help="Show cache effectiveness report",
        description="Display hit rates, time saved, and entry counts for L1/L2/L3 caches.",
    )
    add_scope_args(p)
    p.add_argument(
        "--hours",
        type=int,
        default=24,
        metavar="N",
        help="Time period to analyze in hours (default: 24)",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Clear all caches instead of showing report",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default for AI/machine use)",
    )
    p.add_argument(
        "--no-json",
        action="store_false",
        dest="json",
        help="Output as human-readable text instead of JSON",
    )
    p.set_defaults(func=cmd_cache_report)


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser with all auggie subcommands.
    """
    p = argparse.ArgumentParser(
        prog="auggie",
        description="Semantic grep for codebases. Search by meaning, not just keywords. "
        "Supports hybrid (semantic + keyword), pure semantic, or FTS5 fulltext modes. "
        "JSON output is default for AI/machine use. Use --no-json for human-readable output.",
        epilog="Search modes: --mode hybrid (default), semantic, fulltext.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    # Add all subcommands
    _add_index_command(sub)
    _add_query_command(sub)
    _add_chunk_command(sub)
    _add_reset_command(sub)
    _add_reindex_command(sub)
    _add_clean_command(sub)
    _add_cache_report_command(sub)
    _add_delete_command(sub)
    _add_log_command(sub)
    _add_status_command(sub)
    _add_health_command(sub)
    _add_tune_command(sub)
    _add_watch_command(sub)

    return p


def main() -> None:
    """
    Main entry point for the auggie CLI.

    Parses command-line arguments and dispatches to the appropriate
    command handler. Exit code is determined by the command's return value.
    """
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
