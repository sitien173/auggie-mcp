"""
Query command for auggie.

Performs search against an indexed codebase, returning
the most relevant code chunks ranked by similarity score.

Supports three search modes:
- semantic: Embedding similarity only (original behavior)
- fulltext: SQLite FTS5 keyword matching only
- hybrid: Combined score (default) - best of both worlds

Supports --refresh flag to automatically reindex changed files before
querying, ensuring search results reflect the current codebase state.

Supports --json flag for structured output suitable for AI tools and
programmatic use, including full chunk text and metadata.

Supports --rerank flag for cross-encoder reranking, which improves
result ordering by processing (query, document) pairs together.
Requires Voyage reranking support.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from ..search import FUSION_METHOD, Hit, PathFilter, filter_hits_by_path
from ..search import query as query_db
from ._common import detect_language, get_current_branch, require_embedding_config, resolve_db_path

# Over-fetch multiplier when path filtering is active
# We fetch more results to ensure we have enough after filtering
PATH_FILTER_OVERFETCH = 3


def _get_index_info(db_path: Path) -> tuple[str, int] | None:
    """
    Get model and dimensions from the index.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Tuple of (model_name, dimensions) or None if index is empty.
    """
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT model, dim FROM chunks LIMIT 1").fetchone()
        if row:
            return row[0], row[1]
        return None
    finally:
        con.close()


def _format_json_result(hit: Hit, repo_root: Path, rank: int) -> dict:
    """
    Format a single Hit for JSON output.

    Args:
        hit: The search result Hit object.
        repo_root: Repository root for relative path calculation.
        rank: The 1-indexed rank of this result.

    Returns:
        Dictionary with formatted result fields.
    """
    # Calculate relative path from repo root
    try:
        rel_path = str(Path(hit.path).relative_to(repo_root))
    except ValueError:
        rel_path = hit.path  # Fallback to absolute if not relative to root

    # Build chunk_ref: relative_path:chunk_index
    chunk_ref = f"{rel_path}:{hit.chunk_index}"

    # Build confidence output - use detailed object if available
    if hit.confidence_details is not None:
        confidence = hit.confidence_details.to_dict()
    else:
        # Fallback for legacy mode (absolute confidence)
        confidence = hit.confidence

    return {
        "rank": rank,
        "chunk_ref": chunk_ref,
        "chunk_id": hit.chunk_id,
        "path": hit.path,
        "relative_path": rel_path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "score": round(hit.score, 4),
        "confidence": confidence,
        "language": detect_language(hit.path),
        "text": hit.text,
    }


def _format_text_output(hits: list[Hit]) -> None:
    """
    Print hits in human-readable text format.

    Args:
        hits: List of Hit objects to format and print.
    """
    for h in hits:
        print(f"{h.path}:{h.start_line}-{h.end_line}  score={h.score:0.4f} ({h.confidence})")
        snippet = h.text.strip().replace("\n", "\\n")
        print(f"  {snippet[:240]}")


def _aggregate_to_summary(hits: list[Hit], repo_root: Path) -> list[dict]:
    """
    Aggregate hits into file-level summary.

    Groups chunks by file and computes statistics for each file.

    Args:
        hits: List of Hit objects (already filtered/sorted).
        repo_root: Repository root for relative path calculation.

    Returns:
        List of file summary dictionaries, sorted by best_score descending.
    """
    from collections import defaultdict

    from ..search import assign_confidence

    # Group hits by file path
    by_file: dict[str, list[Hit]] = defaultdict(list)
    for hit in hits:
        by_file[hit.path].append(hit)

    # Build summary for each file
    summaries = []
    for path, file_hits in by_file.items():
        # Calculate relative path
        try:
            rel_path = str(Path(path).relative_to(repo_root))
        except ValueError:
            rel_path = path

        # Compute statistics
        scores = [h.score for h in file_hits]
        best_score = max(scores)
        min_score = min(scores)

        # Get confidence for best score (using relative to top result of all hits)
        top_overall_score = hits[0].score if hits else 0.0
        level, _ = assign_confidence(best_score, top_overall_score, rank=1)

        # Extract line ranges
        lines_covered = [
            [h.start_line, h.end_line] for h in sorted(file_hits, key=lambda x: x.start_line)
        ]

        summaries.append(
            {
                "path": path,
                "relative_path": rel_path,
                "chunks_matched": len(file_hits),
                "best_score": round(best_score, 4),
                "score_range": [round(min_score, 4), round(best_score, 4)],
                "confidence": level,
                "language": detect_language(path),
                "lines_covered": lines_covered,
            }
        )

    # Sort by best_score descending
    summaries.sort(key=lambda x: x["best_score"], reverse=True)
    return summaries


def _format_summary_text(summaries: list[dict]) -> None:
    """
    Print file summaries in human-readable text format.

    Args:
        summaries: List of file summary dictionaries.
    """
    for s in summaries:
        line_ranges = ", ".join(f"{r[0]}-{r[1]}" for r in s["lines_covered"])
        print(
            f"{s['relative_path']}  {s['chunks_matched']} chunks  "
            f"score={s['best_score']:.4f} ({s['confidence']})"
        )
        print(f"  lines: {line_ranges}")


def _check_stale_files(db_path: Path, repo_root: Path) -> list[Path]:
    """
    Check for files that have changed since last indexing.

    Compares current file mtime/size against stored values to detect
    files that need reindexing.

    Args:
        db_path: Path to the SQLite database.
        repo_root: Repository root to resolve relative paths.

    Returns:
        List of file paths that have changed or been deleted.
    """
    stale: list[Path] = []
    con = sqlite3.connect(str(db_path))

    try:
        rows = con.execute("SELECT path, mtime_ns, size FROM files").fetchall()
        for path_str, mtime_ns, size in rows:
            file_path = Path(path_str)
            if not file_path.exists():
                # File was deleted
                stale.append(file_path)
            else:
                # Check if file was modified
                stat = file_path.stat()
                if stat.st_mtime_ns != mtime_ns or stat.st_size != size:
                    stale.append(file_path)
    finally:
        con.close()

    return stale


MIN_QUERY_LENGTH = 2  # Minimum characters for a meaningful query


def cmd_query(args: argparse.Namespace) -> int:
    """
    Run a search query against the index.

    Supports three search modes:
    - semantic: Embedding similarity only (original behavior)
    - fulltext: SQLite FTS5 keyword matching only
    - hybrid: Combined score (default) - best of both worlds

    Args:
        args: Parsed command-line arguments containing:
            - query: Natural language search query
            - top: Number of results to return
            - mode: Search mode (semantic, fulltext, hybrid)
            - refresh: Whether to check for and reindex changed files
            - json: Whether to output results as JSON
            - db, profile, global_cache, repo_root: Scope options
            - model: Voyage embedding model (must match indexed model)
            - dimensions: Embedding dimensions (must match indexed dimensions)

    Returns:
        Exit code (0 for success, 1 if database not found).

    Note:
        When --refresh is used, auggie checks all indexed files for changes
        (mtime/size) and runs an incremental reindex before querying.
        This ensures search results reflect the current codebase state.

        When --json is used, output is structured JSON with full chunk text,
        language detection, and metadata. Recommended for AI tools.

        IMPORTANT: Without --refresh, queries may return stale results if
        files have been modified since the last index. AI tools and skills
        should always use --refresh to ensure accurate results.

        If FTS5 is unavailable and mode is hybrid/fulltext, falls back
        to semantic search with a warning.
    """
    use_json = getattr(args, "json", False)

    # Validate query length before any expensive operations
    query_text = args.query.strip() if args.query else ""
    if len(query_text) < MIN_QUERY_LENGTH:
        error_msg = (
            f"Query too short: '{query_text}' ({len(query_text)} chars). "
            f"Minimum is {MIN_QUERY_LENGTH} characters."
        )
        if use_json:
            print(json.dumps({"error": error_msg, "error_code": "QUERY_TOO_SHORT"}))
        else:
            print(f"Error: {error_msg}", file=sys.stderr)
        return 1

    if not require_embedding_config():
        return 1

    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)

    if not db.exists():
        if use_json:
            print(json.dumps({"error": f"Database not found at {db}"}))
        else:
            print(f"Error: Database not found at {db}", file=sys.stderr)
            print("Run 'auggie index .' first to create the index.", file=sys.stderr)
        return 1

    # Track refresh stats for JSON output
    refreshed_files = 0

    # Get index model/dimensions BEFORE any operations
    # This ensures --refresh uses the correct model, not CLI defaults
    index_info = _get_index_info(db)
    if index_info:
        index_model, index_dim = index_info
    else:
        # Empty index - use CLI args
        index_model, index_dim = args.model, args.dimensions

    # Check AST mode metadata
    index_ast_mode = None
    try:
        con = sqlite3.connect(str(db))
        ast_row = con.execute("SELECT value FROM index_metadata WHERE key = 'ast_mode'").fetchone()
        if ast_row:
            index_ast_mode = ast_row[0] == "true"
        con.close()
    except sqlite3.OperationalError:
        # index_metadata table doesn't exist (older index)
        pass

    # Handle --refresh: check for stale files and reindex if needed
    if getattr(args, "refresh", False):
        stale_files = _check_stale_files(db, repo_root)
        if stale_files:
            # Import here to avoid circular imports
            from ..indexer import index_path

            # Warn if user specified a different model than the index uses
            if args.model != index_model and index_info is not None:
                if not use_json:
                    print(
                        f"Note: Using index model ({index_model}), not -m {args.model}",
                        file=sys.stderr,
                    )

            if not use_json:
                print(f"Refreshing index ({len(stale_files)} changed files)...", file=sys.stderr)

            try:
                stats = index_path(
                    root=repo_root,
                    db_path=db,
                    model=index_model,
                    dimensions=index_dim,
                )
            except KeyboardInterrupt:
                if use_json:
                    print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
                else:
                    print("\n\nInterrupted by user (Ctrl-C).", file=sys.stderr)
                    print("Partial refresh may have been saved.", file=sys.stderr)
                return 130  # Standard SIGINT exit code (128 + 2)

            refreshed_files = stats.files_indexed

            # Update history action from "index" to "refresh" (AI tool integration)
            if stats.files_indexed > 0:
                try:
                    con = sqlite3.connect(str(db))
                    con.execute(
                        "UPDATE index_history SET action = 'refresh' "
                        "WHERE id = (SELECT MAX(id) FROM index_history)"
                    )
                    con.commit()
                    con.close()
                except sqlite3.OperationalError:
                    pass  # History table might not exist in older databases

            if not use_json and (stats.files_indexed > 0 or stats.chunks_reused > 0):
                print(
                    f"  Updated: {stats.files_indexed} files, "
                    f"{stats.chunks_embedded} new chunks "
                    f"({stats.chunks_reused} reused)",
                    file=sys.stderr,
                )

    # Get search mode
    search_mode = getattr(args, "mode", None)

    # Check reranking options
    do_rerank = getattr(args, "rerank", False)
    rerank_top = getattr(args, "rerank_top", None)
    rerank_model = getattr(args, "rerank_model", None)

    # --rerank-top and --rerank-model imply --rerank
    if rerank_top is not None or rerank_model is not None:
        do_rerank = True

        # Validate: --rerank-top must be >= -n (only if explicitly specified)
        # Otherwise we can't return -n results from a smaller rerank pool
        if rerank_top is not None and rerank_top < args.top:
            error_msg = f"--rerank-top ({rerank_top}) must be >= -n ({args.top})"
            if use_json:
                print(
                    json.dumps(
                        {
                            "error": error_msg,
                            "error_code": "INVALID_RERANK_ARGS",
                            "suggestion": (
                                f"Use --rerank-top {args.top} or higher, "
                                f"or reduce -n to {rerank_top} or lower"
                            ),
                        }
                    )
                )
            else:
                print(f"Error: {error_msg}", file=sys.stderr)
                print(
                    f"Suggestion: Use --rerank-top {args.top} or higher, "
                    f"or reduce -n to {rerank_top} or lower",
                    file=sys.stderr,
                )
            return 1

    # Build path filter from CLI args
    glob_patterns = getattr(args, "glob", None)
    exclude_patterns = getattr(args, "exclude", None)
    path_filter = PathFilter(
        glob_patterns=glob_patterns,
        exclude_patterns=exclude_patterns,
    )
    has_path_filter = not path_filter.is_empty()

    # Time the search
    start_time = time.perf_counter()

    # Calculate fetch_limit with over-fetching for reranking and/or filtering
    fetch_limit = args.top
    if do_rerank:
        # Fetch enough for reranking (default 50, or rerank_top if specified)
        from ..rerank import DEFAULT_RERANK_TOPN

        rerank_n = rerank_top if rerank_top is not None else DEFAULT_RERANK_TOPN
        fetch_limit = max(args.top, rerank_n)

    # Over-fetch when path filtering is active to ensure enough results after filtering
    if has_path_filter:
        fetch_limit = max(fetch_limit, args.top * PATH_FILTER_OVERFETCH)

    # Get branch - from CLI arg or auto-detect
    branch = getattr(args, "branch", None) or get_current_branch(repo_root)

    hits, fts_available = query_db(
        db_path=db,
        q=query_text,
        top_k=fetch_limit,
        model=index_model,
        dimensions=index_dim,
        mode=search_mode,
        branch=branch,
    )

    # Apply reranking if requested
    reranked = False
    rerank_skipped = False
    rerank_skip_reason: str | None = None
    rerank_warnings: list[str] = []

    if do_rerank and hits:
        try:
            from ..rerank import (
                clear_captured_warnings,
                get_captured_warnings,
                is_reranker_available,
                rerank_results,
            )

            if is_reranker_available(rerank_model):
                rerank_n = rerank_top if rerank_top is not None else DEFAULT_RERANK_TOPN
                try:
                    # Get cache path for L3 caching (if caching enabled)
                    cache_path = None
                    no_cache = getattr(args, "no_cache", False)
                    if not no_cache:
                        from ..cache import get_cache_path

                        cache_path = get_cache_path(db)

                    hits = rerank_results(
                        query_text,
                        hits,
                        top_n=rerank_n,
                        model_name=rerank_model,
                        cache_path=cache_path,
                    )
                    # Trim to requested number after reranking
                    hits = hits[: args.top]
                    reranked = True

                    # Collect any warnings from reranker initialization (CUDA, etc.)
                    rerank_warnings = get_captured_warnings()
                    clear_captured_warnings()
                except Exception as e:
                    # Reranking failed at runtime - fall back to non-reranked results
                    rerank_skipped = True
                    rerank_skip_reason = f"Reranking failed: {e}"
                    rerank_warnings = get_captured_warnings()
                    clear_captured_warnings()
                    if not use_json:
                        print(f"Warning: {rerank_skip_reason}", file=sys.stderr)
                        print("Returning results without reranking.", file=sys.stderr)
            else:
                rerank_skipped = True
                rerank_skip_reason = "voyage reranker unavailable"

                if not use_json:
                    print(
                        f"Warning: Reranking requested but {rerank_skip_reason}.",
                        file=sys.stderr,
                    )
                    print(
                        "Set VOYAGE_API_KEY and install the dependencies from requirements.txt.",
                        file=sys.stderr,
                    )
                    print("Returning results without reranking.", file=sys.stderr)
        except ImportError as e:
            # Import failed - fall back to non-reranked results
            rerank_skipped = True
            rerank_skip_reason = f"Import error: {e}"
            if not use_json:
                print(f"Warning: Reranking unavailable: {e}", file=sys.stderr)
                print("Returning results without reranking.", file=sys.stderr)

    search_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Apply path filtering if configured
    filter_stats = None
    if has_path_filter:
        hits, filter_stats = filter_hits_by_path(hits, path_filter, args.top)
        # Trim to requested number after filtering
        hits = hits[: args.top]
        # Add filter warning if applicable
        if filter_stats.warning and not use_json:
            print(f"Warning: {filter_stats.warning}", file=sys.stderr)

    # Get total chunk count for stats (model/dim already fetched above)
    con = sqlite3.connect(str(db))
    try:
        total_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        con.close()

    # Collect all warnings for structured output
    all_warnings: list[str] = []
    all_warnings.extend(rerank_warnings)

    # Add FTS5 warning if applicable
    if search_mode in ("hybrid", "fulltext") and not fts_available:
        all_warnings.append(
            "FTS5 index not available, using semantic search only. "
            "Run 'auggie reindex .' to enable hybrid search."
        )

    # Check if summary mode is requested
    use_summarize = getattr(args, "summarize", False)

    if use_json:
        if use_summarize:
            # Summary mode: aggregate to file-level
            file_summaries = _aggregate_to_summary(hits, repo_root)

            output: dict[str, Any] = {
                "query": query_text,
                "mode": search_mode or "hybrid",
                "summary": True,
                "total_chunks_matched": len(hits),
                "files": file_summaries,
                "search_time_ms": search_time_ms,
                "recommendation": "Use 'auggie chunk <path>:<N>' to expand specific files",
            }

            # Add basic stats
            output["stats"] = {
                "files_matched": len(file_summaries),
                "total_chunks": total_chunks,
                "search_mode": search_mode or "hybrid",
                "reranked": reranked,
                "fts_available": fts_available,
                "branch": branch,
            }
        else:
            # Full results mode (default)
            results = [_format_json_result(h, repo_root, rank) for rank, h in enumerate(hits, 1)]

            # Calculate confidence distribution
            confidence_summary = {"high": 0, "medium": 0, "low": 0, "very_low": 0}
            for h in hits:
                confidence_summary[h.confidence] += 1

            output = {
                "query": query_text,
                "branch": branch,
                "results": results,
                "stats": {
                    "total_results": len(hits),
                    "total_chunks": total_chunks,
                    "search_time_ms": search_time_ms,
                    "search_mode": search_mode or "hybrid",
                    "fusion_method": FUSION_METHOD
                    if (search_mode or "hybrid") == "hybrid"
                    else None,
                    "reranked": reranked,
                    "rerank_requested": do_rerank,
                    "fts_available": fts_available,
                    "index_model": index_model,
                    "index_dimensions": index_dim,
                    "refreshed_files": refreshed_files,
                    "confidence_summary": confidence_summary,
                    "branch": branch,
                },
            }

            # Add rerank model info if reranking was performed
            if reranked:
                output["stats"]["rerank_model"] = rerank_model or "voyage"

                # Hint: reranking typically doesn't help with high-quality embeddings
                high_quality_models = (
                    "voyage-code-3",
                    "voyage-3",
                    "voyage-code",
                    "voyage",
                    "voyage-3-lite",
                    "voyage-lite",
                )
                if index_model and index_model.lower() in high_quality_models:
                    output["rerank_hint"] = (
                        f"Note: Reranking typically does not improve results with {index_model}. "
                        "Benchmarks show it can degrade quality. Consider querying without "
                        "--rerank for optimal results."
                    )

            # Add rerank skip info if reranking was requested but couldn't be performed
            if rerank_skipped:
                output["stats"]["rerank_skipped"] = True
                output["stats"]["rerank_skip_reason"] = rerank_skip_reason

                output["suggestion"] = (
                    "Voyage reranking failed. Results were returned without reranking. "
                    "Check VOYAGE_API_KEY and Voyage connectivity."
                )

            # Add AST mode info if available
            if index_ast_mode is not None:
                output["stats"]["ast_mode"] = index_ast_mode
                if not index_ast_mode:
                    output["hint"] = (
                        "Index was built without AST chunking. For semantic boundaries, "
                        "install tree-sitter support and reindex: "
                        "pip install tree-sitter-<language> && auggie reindex ."
                    )

        # Add path filter stats if filtering was applied (both modes)
        if filter_stats is not None:
            output["stats"]["path_filter"] = filter_stats.to_dict()
            # Add filter warning to all_warnings if present
            if filter_stats.warning:
                all_warnings.append(filter_stats.warning)

        # Add warnings if any (includes CUDA, FTS5, reranker issues)
        if all_warnings:
            output["warnings"] = all_warnings

        print(json.dumps(output, indent=2))
    else:
        # Print warnings first for text output (to stderr)
        for warning in all_warnings:
            print(f"Warning: {warning}", file=sys.stderr)

        # Print rerank hint for high-quality embeddings
        if reranked and index_model:
            high_quality_models = (
                "voyage-code-3",
                "voyage-3",
                "voyage-code",
                "voyage",
                "voyage-3-lite",
                "voyage-lite",
            )
            if index_model.lower() in high_quality_models:
                print(
                    f"Hint: Reranking may not improve results with {index_model}. "
                    "Try without --rerank for potentially better results.",
                    file=sys.stderr,
                )

        if use_summarize:
            # Summary mode: aggregate to file-level
            file_summaries = _aggregate_to_summary(hits, repo_root)
            _format_summary_text(file_summaries)
        else:
            # Human-readable output using helper
            _format_text_output(hits)

    return 0
