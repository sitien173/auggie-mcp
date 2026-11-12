"""
Index command for auggie.

Indexes a directory by scanning files, chunking text, and storing
embeddings in a local SQLite database for semantic search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..filetype import FileTypeResult, detect_file_types_batch, has_file_command
from ..indexer import IndexStats, index_path, iter_files, load_auggieignore
from ..models import get_optimal_chunk_lines, get_optimal_overlap
from ._common import find_git_root, require_embedding_config, resolve_db_path


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"


def _safe_relative_path(path: Path, root: Path) -> Path:
    """
    Get relative path, falling back to absolute if outside root.

    Args:
        path: The file path to make relative.
        root: The root directory to calculate relative path from.

    Returns:
        Relative path if within root, otherwise the original path.
    """
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _accumulate_stat(stats: dict[str, tuple[int, int]], key: str, size: int) -> None:
    """
    Accumulate file count and size statistics.

    Args:
        stats: Dictionary mapping key to (count, total_size) tuple.
        key: The key to accumulate (e.g., extension, category).
        size: File size to add.
    """
    if key not in stats:
        stats[key] = (0, 0)
    count, total = stats[key]
    stats[key] = (count + 1, total + size)


def _extract_mime_category(mime_type: str | None) -> str:
    """
    Extract category from MIME type (e.g., "text/x-python" -> "text").

    Args:
        mime_type: Full MIME type string or None.

    Returns:
        Category portion, or "text" for None/empty.
    """
    if not mime_type:
        return "text"
    if "/" in mime_type:
        return mime_type.split("/")[0]
    return mime_type


def _print_model_mismatch_help(error_msg: str, requested_model: str | None) -> None:
    """Print friendly help message for model mismatch errors."""
    print(f"\nError: {error_msg}\n")

    print("Options:")
    print()
    print("  1. Use the same model as the existing index:")
    print("     auggie index .")
    print()
    print("  2. Switch to new model (rebuilds entire index):")
    print("     auggie reindex . --force")
    print()
    print("  3. Start fresh with new model:")
    print("     auggie reset -f")
    print("     auggie index .")
    print()


# Extensions that suggest non-code content (logs, data, dumps)
_REVIEW_EXTENSIONS = frozenset(
    {
        ".log",
        ".log_save",
        ".old",
        ".bak",
        ".backup",
        ".tmp",
        ".dump",
        ".sql",
        ".sqlt",
        ".csv",
        ".tsv",
        ".dat",
        ".out",
        ".err",
        ".trace",
        ".prof",
    }
)

# Size threshold for review suggestion (500KB)
_REVIEW_SIZE_THRESHOLD = 500 * 1024


def _should_review(path: Path, size: int) -> str | None:
    """
    Check if a file should be flagged for manual review.

    Returns a reason string if the file should be reviewed, None otherwise.
    """
    name = path.name.lower()
    ext = path.suffix.lower()

    # Check extension
    if ext in _REVIEW_EXTENSIONS:
        return f"extension '{ext}'"

    # Check for common non-code patterns in filename
    if any(pattern in name for pattern in (".log", "_log", "-log", ".old", ".bak")):
        return "filename suggests log/backup"

    # Large files with suspicious extensions
    if size > _REVIEW_SIZE_THRESHOLD:
        if ext in {".txt", ".text", ""} or not ext:
            return f"large file ({_format_size(size)}) without code extension"

    return None


def _list_files(
    root: Path, exclude: list[str], include: list[str], detect: bool = True, use_json: bool = False
) -> int:
    """
    List files that would be indexed, sorted by extension then size.

    Shows file type detection results when detect=True, marking binary
    files with [BINARY: mime/type] prefix.

    Args:
        root: Root directory to scan.
        exclude: Additional exclude patterns.
        include: Include patterns (override excludes).
        detect: If True, run file type detection and show results.
        use_json: If True, output as JSON.

    Returns:
        Exit code (0 for success).
    """
    # Load .auggieignore patterns
    ignore_patterns = load_auggieignore(root)
    all_exclude = list(exclude) + ignore_patterns

    # Collect files with their stats
    file_info: list[tuple[Path, str, int]] = []  # (path, extension, size)

    for p in iter_files(root, exclude=all_exclude, include=include):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
            ext = p.suffix.lower() if p.suffix else "(no extension)"
            file_info.append((p, ext, size))
        except (OSError, FileNotFoundError):
            continue

    if not file_info:
        if use_json:
            print(json.dumps({"files": [], "indexable_count": 0, "indexable_size": 0}))
        else:
            print("No files would be indexed.")
        return 0

    # Run file type detection if enabled
    detection_results: dict[Path, FileTypeResult] = {}
    if detect and has_file_command():
        paths = [p for p, _, _ in file_info]
        detection_results = detect_file_types_batch(paths)

    # Sort by extension, then by size (ascending, so biggest last)
    file_info.sort(key=lambda x: (x[1], x[2]))

    # Separate indexable and excluded files
    indexable: list[tuple[Path, str, int]] = []
    excluded: list[tuple[Path, str, int, str]] = []  # + mime_type

    for p, ext, size in file_info:
        result = detection_results.get(p)
        if result and not result.is_text:
            excluded.append((p, ext, size, result.mime_type or "unknown"))
        else:
            indexable.append((p, ext, size))

    # Group by extension for summary (indexable only)
    ext_stats: dict[str, tuple[int, int]] = {}  # ext -> (count, total_size)
    for _, ext, size in indexable:
        _accumulate_stat(ext_stats, ext, size)

    # Count files per directory
    dir_counts: dict[Path, int] = {}
    for p, _, _ in indexable:
        try:
            rel = p.relative_to(root)
            # Get top-level directory or parent
            parts = rel.parts
            if len(parts) > 1:
                dir_key = root / parts[0]
            else:
                dir_key = root
            dir_counts[dir_key] = dir_counts.get(dir_key, 0) + 1
        except ValueError:
            pass

    # JSON output
    if use_json:
        indexable_size = sum(size for _, _, size in indexable)
        excluded_size = sum(size for _, _, size, _ in excluded)

        output = {
            "indexable": [
                {
                    "path": str(_safe_relative_path(p, root)),
                    "extension": ext,
                    "size": size,
                }
                for p, ext, size in indexable
            ],
            "excluded": [
                {
                    "path": str(_safe_relative_path(p, root)),
                    "extension": ext,
                    "size": size,
                    "mime_type": mime,
                }
                for p, ext, size, mime in excluded
            ],
            "summary": {
                "indexable_count": len(indexable),
                "indexable_size": indexable_size,
                "excluded_count": len(excluded),
                "excluded_size": excluded_size,
            },
            "by_extension": {
                ext: {"count": count, "size": total} for ext, (count, total) in ext_stats.items()
            },
        }
        print(json.dumps(output))
        return 0

    # Print file list grouped by extension
    current_ext = None
    for p, ext, size in file_info:
        if ext != current_ext:
            current_ext = ext
            # Count total files in this extension (including excluded)
            ext_files = [(fp, fs) for fp, fe, fs in file_info if fe == ext]
            ext_total_size = sum(s for _, s in ext_files)
            print(f"\n── {ext} ({len(ext_files)} files, {_format_size(ext_total_size)}) ──")

        rel_path = _safe_relative_path(p, root)
        result = detection_results.get(p)

        if result and not result.is_text:
            # Show binary marker
            mime_short = result.mime_type or "unknown"
            # Truncate long mime types
            if len(mime_short) > 30:
                mime_short = mime_short[:27] + "..."
            print(f"  [BINARY: {mime_short}] {_format_size(size):>8}  {rel_path}")
        else:
            print(f"  {_format_size(size):>8}  {rel_path}")

    # Print summary
    indexable_size = sum(size for _, _, size in indexable)
    excluded_size = sum(size for _, _, size, _ in excluded)

    print(f"\n{'─' * 50}")
    print(f"Would index: {len(indexable)} files, {_format_size(indexable_size)}")

    if excluded:
        print(f"Excluded by detection: {len(excluded)} files, {_format_size(excluded_size)}")

    if not detect:
        print("(Detection disabled, use without --no-detect for MIME checking)")
    elif not has_file_command():
        print("(file command not available, using null-byte detection only)")

    # Breakdown by extension (indexable files only)
    if ext_stats and len(ext_stats) > 1:
        print("\nBreakdown by extension:")
        sorted_exts = sorted(ext_stats.items(), key=lambda x: x[1][1], reverse=True)
        print(f"  {'Extension':<15} {'Files':>7} {'Size':>10} {'%':>6}")
        print(f"  {'-' * 15} {'-' * 7} {'-' * 10} {'-' * 6}")
        for ext, (count, total) in sorted_exts:
            pct = (total / indexable_size * 100) if indexable_size > 0 else 0
            print(f"  {ext:<15} {count:>7} {_format_size(total):>10} {pct:>5.1f}%")

    # Breakdown by MIME type category (if detection was run)
    if detection_results:
        mime_stats: dict[str, tuple[int, int]] = {}  # category -> (count, size)
        for p, _ext, size in indexable:
            result = detection_results.get(p)
            mime_type = result.mime_type if result else None
            category = _extract_mime_category(mime_type)
            _accumulate_stat(mime_stats, category, size)

        # Add excluded files
        for _p, _ext, size, mime in excluded:
            category = _extract_mime_category(mime)
            _accumulate_stat(mime_stats, category, size)

        if len(mime_stats) > 1:
            total_files = len(indexable) + len(excluded)
            print("\nBreakdown by file type:")
            sorted_mimes = sorted(mime_stats.items(), key=lambda x: x[1][0], reverse=True)
            print(f"  {'Type':<15} {'Files':>7} {'Size':>10} {'%':>6}")
            print(f"  {'-' * 15} {'-' * 7} {'-' * 10} {'-' * 6}")
            for category, (count, total) in sorted_mimes:
                pct = (count / total_files * 100) if total_files > 0 else 0
                print(f"  {category:<15} {count:>7} {_format_size(total):>10} {pct:>5.1f}%")

    # Show top 10 directories by file count
    if dir_counts:
        sorted_dirs = sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop directories by file count:")
        for dir_path, count in sorted_dirs:
            rel_dir = dir_path.relative_to(root) if dir_path != root else Path(".")
            print(f"  {count:>6} files  {rel_dir}/")

    # Show top 5 largest indexable files
    largest_indexable = sorted(indexable, key=lambda x: x[2], reverse=True)[:5]
    if largest_indexable:
        print("\nLargest indexable:")
        for p, _ext, size in largest_indexable:
            rel_path = _safe_relative_path(p, root)
            print(f"  {_format_size(size):>8}  {rel_path}")

    # Show largest excluded files (binary)
    if excluded:
        largest_excluded = sorted(excluded, key=lambda x: x[2], reverse=True)[:5]
        print("\nLargest excluded (binary):")
        for p, _ext, size, mime in largest_excluded:
            rel_path = _safe_relative_path(p, root)
            print(f"  {_format_size(size):>8}  {rel_path} ({mime})")

    # Show files that should be reviewed (pass MIME but may not be code)
    review_files: list[tuple[Path, int, str]] = []  # (path, size, reason)
    for p, _ext, size in indexable:
        reason = _should_review(p, size)
        if reason:
            review_files.append((p, size, reason))

    if review_files:
        # Sort by size descending, show top 10
        review_files.sort(key=lambda x: x[1], reverse=True)
        print("\n⚠ Review suggested (may distort search results):")
        print("  These files pass MIME detection but may not be useful code.")
        print("  Consider adding patterns to .auggieignore:")
        for p, size, reason in review_files[:10]:
            rel_path = _safe_relative_path(p, root)
            print(f"  {_format_size(size):>8}  {rel_path}")
            print(f"           └─ {reason}")
        if len(review_files) > 10:
            print(f"  ... and {len(review_files) - 10} more")

    return 0


def _resolve_chunk_lines(args: argparse.Namespace) -> int:
    """
    Resolve chunk size from args or model-specific default.

    Args:
        args: Parsed command-line arguments with chunk_lines and model.

    Returns:
        Chunk size in lines.
    """
    if args.chunk_lines is not None:
        return args.chunk_lines
    return get_optimal_chunk_lines(args.model)


def _resolve_overlap(args: argparse.Namespace) -> int:
    """
    Resolve overlap size from args or model-specific default.

    Args:
        args: Parsed command-line arguments with overlap and model.

    Returns:
        Overlap size in lines.
    """
    if args.overlap is not None:
        return args.overlap
    return get_optimal_overlap(args.model)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Resolve root directory and database path from arguments.

    The repo_root (where .auggie is created) is determined by:
    1. Explicit --repo-root argument
    2. Git repository root (if in a git repo)
    3. The indexed path itself (fallback for non-git directories)

    This ensures .auggie is always created in the repo root, not in
    subdirectories when indexing from within a subdirectory.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (root_path, db_path).
    """
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
    return root, db


def _print_stats(db: Path, stats: IndexStats) -> None:
    """
    Print indexing statistics to stdout.

    Args:
        db: Path to the database file.
        stats: IndexStats dataclass with indexing results.
    """
    print(f"Indexed into {db}")
    print(f"  Model: {stats.model}")
    print(f"  Files: {stats.files_indexed} indexed, {stats.files_skipped} skipped")

    if stats.chunks_total > 0:
        _print_chunk_stats(stats)


def _print_chunk_stats(stats: IndexStats) -> None:
    """
    Print chunk-level statistics.

    Args:
        stats: IndexStats dataclass with chunk counts.
    """
    msg = f"  Chunks: {stats.chunks_total} total"

    if stats.chunks_reused > 0:
        msg += f" ({stats.chunks_reused} reused, ~{stats.tokens_saved_estimate} tokens saved)"
    else:
        msg += f" ({stats.chunks_embedded} embedded)"

    print(msg)


def cmd_index(args: argparse.Namespace) -> int:
    """
    Index a directory for semantic search.

    Scans the target directory for text files, splits them into chunks,
    generates embeddings via Voyage API, and stores everything in a
    local SQLite database.

    Args:
        args: Parsed command-line arguments containing:
            - path: Directory to index (default: current directory)
            - db, profile, global_cache, repo_root: Scope options
            - model: Voyage embedding model name
            - dimensions: Embedding dimensions (model-specific)
            - chunk_lines: Lines per chunk (None = model-specific default)
            - overlap: Overlapping lines between chunks
            - max_bytes: Maximum file size to index
            - exclude: Additional glob patterns to exclude
            - include: Glob patterns to include (override excludes)
            - list: If True, list files that would be indexed (dry run)
            - json: Whether to output as JSON

    Returns:
        Exit code (0 for success, 1 for configuration error, 130 for interrupt).
    """
    root = Path(args.path).resolve()
    detect = not getattr(args, "no_detect", False)
    use_json = getattr(args, "json", False)

    # Handle --list flag (doesn't require embedding config)
    if getattr(args, "list", False):
        try:
            return _list_files(root, args.exclude, args.include, detect=detect, use_json=use_json)
        except KeyboardInterrupt:
            if use_json:
                print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
            else:
                print("\n\nInterrupted by user (Ctrl-C).")
            return 130

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

    root, db = _resolve_paths(args)
    chunk_lines = _resolve_chunk_lines(args)
    overlap = _resolve_overlap(args)
    verbose = getattr(args, "verbose", False)

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
            detect=detect,
            verbose=verbose,
            ast=getattr(args, "ast", False),
        )
    except KeyboardInterrupt:
        if use_json:
            print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
        else:
            print("\n\nInterrupted by user (Ctrl-C).")
            print("Partial progress may have been saved to the index.")
            print("Run 'auggie index .' again to continue from where you left off.")
        return 130  # Standard SIGINT exit code (128 + 2)
    except ValueError as e:
        if "Model mismatch" in str(e):
            if use_json:
                print(json.dumps({"error": str(e), "error_code": "MODEL_MISMATCH"}))
            else:
                _print_model_mismatch_help(str(e), args.model)
            return 1
        raise

    if use_json:
        output = {
            "database": str(db),
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
            output["ast_hint"] = stats.ast_hint
        if verbose and stats.indexed_files is not None:
            output["indexed_files"] = stats.indexed_files
        print(json.dumps(output))
    else:
        _print_stats(db, stats)
        if verbose and stats.indexed_files:
            print("\nFiles indexed:")
            for f in stats.indexed_files:
                print(f"  {f}")

    return 0
