"""
Tune command for auggie.

Auto-detects optimal chunk size for a codebase by sampling significant
code patterns and testing retrieval accuracy across different chunk sizes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from pathlib import Path

from ..indexer import index_path, iter_files
from ..models import resolve_model
from ..search import query as search_query


def _save_chunk_lines_to_env(env_file: Path, chunk_lines: int) -> None:
    """
    Save AUGGIE_CHUNK_LINES to a .env file.

    Creates the file if it doesn't exist, or updates the existing value.
    """
    env_var = f"AUGGIE_CHUNK_LINES={chunk_lines}"

    if env_file.exists():
        content = env_file.read_text()
        lines = content.splitlines()
        updated = False

        for i, line in enumerate(lines):
            if line.startswith("AUGGIE_CHUNK_LINES="):
                lines[i] = env_var
                updated = True
                break

        if not updated:
            lines.append(env_var)

        env_file.write_text("\n".join(lines) + "\n")
    else:
        env_file.write_text(env_var + "\n")


# Patterns to identify significant code lines
SIGNIFICANT_PATTERNS = [
    # Python
    (r"^\s*def\s+(\w+)\s*\(", "function {0}"),
    (r"^\s*class\s+(\w+)", "class {0}"),
    (r"^\s*async\s+def\s+(\w+)", "async function {0}"),
    # JavaScript/TypeScript
    (r"^\s*function\s+(\w+)\s*\(", "function {0}"),
    (r"^\s*(?:export\s+)?class\s+(\w+)", "class {0}"),
    (r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function {0}"),
    (r"^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*\{", "object {0}"),
    # Go
    (r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", "function {0}"),
    (r"^\s*type\s+(\w+)\s+struct", "struct {0}"),
    (r"^\s*type\s+(\w+)\s+interface", "interface {0}"),
    # Rust
    (r"^\s*(?:pub\s+)?fn\s+(\w+)", "function {0}"),
    (r"^\s*(?:pub\s+)?struct\s+(\w+)", "struct {0}"),
    (r"^\s*(?:pub\s+)?trait\s+(\w+)", "trait {0}"),
    (r"^\s*impl\s+(?:\w+\s+for\s+)?(\w+)", "implementation {0}"),
    # Java/Kotlin
    (r"^\s*(?:public|private|protected)?\s*class\s+(\w+)", "class {0}"),
    (r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)", "interface {0}"),
    # Ruby
    (r"^\s*def\s+(\w+)", "method {0}"),
    (r"^\s*class\s+(\w+)", "class {0}"),
    (r"^\s*module\s+(\w+)", "module {0}"),
]


def _extract_significant_lines(
    root: Path,
    max_samples: int = 10,
) -> list[tuple[Path, int, str, str]]:
    """
    Extract significant code lines from source files.

    Returns:
        List of (file_path, line_number, original_line, semantic_query) tuples.
    """
    candidates: list[tuple[Path, int, str, str]] = []

    for file_path in iter_files(root):
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(errors="ignore")
        except Exception:
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            for pattern, query_template in SIGNIFICANT_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    # Skip dunder methods and test functions
                    if name.startswith("__") or name.startswith("test_"):
                        continue
                    # Create semantic query
                    query = f"where is the {query_template.format(name)} defined"
                    candidates.append((file_path, line_num, line.strip(), query))
                    break

    # Sample randomly if we have too many
    if len(candidates) > max_samples:
        candidates = random.sample(candidates, max_samples)

    return candidates


def _test_chunk_size(
    root: Path,
    db_path: Path,
    chunk_size: int,
    samples: list[tuple[Path, int, str, str]],
    model: str | None,
) -> tuple[float, int]:
    """
    Test a specific chunk size and return accuracy metrics.

    Returns:
        (accuracy_score, hit_count) - accuracy is 0.0-1.0
    """
    # Resolve model from arg, env, or default (avoid passing None to search)
    resolved_model = resolve_model(model)

    # Index with this chunk size (use AST when available)
    index_path(
        root=root,
        db_path=db_path,
        model=model,
        chunk_lines=chunk_size,
        overlap=max(5, chunk_size // 6),
        ast=None,  # Auto-detect: use AST chunking when tree-sitter is available
    )

    hits = 0
    total_score = 0.0

    for file_path, line_num, _original, query in samples:
        results, _ = search_query(db_path=db_path, q=query, top_k=5, model=resolved_model)

        # Check if correct file is in top results
        # DB stores absolute paths (resolved)
        file_str = str(file_path.resolve())
        for i, hit in enumerate(results):
            if hit.path == file_str:
                # Check if line number is within the chunk
                if hit.start_line <= line_num <= hit.end_line:
                    hits += 1
                    # Weight by position (top result = 1.0, 5th = 0.2)
                    total_score += (5 - i) / 5
                    break

    accuracy = total_score / len(samples) if samples else 0.0
    return accuracy, hits


def cmd_tune(args: argparse.Namespace) -> int:
    """
    Auto-tune chunk size for optimal search relevance.

    Samples significant code patterns from the codebase, tests different
    chunk sizes, and recommends the optimal configuration.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    root = Path(args.path).resolve()
    use_json = getattr(args, "json", False)

    if not use_json:
        print("Analyzing codebase for significant patterns...")
    samples = _extract_significant_lines(root, max_samples=args.samples)

    if len(samples) < 3:
        if use_json:
            print(
                json.dumps(
                    {
                        "error": "Not enough significant code patterns found for tuning",
                        "samples_found": len(samples),
                        "samples_required": 3,
                    }
                )
            )
        else:
            print("Not enough significant code patterns found for tuning.")
            print("Need at least 3 function/class definitions.")
        return 1

    if not use_json:
        print(f"Found {len(samples)} test patterns:")
        for file_path, line_num, _original, query in samples[:5]:
            rel_path = file_path.relative_to(root) if file_path.is_relative_to(root) else file_path
            print(f'  {rel_path}:{line_num} -> "{query[:50]}..."')
        if len(samples) > 5:
            print(f"  ... and {len(samples) - 5} more")
        print()

    # Test different chunk sizes
    chunk_sizes = [30, 45, 60, 90, 120]
    results: list[tuple[int, float, int]] = []

    # Use temp directory for test indexes
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for chunk_size in chunk_sizes:
                db_path = Path(tmpdir) / f"test_{chunk_size}.sqlite"
                if not use_json:
                    print(f"Testing chunk size {chunk_size}...", end=" ", flush=True)

                try:
                    accuracy, hits = _test_chunk_size(
                        root=root,
                        db_path=db_path,
                        chunk_size=chunk_size,
                        samples=samples,
                        model=args.model,
                    )
                    results.append((chunk_size, accuracy, hits))
                    if not use_json:
                        print(f"accuracy={accuracy:.2f} ({hits}/{len(samples)} hits)")
                except Exception as e:
                    if not use_json:
                        print(f"failed: {e}")
                    results.append((chunk_size, 0.0, 0))
    except KeyboardInterrupt:
        if use_json:
            print(json.dumps({"error": "Interrupted by user (Ctrl-C)"}))
        else:
            print("\n\nInterrupted by user (Ctrl-C).")
            print("Tuning cancelled. No results saved.")
        return 130  # Standard SIGINT exit code (128 + 2)

    best_chunk = 60  # default
    best_accuracy = 0.0

    for chunk_size, accuracy, _hits in results:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_chunk = chunk_size

    if use_json:
        output = {
            "samples_tested": len(samples),
            "results": [
                {"chunk_size": cs, "accuracy": acc, "hits": h, "total": len(samples)}
                for cs, acc, h in results
            ],
            "recommended_chunk_size": best_chunk,
            "best_accuracy": best_accuracy,
        }
        print(json.dumps(output))
        return 0

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"{'Chunk Size':<12} {'Accuracy':<10} {'Hits':<8}")
    print("-" * 30)

    for chunk_size, accuracy, hits in results:
        print(f"{chunk_size:<12} {accuracy:<10.2f} {hits}/{len(samples)}")

    print("-" * 30)
    print(f"\nRecommended chunk size: {best_chunk} lines")

    # Save to .env if requested
    if args.save:
        env_file = root / ".env"
        _save_chunk_lines_to_env(env_file, best_chunk)
        print(f"\nSaved AUGGIE_CHUNK_LINES={best_chunk} to {env_file}")
    else:
        print("\nTo use this setting:")
        print(f"  export AUGGIE_CHUNK_LINES={best_chunk}")
        print(f"  # Or add to .env: AUGGIE_CHUNK_LINES={best_chunk}")
        print(f"  # Or use: auggie index . --chunk-lines {best_chunk}")
        print("\n  Tip: Use --save to automatically save to .env")

    # Offer to reindex with optimal settings
    if args.apply:
        from ._common import resolve_db_path

        # If root is a file, use its parent directory for repo_root
        if args.repo_root:
            repo_root = args.repo_root.resolve()
        elif root.is_file():
            repo_root = root.parent
        else:
            repo_root = root
        db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)

        print(f"\nReindexing with optimal chunk size ({best_chunk})...")
        if db.exists():
            db.unlink()

            # Also delete cache file if it exists
            from ..cache import get_cache_path

            cache_path = get_cache_path(db)
            if cache_path.exists():
                cache_path.unlink()

        try:
            stats = index_path(
                root=root,
                db_path=db,
                model=args.model,
                chunk_lines=best_chunk,
                overlap=max(5, best_chunk // 6),
                ast=None,  # Auto-detect: use AST chunking when tree-sitter is available
            )
            print(f"Indexed into {db}")
            print(f"  Files: {stats.files_indexed} indexed")
            print(f"  Chunks: {stats.chunks_total} ({stats.chunks_embedded} embedded)")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl-C).")
            print("Reindex cancelled. Partial progress may have been saved.")
            return 130  # Standard SIGINT exit code (128 + 2)

    return 0
