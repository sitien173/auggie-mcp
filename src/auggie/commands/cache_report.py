"""
Cache report command for auggie.

Shows cache effectiveness metrics including hit rates,
time saved, and entry counts for all cache levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import resolve_db_path


def cmd_cache_report(args: argparse.Namespace) -> int:
    """
    Show cache effectiveness report.

    Displays hit rates, time saved, and entry counts for L1, L2, and L3 caches.
    Useful for understanding cache performance and tuning.

    Args:
        args: Parsed command-line arguments containing:
            - hours: Time period to analyze (default: 24)
            - json: Whether to output as JSON (default: True)
            - clear: Whether to clear all caches
            - db, profile, global_cache, repo_root: Scope options

    Returns:
        Exit code (0 for success).
    """
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd()
    db = resolve_db_path(args.db, args.profile, args.global_cache, repo_root)
    use_json = getattr(args, "json", True)
    hours = getattr(args, "hours", 24)
    clear = getattr(args, "clear", False)

    from ..cache import get_cache_path

    cache_path = get_cache_path(db)

    if not cache_path.exists():
        if use_json:
            print(json.dumps({"error": "No cache database found", "path": str(cache_path)}))
        else:
            print(f"No cache database found at {cache_path}")
        return 1

    from ..cache import clear_all_caches, connect_cache, get_cache_report

    cache_con = connect_cache(cache_path)

    if clear:
        # Clear all caches
        counts = clear_all_caches(cache_con)
        if use_json:
            print(
                json.dumps(
                    {
                        "status": "cleared",
                        "cleared": counts,
                        "cache_path": str(cache_path),
                    }
                )
            )
        else:
            print("Cache cleared:")
            print(f"  L1 (embeddings): {counts.get('L1', 0)} entries removed")
            print(f"  L2 (search): {counts.get('L2', 0)} entries removed")
            print(f"  L3 (rerank): {counts.get('L3', 0)} entries removed")
        return 0

    # Get cache report
    report = get_cache_report(cache_con, since_hours=hours)

    if use_json:
        report["cache_path"] = str(cache_path)
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report, hours)

    return 0


def _print_human_report(report: dict, hours: int) -> None:
    """Print cache report in human-readable format."""
    print(f"\n{'─' * 50}")
    print(f"  Cache Effect Report (last {hours} hours)")
    print(f"{'─' * 50}\n")

    # Stats by level
    print(f"{'Level':<10} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10} {'Time Saved':>12}")
    print("─" * 50)

    total_hits = 0
    total_misses = 0
    total_time_saved = 0

    for level in ["L1", "L2", "L3"]:
        stats = report.get("levels", {}).get(level, {})
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        time_saved_ms = stats.get("time_saved_ms", 0)

        total_hits += hits
        total_misses += misses
        total_time_saved += time_saved_ms

        if hits + misses > 0:
            hit_rate = f"{hits / (hits + misses) * 100:.1f}%"
        else:
            hit_rate = "N/A"

        level_name = {"L1": "Embed", "L2": "Search", "L3": "Rerank"}.get(level, level)
        time_saved_str = (
            f"{time_saved_ms / 1000:.1f}s" if time_saved_ms > 1000 else f"{time_saved_ms}ms"
        )

        print(f"{level_name:<10} {hits:>8} {misses:>8} {hit_rate:>10} {time_saved_str:>12}")

    print("─" * 50)

    # Total
    if total_hits + total_misses > 0:
        total_hit_rate = f"{total_hits / (total_hits + total_misses) * 100:.1f}%"
    else:
        total_hit_rate = "N/A"
    total_time_str = (
        f"{total_time_saved / 1000:.1f}s" if total_time_saved > 1000 else f"{total_time_saved}ms"
    )
    print(
        f"{'Total':<10} {total_hits:>8} {total_misses:>8} {total_hit_rate:>10} {total_time_str:>12}"
    )

    # Entry counts
    print(f"\n{'─' * 50}")
    print("  Cache Size")
    print("─" * 50)

    entries = report.get("entries", {})
    for level in ["L1", "L2", "L3"]:
        count = entries.get(level, 0)
        level_name = {"L1": "L1 (embeddings)", "L2": "L2 (search)", "L3": "L3 (rerank)"}.get(
            level, level
        )
        print(f"  {level_name}: {count} entries")

    # Recommendations
    recommendations = []
    for level in ["L1", "L2", "L3"]:
        stats = report.get("levels", {}).get(level, {})
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        if hits + misses >= 10:  # Only suggest if we have enough data
            hit_rate = hits / (hits + misses)
            if hit_rate < 0.5:
                level_name = {"L1": "embedding", "L2": "search", "L3": "rerank"}.get(level, level)
                recommendations.append(
                    f"  - {level} hit rate is low ({hit_rate:.0%}). Queries may be too varied."
                )

    if recommendations:
        print(f"\n{'─' * 50}")
        print("  Recommendations")
        print("─" * 50)
        for rec in recommendations:
            print(rec)

    print()
