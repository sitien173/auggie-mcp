"""
Watch command for auggie.
"""

from __future__ import annotations

import argparse
import sys
import time

from ..index_lock import IndexLock, IndexLockError
from ..indexer import index_path
from ..watcher import (
    AuggieWatchService,
    WatchEventHandler,
    create_observer,
)
from ._common import require_embedding_config
from .index import _resolve_chunk_lines, _resolve_overlap, _resolve_paths


def _print_pass_log(stats, duration_s: float) -> None:
    """Print a single watch pass summary to stderr."""
    print(
        f"[watch] refresh complete: {stats.files_indexed} indexed, "
        f"{stats.files_skipped} skipped in {duration_s:.2f}s",
        file=sys.stderr,
    )


def _wait_for_interrupt(observer) -> None:
    """Block until Ctrl-C stops the foreground daemon."""
    while observer.is_alive():
        observer.join(timeout=1.0)


def cmd_watch(args: argparse.Namespace) -> int:
    """
    Run the foreground watcher daemon.
    """
    if not require_embedding_config():
        return 1

    root, db = _resolve_paths(args)
    chunk_lines = _resolve_chunk_lines(args)
    overlap = _resolve_overlap(args)
    detect = not getattr(args, "no_detect", False)
    quiet = getattr(args, "quiet", False)

    # Enforce one watcher per index. The lock lives next to the SQLite db and is
    # released by the OS when this process exits, even on a crash.
    lock = IndexLock(db.parent / "index.lock")
    try:
        lock.acquire()
    except IndexLockError as e:
        print(f"Error: {e}. Only one writer per index is allowed.", file=sys.stderr)
        return 1

    def run_incremental_pass(_dirty_paths: set[str] | None = None) -> None:
        start = time.perf_counter()
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
            ast=getattr(args, "ast", False),
        )
        if not quiet:
            _print_pass_log(stats, time.perf_counter() - start)

    try:
        try:
            run_incremental_pass()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl-C).", file=sys.stderr)
            return 130

        service = AuggieWatchService(root=root, index_callback=run_incremental_pass)
        observer = None
        try:
            handler = WatchEventHandler(service)
            observer = create_observer(root, handler)
            service.start()
            observer.start()
            _wait_for_interrupt(observer)
        except KeyboardInterrupt:
            pass
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        finally:
            service.stop()
            service.join()
            if observer is not None:
                observer.stop()
                observer.join()

        return 0
    finally:
        lock.release()
