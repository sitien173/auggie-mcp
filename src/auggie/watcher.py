"""
Watcher daemon support for auggie.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .indexer import DEFAULT_SKIP_DIRS

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - exercised indirectly in tests
    class FileSystemEventHandler:  # type: ignore[no-redef]
        pass

    Observer = None  # type: ignore[assignment]


DEFAULT_WATCH_DEBOUNCE_MS = int(os.environ.get("AUGGIE_WATCH_DEBOUNCE_MS", "500"))


class WatchLockError(Exception):
    """Raised when another watcher already holds the lock for an index."""

    def __init__(self, lock_path: Path, holder_pid: str | None):
        self.lock_path = lock_path
        self.holder_pid = holder_pid
        who = f" (PID {holder_pid})" if holder_pid else ""
        super().__init__(
            f"another `auggie watch` is already running for this index{who}"
        )


def _lock_handle(fh) -> None:
    """Take a non-blocking exclusive lock on the file; raise OSError if held."""
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(fh) -> None:
    """Release the lock taken by _lock_handle."""
    if sys.platform == "win32":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class WatchLock:
    """
    OS-level single-instance lock for one index.

    Held for the lifetime of the process and released by the OS on exit, so a
    crashed watcher never leaves a stale lock. A second `auggie watch` on the
    same index fails fast with WatchLockError instead of running a duplicate.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fh = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+")
        fh.seek(0)
        try:
            _lock_handle(fh)
        except OSError:
            holder = None
            try:
                fh.seek(0)
                holder = fh.read().strip() or None
            except OSError:
                pass
            fh.close()
            raise WatchLockError(self.lock_path, holder)
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            _unlock_handle(self._fh)
        except OSError:
            pass
        self._fh.close()
        self._fh = None
        try:
            self.lock_path.unlink()
        except OSError:
            pass


def _resolve_event_path(path: str | Path) -> str:
    """Normalize an event path to an absolute string."""
    return str(Path(path).resolve())


def _has_skipped_component(root: Path, path: Path) -> bool:
    """Return True when the path lives in a skipped directory."""
    try:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        rel_parts = path.resolve().parts
    return any(part in DEFAULT_SKIP_DIRS for part in rel_parts)


class AuggieWatchService:
    """Debounced single-writer index scheduler."""

    def __init__(
        self,
        *,
        root: Path,
        debounce_ms: int = DEFAULT_WATCH_DEBOUNCE_MS,
        index_callback: Callable[[set[str]], None],
    ) -> None:
        self.root = root.resolve()
        self.debounce_ms = debounce_ms
        self._index_callback = index_callback
        self._dirty_paths: set[str] = set()
        self._last_event_at: float | None = None
        self._stop_requested = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="auggie-watch-indexer", daemon=True)

    def should_watch_path(self, path: str | Path) -> bool:
        """Cheap path filter to avoid obvious watcher noise."""
        resolved = Path(path).resolve()
        auggie_dir = self.root / ".auggie"
        try:
            resolved.relative_to(auggie_dir)
            return False
        except ValueError:
            pass
        return not _has_skipped_component(self.root, resolved)

    def mark_dirty(self, path: str | Path) -> bool:
        """Queue a path for the next debounced indexing pass."""
        if not self.should_watch_path(path):
            return False

        resolved = _resolve_event_path(path)
        with self._condition:
            self._dirty_paths.add(resolved)
            self._last_event_at = time.monotonic()
            self._condition.notify_all()
        return True

    def start(self) -> None:
        """Start the background scheduler."""
        self._thread.start()

    def stop(self) -> None:
        """Request scheduler shutdown."""
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background scheduler to exit."""
        self._thread.join(timeout)

    def _run(self) -> None:
        debounce_seconds = self.debounce_ms / 1000.0

        while True:
            with self._condition:
                while not self._stop_requested and not self._dirty_paths:
                    self._condition.wait()
                if self._stop_requested:
                    return

                while not self._stop_requested:
                    assert self._last_event_at is not None
                    delay = debounce_seconds - (time.monotonic() - self._last_event_at)
                    if delay <= 0:
                        break
                    self._condition.wait(timeout=delay)
                if self._stop_requested:
                    return

                batch = set(self._dirty_paths)
                self._dirty_paths.clear()

            self._index_callback(batch)


class WatchEventHandler(FileSystemEventHandler):
    """Watchdog event adapter that feeds the scheduler."""

    def __init__(self, service: AuggieWatchService) -> None:
        self._service = service

    def on_created(self, event) -> None:
        self._handle_event(event)

    def on_modified(self, event) -> None:
        self._handle_event(event)

    def on_deleted(self, event) -> None:
        self._handle_event(event)

    def on_moved(self, event) -> None:
        self._handle_event(event)
        dest_path = getattr(event, "dest_path", None)
        if dest_path:
            self._service.mark_dirty(dest_path)

    def _handle_event(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        src_path = getattr(event, "src_path", None)
        if src_path:
            self._service.mark_dirty(src_path)


def create_observer(root: Path, handler: WatchEventHandler):
    """Create and configure the watchdog observer."""
    if Observer is None:
        raise ImportError(
            "File watching requires the watchdog package. "
            "Install the dependencies from requirements.txt."
        )

    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    return observer
