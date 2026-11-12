"""
Watcher daemon support for auggie.
"""

from __future__ import annotations

import os
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
