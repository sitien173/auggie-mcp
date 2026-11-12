"""
Shared OS-level write lock for one semantic index.

`auggie index`, `auggie reindex`, and `auggie watch` take this lock so only one
process writes a given index at a time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class IndexLockError(Exception):
    """Raised when another process already holds the lock for an index."""

    def __init__(self, lock_path: Path, holder_pid: str | None):
        self.lock_path = lock_path
        self.holder_pid = holder_pid
        who = f" (PID {holder_pid})" if holder_pid else ""
        super().__init__(f"index is locked by another auggie process{who}")


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


class IndexLock:
    """
    OS-level single-writer lock for one index.

    Released by the OS on process exit, so a crashed writer never leaves a
    stale lock. A second writer on the same index fails fast with
    IndexLockError instead of running concurrently.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fh = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+")
        try:
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
                raise IndexLockError(self.lock_path, holder)
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
            self._fh = fh
        except BaseException:
            # Closing the handle drops the OS lock if it was taken.
            fh.close()
            raise

    def release(self) -> None:
        # The lock file is never unlinked: removing it would let a waiter lock
        # the old inode while a new writer locks a fresh file at the same path.
        if self._fh is None:
            return
        try:
            _unlock_handle(self._fh)
        except OSError:
            pass
        self._fh.close()
        self._fh = None
