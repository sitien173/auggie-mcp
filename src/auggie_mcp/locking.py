"""Advisory file locking for the repository index."""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class IndexBusyError(BlockingIOError):
    """Raised immediately when the repository index lock is already held."""


def lock_path(repo_root: Path | str) -> Path:
    """Return the path to the index lock file for a repository."""
    return Path(repo_root) / ".auggie" / "index.lock"


@contextmanager
def index_lock(repo_root: Path | str) -> Generator[Path, None, None]:
    """Acquire an exclusive non-blocking flock on ``<repo>/.auggie/index.lock``."""
    target = lock_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise IndexBusyError(
                    f"Index lock is held by another process: {target}"
                ) from exc
            raise
        try:
            yield target
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
