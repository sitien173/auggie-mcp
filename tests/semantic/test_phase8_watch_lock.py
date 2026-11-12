"""Single-instance watch lock tests."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import mkdtemp

sys.modules.setdefault(
    "tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable)
)

from auggie.index_lock import IndexLock, IndexLockError


class WatchLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(mkdtemp(prefix="auggie-watchlock-"))
        self.lock_path = self.root / ".auggie" / "index.lock"

    def test_second_acquire_same_index_is_refused(self) -> None:
        first = IndexLock(self.lock_path)
        first.acquire()
        try:
            second = IndexLock(self.lock_path)
            with self.assertRaises(IndexLockError):
                second.acquire()
        finally:
            first.release()

    def test_release_frees_lock_for_next_acquire(self) -> None:
        first = IndexLock(self.lock_path)
        first.acquire()
        first.release()

        second = IndexLock(self.lock_path)
        second.acquire()  # must not raise now
        second.release()

    def test_different_index_is_independent(self) -> None:
        a = IndexLock(self.lock_path)
        b = IndexLock(self.root / "other" / "index.lock")
        a.acquire()
        try:
            b.acquire()  # different file, must not raise
            b.release()
        finally:
            a.release()


if __name__ == "__main__":
    unittest.main()
