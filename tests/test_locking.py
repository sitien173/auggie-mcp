"""Tests for auggie_mcp.locking."""

import multiprocessing
import tempfile
import unittest
from pathlib import Path

from auggie_mcp import locking


def _try_acquire_in_child(repo_str: str, queue: multiprocessing.Queue) -> None:
    try:
        with locking.index_lock(Path(repo_str)):
            queue.put("acquired")
    except locking.IndexBusyError:
        queue.put("busy")


class LockingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)

    def test_second_holder_fails_immediately_then_succeeds_after_release(self):
        ctx = multiprocessing.get_context("spawn")
        with locking.index_lock(self.repo):
            q1: multiprocessing.Queue = ctx.Queue()
            p1 = ctx.Process(target=_try_acquire_in_child, args=(str(self.repo), q1))
            p1.start()
            p1.join(timeout=5)
            self.assertEqual(p1.exitcode, 0)
            self.assertEqual(q1.get(timeout=1), "busy")

        q2: multiprocessing.Queue = ctx.Queue()
        p2 = ctx.Process(target=_try_acquire_in_child, args=(str(self.repo), q2))
        p2.start()
        p2.join(timeout=5)
        self.assertEqual(p2.exitcode, 0)
        self.assertEqual(q2.get(timeout=1), "acquired")
