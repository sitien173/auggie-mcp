"""Shared index write lock tests for index, reindex, and watch."""

from __future__ import annotations

import io
import json
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from auggie.commands.index import cmd_index
from auggie.commands.reindex import cmd_reindex
from auggie.commands.watch import cmd_watch
from auggie.index_lock import IndexLock, IndexLockError


def _stats():
    return types.SimpleNamespace(
        model="voyage-code-3",
        files_indexed=1,
        files_skipped=0,
        files_scanned=1,
        chunks_total=1,
        chunks_embedded=1,
        chunks_reused=0,
        chunks_reused_global=0,
        chunks_reused_local=0,
        tokens_saved_estimate=0,
        dedup_ratio=0.0,
        ast_mode="off",
        ast_hint=None,
        indexed_files=None,
    )


def _args(root: Path, db: Path, **extra) -> Namespace:
    values = dict(
        path=str(root),
        db=str(db),
        profile=None,
        global_cache=False,
        repo_root=root,
        model="voyage-code-3",
        dimensions=None,
        chunk_lines=30,
        overlap=15,
        max_bytes=2_000_000,
        exclude=[],
        include=[],
        ast=None,
        no_detect=False,
        quiet=True,
        verbose=False,
        json=True,
        list=False,
    )
    values.update(extra)
    return Namespace(**values)


class IndexLockCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(mkdtemp(prefix="auggie-index-lock-"))
        self.db = self.root / ".auggie" / "index.sqlite"
        self.db.parent.mkdir(parents=True)
        self.lock_path = self.db.parent / "index.lock"
        self.held = IndexLock(self.lock_path)

    def tearDown(self) -> None:
        self.held.release()

    def _assert_free(self) -> None:
        probe = IndexLock(self.lock_path)
        probe.acquire()
        probe.release()

    def _run_index(self, args, index_path):
        with (
            patch("auggie.commands.index.require_embedding_config", return_value=True),
            patch("auggie.commands.index._resolve_paths", return_value=(self.root, self.db)),
            patch("auggie.commands.index.index_path", index_path),
        ):
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                try:
                    code = cmd_index(args)
                except KeyboardInterrupt:
                    code = "raised"
        return code, out.getvalue()

    def test_index_refuses_when_lock_held(self) -> None:
        self.held.acquire()
        calls: list[dict] = []
        code, out = self._run_index(
            _args(self.root, self.db), lambda **kw: calls.append(kw) or _stats()
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(out)["error_code"], "INDEX_BUSY")
        self.assertEqual(calls, [])

    def test_index_non_json_busy_prints_stderr(self) -> None:
        self.held.acquire()
        err = io.StringIO()
        with (
            patch("auggie.commands.index.require_embedding_config", return_value=True),
            patch("auggie.commands.index._resolve_paths", return_value=(self.root, self.db)),
            patch("auggie.commands.index.index_path") as ip,
            redirect_stderr(err),
            redirect_stdout(io.StringIO()),
        ):
            code = cmd_index(_args(self.root, self.db, json=False))
        self.assertEqual(code, 1)
        self.assertIn("locked", err.getvalue())
        ip.assert_not_called()

    def test_reindex_refuses_when_lock_held_and_keeps_db(self) -> None:
        self.db.write_text("existing", encoding="utf-8")
        self.held.acquire()
        out = io.StringIO()
        with (
            patch("auggie.commands.reindex.require_embedding_config", return_value=True),
            patch("auggie.commands.reindex.index_path") as ip,
            redirect_stdout(out),
            redirect_stderr(io.StringIO()),
        ):
            code = cmd_reindex(_args(self.root, self.db))
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["error_code"], "INDEX_BUSY")
        self.assertTrue(self.db.exists())
        self.assertEqual(self.db.read_text(encoding="utf-8"), "existing")
        ip.assert_not_called()

    def test_watch_refuses_when_lock_held(self) -> None:
        self.held.acquire()
        err = io.StringIO()
        with (
            patch("auggie.commands.watch.require_embedding_config", return_value=True),
            patch("auggie.commands.watch._resolve_paths", return_value=(self.root, self.db)),
            patch("auggie.commands.watch.index_path") as ip,
            patch("auggie.commands.watch.AuggieWatchService"),
            patch("auggie.commands.watch.create_observer"),
            patch("auggie.commands.watch._wait_for_interrupt", side_effect=KeyboardInterrupt),
            redirect_stderr(err),
        ):
            code = cmd_watch(_args(self.root, self.db))
        self.assertEqual(code, 1)
        ip.assert_not_called()

    def test_lock_released_after_normal_index(self) -> None:
        code, _ = self._run_index(_args(self.root, self.db), lambda **kw: _stats())
        self.assertEqual(code, 0)
        self._assert_free()

    def test_lock_released_after_index_path_raises(self) -> None:
        def boom(**kw):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            self._run_index(_args(self.root, self.db), boom)
        self._assert_free()

    def test_lock_released_after_keyboard_interrupt(self) -> None:
        def interrupt(**kw):
            raise KeyboardInterrupt

        code, _ = self._run_index(_args(self.root, self.db), interrupt)
        self.assertEqual(code, 130)
        self._assert_free()

    def test_lock_held_during_index_run(self) -> None:
        seen: list[bool] = []

        def probe(**kw):
            try:
                IndexLock(self.lock_path).acquire()
                seen.append(False)
            except IndexLockError:
                seen.append(True)
            return _stats()

        code, _ = self._run_index(_args(self.root, self.db), probe)
        self.assertEqual(code, 0)
        self.assertEqual(seen, [True])

    def test_list_mode_succeeds_while_lock_held(self) -> None:
        (self.root / "a.txt").write_text("hello\n", encoding="utf-8")
        self.held.acquire()
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cmd_index(_args(self.root, self.db, list=True))
        self.assertEqual(code, 0)


class IndexLockUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock_path = Path(mkdtemp(prefix="auggie-index-lock-unit-")) / "index.lock"

    def test_release_keeps_lock_file_and_next_holder_excludes_others(self) -> None:
        first = IndexLock(self.lock_path)
        first.acquire()
        first.release()
        self.assertTrue(self.lock_path.exists())

        second = IndexLock(self.lock_path)
        second.acquire()
        try:
            with self.assertRaises(IndexLockError):
                IndexLock(self.lock_path).acquire()
        finally:
            second.release()

    def test_failure_after_lock_closes_handle_and_frees_lock(self) -> None:
        first = IndexLock(self.lock_path)
        raised = None
        with patch("auggie.index_lock.os.getpid", side_effect=KeyboardInterrupt):
            try:
                first.acquire()
            except KeyboardInterrupt as exc:
                # Keeping the exception keeps its traceback frames alive, so a
                # leaked handle is not freed by garbage collection.
                raised = exc
        self.assertIsNotNone(raised)
        second = IndexLock(self.lock_path)
        second.acquire()
        second.release()


if __name__ == "__main__":
    unittest.main()
