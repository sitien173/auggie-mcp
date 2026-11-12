from __future__ import annotations

import io
import sys
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie.commands.clean import cmd_clean
from auggie.commands.reset import cmd_reset
from auggie.db import connect


def _seed_chunk(
    db_path: Path,
    *,
    file_id: int,
    path: str,
    branch: str,
    chunk_id: int,
) -> None:
    con = connect(db_path, init_fts=True)
    try:
        con.execute(
            """
            INSERT INTO files (id, path, branch, mtime_ns, size, sha256)
            VALUES (?, ?, ?, 1, 10, ?)
            """,
            (file_id, path, branch, f"sha-{file_id}"),
        )
        con.execute(
            """
            INSERT INTO chunks (id, file_id, chunk_index, start_line, end_line, text, text_sha256, dim, model)
            VALUES (?, ?, 0, 1, 2, ?, ?, 2, 'voyage-code-3')
            """,
            (chunk_id, file_id, f"text-{chunk_id}", f"text-sha-{chunk_id}"),
        )
        con.commit()
    finally:
        con.close()


class Phase4MaintenanceTests(unittest.TestCase):
    def test_branch_reset_deletes_lance_vectors_for_current_branch(self) -> None:
        tmp_path = Path(mkdtemp(prefix="auggie-phase4-reset-"))
        db_path = tmp_path / ".auggie" / "index.sqlite"

        _seed_chunk(db_path, file_id=1, path=str(tmp_path / "main.py"), branch="main", chunk_id=11)
        _seed_chunk(db_path, file_id=2, path=str(tmp_path / "util.py"), branch="main", chunk_id=12)
        _seed_chunk(db_path, file_id=3, path=str(tmp_path / "other.py"), branch="other", chunk_id=21)

        deleted_ids: list[list[int]] = []

        class FakeStore:
            def __init__(self, root):  # noqa: ARG002
                pass

            def delete_chunk_ids(self, chunk_ids):
                deleted_ids.append(sorted(chunk_ids))

        args = Namespace(
            repo_root=tmp_path,
            db=db_path,
            profile=None,
            global_cache=False,
            json=False,
            force=True,
            all=False,
        )

        with (
            patch("auggie.commands.reset.get_current_branch", return_value="main"),
            patch("auggie.commands.reset.LanceStore", FakeStore),
        ):
            with redirect_stdout(io.StringIO()):
                exit_code = cmd_reset(args)

        con = connect(db_path, init_fts=False)
        try:
            remaining = con.execute("SELECT id FROM chunks ORDER BY id").fetchall()
        finally:
            con.close()

        self.assertEqual(exit_code, 0)
        self.assertEqual(deleted_ids, [[11, 12]])
        self.assertEqual(remaining, [(21,)])

    def test_reset_all_removes_sqlite_and_lance_store(self) -> None:
        tmp_path = Path(mkdtemp(prefix="auggie-phase4-reset-all-"))
        db_path = tmp_path / ".auggie" / "index.sqlite"
        lance_dir = db_path.parent / "lance"
        cache_path = db_path.parent / "cache.sqlite"

        _seed_chunk(db_path, file_id=1, path=str(tmp_path / "main.py"), branch="main", chunk_id=11)
        lance_dir.mkdir(parents=True, exist_ok=True)
        (lance_dir / "sentinel.txt").write_text("vector store", encoding="utf-8")
        cache_path.write_text("cache", encoding="utf-8")

        args = Namespace(
            repo_root=tmp_path,
            db=db_path,
            profile=None,
            global_cache=False,
            json=False,
            force=True,
            all=True,
        )

        with patch("auggie.cache.get_cache_path", return_value=cache_path):
            with redirect_stdout(io.StringIO()):
                exit_code = cmd_reset(args)

        self.assertEqual(exit_code, 0)
        self.assertFalse(db_path.exists())
        self.assertFalse(lance_dir.exists())
        self.assertFalse(cache_path.exists())

    def test_clean_deletes_lance_vectors_for_stale_files_and_pruned_branches(self) -> None:
        tmp_path = Path(mkdtemp(prefix="auggie-phase4-clean-"))
        db_path = tmp_path / ".auggie" / "index.sqlite"
        existing_file = tmp_path / "live.py"
        existing_file.write_text("print('ok')\n", encoding="utf-8")

        _seed_chunk(
            db_path,
            file_id=1,
            path=str(tmp_path / "missing.py"),
            branch="main",
            chunk_id=31,
        )
        _seed_chunk(
            db_path,
            file_id=2,
            path=str(existing_file),
            branch="old-branch",
            chunk_id=41,
        )
        _seed_chunk(
            db_path,
            file_id=3,
            path=str(existing_file),
            branch="main",
            chunk_id=51,
        )

        deleted_ids: list[list[int]] = []

        class FakeStore:
            def __init__(self, root):  # noqa: ARG002
                pass

            def delete_chunk_ids(self, chunk_ids):
                deleted_ids.append(sorted(chunk_ids))

        args = Namespace(
            repo_root=tmp_path,
            db=db_path,
            profile=None,
            global_cache=False,
            json=False,
            vacuum=False,
        )

        with (
            patch("auggie.commands.clean.get_git_branches", return_value={"main"}),
            patch("auggie.commands.clean.LanceStore", FakeStore),
        ):
            with redirect_stdout(io.StringIO()):
                exit_code = cmd_clean(args)

        con = connect(db_path, init_fts=False)
        try:
            remaining = con.execute("SELECT id FROM chunks ORDER BY id").fetchall()
        finally:
            con.close()

        self.assertEqual(exit_code, 0)
        self.assertEqual(deleted_ids, [[31], [41]])
        self.assertEqual(remaining, [(51,)])


if __name__ == "__main__":
    unittest.main()
