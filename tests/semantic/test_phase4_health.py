from __future__ import annotations

import io
import json
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

from auggie.commands.health import cmd_health
from auggie.db import connect


def _seed_chunk(db_path: Path, *, file_id: int, chunk_id: int) -> None:
    con = connect(db_path, init_fts=True)
    try:
        con.execute(
            """
            INSERT INTO files (id, path, branch, mtime_ns, size, sha256)
            VALUES (?, ?, 'main', 1, 10, ?)
            """,
            (file_id, str(db_path.parent / f"file-{file_id}.py"), f"sha-{file_id}"),
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


class Phase4HealthTests(unittest.TestCase):
    def test_health_reports_and_fixes_lance_sqlite_orphans(self) -> None:
        tmp_path = Path(mkdtemp(prefix="auggie-phase4-health-"))
        db_path = tmp_path / ".auggie" / "index.sqlite"
        _seed_chunk(db_path, file_id=1, chunk_id=11)
        _seed_chunk(db_path, file_id=2, chunk_id=12)

        deleted_ids: list[list[int]] = []

        class FakeStore:
            def __init__(self, root):  # noqa: ARG002
                pass

            def count(self):
                return 2

            def list_chunk_ids(self):
                return [12, 13]

            def delete_chunk_ids(self, chunk_ids):
                deleted_ids.append(sorted(chunk_ids))

        base_args = dict(
            repo_root=tmp_path,
            db=db_path,
            profile=None,
            global_cache=False,
            vacuum=False,
            rebuild_fts=False,
            reindex=False,
            integrity=False,
            full=False,
            json=True,
        )

        stdout = io.StringIO()
        with patch("auggie.commands.health.LanceStore", FakeStore, create=True):
            with redirect_stdout(stdout):
                exit_code = cmd_health(Namespace(**base_args, fix=False))
        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["lance"]["sqlite_chunk_count"], 2)
        self.assertEqual(report["lance"]["lance_chunk_count"], 2)
        self.assertEqual(report["lance"]["sqlite_missing_in_lance"], [11])
        self.assertEqual(report["lance"]["lance_missing_in_sqlite"], [13])

        stdout = io.StringIO()
        with patch("auggie.commands.health.LanceStore", FakeStore, create=True):
            with redirect_stdout(stdout):
                fix_exit_code = cmd_health(Namespace(**base_args, fix=True))
        fixed_report = json.loads(stdout.getvalue())

        self.assertEqual(fix_exit_code, 1)
        self.assertEqual(deleted_ids, [[13]])
        self.assertEqual(fixed_report["lance"]["deleted_lance_orphans"], [13])
        self.assertEqual(fixed_report["lance"]["sqlite_missing_in_lance"], [11])


if __name__ == "__main__":
    unittest.main()
