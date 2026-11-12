from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie.db import connect
from auggie.indexer import index_path


class _FakeTqdm:
    def __init__(self, iterable=None, total=None, **kwargs):  # noqa: ARG002
        self._iterable = iterable
        self.total = total

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):  # noqa: ARG002
        return None


class _FakeLanceStore:
    rows: dict[int, tuple[bytes, int]]
    deleted_batches: list[list[int]]

    def __init__(self, root):  # noqa: ARG002
        if not hasattr(type(self), "rows"):
            type(self).rows = {}
        if not hasattr(type(self), "deleted_batches"):
            type(self).deleted_batches = []

    def fetch_vectors_by_chunk_ids(self, chunk_ids):
        return {chunk_id: type(self).rows[chunk_id] for chunk_id in chunk_ids if chunk_id in type(self).rows}

    def upsert_chunk_vectors(self, rows, *, model):  # noqa: ARG002
        for chunk_id, emb, dim in rows:
            type(self).rows[chunk_id] = (emb, dim)

    def delete_chunk_ids(self, chunk_ids):
        if chunk_ids:
            type(self).deleted_batches.append(list(chunk_ids))
        for chunk_id in chunk_ids:
            type(self).rows.pop(chunk_id, None)


class Phase3IndexerRegressionTests(unittest.TestCase):
    def test_indexed_file_that_becomes_binary_is_removed(self) -> None:
        tmp_path = Path(mkdtemp(prefix="auggie-phase3-regression-"))
        db_path = tmp_path / ".auggie" / "index.sqlite"
        source_path = tmp_path / "sample.py"
        source_path.write_text("print('hello')\n", encoding="utf-8")

        _FakeLanceStore.rows = {}
        _FakeLanceStore.deleted_batches = []

        with (
            patch("auggie.indexer.tqdm", _FakeTqdm),
            patch("auggie.indexer.LanceStore", _FakeLanceStore),
            patch(
                "auggie.indexer.embed_texts",
                return_value=([b"\x00\x00\x80?\x00\x00\x00@"], 2),
            ),
        ):
            first = index_path(
                root=tmp_path,
                db_path=db_path,
                model="voyage-code-3",
                detect=False,
                branch="main",
            )

        self.assertEqual(first.files_indexed, 1)

        con = connect(db_path, init_fts=False)
        try:
            before_rows = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            before_chunks = con.execute("SELECT id FROM chunks").fetchall()
        finally:
            con.close()
        self.assertEqual(before_rows, 1)
        self.assertTrue(before_chunks)

        source_path.write_bytes(b"\x00\x01\x02\x03")

        with (
            patch("auggie.indexer.tqdm", _FakeTqdm),
            patch("auggie.indexer.LanceStore", _FakeLanceStore),
            patch(
                "auggie.indexer.embed_texts",
                return_value=([b"\x00\x00\x80?\x00\x00\x00@"], 2),
            ),
        ):
            refresh = index_path(
                root=tmp_path,
                db_path=db_path,
                model="voyage-code-3",
                detect=False,
                branch="main",
            )

        con = connect(db_path, init_fts=False)
        try:
            after_rows = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            after_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            con.close()

        self.assertEqual(refresh.files_indexed, 0)
        self.assertEqual(after_rows, 0)
        self.assertEqual(after_chunks, 0)
        self.assertTrue(_FakeLanceStore.deleted_batches)


if __name__ == "__main__":
    unittest.main()
