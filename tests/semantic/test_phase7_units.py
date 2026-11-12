from __future__ import annotations

import array
import hashlib
import importlib.util
import pathlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

import auggie
from auggie import db as db_module
from auggie import indexer, search
from auggie.indexer import IndexStats


def test_working_tree_import_path() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    assert pathlib.Path(auggie.__file__).resolve().is_relative_to(root)


def test_stat_gate_mtime_size_short_circuit() -> None:
    file_states = {str(Path("sample.py").resolve()): (1, 100, 20, "sha")}
    stats = IndexStats()
    path = Path("sample.py")

    with patch("auggie.indexer._read_file_if_indexable", side_effect=AssertionError("should not read")):
        existing_state = file_states[str(path.resolve())]
        st = type("Stat", (), {"st_mtime_ns": 100, "st_size": 20})()
        skipped = (
            existing_state is not None
            and existing_state[1] == st.st_mtime_ns
            and existing_state[2] == st.st_size
        )
        assert skipped is True


def test_stat_gate_sha_confirm_updates_only_metadata() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / ".auggie" / "index.sqlite"
        con = db_module.connect(db_path, init_fts=False)
        try:
            cur = con.execute(
                "INSERT INTO files(path, branch, mtime_ns, size, sha256) VALUES(?,?,?,?,?)",
                ("sample.py", "main", 1, 10, hashlib.sha256(b"alpha").hexdigest()),
            )
            file_id = int(cur.lastrowid)
            con.commit()

            indexer.touch_file_state(con, file_id, 99, 15)
            row = con.execute("SELECT mtime_ns, size, sha256 FROM files WHERE id = ?", (file_id,)).fetchone()
        finally:
            con.close()

        assert row == (99, 15, hashlib.sha256(b"alpha").hexdigest())


def test_chunk_diff_renumber_collision_keeps_contiguous_indices() -> None:
    class Chunk:
        def __init__(self, chunk_index: int, start_line: int, end_line: int, text: str) -> None:
            self.chunk_index = chunk_index
            self.start_line = start_line
            self.end_line = end_line
            self.text = text

    existing_chunks = [
        (101, 0, 1, 2, "alpha", hashlib.sha256(b"alpha").hexdigest(), 2, "voyage-code-3"),
        (102, 1, 3, 4, "beta", hashlib.sha256(b"beta").hexdigest(), 2, "voyage-code-3"),
    ]
    existing_embeddings = {
        hashlib.sha256(b"alpha").hexdigest(): (array.array("f", [1.0, 0.0]).tobytes(), 2),
        hashlib.sha256(b"beta").hexdigest(): (array.array("f", [0.0, 1.0]).tobytes(), 2),
    }
    chunks = [
        Chunk(0, 1, 2, "intro"),
        Chunk(1, 3, 4, "alpha"),
        Chunk(2, 5, 6, "beta"),
    ]
    chunk_hashes = [hashlib.sha256(c.text.encode()).hexdigest() for c in chunks]
    stats = IndexStats()

    preserved_rows, reusable_new_rows, chunks_to_embed, removed_chunk_ids = indexer._plan_chunk_diff(
        chunks,
        chunk_hashes,
        existing_chunks,
        existing_embeddings,
        {},
        stats,
    )

    assert preserved_rows == {}
    assert [idx for idx, _emb, _dim in reusable_new_rows] == [1, 2]
    assert chunks_to_embed == [(0, "intro")]
    assert removed_chunk_ids == []

    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / ".auggie" / "index.sqlite"
        con = db_module.connect(db_path, init_fts=False)
        try:
            cur = con.execute(
                "INSERT INTO files(path, branch, mtime_ns, size, sha256) VALUES(?,?,?,?,?)",
                ("sample.py", "main", 1, 10, "sha"),
            )
            file_id = int(cur.lastrowid)
            con.executemany(
                """
                INSERT INTO chunks(id, file_id, chunk_index, start_line, end_line, text, text_sha256, dim, model)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    (101, file_id, 0, 1, 2, "alpha", hashlib.sha256(b"alpha").hexdigest(), 2, "voyage-code-3"),
                    (102, file_id, 1, 3, 4, "beta", hashlib.sha256(b"beta").hexdigest(), 2, "voyage-code-3"),
                ],
            )
            con.commit()

            con.execute("DELETE FROM chunks WHERE id IN (?, ?)", (101, 102))
            vector_rows = indexer._store_chunk_metadata(
                con,
                file_id,
                chunks,
                chunk_hashes,
                preserved_rows,
                {0: (array.array("f", [0.5, 0.5]).tobytes(), 2)},
                reusable_new_rows,
                "voyage-code-3",
            )
            rows = con.execute("SELECT chunk_index FROM chunks ORDER BY chunk_index").fetchall()
        finally:
            con.close()

    assert rows == [(0,), (1,), (2,)]
    assert all(chunk_index >= 0 for (chunk_index,) in rows)
    assert len(vector_rows) == 3


@pytest.mark.skipif(importlib.util.find_spec("lancedb") is None, reason="lancedb not installed")
def test_lance_round_trip() -> None:
    from auggie.lance_store import LanceStore

    with TemporaryDirectory() as tmp:
        store = LanceStore(Path(tmp))
        rows = [
            (1, array.array("f", [1.0, 0.0]).tobytes(), 2),
            (2, array.array("f", [0.0, 1.0]).tobytes(), 2),
        ]
        store.upsert_chunk_vectors(rows, model="voyage-code-3")
        fetched = store.fetch_vectors_by_chunk_ids([1, 2])
        hits = store.ann_query(array.array("f", [1.0, 0.0]).tobytes(), limit=2)
        store.delete_chunk_ids([2])
        after_delete = store.fetch_vectors_by_chunk_ids([1, 2])

    assert set(fetched) == {1, 2}
    assert hits[0].chunk_id == 1
    assert set(after_delete) == {1}

