from __future__ import annotations

import array
import sys
import types
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie.db import connect
from auggie.search import query


def _insert_chunk(
    db_path: Path,
    *,
    file_id: int,
    path: str,
    branch: str,
    chunk_id: int,
    chunk_index: int,
    text: str,
) -> None:
    con = connect(db_path, init_fts=False)
    try:
        con.execute(
            """
            INSERT INTO files (id, path, branch, mtime_ns, size, sha256)
            VALUES (?, ?, ?, 1, ?, ?)
            """,
            (file_id, path, branch, len(text), f"sha-{file_id}"),
        )
        con.execute(
            """
            INSERT INTO chunks (id, file_id, chunk_index, start_line, end_line, text, text_sha256, dim, model)
            VALUES (?, ?, ?, 1, 3, ?, ?, 2, 'voyage-code-3')
            """,
            (chunk_id, file_id, chunk_index, text, f"text-{chunk_id}"),
        )
        con.commit()
    finally:
        con.close()


class Phase4SearchTests(unittest.TestCase):
    def test_query_uses_lance_ann_and_overfetches_branch_candidates(self) -> None:
        tmp_path = Path(mkdtemp(prefix="auggie-phase4-search-"))
        db_path = tmp_path / ".auggie" / "index.sqlite"

        _insert_chunk(
            db_path,
            file_id=1,
            path=str(tmp_path / "src" / "main.py"),
            branch="main",
            chunk_id=101,
            chunk_index=0,
            text="main branch winner",
        )
        _insert_chunk(
            db_path,
            file_id=2,
            path=str(tmp_path / "src" / "other.py"),
            branch="other",
            chunk_id=202,
            chunk_index=0,
            text="other branch result",
        )
        _insert_chunk(
            db_path,
            file_id=3,
            path=str(tmp_path / "src" / "fallback.py"),
            branch="main",
            chunk_id=303,
            chunk_index=1,
            text="main branch fallback",
        )

        call_limits: list[int] = []

        class FakeStore:
            def __init__(self, root):  # noqa: ARG002
                pass

            def count(self):
                return 3

            def ann_query(self, vector, limit):
                call_limits.append(limit)
                hits = [
                    {"chunk_id": 202, "distance": 0.0},
                    {"chunk_id": 101, "distance": 0.2},
                    {"chunk_id": 303, "distance": 0.6},
                ]
                return [type("Hit", (), hit)() for hit in hits[:limit]]

            def fetch_vectors_by_chunk_ids(self, chunk_ids):  # noqa: ARG002
                raise AssertionError("query() should not fetch all vectors for brute-force search")

        with (
            patch(
                "auggie.search.embed_texts",
                return_value=([array.array("f", [1.0, 0.0]).tobytes()], 2),
            ),
            patch("auggie.search.LanceStore", FakeStore),
        ):
            hits, fts_available = query(
                db_path=db_path,
                q="main branch",
                top_k=2,
                mode="semantic",
                no_cache=True,
                branch="main",
            )

        self.assertFalse(fts_available)
        self.assertEqual([hit.chunk_id for hit in hits], [101, 303])
        self.assertEqual(
            [hit.text for hit in hits],
            ["main branch winner", "main branch fallback"],
        )
        self.assertTrue(call_limits)
        self.assertGreater(call_limits[0], 2)


if __name__ == "__main__":
    unittest.main()
