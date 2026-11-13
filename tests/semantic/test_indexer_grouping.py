from __future__ import annotations

import struct
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from auggie.db import connect
from auggie.embed import _estimate_voyage_tokens
from auggie.indexer import index_path


class _FakeTqdm:
    def __init__(self, iterable=None, total=None, **kwargs):  # noqa: ARG002
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable if self._iterable is not None else ())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):  # noqa: ARG002
        return None


class _FakeLanceStore:
    rows: dict[int, tuple[bytes, int]] = {}

    def __init__(self, root):  # noqa: ARG002
        pass

    def fetch_vectors_by_chunk_ids(self, chunk_ids):
        return {i: type(self).rows[i] for i in chunk_ids if i in type(self).rows}

    def upsert_chunk_vectors(self, rows, *, model):  # noqa: ARG002
        for chunk_id, emb, dim in rows:
            type(self).rows[chunk_id] = (emb, dim)

    def delete_chunk_ids(self, chunk_ids):
        for chunk_id in chunk_ids:
            type(self).rows.pop(chunk_id, None)


def _vector(text: str) -> bytes:
    return struct.pack("2f", float(len(text)), float(sum(text.encode()) % 997))


class _FakeEmbed:
    """Records every call; optionally raises on the Nth call."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, texts, model=None, dimensions=None):  # noqa: ARG002
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("embed failed")
        return [_vector(t) for t in texts], 2


class IndexerGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(mkdtemp(prefix="auggie-grouping-"))
        self.db_path = self.root / ".auggie" / "index.sqlite"
        _FakeLanceStore.rows = {}

    def _write(self, name: str, text: str) -> None:
        (self.root / name).write_text(text, encoding="utf-8")

    def _index(self, embed: _FakeEmbed):
        with (
            patch("auggie.indexer.tqdm", _FakeTqdm),
            patch("auggie.indexer.LanceStore", _FakeLanceStore),
            patch("auggie.indexer.embed_texts", side_effect=embed),
        ):
            return index_path(
                self.root,
                self.db_path,
                model="voyage-code-3",
                chunk_lines=50,
                overlap=0,
                detect=False,
                branch="main",
            )

    def _stored_files(self) -> set[str]:
        con = connect(self.db_path, init_fts=False)
        try:
            return {Path(r[0]).name for r in con.execute("SELECT path FROM files")}
        finally:
            con.close()

    def test_small_files_share_one_embed_call(self) -> None:
        for name in ("a.py", "b.py", "c.py"):
            self._write(name, f"print('{name}')\n")
        embed = _FakeEmbed()

        stats = self._index(embed)

        self.assertEqual(stats.files_indexed, 3)
        self.assertEqual(len(embed.calls), 1)
        self.assertEqual(len(embed.calls[0]), 3)
        self.assertEqual(self._stored_files(), {"a.py", "b.py", "c.py"})

    def test_identical_chunk_is_embedded_once_and_stored_for_both_files(self) -> None:
        shared = "def shared():\n    return 1\n"
        self._write("a.py", shared)
        self._write("b.py", shared)
        self._write("c.py", "print('unique')\n")
        embed = _FakeEmbed()

        stats = self._index(embed)

        self.assertEqual(len(embed.calls), 1)
        sent = embed.calls[0]
        self.assertEqual(sum(1 for t in sent if t.strip() == shared.strip()), 1)
        self.assertEqual(len(sent), 2)
        self.assertEqual(stats.chunks_embedded, 2)
        self.assertEqual(stats.chunks_reused_global, 1)

        con = connect(self.db_path, init_fts=False)
        try:
            rows = con.execute(
                "SELECT f.path, c.id FROM chunks c JOIN files f ON f.id = c.file_id"
            ).fetchall()
        finally:
            con.close()
        by_name = {Path(p).name: cid for p, cid in rows}
        self.assertEqual(set(by_name), {"a.py", "b.py", "c.py"})
        expected = (_vector(sent[0] if sent[0].strip() == shared.strip() else sent[1]), 2)
        self.assertEqual(_FakeLanceStore.rows[by_name["a.py"]], expected)
        self.assertEqual(_FakeLanceStore.rows[by_name["b.py"]], expected)

    def _split_group_setup(self) -> int:
        body = "x = 1\n" * 40
        for name in ("a.py", "b.py", "c.py", "d.py"):
            self._write(name, f"# {name}\n{body}")
        one_file = _estimate_voyage_tokens((self.root / "a.py").read_text(encoding="utf-8"))
        # One file stays below the threshold, two files reach it.
        return one_file + 1

    def test_failure_in_second_group_keeps_first_group_committed(self) -> None:
        threshold = self._split_group_setup()
        embed = _FakeEmbed(fail_on_call=2)

        with patch("auggie.indexer.EMBED_GROUP_TOKENS", threshold, create=True):
            with self.assertRaises(RuntimeError):
                self._index(embed)

        self.assertEqual(len(embed.calls), 2)
        first_group = {Path(t.splitlines()[0][2:]).name for t in embed.calls[0]}
        self.assertEqual(len(first_group), 2)
        self.assertEqual(self._stored_files(), first_group)

    def test_rerun_after_failure_indexes_missing_files(self) -> None:
        threshold = self._split_group_setup()

        with patch("auggie.indexer.EMBED_GROUP_TOKENS", threshold, create=True):
            with self.assertRaises(RuntimeError):
                self._index(_FakeEmbed(fail_on_call=2))
            committed = self._stored_files()
            embed = _FakeEmbed()
            stats = self._index(embed)

        self.assertEqual(self._stored_files(), {"a.py", "b.py", "c.py", "d.py"})
        self.assertEqual(stats.files_indexed, 4 - len(committed))
        sent = {t.splitlines()[0][2:] for call in embed.calls for t in call}
        self.assertEqual(sent, {"a.py", "b.py", "c.py", "d.py"} - committed)

    def test_cross_file_shared_chunk_counts_as_global_reuse(self) -> None:
        # Four-line files with chunk_lines=2 give two chunks each; the first
        # chunk is shared. Writing A before planning B, B reuses it globally.
        self._write("a.py", "s1\ns2\na3\na4\n")
        self._write("b.py", "s1\ns2\nb3\nb4\n")
        embed = _FakeEmbed()

        with (
            patch("auggie.indexer.tqdm", _FakeTqdm),
            patch("auggie.indexer.LanceStore", _FakeLanceStore),
            patch("auggie.indexer.embed_texts", side_effect=embed),
        ):
            stats = index_path(
                self.root,
                self.db_path,
                model="voyage-code-3",
                chunk_lines=2,
                overlap=0,
                detect=False,
                branch="main",
            )

        self.assertEqual(stats.chunks_total, 4)
        self.assertEqual(stats.chunks_embedded, 3)
        self.assertEqual(stats.chunks_reused, 1)
        self.assertEqual(stats.chunks_reused_global, 1)
        self.assertEqual(stats.chunks_reused_local, 0)
        self.assertEqual(len(embed.calls), 1)
        self.assertEqual(len(embed.calls[0]), 3)

    def test_group_flushes_on_file_count_without_new_embeddings(self) -> None:
        self._write("seed.py", "print('same')\n")
        self._index(_FakeEmbed())
        for name in ("a.py", "b.py", "c.py", "d.py"):
            self._write(name, "print('same')\n")

        import auggie.indexer as indexer

        real_flush = indexer._flush_pending
        flush_sizes: list[int] = []

        def spy(con, lance_store, pending, *args, **kwargs):
            flush_sizes.append(len(pending))
            return real_flush(con, lance_store, pending, *args, **kwargs)

        embed = _FakeEmbed()
        with (
            patch("auggie.indexer.EMBED_GROUP_MAX_FILES", 2, create=True),
            patch("auggie.indexer._flush_pending", side_effect=spy),
        ):
            stats = self._index(embed)

        self.assertEqual(stats.files_indexed, 4)
        self.assertEqual(embed.calls, [])
        self.assertEqual([n for n in flush_sizes if n], [2, 2])


if __name__ == "__main__":
    unittest.main()
