from __future__ import annotations

import array
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auggie import db
from auggie.indexer import index_path
import auggie.mcp_tools as mcp_tools
from auggie.search import Hit


class FakeLanceStore:
    registry: dict[str, dict[int, tuple[bytes, int]]] = {}

    def __init__(self, root: Path):
        self.root = str(Path(root))
        type(self).registry.setdefault(self.root, {})

    @property
    def _rows(self) -> dict[int, tuple[bytes, int]]:
        return type(self).registry[self.root]

    def upsert_chunk_vectors(self, rows, *, model):  # noqa: ARG002
        Path(self.root).mkdir(parents=True, exist_ok=True)
        for chunk_id, emb, dim in rows:
            self._rows[int(chunk_id)] = (emb, int(dim))

    def fetch_vectors_by_chunk_ids(self, chunk_ids):
        return {
            int(chunk_id): self._rows[int(chunk_id)]
            for chunk_id in chunk_ids
            if int(chunk_id) in self._rows
        }

    def count(self) -> int:
        return len(self._rows)

    def ann_query(self, vector, limit):
        query_vec = array.array("f")
        query_vec.frombytes(vector)
        q = list(query_vec)
        hits = []
        for chunk_id, (emb, _dim) in self._rows.items():
            doc_vec = array.array("f")
            doc_vec.frombytes(emb)
            dot = sum(float(a) * float(b) for a, b in zip(q, doc_vec, strict=True))
            distance = math.sqrt(max(0.0, 2.0 - 2.0 * dot))
            hits.append(type("Hit", (), {"chunk_id": chunk_id, "distance": distance})())
        hits.sort(key=lambda item: item.distance)
        return hits[:limit]


def fake_embed_texts(texts, model=None, dimensions=None):  # noqa: ARG001
    vectors = []
    for text in texts:
        if "database" in text or "open_database_connection" in text:
            vector = [1.0, 0.0]
        elif "cache" in text:
            vector = [0.0, 1.0]
        else:
            vector = [0.5, 0.5]
        vectors.append(array.array("f", vector).tobytes())
    return vectors, 2


def _create_index(root: Path) -> tuple[Path, Path]:
    source = root / "sample.py"
    source.write_text(
        "alpha setup\n"
        "beta database connection\n"
        "gamma cache lookup\n"
        "delta context line\n",
        encoding="utf-8",
    )
    db_path = root / ".auggie" / "index.sqlite"
    con = db.connect(db_path, init_fts=False)
    try:
        cur = con.execute(
            """
            INSERT INTO files (path, branch, mtime_ns, size, sha256)
            VALUES (?, 'default', 1, 10, 'file-sha')
            """,
            (str(source.resolve()),),
        )
        con.execute(
            """
            INSERT INTO chunks
              (file_id, chunk_index, start_line, end_line, text, text_sha256, dim, model)
            VALUES
              (?, 0, 1, 2, 'alpha setup beta database connection', 'chunk-sha-1', 2, 'test-model'),
              (?, 1, 3, 4, 'gamma cache lookup delta context line', 'chunk-sha-2', 2, 'test-model')
            """,
            (cur.lastrowid, cur.lastrowid),
        )
        con.commit()
    finally:
        con.close()
    return db_path, source


def _hit(source: Path) -> Hit:
    return Hit(
        score=0.9876543,
        path=str(source.resolve()),
        start_line=1,
        end_line=2,
        text="alpha setup beta database connection",
        chunk_id=1,
        chunk_index=0,
        confidence="high",
    )


class McpServerUnitTests(unittest.TestCase):
    def test_status_success_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _source = _create_index(root)

            with (
                patch("auggie.mcp_tools._resolve_root", return_value=root),
                patch("auggie.mcp_tools._resolve_db", return_value=db_path),
            ):
                result = mcp_tools.auggie_status(str(root))

        self.assertTrue(result["exists"])
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["dim"], 2)
        self.assertEqual(result["branches"], {"default": 1})

    def test_query_success_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, source = _create_index(root)
            hit = _hit(source)

            with (
                patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}),
                patch("auggie.mcp_tools._resolve_root", return_value=root),
                patch("auggie.mcp_tools._resolve_db", return_value=db_path),
                patch("auggie.mcp_tools.get_current_branch", return_value="default"),
                patch("auggie.mcp_tools.query_db", return_value=([hit], True)) as query_db,
            ):
                result = mcp_tools.auggie_query(
                    "database connection",
                    str(root),
                    top_k=3,
                    mode="semantic",
                )

        self.assertEqual(set(result), {"hits", "fts_available"})
        self.assertTrue(result["fts_available"])
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["path"], str(source.resolve()))
        self.assertEqual(result["hits"][0]["score"], 0.987654)
        query_db.assert_called_once()
        call = query_db.call_args.kwargs
        self.assertEqual(call["db_path"], db_path)
        self.assertEqual(call["q"], "database connection")
        self.assertEqual(call["top_k"], 3)
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["dimensions"], 2)
        self.assertEqual(call["mode"], "semantic")
        self.assertTrue(call["no_cache"])
        self.assertEqual(call["branch"], "default")

    def test_context_success_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, source = _create_index(root)
            hit = _hit(source)
            expanded = [
                {
                    "chunk_id": 1,
                    "path": str(source.resolve()),
                    "start_line": 1,
                    "end_line": 4,
                    "score": 0.987654,
                    "match_source": "semantic",
                    "text": source.read_text(encoding="utf-8"),
                }
            ]

            with (
                patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}),
                patch("auggie.mcp_tools._resolve_root", return_value=root),
                patch("auggie.mcp_tools._resolve_db", return_value=db_path),
                patch("auggie.mcp_tools.get_current_branch", return_value="default"),
                patch("auggie.mcp_tools.query_db", return_value=([hit], True)) as query_db,
                patch("auggie.mcp_tools._expand_results", return_value=expanded) as expand_results,
            ):
                result = mcp_tools.auggie_context(
                    "database connection",
                    str(root),
                    top_k=1,
                    context_lines=2,
                )

        self.assertEqual(set(result), {"hits", "fts_available"})
        self.assertTrue(result["fts_available"])
        self.assertEqual(result["hits"][0]["start_line"], 1)
        self.assertEqual(result["hits"][0]["end_line"], 4)
        self.assertIn("delta context line", result["hits"][0]["text"])
        self.assertEqual(result["hits"][0]["match_source"], "semantic")
        query_db.assert_called_once()
        self.assertEqual(query_db.call_args.kwargs["mode"], "semantic")
        self.assertTrue(query_db.call_args.kwargs["no_cache"])
        expand_results.assert_called_once()
        self.assertEqual(expand_results.call_args.args[1], 2)

    def test_missing_index_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_db = root / ".auggie" / "index.sqlite"

            with (
                patch("auggie.mcp_tools._resolve_root", return_value=root),
                patch("auggie.mcp_tools._resolve_db", return_value=missing_db),
            ):
                status = mcp_tools.auggie_status(str(root))
                query = mcp_tools.auggie_query("database connection", str(root))
                context = mcp_tools.auggie_context("database connection", str(root))

        for result in (status, query, context):
            self.assertEqual(set(result), {"error", "hint"})
            self.assertIn("No auggie index found", result["error"])
            self.assertIn("auggie index", result["hint"])

    def test_missing_key_error_for_search_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _source = _create_index(root)

            with (
                patch.dict(os.environ, {"VOYAGE_API_KEY": ""}),
                patch("auggie.mcp_tools._resolve_root", return_value=root),
                patch("auggie.mcp_tools._resolve_db", return_value=db_path),
                patch("auggie.mcp_tools.query_db") as query_db,
            ):
                query = mcp_tools.auggie_query("database connection", str(root))
                context = mcp_tools.auggie_context("database connection", str(root))

        for result in (query, context):
            self.assertEqual(result["error"], "VOYAGE_API_KEY is not set")
            self.assertIn("Set VOYAGE_API_KEY", result["hint"])
        query_db.assert_not_called()

    def test_bad_repo_path_error_for_search_tools(self) -> None:
        for tool in (mcp_tools.auggie_query, mcp_tools.auggie_context):
            result = tool("database connection", "\0")
            self.assertEqual(set(result), {"error"})
            self.assertIn("Invalid repo_path", result["error"])

    def test_embed_layer_missing_key_error_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, _source = _create_index(root)

            with (
                patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}),
                patch("auggie.mcp_tools._resolve_root", return_value=root),
                patch("auggie.mcp_tools._resolve_db", return_value=db_path),
                patch(
                    "auggie.mcp_tools.query_db",
                    side_effect=ValueError("VOYAGE_API_KEY environment variable is not set"),
                ),
            ):
                query = mcp_tools.auggie_query("database connection", str(root))
                context = mcp_tools.auggie_context("database connection", str(root))

        for result in (query, context):
            self.assertEqual(result["error"], "VOYAGE_API_KEY is not set")
            self.assertIn("Set VOYAGE_API_KEY", result["hint"])


@unittest.skipUnless(os.environ.get("VOYAGE_API_KEY"), "VOYAGE_API_KEY is not set")
class McpServerIntegrationTests(unittest.TestCase):
    def test_real_index_status_query_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "service.py"
            source.write_text(
                "def open_database_connection():\n"
                "    dsn = 'primary database'\n"
                "    # sentinel context neighbor\n"
                "    return f'connect to {dsn}'\n"
                "\n"
                "def unrelated_cache_lookup():\n"
                "    return 'cache'\n",
                encoding="utf-8",
            )
            db_path = root / ".auggie" / "index.sqlite"

            FakeLanceStore.registry = {}
            with (
                patch("auggie.indexer.embed_texts", side_effect=fake_embed_texts),
                patch("auggie.indexer.LanceStore", FakeLanceStore),
                patch("auggie.search.embed_texts", side_effect=fake_embed_texts),
                patch("auggie.search.LanceStore", FakeLanceStore),
                patch("auggie.mcp_tools.LanceStore", FakeLanceStore),
            ):
                stats = index_path(
                    root=root,
                    db_path=db_path,
                    model="voyage-code-3",
                    chunk_lines=2,
                    overlap=0,
                    detect=False,
                    branch="default",
                )
                self.assertGreater(stats.chunks_embedded, 0)

                status = mcp_tools.auggie_status(str(root))
                self.assertTrue(status["exists"])
                self.assertEqual(status["total_files"], 1)
                self.assertGreater(status["chunks"], 0)
                self.assertEqual(status["model"], "voyage-code-3")
                self.assertEqual(status["dim"], 2)

                query = mcp_tools.auggie_query(
                    "open database connection",
                    str(root),
                    top_k=3,
                    mode="semantic",
                )
                self.assertIn("hits", query)
                self.assertGreater(len(query["hits"]), 0)
                self.assertEqual(query["hits"][0]["path"], str(source.resolve()))

                context = mcp_tools.auggie_context(
                    "open database connection",
                    str(root),
                    top_k=1,
                    context_lines=1,
                )
                self.assertIn("hits", context)
                self.assertGreater(len(context["hits"]), 0)
                self.assertEqual(context["hits"][0]["path"], str(source.resolve()))
                self.assertGreaterEqual(
                    len(context["hits"][0]["text"]),
                    len(query["hits"][0]["text"]),
                )
                self.assertIn("sentinel context neighbor", context["hits"][0]["text"])


if __name__ == "__main__":
    unittest.main()
