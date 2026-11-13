"""Tests for auggie_mcp.semantic_search and codebase_search mode=semantic."""

import array
import asyncio
import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp.server.mcpserver.exceptions import ToolError

from auggie import db as auggie_db
from auggie.lance_store import get_lance_dir
from auggie.search import Hit
from auggie_mcp import semantic_search, server

TESTS_DIR = Path(__file__).resolve().parent
FAKE_RG = TESTS_DIR / "fake_rg.py"
FAKE_AUGGIE = TESTS_DIR / "fake_auggie.py"

MODEL = "test-model"
DIM = 2


class FakeLanceStore:
    registry: dict[str, dict[int, tuple[bytes, int]]] = {}

    def __init__(self, root):
        self.root = str(Path(root))
        type(self).registry.setdefault(self.root, {})

    @property
    def _rows(self):
        return type(self).registry[self.root]

    def count(self):
        return len(self._rows)

    def fetch_vectors_by_chunk_ids(self, chunk_ids):
        return {int(c): self._rows[int(c)] for c in chunk_ids if int(c) in self._rows}

    def ann_query(self, vector, limit):
        query_vec = array.array("f")
        query_vec.frombytes(vector)
        hits = []
        for chunk_id, (emb, _dim) in self._rows.items():
            doc = array.array("f")
            doc.frombytes(emb)
            dot = sum(float(a) * float(b) for a, b in zip(query_vec, doc, strict=True))
            distance = math.sqrt(max(0.0, 2.0 - 2.0 * dot))
            hits.append(type("Hit", (), {"chunk_id": chunk_id, "distance": distance})())
        hits.sort(key=lambda item: item.distance)
        return hits[:limit]


def fake_embed_texts(texts, model=None, dimensions=None):  # noqa: ARG001
    vectors = [array.array("f", [1.0, 0.0]).tobytes() for _ in texts]
    return vectors, DIM


class SemanticCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(os.path.realpath(self.tmp.name)) / "repo"
        self.repo.mkdir()
        self.db = self.repo / ".auggie" / "index.sqlite"
        env = patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def write(self, name, text="one\ntwo\nthree\nfour\n"):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def build_index(self, names, history=("2026-11-01 10:00:00",)):
        con = auggie_db.connect(self.db, init_fts=False)
        try:
            for name in names:
                path = self.repo / name
                st = path.stat()
                cur = con.execute(
                    "INSERT INTO files(path, branch, mtime_ns, size, sha256) "
                    "VALUES(?, 'default', ?, ?, 'sha')",
                    (str(path), st.st_mtime_ns, st.st_size),
                )
                con.execute(
                    "INSERT INTO chunks(file_id, chunk_index, start_line, end_line, "
                    "text, text_sha256, dim, model) VALUES(?, 0, 2, 3, ?, ?, ?, ?)",
                    (cur.lastrowid, f"chunk of {name}", f"h{name}", DIM, MODEL),
                )
            for stamp in history:
                con.execute(
                    "INSERT INTO index_history(timestamp, action) VALUES(?, 'index')",
                    (stamp,),
                )
            con.commit()
        finally:
            con.close()

    def hit(self, name, score=0.9876543, start=2, end=3, text="the code"):
        return Hit(
            score=score,
            path=str(self.repo / name),
            start_line=start,
            end_line=end,
            text=text,
            chunk_id=1,
            chunk_index=0,
            confidence="high",
        )

    def fake_vectors(self):
        """Seed the fake Lance store for the two indexed files and return the patches."""
        con = sqlite3.connect(self.db)
        try:
            ids = [r[0] for r in con.execute("SELECT id FROM chunks ORDER BY id")]
        finally:
            con.close()
        lance = get_lance_dir(self.db)
        FakeLanceStore.registry[str(lance)] = {
            ids[0]: (array.array("f", [1.0, 0.0]).tobytes(), DIM),
            ids[1]: (array.array("f", [0.0, 1.0]).tobytes(), DIM),
        }
        self.addCleanup(FakeLanceStore.registry.clear)
        lance.mkdir(parents=True, exist_ok=True)
        return (
            patch("auggie.search.embed_texts", fake_embed_texts),
            patch("auggie.search.LanceStore", FakeLanceStore),
            patch("auggie.search.DEFAULT_SEARCH_MODE", "semantic"),
        )

    def journal_mode(self):
        con = sqlite3.connect(self.db)
        try:
            return con.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            con.close()

    def run_search(self, hits, query="q", path=".", top_k=10, context_lines=0):
        with patch(
            "auggie_mcp.semantic_search.query_db", return_value=(hits, True)
        ) as query_db:
            result = semantic_search.run_semantic(
                self.repo, query, path, top_k, context_lines
            )
        return result, query_db


class ValidateTest(unittest.TestCase):
    def check(self, mode, **kw):
        args = {"flags": None, "top_k": 10, "context_lines": 0, **kw}
        semantic_search.validate_params(mode, **args)

    def test_defaults_pass_in_both_modes(self):
        self.check("exact")
        self.check("semantic")

    def test_flags_rejected_in_semantic_mode(self):
        with self.assertRaisesRegex(ToolError, "flags apply only to mode=exact") as cm:
            self.check("semantic", flags=["-F"])
        self.assertIn("mode='exact'", str(cm.exception))

    def test_validation_errors_state_accepted_values_or_mode(self):
        cases = (
            ("exact", {"top_k": 5}, "mode='semantic'"),
            ("exact", {"context_lines": 2}, "mode='semantic'"),
            ("semantic", {"top_k": 0}, "1 to 50"),
            ("semantic", {"context_lines": 21}, "0 to 20"),
            ("fuzzy", {}, "mode='exact'"),
        )
        for mode, kw, guidance in cases:
            with self.subTest(mode=mode, **kw), self.assertRaises(ToolError) as cm:
                self.check(mode, **kw)
            self.assertIn(guidance, str(cm.exception))

    def test_top_k_and_context_rejected_in_exact_mode(self):
        with self.assertRaisesRegex(ToolError, "top_k"):
            self.check("exact", top_k=5)
        with self.assertRaisesRegex(ToolError, "context_lines"):
            self.check("exact", context_lines=2)

    def test_bounds(self):
        for kw in ({"top_k": 0}, {"top_k": 51}, {"context_lines": -1}, {"context_lines": 21}):
            with self.subTest(**kw), self.assertRaises(ToolError):
                self.check("semantic", **kw)
        self.check("semantic", top_k=50, context_lines=20)
        self.check("semantic", top_k=1)

    def test_unknown_mode(self):
        with self.assertRaisesRegex(ToolError, "mode"):
            self.check("fuzzy")


class RunSemanticTest(SemanticCase):
    def test_renders_blocks_with_repo_relative_paths(self):
        self.write("a.py")
        self.write("pkg/b.py")
        self.build_index(["a.py", "pkg/b.py"])
        hits = [
            self.hit("a.py", text="alpha"),
            self.hit("pkg/b.py", score=0.5, start=5, end=9, text="beta\nline"),
        ]
        result, _ = self.run_search(hits)
        self.assertEqual(
            result["stdout"],
            "a.py:2-3 score=0.988\nalpha\n--\npkg/b.py:5-9 score=0.500\nbeta\nline",
        )
        self.assertEqual(
            result["blocks"],
            ["a.py:2-3 score=0.988\nalpha", "pkg/b.py:5-9 score=0.500\nbeta\nline"],
        )
        self.assertEqual("\n--\n".join(result["blocks"]), result["stdout"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stderr"], "")

    def test_display_root_makes_paths_subdirectory_relative(self):
        self.write("a.py")
        self.write("pkg/b.py")
        self.build_index(["a.py", "pkg/b.py"])
        hits = [self.hit("pkg/b.py", text="beta"), self.hit("a.py", text="alpha")]
        with patch("auggie_mcp.semantic_search.query_db", return_value=(hits, True)):
            result = semantic_search.run_semantic(
                self.repo, "q", ".", 10, 0, display_root=self.repo / "pkg"
            )
        blocks = result["stdout"].split("\n--\n")
        self.assertEqual(blocks[0], "b.py:2-3 score=0.988\nbeta")
        self.assertEqual(blocks[1], f"{self.repo / 'a.py'}:2-3 score=0.988\nalpha")

    def test_omitted_display_root_keeps_git_root_relative_paths(self):
        self.write("pkg/b.py")
        self.build_index(["pkg/b.py"])
        hits = [self.hit("pkg/b.py", text="beta")]
        with patch("auggie_mcp.semantic_search.query_db", return_value=(hits, True)):
            result = semantic_search.run_semantic(self.repo, "q", "pkg", 10, 0)
        self.assertEqual(result["stdout"], "pkg/b.py:2-3 score=0.988\nbeta")

    def test_no_hits_is_exit_code_1(self):
        self.write("a.py")
        self.build_index(["a.py"])
        result, _ = self.run_search([])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")

    def test_query_uses_stored_model_and_no_cache(self):
        self.write("a.py")
        self.build_index(["a.py"])
        _, query_db = self.run_search([self.hit("a.py")], query="find it", top_k=3)
        call = query_db.call_args.kwargs
        self.assertEqual(call["db_path"], self.db)
        self.assertEqual(call["q"], "find it")
        self.assertEqual(call["top_k"], 3)
        self.assertEqual(call["model"], MODEL)
        self.assertEqual(call["dimensions"], DIM)
        self.assertEqual(call["branch"], "default")
        self.assertTrue(call["no_cache"])
        self.assertIsNone(call["mode"])

    def test_subtree_filters_whole_path_components(self):
        for name in ("src/a/x.py", "src/ab/y.py", "src/a.py"):
            self.write(name)
        self.build_index(["src/a/x.py", "src/ab/y.py", "src/a.py"])
        hits = [self.hit("src/ab/y.py"), self.hit("src/a.py"), self.hit("src/a/x.py")]
        result, query_db = self.run_search(hits, path="src/a", top_k=5)
        self.assertEqual(query_db.call_args.kwargs["top_k"], 25)
        self.assertEqual(result["stdout"].splitlines()[0], "src/a/x.py:2-3 score=0.988")
        self.assertNotIn("src/ab", result["stdout"])
        self.assertNotIn("src/a.py", result["stdout"])

    def test_file_path_matches_only_that_file(self):
        self.write("src/a.py")
        self.write("src/a.pyc")
        self.build_index(["src/a.py", "src/a.pyc"])
        result, _ = self.run_search(
            [self.hit("src/a.pyc"), self.hit("src/a.py")], path="src/a.py"
        )
        self.assertTrue(result["stdout"].startswith("src/a.py:2-3"))
        self.assertNotIn("a.pyc", result["stdout"])

    def test_candidates_capped_at_250_and_truncated_to_top_k(self):
        self.write("src/a/x.py")
        self.build_index(["src/a/x.py"])
        hits = [self.hit("src/a/x.py", score=1 - i / 100) for i in range(4)]
        result, query_db = self.run_search(hits, path="src", top_k=2)
        self.assertEqual(query_db.call_args.kwargs["top_k"], 10)
        self.assertEqual(result["stdout"].count("src/a/x.py:"), 2)
        _, query_db = self.run_search(hits, path="src", top_k=50)
        self.assertEqual(query_db.call_args.kwargs["top_k"], 250)

    def test_root_path_requests_top_k_only(self):
        self.write("a.py")
        self.build_index(["a.py"])
        _, query_db = self.run_search([self.hit("a.py")], top_k=7)
        self.assertEqual(query_db.call_args.kwargs["top_k"], 7)

    def test_path_outside_repo_root_is_rejected(self):
        self.write("a.py")
        with self.assertRaises(ToolError):
            self.run_search([], path="..")

    def test_missing_index_names_index_update(self):
        self.write("a.py")
        with self.assertRaises(ToolError) as cm:
            self.run_search([])
        self.assertIn("index_update(repository_path=", str(cm.exception))
        self.assertFalse(self.db.exists())

    def test_missing_api_key(self):
        self.write("a.py")
        self.build_index(["a.py"])
        with patch.dict(os.environ):
            os.environ.pop("VOYAGE_API_KEY")
            with self.assertRaisesRegex(ToolError, "VOYAGE_API_KEY"):
                self.run_search([])

    def test_search_failure_becomes_tool_error(self):
        self.write("a.py")
        self.build_index(["a.py"])
        with patch("auggie_mcp.semantic_search.query_db", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(ToolError, "boom") as cm:
                semantic_search.run_semantic(self.repo, "q", ".", 10, 0)
        message = str(cm.exception)
        for step in ("VOYAGE_API_KEY", "index_status", "index_update(repository_path=", "retry"):
            self.assertIn(step, message)

    def downgrade_files_table(self):
        """Replace the files table with the pre-branch layout."""
        con = sqlite3.connect(self.db)
        try:
            con.executescript(
                "DROP TABLE files;"
                "CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, "
                "mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL);"
            )
        finally:
            con.close()

    def set_schema_version(self, value):
        con = sqlite3.connect(self.db)
        try:
            con.execute("UPDATE index_metadata SET value = ? WHERE key = 'schema_version'", (value,))
            con.commit()
        finally:
            con.close()

    def assert_outdated_schema_rejected(self):
        before = self.db.read_bytes()
        with patch("auggie_mcp.semantic_search.query_db") as query_db:
            with self.assertRaises(ToolError) as cm:
                semantic_search.run_semantic(self.repo, "q", ".", 10, 0)
        message = str(cm.exception)
        self.assertIn("schema", message)
        self.assertIn("outdated", message)
        self.assertIn("index_update(repository_path=", message)
        query_db.assert_not_called()
        self.assertEqual(self.db.read_bytes(), before)

    def test_old_schema_version_is_rejected_without_migrating(self):
        self.write("a.py")
        self.build_index(["a.py"])
        self.set_schema_version("1")
        self.assert_outdated_schema_rejected()

    def test_old_files_schema_is_rejected_without_migrating(self):
        self.write("a.py")
        self.build_index(["a.py"])
        self.downgrade_files_table()
        self.assert_outdated_schema_rejected()

    def test_current_schema_still_searches_with_bytes_unchanged(self):
        self.write("a.py")
        self.build_index(["a.py"])
        before = self.db.read_bytes()
        result, _ = self.run_search([self.hit("a.py")])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(self.db.read_bytes(), before)

    def test_index_info_reports_stale_counts_and_last_indexed(self):
        a = self.write("a.py")
        self.write("b.py")
        self.write("c.py")
        self.build_index(["a.py", "b.py", "c.py"], history=("2026-11-01 10:00:00", "2026-11-02 09:30:00"))
        a.write_text("changed\n")
        (self.repo / "b.py").unlink()
        result, _ = self.run_search([self.hit("c.py")])
        self.assertEqual(
            result["index"],
            {"stale_modified": 1, "stale_deleted": 1, "last_indexed": "2026-11-02 09:30:00"},
        )

    def test_context_expansion(self):
        self.write("a.py", "l1\nl2\nl3\nl4\nl5\nl6\n")
        self.build_index(["a.py"])
        result, _ = self.run_search([self.hit("a.py", start=3, end=3, text="l3")], context_lines=1)
        self.assertEqual(result["stdout"], "a.py:2-4 score=0.988\nl2\nl3\nl4")

    def test_context_expansion_skipped_for_modified_files(self):
        a = self.write("a.py", "l1\nl2\nl3\nl4\nl5\nl6\n")
        self.write("b.py", "m1\nm2\nm3\nm4\n")
        self.build_index(["a.py", "b.py"])
        a.write_text("l1\nl2\nl3 edited\nl4\nl5\nl6\nl7\n")
        hits = [
            self.hit("a.py", start=3, end=3, text="l3"),
            self.hit("b.py", start=2, end=2, text="m2"),
        ]
        result, _ = self.run_search(hits, context_lines=1)
        self.assertEqual(
            result["stdout"],
            "a.py:3-3 score=0.988\nl3\n--\nb.py:1-3 score=0.988\nm1\nm2\nm3",
        )

    def real_search(self):
        patches = self.fake_vectors()
        with patches[0], patches[1], patches[2]:
            return semantic_search.run_semantic(self.repo, "anything", ".", 5, 0)

    def test_database_bytes_unchanged_by_real_query(self):
        self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"])
        before = self.db.read_bytes()
        result = self.real_search()
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("a.py:2-3", result["stdout"])
        self.assertEqual(self.db.read_bytes(), before)

    def test_journal_mode_delete_index_is_not_changed_by_search(self):
        self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"])
        con = sqlite3.connect(self.db)
        try:
            con.execute("PRAGMA journal_mode=DELETE")
        finally:
            con.close()
        self.assertEqual(self.journal_mode(), "delete")
        before = self.db.read_bytes()
        result = self.real_search()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(self.journal_mode(), "delete")
        self.assertEqual(self.db.read_bytes(), before)

    def test_old_vector_schema_directs_to_rebuild(self):
        self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"])
        con = sqlite3.connect(self.db)
        try:
            con.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
            con.commit()
        finally:
            con.close()
        before = self.db.read_bytes()
        with self.assertRaises(ToolError) as cm:
            self.real_search()
        self.assertIn("rebuild=True", str(cm.exception))
        self.assertEqual(self.db.read_bytes(), before)

    def test_old_schema_version_real_query_directs_to_incremental_update(self):
        self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"])
        self.set_schema_version("1")
        before = self.db.read_bytes()
        with self.assertRaises(ToolError) as cm:
            self.real_search()
        self.assertIn("index_update(repository_path=", str(cm.exception))
        self.assertNotIn("rebuild=True", str(cm.exception))
        self.assertEqual(self.db.read_bytes(), before)

    def test_readonly_operational_error_names_index_update(self):
        self.write("a.py")
        self.build_index(["a.py"])
        error = sqlite3.OperationalError("attempt to write a readonly database")
        with patch("auggie_mcp.semantic_search.query_db", side_effect=error):
            with self.assertRaises(ToolError) as cm:
                semantic_search.run_semantic(self.repo, "q", ".", 10, 0)
        self.assertIn("index_update(repository_path=", str(cm.exception))
        self.assertIn("needs an update", str(cm.exception))

    def test_query_passes_readonly_true(self):
        self.write("a.py")
        self.build_index(["a.py"])
        _, query_db = self.run_search([self.hit("a.py")])
        self.assertIs(query_db.call_args.kwargs["readonly"], True)


class QueryConnectionTest(SemanticCase):
    def query(self, **kwargs):
        from auggie import search

        self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"])
        patches = self.fake_vectors()
        with patches[0], patches[1], patches[2]:
            return search.query(
                self.db, "q", top_k=2, model=MODEL, dimensions=DIM,
                no_cache=True, branch="default", **kwargs,
            )  # fmt: skip

    def test_default_path_still_uses_connect(self):
        with (
            patch("auggie.search.connect", wraps=auggie_db.connect) as connect,
            patch("auggie.search.connect_readonly", wraps=auggie_db.connect_readonly) as ro,
        ):
            self.query()
        connect.assert_called_once()
        ro.assert_not_called()

    def test_readonly_path_uses_connect_readonly_only(self):
        with (
            patch("auggie.search.connect", wraps=auggie_db.connect) as connect,
            patch("auggie.search.connect_readonly", wraps=auggie_db.connect_readonly) as ro,
        ):
            self.query(readonly=True)
        ro.assert_called_once()
        connect.assert_not_called()


class CodebaseSearchSemanticToolTest(SemanticCase):
    def make_server(self):
        return server.create_server(
            rg_executable=str(FAKE_RG),
            index_command=[os.sys.executable, str(FAKE_AUGGIE)],
            repository_root=self.repo,
            client_factory=lambda: None,
        )

    def call(self, arguments):
        srv = self.make_server()
        return asyncio.run(srv.call_tool("codebase_search", arguments))

    def test_semantic_returns_envelope_with_mode_and_index(self):
        self.write("a.py")
        self.build_index(["a.py"])
        with patch(
            "auggie_mcp.semantic_search.query_db",
            return_value=([self.hit("a.py", text="alpha")], True),
        ):
            result = self.call({"query": "alpha", "mode": "semantic"})
        data = result.structured_content if hasattr(result, "structured_content") else result[1]
        self.assertEqual(data["tool"], "codebase_search")
        self.assertEqual(data["data"]["mode"], "semantic")
        self.assertEqual(data["data"]["stdout"], "a.py:2-3 score=0.988\nalpha")
        self.assertEqual(data["data"]["exit_code"], 0)
        self.assertEqual(
            data["data"]["index"],
            {"stale_modified": 0, "stale_deleted": 0, "last_indexed": "2026-11-01 10:00:00"},
        )

    def test_semantic_no_hits_is_a_normal_result(self):
        self.write("a.py")
        self.build_index(["a.py"])
        with patch("auggie_mcp.semantic_search.query_db", return_value=([], True)):
            result = self.call({"query": "zzz", "mode": "semantic"})
        data = result.structured_content if hasattr(result, "structured_content") else result[1]
        self.assertEqual(data["data"]["exit_code"], 1)

    def test_exact_mode_reports_mode_and_null_index(self):
        self.write("hello.txt", "hello world\n")
        result = self.call({"query": "hello"})
        data = result.structured_content if hasattr(result, "structured_content") else result[1]
        self.assertEqual(data["data"]["mode"], "exact")
        self.assertIsNone(data["data"]["index"])

    def test_wrong_mode_parameters_are_tool_errors(self):
        for args, message in (
            ({"query": "x", "mode": "semantic", "flags": ["-F"]}, "flags apply only to mode=exact"),
            ({"query": "x", "top_k": 5}, "top_k"),
            ({"query": "x", "context_lines": 3}, "context_lines"),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(Exception, message):
                self.call(args)

    def test_missing_index_is_tool_error(self):
        self.write("a.py")
        with self.assertRaisesRegex(Exception, r"index_update\(repository_path="):
            self.call({"query": "x", "mode": "semantic"})

    def test_semantic_search_runs_off_the_event_loop(self):
        self.write("a.py")
        self.build_index(["a.py"])
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append(func)
            return func(*args, **kwargs)

        with (
            patch("auggie_mcp.semantic_search.query_db", return_value=([], True)),
            patch("auggie_mcp.server.asyncio.to_thread", fake_to_thread),
        ):
            self.call({"query": "x", "mode": "semantic"})
        self.assertIn(semantic_search.run_semantic, calls)


if __name__ == "__main__":
    unittest.main()
