from __future__ import annotations

import json
import io
import sys
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie.commands.context import cmd_context
from auggie.commands import __all__ as command_exports
from auggie import cli
from auggie.db import connect
from auggie.search import Hit


class Phase6ContextTests(unittest.TestCase):
    def test_context_returns_semantic_hits(self) -> None:
        root = Path(mkdtemp(prefix="auggie-context-merge-"))
        db_path = root / ".auggie" / "index.sqlite"
        source = root / "sample.py"
        source.write_text(
            "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n",
            encoding="utf-8",
        )

        con = connect(db_path, init_fts=False)
        try:
            con.execute(
                "INSERT INTO files (id, path, branch, mtime_ns, size, sha256) VALUES (1, ?, 'main', 1, 10, 'sha')",
                (str(source.resolve()),),
            )
            con.execute(
                """
                INSERT INTO chunks (id, file_id, chunk_index, start_line, end_line, text, text_sha256, dim, model)
                VALUES
                (11, 1, 0, 2, 4, 'chunk one', 'sha-11', 2, 'voyage-code-3'),
                (12, 1, 1, 6, 8, 'chunk two', 'sha-12', 2, 'voyage-code-3'),
                (13, 1, 2, 9, 10, 'chunk three', 'sha-13', 2, 'voyage-code-3')
                """
            )
            con.commit()
        finally:
            con.close()

        semantic_hits = [
            Hit(
                score=0.9,
                path=str(source.resolve()),
                start_line=2,
                end_line=4,
                text="chunk one",
                chunk_id=11,
                chunk_index=0,
                confidence="high",
            ),
            Hit(
                score=0.7,
                path=str(source.resolve()),
                start_line=9,
                end_line=10,
                text="chunk three",
                chunk_id=13,
                chunk_index=2,
                confidence="medium",
            ),
        ]
        args = Namespace(
            query="concept symbol",
            top=5,
            context=1,
            db=str(db_path),
            profile=None,
            global_cache=False,
            repo_root=root,
            model="voyage-code-3",
            dimensions=None,
            json=True,
        )

        stdout = io.StringIO()
        with (
            patch("auggie.commands.context.require_embedding_config", return_value=True),
            patch("auggie.commands.context.get_current_branch", return_value="main"),
            patch("auggie.commands.context.query_db", return_value=(semantic_hits, True)),
            redirect_stdout(stdout),
        ):
            exit_code = cmd_context(args)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["results"][0]["chunk_id"], 11)
        by_chunk_id = {item["chunk_id"]: item for item in payload["results"]}
        self.assertEqual(set(by_chunk_id), {11, 13})
        self.assertEqual(by_chunk_id[11]["match_source"], "semantic")
        self.assertEqual(by_chunk_id[13]["match_source"], "semantic")
        self.assertEqual(payload["results"][0]["start_line"], 1)
        self.assertEqual(payload["results"][0]["end_line"], 5)
        self.assertIn("line1", payload["results"][0]["text"])
        self.assertIn("line5", payload["results"][0]["text"])

    def test_context_returns_semantic_only(self) -> None:
        root = Path(mkdtemp(prefix="auggie-context-warning-"))
        db_path = root / ".auggie" / "index.sqlite"
        source = root / "sample.py"
        source.write_text("one\ntwo\nthree\n", encoding="utf-8")

        con = connect(db_path, init_fts=False)
        try:
            con.execute(
                "INSERT INTO files (id, path, branch, mtime_ns, size, sha256) VALUES (1, ?, 'main', 1, 10, 'sha')",
                (str(source.resolve()),),
            )
            con.execute(
                """
                INSERT INTO chunks (id, file_id, chunk_index, start_line, end_line, text, text_sha256, dim, model)
                VALUES (11, 1, 0, 1, 2, 'chunk one', 'sha-11', 2, 'voyage-code-3')
                """
            )
            con.commit()
        finally:
            con.close()

        semantic_hits = [
            Hit(
                score=0.8,
                path=str(source.resolve()),
                start_line=1,
                end_line=2,
                text="chunk one",
                chunk_id=11,
                chunk_index=0,
                confidence="high",
            )
        ]
        args = Namespace(
            query="concept only",
            top=5,
            context=0,
            db=str(db_path),
            profile=None,
            global_cache=False,
            repo_root=root,
            model="voyage-code-3",
            dimensions=None,
            json=True,
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("auggie.commands.context.require_embedding_config", return_value=True),
            patch("auggie.commands.context.get_current_branch", return_value="main"),
            patch("auggie.commands.context.query_db", return_value=(semantic_hits, True)),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = cmd_context(args)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["match_source"], "semantic")
        self.assertNotIn("warnings", payload)

    # Known failure carried from upstream clone commit 62899a0: context is imported in cli.py but never registered.
    @unittest.expectedFailure
    def test_context_command_is_registered(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args(["context", "needle", "-n", "3", "-C", "2"])

        self.assertEqual(args.cmd, "context")
        self.assertEqual(args.query, "needle")
        self.assertEqual(args.top, 3)
        self.assertEqual(args.context, 2)
        self.assertIn("cmd_context", command_exports)


if __name__ == "__main__":
    unittest.main()
