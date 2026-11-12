"""Tests for auggie_mcp.index_tools and the index_status / index_update tools."""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from auggie import db as auggie_db
from auggie_mcp import index_tools, server

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"
FAKE_RG = TESTS_DIR / "fake_rg.py"
FAKE_AUGGIE = TESTS_DIR / "fake_auggie.py"
FAKE_COMMAND = [sys.executable, str(FAKE_AUGGIE)]

MODEL = "voyage-code-3"
DIM = 512


def pid_alive(pid):
    """True while the process exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class RepoCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(os.path.realpath(self.tmp.name)) / "repo"
        self.repo.mkdir()
        self.db = self.repo / ".auggie" / "index.sqlite"
        env = patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def write(self, name, text="x = 1\n"):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def git(self, *args):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=self.repo, check=True, capture_output=True,
        )

    def build_index(
        self, names, model=MODEL, dim=DIM, history=("2026-11-01 10:00:00",), branch="default"
    ):
        """Create a real-schema index recording the current stat of each file."""
        con = auggie_db.connect(self.db)
        try:
            for name in names:
                path = self.repo / name
                st = path.stat()
                cur = con.execute(
                    "INSERT INTO files(path, branch, mtime_ns, size, sha256) "
                    "VALUES(?, ?, ?, ?, 'sha')",
                    (str(path), branch, st.st_mtime_ns, st.st_size),
                )
                for i in range(2):
                    con.execute(
                        "INSERT INTO chunks(file_id, chunk_index, start_line, end_line, "
                        "text, text_sha256, dim, model) VALUES(?, ?, 1, 1, 't', ?, ?, ?)",
                        (cur.lastrowid, i, f"h{name}{i}", dim, model),
                    )
            for stamp in history:
                con.execute(
                    "INSERT INTO index_history(timestamp, action) VALUES(?, 'index')",
                    (stamp,),
                )
            con.commit()
        finally:
            con.close()


class IndexStatusTest(RepoCase):
    def test_missing_index(self):
        self.write("a.py")
        result = index_tools.index_status(self.repo)
        self.assertFalse(result["exists"])
        self.assertIn("index_update", result["next_step"])
        self.assertFalse(self.db.exists())

    def test_counts(self):
        self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"], history=("2026-11-01 10:00:00", "2026-11-02 09:30:00"))
        result = index_tools.index_status(self.repo)
        self.assertTrue(result["exists"])
        self.assertEqual(result["model"], MODEL)
        self.assertEqual(result["dim"], DIM)
        self.assertEqual(result["files"], 2)
        self.assertEqual(result["chunks"], 4)
        self.assertEqual(result["last_indexed"], "2026-11-02 09:30:00")
        self.assertEqual(
            result["stale"], {"modified": 0, "deleted": 0, "candidate_added": 0}
        )

    def test_stale_modified_deleted_and_candidate_added(self):
        a = self.write("a.py")
        self.write("b.py")
        self.write("c.py")
        self.build_index(["a.py", "b.py", "c.py"])
        a.write_text("x = 1\ny = 2\n")
        (self.repo / "b.py").unlink()
        self.write("new.py")
        result = index_tools.index_status(self.repo)
        self.assertEqual(
            result["stale"], {"modified": 1, "deleted": 1, "candidate_added": 1}
        )
        self.assertIn("index_update", result["next_step"])

    def test_candidate_added_applies_indexer_ignore_rules(self):
        self.write("a.py")
        self.build_index(["a.py"])
        self.write("notes.md", "docs\n")
        self.write("node_modules/dep.js", "x\n")
        self.write("big.py", "x" * 2_000_001)
        self.write("empty.py", "")
        self.write("kept.py")
        result = index_tools.index_status(self.repo)
        self.assertEqual(result["stale"]["candidate_added"], 1)

    def test_database_is_not_written(self):
        self.write("a.py")
        self.build_index(["a.py"])
        (self.repo / "a.py").write_text("changed\n")
        before_bytes = self.db.read_bytes()
        before_stat = self.db.stat()
        index_tools.index_status(self.repo)
        self.assertEqual(self.db.read_bytes(), before_bytes)
        self.assertEqual(self.db.stat().st_mtime_ns, before_stat.st_mtime_ns)

    def test_does_not_migrate_old_schema(self):
        self.write("a.py")
        self.db.parent.mkdir(parents=True)
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT, mtime_ns INT, size INT)")
        con.commit()
        con.close()
        before = self.db.read_bytes()
        with self.assertRaisesRegex(ToolError, "index_update"):
            index_tools.index_status(self.repo)
        self.assertEqual(self.db.read_bytes(), before)

    def test_invalid_path_is_tool_error(self):
        with self.assertRaisesRegex(ToolError, "does not exist"):
            index_tools.index_status(self.repo / "nope")

    def test_subdirectory_reports_same_as_root(self):
        self.git("init", "-q", "-b", "main")
        self.write("a.py")
        self.write("sub/b.py")
        self.build_index(["a.py", "sub/b.py"], branch="main")
        self.write("sub/new.py")
        self.write("outside.py")
        at_root = index_tools.index_status(self.repo)
        at_sub = index_tools.index_status(self.repo / "sub")
        self.assertEqual(at_root["stale"]["candidate_added"], 2)
        self.assertEqual(at_sub, at_root)

    def test_branch_is_not_cached_across_checkouts(self):
        self.git("init", "-q", "-b", "main")
        self.write("a.py")
        self.build_index(["a.py"], branch="main")
        self.assertEqual(index_tools.index_status(self.repo)["files"], 1)
        self.git("checkout", "-q", "-b", "other")
        self.assertEqual(index_tools.index_status(self.repo)["files"], 0)

    def test_stale_files_is_reusable(self):
        a = self.write("a.py")
        self.write("b.py")
        self.build_index(["a.py", "b.py"])
        a.write_text("changed\n")
        (self.repo / "b.py").unlink()
        con = index_tools.open_readonly(self.db)
        try:
            modified, deleted = index_tools.stale_files(con, "default")
        finally:
            con.close()
        self.assertEqual(modified, [str(a)])
        self.assertEqual(deleted, [str(self.repo / "b.py")])


class IndexUpdateTest(RepoCase):
    def setUp(self):
        super().setUp()
        self.log = Path(self.tmp.name) / "index.log"
        self.pidfile = Path(self.tmp.name) / "pids"
        self.progress = []

        async def record(progress, total, message):
            self.progress.append((progress, total, message))

        self.record = record

    def update(self, rebuild=False, env=None, command=FAKE_COMMAND):
        environ = {"FAKE_INDEX_LOG": str(self.log), **(env or {})}
        with patch.dict(os.environ, environ):
            return asyncio.run(
                index_tools.run_index_update(
                    self.repo,
                    rebuild=rebuild,
                    command=command,
                    report_progress=self.record,
                )
            )

    def log_entries(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_default_command_uses_interpreter_module(self):
        self.assertEqual(
            index_tools.default_index_command(), [sys.executable, "-m", "auggie"]
        )

    def test_first_build_command_line(self):
        self.update()
        (entry,) = self.log_entries()
        self.assertEqual(entry, ["index", ["index", str(self.repo), "--json"], str(self.repo)])

    def test_incremental_passes_stored_model_and_dimensions(self):
        self.write("a.py")
        self.build_index(["a.py"])
        self.update()
        (entry,) = self.log_entries()
        self.assertEqual(
            entry[1],
            ["index", str(self.repo), "--json", "--model", MODEL, "--dimensions", str(DIM)],
        )

    def test_rebuild_runs_reindex_without_stored_model(self):
        self.write("a.py")
        self.build_index(["a.py"])
        self.update(rebuild=True)
        (entry,) = self.log_entries()
        self.assertEqual(entry[0], "reindex")
        self.assertEqual(entry[1], ["reindex", str(self.repo), "--json"])

    def test_result_has_indexer_fields_and_duration(self):
        result = self.update()
        self.assertEqual(result["files_indexed"], 2)
        self.assertEqual(result["chunks_total"], 4)
        self.assertIsInstance(result["duration_ms"], int)

    def test_progress_at_start_and_finish(self):
        self.update()
        self.assertEqual(len(self.progress), 2)
        self.assertEqual(self.progress[0][0], 0)
        self.assertEqual(self.progress[1][0], self.progress[1][1])

    def test_busy_maps_to_tool_error(self):
        with self.assertRaisesRegex(ToolError, "index update already running"):
            self.update(env={"FAKE_INDEX_BUSY": "1"})

    def test_other_failure_includes_stderr_tail(self):
        with self.assertRaisesRegex(ToolError, "index boom"):
            self.update(env={"FAKE_INDEX_FAIL": "1"})

    def test_missing_api_key_checked_before_launch(self):
        with patch.dict(os.environ):
            os.environ.pop("VOYAGE_API_KEY")
            with self.assertRaisesRegex(ToolError, "VOYAGE_API_KEY"):
                self.update()
        self.assertFalse(self.log.exists())

    def test_launch_failure_is_tool_error(self):
        with self.assertRaisesRegex(ToolError, "could not start"):
            self.update(command=["/definitely/missing/python"])

    def test_invalid_timeout_is_tool_error(self):
        with self.assertRaisesRegex(ToolError, "AUGGIE_INDEX_TIMEOUT_S"):
            self.update(env={"AUGGIE_INDEX_TIMEOUT_S": "soon"})

    def test_subdirectory_input_indexes_git_root(self):
        self.git("init", "-q", "-b", "main")
        (self.repo / "sub").mkdir()
        self.update_at(self.repo / "sub")
        (entry,) = self.log_entries()
        self.assertEqual(entry[1][1], str(self.repo))
        self.assertEqual(entry[2], str(self.repo))

    def update_at(self, path):
        with patch.dict(os.environ, {"FAKE_INDEX_LOG": str(self.log)}):
            return asyncio.run(
                index_tools.run_index_update(
                    path, command=FAKE_COMMAND, report_progress=self.record
                )
            )

    def test_timeout_kills_children_after_leader_exits(self):
        env = {
            "FAKE_INDEX_LEADER_EXIT": "1",
            "FAKE_INDEX_PIDFILE": str(self.pidfile),
            "AUGGIE_INDEX_TIMEOUT_S": "1",
        }
        with self.assertRaisesRegex(ToolError, "timed out"):
            self.update(env=env)
        for pid in self.read_pids():
            self.assertFalse(pid_alive(pid), pid)

    def read_pids(self):
        return [int(p) for p in self.pidfile.read_text().split()]

    def test_timeout_kills_process_group(self):
        env = {
            "FAKE_INDEX_SLEEP": "30",
            "FAKE_INDEX_PIDFILE": str(self.pidfile),
            "AUGGIE_INDEX_TIMEOUT_S": "1",
        }
        with self.assertRaisesRegex(ToolError, "timed out"):
            self.update(env=env)
        for pid in self.read_pids():
            self.assertFalse(pid_alive(pid), pid)

    def test_cancellation_kills_process_group(self):
        env = {"FAKE_INDEX_SLEEP": "30", "FAKE_INDEX_PIDFILE": str(self.pidfile)}

        async def scenario():
            with patch.dict(os.environ, env):
                task = asyncio.create_task(
                    index_tools.run_index_update(
                        self.repo,
                        rebuild=False,
                        command=FAKE_COMMAND,
                        report_progress=self.record,
                    )
                )
                for _ in range(100):
                    if self.pidfile.exists() and len(self.pidfile.read_text().split()) == 2:
                        break
                    await asyncio.sleep(0.1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(scenario())
        for pid in self.read_pids():
            self.assertFalse(pid_alive(pid), pid)


class ServerToolsTest(RepoCase):
    def make_server(self):
        return server.create_server(
            executable=str(FAKE_PI),
            rg_executable=str(FAKE_RG),
            index_command=FAKE_COMMAND,
        )

    def test_tools_and_annotations(self):
        tools = {t.name: t for t in asyncio.run(self.make_server().list_tools())}
        status = tools["index_status"].annotations
        self.assertTrue(status.read_only_hint)
        self.assertTrue(status.idempotent_hint)
        self.assertFalse(status.open_world_hint)
        update = tools["index_update"].annotations
        self.assertFalse(update.read_only_hint)
        self.assertTrue(update.destructive_hint)
        self.assertTrue(update.open_world_hint)
        for text in ("rebuild=True", "not recovered"):
            self.assertIn(text, tools["index_update"].description)
        self.assertEqual(
            set(tools["index_update"].input_schema["properties"]),
            {"repository_path", "rebuild"},
        )
        self.assertEqual(
            set(tools["index_status"].input_schema["properties"]), {"repository_path"}
        )

    def test_status_tool_missing_index(self):
        result = asyncio.run(
            self.make_server().call_tool("index_status", {"repository_path": str(self.repo)})
        )
        self.assertFalse(result.is_error)
        self.assertFalse(result.structured_content["exists"])

    def test_update_tool_reports_progress_and_result(self):
        srv = self.make_server()
        with patch.object(Context, "report_progress", new_callable=AsyncMock) as progress:
            result = asyncio.run(
                srv.call_tool("index_update", {"repository_path": str(self.repo)})
            )
        self.assertEqual(result.structured_content["files_indexed"], 2)
        self.assertEqual(progress.await_count, 2)

    def test_update_tool_busy_is_tool_error(self):
        srv = self.make_server()
        with patch.dict(os.environ, {"FAKE_INDEX_BUSY": "1"}):
            with patch.object(Context, "report_progress", new_callable=AsyncMock):
                with self.assertRaisesRegex(ToolError, "index update already running"):
                    asyncio.run(
                        srv.call_tool("index_update", {"repository_path": str(self.repo)})
                    )


if __name__ == "__main__":
    unittest.main()
