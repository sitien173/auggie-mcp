"""Contract tests for the five-tool MCP server."""

import asyncio
import importlib.metadata
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import Implementation

from auggie_mcp import (
    config,
    embed,
    locking,
    output_limits,
    retrieval,
    semantic,
    server,
    telemetry,
)

FAKE_RG = Path(__file__).parent / "fake_rg.py"


def make_server(repo=None):
    cfg = config.Config(agent=config.AgentConfig(driver="pi"))
    return server.create_server(
        app_config=cfg,
        rg_executable=str(FAKE_RG),
        repository_root=repo,
    )


class ServerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def call(self, srv, name, args):
        return asyncio.run(srv.call_tool(name, args))

    def test_five_tools_have_expected_parameters_and_output_schema(self):
        tools = asyncio.run(make_server(self.repo).list_tools())
        self.assertEqual([tool.name for tool in tools], [
            "grep_search", "semantic_search", "index_status", "index_update", "codebase_retrieval"
        ])
        expected = {
            "grep_search": {"repository_path", "pattern", "path", "flags", "max_results"},
            "semantic_search": {"repository_path", "query", "path", "top_k", "context_lines"},
            "index_status": {"repository_path"},
            "index_update": {"repository_path", "rebuild"},
            "codebase_retrieval": {"repository_path", "query"},
        }
        for tool in tools:
            self.assertEqual(set(tool.input_schema["properties"]), expected[tool.name])
            self.assertIsNotNone(tool.output_schema)

    def test_metadata_version_comes_from_distribution(self):
        self.assertEqual(make_server().version, importlib.metadata.version("auggie-mcp"))

    def test_grep_structured_and_text_output_agree(self):
        result = self.call(make_server(self.repo), "grep_search", {
            "repository_path": str(self.repo), "pattern": "hello", "flags": ["-F"]
        })
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, json.loads(result.content[0].text))
        self.assertEqual(set(result.structured_content), {"matches", "truncated", "archive_path"})

    def test_index_update_and_retrieval_shapes(self):
        with patch.object(server, "_run_index_update", AsyncMock(return_value={
            "files_scanned": 1, "files_changed": 1, "files_deleted": 0,
            "chunks_embedded": 1, "chunks_reused": 0, "duration_ms": 1,
        })):
            result = self.call(make_server(self.repo), "index_update", {"repository_path": str(self.repo)})
        self.assertEqual(set(result.structured_content), {
            "files_scanned", "files_changed", "files_deleted", "chunks_embedded", "chunks_reused", "duration_ms"
        })
        with patch.object(retrieval, "run_agent", AsyncMock(return_value={
            "answer": "answer", "driver": "pi", "duration_ms": 1,
            "repo_modified": False, "git_check": "skipped",
        })):
            result = self.call(make_server(self.repo), "codebase_retrieval", {
                "repository_path": str(self.repo), "query": "q"
            })
        self.assertEqual(result.structured_content["answer"], "answer")
        self.assertFalse(result.structured_content["truncated"])
        self.assertIsNone(result.structured_content["archive_path"])
        self.assertEqual(result.structured_content, json.loads(result.content[0].text))

    def test_configured_index_timeout_maps_to_timeout(self):
        cfg = config.Config(
            index=config.IndexConfig(timeout_s=0.01),
            telemetry=config.TelemetryConfig(enabled=False),
        )
        process = type("Process", (), {"pid": 987654, "returncode": None})()
        process.communicate = AsyncMock()

        async def wait_for_timeout(awaitable, *, timeout):
            awaitable.close()
            if timeout > 1:
                raise AssertionError(f"unexpected duration: {timeout}")
            raise TimeoutError

        with (
            patch.object(server.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)),
            patch.object(server.asyncio, "wait_for", side_effect=wait_for_timeout),
            patch.object(server, "_kill_process_group", AsyncMock()),
            self.assertRaisesRegex(ToolError, "TIMEOUT"),
        ):
            self.call(
                server.create_server(app_config=cfg, repository_root=self.repo),
                "index_update",
                {"repository_path": str(self.repo)},
            )

    def test_semantic_parameters_validate_as_invalid_argument(self):
        for params in ({"top_k": 0}, {"context_lines": 21}):
            with self.subTest(params=params), self.assertRaisesRegex(ToolError, "INVALID_ARGUMENT"):
                self.call(make_server(self.repo), "semantic_search", {
                    "repository_path": str(self.repo), "query": "q", **params
                })

    def test_error_mapping(self):
        cases = [
            ("INVALID_PATH", "grep_search", {"repository_path": str(self.repo), "pattern": "x", "path": "missing"}, None),
            ("INVALID_ARGUMENT", "semantic_search", {"repository_path": str(self.repo), "query": "x", "top_k": 0}, None),
            ("INDEX_MISSING", "semantic_search", {"repository_path": str(self.repo), "query": "x"}, semantic.IndexMissingError("missing index")),
            ("INDEX_MODEL_MISMATCH", "semantic_search", {"repository_path": str(self.repo), "query": "x"}, semantic.ModelMismatchError("wrong model")),
            ("INDEX_BUSY", "index_update", {"repository_path": str(self.repo)}, locking.IndexBusyError("busy")),
            ("MISSING_API_KEY", "index_update", {"repository_path": str(self.repo)}, embed.MissingApiKeyError("missing key")),
            ("EMBED_FAILED", "index_update", {"repository_path": str(self.repo)}, embed.EmbedError("HTTP 503")),
            ("TIMEOUT", "grep_search", {"repository_path": str(self.repo), "pattern": "x"}, TimeoutError("too slow")),
            ("RETRIEVAL_BUSY", "codebase_retrieval", {"repository_path": str(self.repo), "query": "x"}, retrieval.RetrievalBusyError("busy")),
            ("REPO_MODIFIED", "codebase_retrieval", {"repository_path": str(self.repo), "query": "x"}, retrieval.RepoModifiedError("changed")),
            ("BINARY_NOT_FOUND", "grep_search", {"repository_path": str(self.repo), "pattern": "x"}, FileNotFoundError("rg executable not found")),
        ]
        for code, name, args, error in cases:
            with self.subTest(code=code):
                if error is not None:
                    target = {
                        "semantic_search": (semantic, "semantic_search"),
                        "index_update": (server, "_run_index_update"),
                        "codebase_retrieval": (retrieval, "run_agent"),
                        "grep_search": (server.grep, "grep_search"),
                    }[name]
                    with patch.object(target[0], target[1], side_effect=error), self.assertRaisesRegex(
                        ToolError, code
                    ):
                        self.call(make_server(self.repo), name, args)
                else:
                    with self.assertRaisesRegex(ToolError, code):
                        self.call(make_server(self.repo), name, args)

    def test_retrieval_truncation_archives_full_answer(self):
        full = ("line\n" * 20000)
        with patch.object(retrieval, "run_agent", AsyncMock(return_value={
            "answer": full, "driver": "pi", "duration_ms": 1,
            "repo_modified": False, "git_check": "skipped",
        })):
            result = self.call(make_server(self.repo), "codebase_retrieval", {
                "repository_path": str(self.repo), "query": "q"
            })
        data = result.structured_content
        self.assertTrue(data["truncated"])
        self.assertTrue(Path(data["archive_path"]).is_file())
        self.assertLess(len(data["answer"].encode()), 64 * 1024)

    def test_disabled_config_telemetry_writes_no_server_event(self):
        telemetry_dir = self.repo / "telemetry"
        cfg = config.Config(telemetry=config.TelemetryConfig(
            enabled=False, dir=telemetry_dir, retention_days=2
        ))
        with patch.dict(os.environ, {"AUGGIE_TELEMETRY": "1", "AUGGIE_TELEMETRY_DIR": str(telemetry_dir / "env")}):
            self.call(server.create_server(app_config=cfg, rg_executable=str(FAKE_RG)), "grep_search", {
                "repository_path": str(self.repo), "pattern": "hello"
            })
        self.assertFalse(telemetry_dir.exists())

    def test_toml_telemetry_directory_is_used_for_server_events(self):
        telemetry_dir = self.repo / "from-config"
        config_file = self.repo / "config.toml"
        config_file.write_text(
            f'[telemetry]\nenabled = true\ndir = "{telemetry_dir}"\nretention_days = 4\n'
        )
        cfg = config.load(config_file, environ={})
        with patch.dict(os.environ, {"AUGGIE_TELEMETRY_DIR": str(self.repo / "from-env")}):
            self.call(server.create_server(app_config=cfg, rg_executable=str(FAKE_RG)), "grep_search", {
                "repository_path": str(self.repo), "pattern": "hello"
            })
        self.assertEqual(len(list(telemetry_dir.glob("events-*.jsonl"))), 1)
        self.assertFalse((self.repo / "from-env").exists())

    def test_server_startup_prunes_configured_telemetry_directory_and_retention(self):
        telemetry_dir = self.repo / "configured-telemetry"
        config_file = self.repo / "config.toml"
        config_file.write_text(
            f'[telemetry]\nenabled = true\ndir = "{telemetry_dir}"\nretention_days = 4\n'
        )
        cfg = config.load(config_file, environ={})
        srv = server.create_server(
            app_config=cfg, repository_root=self.repo, rg_executable=str(FAKE_RG)
        )

        async def run():
            async with Client(
                srv,
                mode="2026-07-28",
                client_info=Implementation(name="telemetry-config-test", version="1"),
            ) as client:
                await client.list_tools()

        with patch.object(telemetry, "prune_old_files", wraps=telemetry.prune_old_files) as prune:
            asyncio.run(run())
        prune.assert_called_once_with(enabled=True, directory=telemetry_dir, retention=4)

    def test_unicode_retrieval_response_serialization_fits_output_limit(self):
        full = "é\n" * 12000
        with patch.object(retrieval, "run_agent", AsyncMock(return_value={
            "answer": full, "driver": "pi", "duration_ms": 1,
            "repo_modified": False, "git_check": "skipped",
        })):
            result = self.call(make_server(self.repo), "codebase_retrieval", {
                "repository_path": str(self.repo), "query": "q"
            })
        data = result.structured_content
        self.assertTrue(data["truncated"])
        self.assertLessEqual(
            len(output_limits.serialize_json(data).encode("utf-8")),
            output_limits.MAX_OUTPUT_BYTES,
        )
        archived = json.loads(Path(data["archive_path"]).read_text(encoding="utf-8"))
        self.assertEqual(archived["answer"], full)

    def test_index_subprocess_structured_embed_error_preserves_status(self):
        fake_python = self.repo / "fake-python"
        fake_python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "sys.stdout.write(json.dumps({'error': {'code': 'EMBED_FAILED', "
            "'message': 'quota exceeded', 'http_status': 429}}))\n"
            "sys.exit(2)\n"
        )
        fake_python.chmod(0o755)
        with patch.object(server.sys, "executable", str(fake_python)), self.assertRaises(
            ToolError
        ) as raised:
            self.call(make_server(self.repo), "index_update", {
                "repository_path": str(self.repo)
            })
        self.assertIn("EMBED_FAILED", str(raised.exception))
        self.assertIn("HTTP 429", str(raised.exception))

    def test_real_cli_warning_does_not_break_structured_embed_error(self):
        home = self.repo / "home"
        config_dir = home / ".auggie"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text('[search]\nmode = "classic"\n')
        cfg = config.Config(telemetry=config.TelemetryConfig(enabled=False))

        with patch.dict(os.environ, {
            "HOME": str(home),
            "AUGGIE_TEST_INDEX_EMBED_ERROR_STATUS": "429",
            "AUGGIE_TELEMETRY": "0",
        }), self.assertRaises(ToolError) as raised:
            self.call(server.create_server(app_config=cfg), "index_update", {
                "repository_path": str(self.repo)
            })

        self.assertIn("EMBED_FAILED", str(raised.exception))
        self.assertIn("HTTP 429", str(raised.exception))

    def test_missing_directory_from_cli_subprocess_maps_to_invalid_path(self):
        async def scenario():
            original_spawn = asyncio.create_subprocess_exec

            async def remove_before_cli(*args, **kwargs):
                import shutil
                shutil.rmtree(self.repo)
                return await original_spawn(*args, **kwargs)

            cfg = config.Config(telemetry=config.TelemetryConfig(enabled=False))
            srv = server.create_server(app_config=cfg, repository_root=self.repo)
            with (
                patch.dict(os.environ, {"HOME": str(self.repo.parent), "AUGGIE_TELEMETRY": "0"}),
                patch.object(asyncio, "create_subprocess_exec", side_effect=remove_before_cli),
                self.assertRaisesRegex(ToolError, "INVALID_PATH"),
            ):
                await srv.call_tool("index_update", {"repository_path": str(self.repo)})

        asyncio.run(scenario())

    def test_embed_error_message_includes_status_when_missing(self):
        error = embed.EmbedError("quota exceeded", status=429)
        self.assertIn("HTTP 429", str(server._tool_error(error)))

    def fake_python(self, source, name="fake-python"):
        path = self.repo / name
        path.write_text("#!/usr/bin/env python3\n" + source)
        path.chmod(0o755)
        return path

    @staticmethod
    def process_is_running(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        stat_path = Path(f"/proc/{pid}/stat")
        return not (stat_path.exists() and stat_path.read_text().split()[2] == "Z")

    def test_index_subprocess_timeout_and_cancellation_kill_process_group(self):
        pid_path = self.repo / "grandchild.pid"
        script = self.fake_python(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open({str(pid_path)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(60)\n"
        )

        async def wait_pid():
            for _ in range(200):
                if pid_path.exists():
                    return int(pid_path.read_text())
                await asyncio.sleep(0.01)
            self.fail("fake index child did not start")

        async def scenario():
            with (
                patch.object(server.sys, "executable", str(script)),
                self.assertRaises(TimeoutError),
            ):
                await server._run_index_update(str(self.repo), False, timeout_s=0.1)
            timeout_pid = int(pid_path.read_text())
            self.assertFalse(self.process_is_running(timeout_pid))

            pid_path.unlink()
            with patch.object(server.sys, "executable", str(script)):
                task = asyncio.create_task(server._run_index_update(str(self.repo), False))
                cancel_pid = await wait_pid()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            for _ in range(100):
                if not self.process_is_running(cancel_pid):
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(self.process_is_running(cancel_pid))

        asyncio.run(scenario())

    def test_index_subprocess_busy_and_malformed_json_have_defined_errors(self):
        busy = self.fake_python(
            "import json, sys\n"
            "sys.stderr.write(json.dumps({'error': {'code': 'INDEX_BUSY', 'message': 'locked', 'http_status': None}}))\n"
            "sys.exit(3)\n",
            "fake-busy-python",
        )
        malformed = self.fake_python(
            "import sys\nsys.stdout.write('not-json')\nsys.exit(0)\n",
            "fake-malformed-python",
        )

        async def scenario():
            with patch.object(server.sys, "executable", str(busy)), self.assertRaises(
                locking.IndexBusyError
            ):
                await server._run_index_update(str(self.repo), False)
            with patch.object(server.sys, "executable", str(malformed)):
                with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                    await server._run_index_update(str(self.repo), False)
                with self.assertRaisesRegex(ToolError, "INTERNAL_ERROR"):
                    await make_server(self.repo).call_tool("index_update", {
                        "repository_path": str(self.repo)
                    })
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
