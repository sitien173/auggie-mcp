"""Wire-level protocol tests for both MCP handshakes and tool errors."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client
from mcp.types import Implementation

from auggie_mcp import (
    config,
    embed,
    locking,
    output_limits,
    retrieval,
    semantic,
    server,
)
from tests.test_server import FAKE_RG

PROTOCOL = "2026-07-28"


def make_server(repo, telemetry_dir=None):
    telemetry_config = config.TelemetryConfig(dir=telemetry_dir) if telemetry_dir else config.TelemetryConfig()
    return server.create_server(
        app_config=config.Config(agent=config.AgentConfig(driver="pi"), telemetry=telemetry_config),
        repository_root=repo,
        rg_executable=str(FAKE_RG),
    )


def connect(srv, mode=PROTOCOL):
    return Client(srv, mode=mode, client_info=Implementation(name="test", version="1"))


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def run_client(self, fn, *, mode=PROTOCOL):
        async def run():
            async with connect(make_server(self.repo), mode=mode) as client:
                return await fn(client)
        return asyncio.run(run())

    def test_discover_handshake_and_five_tool_output_schemas(self):
        async def fn(client):
            discovered = await client.session.send_discover(PROTOCOL)
            listed = await client.list_tools()
            return discovered, listed
        discovered, listed = self.run_client(fn)
        self.assertIn(PROTOCOL, discovered["supportedVersions"])
        self.assertEqual([tool.name for tool in listed.tools], [
            "grep_search", "semantic_search", "index_status", "index_update", "codebase_retrieval"
        ])
        self.assertTrue(all(tool.output_schema is not None for tool in listed.tools))

    def test_initialize_handshake_and_structured_text_parity(self):
        async def fn(client):
            listed = await client.list_tools()
            result = await client.call_tool("grep_search", {
                "repository_path": str(self.repo), "pattern": "hello", "flags": ["-F"]
            })
            return listed, result
        listed, result = self.run_client(fn, mode="legacy")
        self.assertEqual([tool.name for tool in listed.tools], [
            "grep_search", "semantic_search", "index_status", "index_update", "codebase_retrieval"
        ])
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, __import__("json").loads(result.content[0].text))
        self.assertEqual(set(result.structured_content), {"matches", "truncated", "archive_path"})

    def test_missing_grep_repository_is_invalid_path(self):
        async def fn(client):
            return await client.call_tool("grep_search", {
                "repository_path": str(self.repo / "missing"), "pattern": "x"
            })
        result = self.run_client(fn)
        self.assertTrue(result.is_error)
        self.assertIn("INVALID_PATH", result.content[0].text)

    def test_server_lifespan_cleans_archives_once_for_repository_root(self):
        srv = server.create_server(
            app_config=config.Config(), repository_root=self.repo, rg_executable=str(FAKE_RG)
        )

        async def run():
            async with connect(srv) as client:
                await client.list_tools()

        with patch.object(output_limits, "cleanup_archives_on_startup") as cleanup:
            asyncio.run(run())
        cleanup.assert_called_once_with(self.repo)

    def test_each_design_error_code_reaches_the_mcp_client(self):
        cases = [
            ("INVALID_PATH", "grep_search", {"repository_path": str(self.repo), "pattern": "x", "path": "missing"}, None, None, None),
            ("INVALID_ARGUMENT", "semantic_search", {"repository_path": str(self.repo), "query": "x", "top_k": 0}, None, None, None),
            ("INDEX_MISSING", "semantic_search", {"repository_path": str(self.repo), "query": "x"}, semantic, "semantic_search", semantic.IndexMissingError("missing")),
            ("INDEX_MODEL_MISMATCH", "semantic_search", {"repository_path": str(self.repo), "query": "x"}, semantic, "semantic_search", semantic.ModelMismatchError("mismatch")),
            ("INDEX_BUSY", "index_update", {"repository_path": str(self.repo)}, server, "_run_index_update", locking.IndexBusyError("busy")),
            ("MISSING_API_KEY", "index_update", {"repository_path": str(self.repo)}, server, "_run_index_update", embed.MissingApiKeyError("key")),
            ("EMBED_FAILED", "index_update", {"repository_path": str(self.repo)}, server, "_run_index_update", embed.EmbedError("HTTP 503")),
            ("TIMEOUT", "grep_search", {"repository_path": str(self.repo), "pattern": "x"}, server.grep, "grep_search", TimeoutError("timeout")),
            ("RETRIEVAL_BUSY", "codebase_retrieval", {"repository_path": str(self.repo), "query": "x"}, retrieval, "run_agent", retrieval.RetrievalBusyError("busy")),
            ("REPO_MODIFIED", "codebase_retrieval", {"repository_path": str(self.repo), "query": "x"}, retrieval, "run_agent", retrieval.RepoModifiedError("changed")),
            ("BINARY_NOT_FOUND", "grep_search", {"repository_path": str(self.repo), "pattern": "x"}, server.grep, "grep_search", FileNotFoundError("rg executable not found")),
        ]
        for code, name, arguments, module, function, error in cases:
            with self.subTest(code=code):
                async def fn(client, error=error, name=name, arguments=arguments,
                             module=module, function=function, code=code):
                    if error is None:
                        result = await client.call_tool(name, arguments)
                    else:
                        with patch.object(module, function, side_effect=error):
                            result = await client.call_tool(name, arguments)
                    self.assertTrue(result.is_error)
                    self.assertIn(code, result.content[0].text)
                self.run_client(fn)

    def test_decorator_records_success_error_and_cancellation_with_context(self):
        telemetry_dir = self.repo / "telemetry"
        started = asyncio.Event()

        original = server.grep.grep_search

        async def controlled(repository_path, pattern, **kwargs):
            if pattern == "wait":
                started.set()
                await asyncio.Event().wait()
            return await original(repository_path, pattern, **kwargs)

        async def run_calls():
            async with connect(make_server(self.repo, telemetry_dir)) as client:
                await client.call_tool("grep_search", {
                    "repository_path": str(self.repo), "pattern": "hello"
                })
                await client.call_tool("grep_search", {
                    "repository_path": str(self.repo / "missing"), "pattern": "x"
                })
                pending = asyncio.create_task(client.call_tool("grep_search", {
                    "repository_path": str(self.repo), "pattern": "wait"
                }))
                await started.wait()
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass

        with patch.object(server.grep, "grep_search", side_effect=controlled):
            asyncio.run(run_calls())
        events = [json.loads(line) for path in telemetry_dir.glob("events-*.jsonl")
                  for line in path.read_text().splitlines()]
        self.assertEqual(len(events), 3)
        self.assertEqual({event["outcome"] for event in events}, {"ok", "error", "cancelled"})
        for event in events:
            self.assertEqual(event["tool"], "grep_search")
            self.assertEqual(event["source"], "mcp")
            self.assertEqual(event["client"], {"name": "test", "version": "1"})
            self.assertIn("repository_path", event["input"])
            self.assertIsNotNone(event["runner"])
        success = next(event for event in events if event["outcome"] == "ok")
        self.assertIsNotNone(success["runner"])

    def test_retrieval_event_records_runner_bytes(self):
        telemetry_dir = self.repo / "retrieval-telemetry"

        async def fake_agent(*args, **kwargs):
            return {
                "answer": "retrieval answer", "driver": "pi", "duration_ms": 5,
                "repo_modified": False, "git_check": "skipped",
            }

        async def fn(client):
            return await client.call_tool("codebase_retrieval", {
                "repository_path": str(self.repo), "query": "q"
            })

        async def run():
            async with connect(make_server(self.repo, telemetry_dir)) as client:
                return await fn(client)

        with patch.object(retrieval, "run_agent", side_effect=fake_agent):
            result = asyncio.run(run())
        self.assertFalse(result.is_error)
        event = json.loads(next(telemetry_dir.glob("events-*.jsonl")).read_text())
        self.assertEqual(event["runner"], {
            "exit_code": 0, "stdout_bytes": len("retrieval answer"), "stderr_bytes": 0
        })

    def test_grep_telemetry_records_real_subprocess_runner_values(self):
        telemetry_dir = self.repo / "rg-telemetry"
        executable = self.repo / "controlled-rg"
        event = {
            "type": "match",
            "data": {
                "path": {"text": "hello.txt"},
                "lines": {"text": "hello world\n"},
                "line_number": 1,
            },
        }
        stdout = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        stderr = b"rg diagnostic\n"
        executable.write_text(
            "#!/usr/bin/env python3\nimport sys\n"
            f"if sys.argv[-2] != 'empty-success': sys.stdout.write({stdout.decode()!r})\n"
            f"sys.stderr.write({stderr.decode()!r})\n"
            "if sys.argv[-2] == 'bad-exit': sys.exit(2)\n"
        )
        executable.chmod(0o755)
        srv = server.create_server(
            app_config=config.Config(telemetry=config.TelemetryConfig(dir=telemetry_dir)),
            repository_root=self.repo,
            rg_executable=str(executable),
        )

        async def call():
            async with connect(srv) as client:
                await client.call_tool("grep_search", {
                    "repository_path": str(self.repo), "pattern": "hello"
                })
                await client.call_tool("grep_search", {
                    "repository_path": str(self.repo), "pattern": "empty-success"
                })
                return await client.call_tool("grep_search", {
                    "repository_path": str(self.repo), "pattern": "bad-exit"
                })

        result = asyncio.run(call())
        self.assertTrue(result.is_error)
        records = [json.loads(line) for path in telemetry_dir.glob("events-*.jsonl")
                   for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["runner"], {
            "exit_code": 0,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
        })
        self.assertEqual(records[1]["runner"], {
            "exit_code": 0, "stdout_bytes": 0, "stderr_bytes": len(stderr)
        })
        self.assertEqual(records[2]["outcome"], "error")
        self.assertEqual(records[2]["runner"], {
            "exit_code": 2,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
        })


if __name__ == "__main__":
    unittest.main()
