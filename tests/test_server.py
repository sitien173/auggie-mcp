"""Unit and contract tests for auggie_mcp.server."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.server.mcpserver.exceptions import ToolError

from auggie_mcp import pi_runner, server

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)


def make_server():
    return server.create_server(executable=str(FAKE_PI))


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def run_async(self, coro):
        return asyncio.run(coro)

    # ---- Task 1: discovery exposes exactly one tool -------------------

    def test_discovery_exposes_exactly_codebase_retrieval(self):
        srv = make_server()
        tools = self.run_async(srv.list_tools())
        names = [t.name for t in tools]
        self.assertEqual(names, ["codebase_retrieval"])

    def test_no_resources_or_prompts(self):
        srv = make_server()
        self.assertEqual(self.run_async(srv.list_resources()), [])
        self.assertEqual(self.run_async(srv.list_prompts()), [])

    def test_tool_accepts_only_repository_path_and_query(self):
        srv = make_server()
        tools = self.run_async(srv.list_tools())
        (tool,) = tools
        props = tool.input_schema["properties"]
        self.assertEqual(set(props), {"repository_path", "query"})
        self.assertEqual(
            set(tool.input_schema["required"]), {"repository_path", "query"}
        )

    def test_tool_description_prioritizes_codebase_retrieval(self):
        srv = make_server()
        (tool,) = self.run_async(srv.list_tools())
        self.assertIn("# `mcp__auggie__codebase-retrieval`", tool.description)
        self.assertIn("**FIRST CHOICE**", tool.description)
        self.assertIn("**ALWAYS** use `codebase-retrieval`", tool.description)
        self.assertIn("Absolute path", tool.description)

    # ---- Task 2: success path -----------------------------------------

    def test_successful_call_returns_final_answer_as_text(self):
        srv = make_server()
        result = self.run_async(
            srv.call_tool(
                "codebase_retrieval",
                {"repository_path": str(self.repo), "query": "what is here?"},
            )
        )
        self.assertFalse(result.is_error)
        text = result.content[0].text
        self.assertEqual(text, "Fake final answer.")

    # ---- Task 2: operational failures become tool errors --------------

    def test_invalid_repository_path_returns_tool_error(self):
        srv = make_server()
        with self.assertRaises(ToolError):
            self.run_async(
                srv.call_tool(
                    "codebase_retrieval",
                    {
                        "repository_path": str(self.tmp.name) + "/missing",
                        "query": "q",
                    },
                )
            )

    def test_pi_failure_returns_tool_error(self):
        error = pi_runner.PiExecutionError(
            "Pi exited with status 3", returncode=3, stderr="fake pi failure"
        )
        with patch.object(pi_runner, "run_pi", AsyncMock(side_effect=error)):
            srv = make_server()
            with self.assertRaises(ToolError) as cm:
                self.run_async(
                    srv.call_tool(
                        "codebase_retrieval",
                        {"repository_path": str(self.repo), "query": "q"},
                    )
                )
            self.assertIn("fake pi failure", str(cm.exception))

    def test_parse_failure_returns_tool_error(self):
        error = pi_runner.PiParseError("Pi produced no final assistant response")
        with patch.object(pi_runner, "run_pi", AsyncMock(side_effect=error)):
            srv = make_server()
            with self.assertRaises(ToolError) as cm:
                self.run_async(
                    srv.call_tool(
                        "codebase_retrieval",
                        {"repository_path": str(self.repo), "query": "q"},
                    )
                )
        self.assertIn("no final assistant response", str(cm.exception))

    def test_timeout_returns_tool_error(self):
        with patch.object(
            pi_runner, "run_pi", AsyncMock(side_effect=asyncio.TimeoutError())
        ):
            srv = make_server()
            with self.assertRaises(ToolError) as cm:
                self.run_async(
                    srv.call_tool(
                        "codebase_retrieval",
                        {"repository_path": str(self.repo), "query": "q"},
                    )
                )
        self.assertIn("TimeoutError", str(cm.exception))

    def test_cancellation_propagates_into_adapter(self):
        async def scenario():
            started = asyncio.Event()
            cancelled = asyncio.Event()

            async def wait_forever(*args, **kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            with patch.object(pi_runner, "run_pi", side_effect=wait_forever):
                srv = make_server()
                task = asyncio.create_task(
                    srv.call_tool(
                        "codebase_retrieval",
                        {"repository_path": str(self.repo), "query": "q"},
                    )
                )
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(cancelled.is_set())

        self.run_async(scenario())

    def test_unknown_tool_returns_error(self):
        srv = make_server()
        with self.assertRaises(ToolError):
            self.run_async(srv.call_tool("nope", {"repository_path": str(self.repo), "query": "q"}))

    # ---- Task 3: entry points -----------------------------------------

    def test_main_module_has_run_entry(self):
        # python -m auggie_mcp must exist and call server.run
        import importlib
        main = importlib.import_module("auggie_mcp.__main__")
        self.assertTrue(callable(getattr(main, "main", None)) or hasattr(main, "main"))

    def test_package_console_entrypoint_configured(self):
        import tomllib
        with open("pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        scripts = data["project"]["scripts"]
        self.assertIn("auggie-mcp", scripts)
        self.assertTrue(scripts["auggie-mcp"].startswith("auggie_mcp.server:"))

    def test_dependency_stays_on_mcp_v2(self):
        import tomllib
        with open("pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        self.assertIn("mcp>=2,<3", data["project"]["dependencies"])

    def test_no_production_pi_environment_switches(self):
        self.assertFalse(hasattr(server, "PI_EXECUTABLE"))
        self.assertFalse(hasattr(server, "PI_EXTRA_ARGS"))

    def test_no_extra_transports(self):
        # server module exposes only the stdio runner
        import auggie_mcp.server as s
        self.assertTrue(hasattr(s, "run"))
        self.assertFalse(hasattr(s, "run_sse"))
        self.assertFalse(hasattr(s, "run_http"))

    # ---- Task 3: contract test with fake Pi over stdio ---------------

    def test_contract_stdio_discovery_and_call(self):
        """Start the real server over stdio and speak minimal JSON-RPC."""
        import json
        import subprocess

        env = dict(os.environ)
        fake_bin_dir = tempfile.TemporaryDirectory()
        self.addCleanup(fake_bin_dir.cleanup)
        fake_pi = Path(fake_bin_dir.name) / "pi"
        fake_pi.symlink_to(FAKE_PI)
        env["PATH"] = fake_bin_dir.name + os.pathsep + env.get("PATH", "")
        proc = subprocess.Popen(
            [os.sys.executable, "-m", "auggie_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(TESTS_DIR.parent),
        )

        def request(req):
            proc.stdin.write((json.dumps(req) + "\n").encode())
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line)

        try:
            # initialize
            resp = request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"},
                    },
                }
            )
            self.assertIn("result", resp)
            # notifications/initialized
            proc.stdin.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "method": "notifications/initialized"}
                    )
                    + "\n"
                ).encode()
            )
            proc.stdin.flush()
            # tools/list
            resp = request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            names = [t["name"] for t in resp["result"]["tools"]]
            self.assertEqual(names, ["codebase_retrieval"])
            # tools/call
            resp = request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "codebase_retrieval",
                        "arguments": {
                            "repository_path": str(self.repo),
                            "query": "what is here?",
                        },
                    },
                }
            )
            self.assertFalse(resp["result"]["isError"])
            text = resp["result"]["content"][0]["text"]
            self.assertEqual(text, "Fake final answer.")
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
