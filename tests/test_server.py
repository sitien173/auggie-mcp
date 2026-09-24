"""Unit and contract tests for auggie_mcp.server."""

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.server.mcpserver.exceptions import ToolError
from typesafe_sdk import ChoiceAnswer

from auggie_mcp import pi_runner, server
from auggie_mcp.output_processing import (
    RetrievalEnvelope,
    SearchEnvelope,
    create_typesafe_client,
    load_config,
)

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"
FAKE_TGREP = TESTS_DIR / "fake_tgrep.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)
if not os.access(FAKE_TGREP, os.X_OK):
    os.chmod(FAKE_TGREP, 0o755)


def make_server(repository_root=None, client_factory=None, processor_factory=None):
    return server.create_server(
        executable=str(FAKE_PI),
        tgrep_executable=str(FAKE_TGREP),
        repository_root=repository_root,
        client_factory=client_factory,
        processor_factory=processor_factory,
    )


class FakeClientWithCounters:
    """Mock TypeSafe client tracking creation, calls, and closure."""

    def __init__(self):
        self.created = 0
        self.system_one_calls = 0
        self.closed = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        self.closed += 1

    async def system_one(self, state, questions, *, model=None, **kwargs):
        self.system_one_calls += 1
        choices = {
            q_id: ChoiceAnswer(choice="reference") for q_id in questions.keys()
        }

        class Resp:
            pass

        r = Resp()
        r.choices = choices
        return r


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def run_async(self, coro):
        return asyncio.run(coro)

    # ---- 1. Tool discovery and schemas ---------------------------------

    def test_discovery_exposes_retrieval_and_search(self):
        srv = make_server()
        tools = self.run_async(srv.list_tools())
        names = [t.name for t in tools]
        self.assertEqual(names, ["codebase_retrieval", "codebase_search"])

    def test_no_resources_or_prompts(self):
        srv = make_server()
        self.assertEqual(self.run_async(srv.list_resources()), [])
        self.assertEqual(self.run_async(srv.list_prompts()), [])

    def test_server_advertised_version_is_0_2_0(self):
        srv = make_server()
        self.assertEqual(srv.version, "0.2.0")

    def test_retrieval_tool_schema_matches_phase3_contract(self):
        srv = make_server()
        tools = self.run_async(srv.list_tools())
        tool = next(t for t in tools if t.name == "codebase_retrieval")
        props = tool.input_schema["properties"]
        self.assertEqual(set(props), {"repository_path", "query", "goal"})
        self.assertEqual(
            set(tool.input_schema["required"]), {"repository_path", "query"}
        )
        self.assertEqual(props["goal"]["default"], None)

        out_props = tool.output_schema["properties"]
        self.assertEqual(set(out_props), {"tool", "data", "processing"})
        self.assertEqual(
            set(tool.output_schema["required"]), {"tool", "data", "processing"}
        )
        defs = tool.output_schema.get("$defs", {})
        self.assertIn("RetrievalData", defs)
        self.assertEqual(defs["RetrievalData"]["required"], ["result"])
        self.assertIn("ProcessingMetadata", defs)
        self.assertEqual(
            set(defs["ProcessingMetadata"]["required"]),
            {"status", "retention", "estimated_tokens", "recovery_path", "chunks"},
        )

    def test_search_tool_schema_matches_phase3_contract(self):
        srv = make_server()
        tools = self.run_async(srv.list_tools())
        tool = next(t for t in tools if t.name == "codebase_search")
        props = tool.input_schema["properties"]
        self.assertEqual(set(props), {"pattern", "path", "flags", "goal"})
        self.assertEqual(set(tool.input_schema["required"]), {"pattern"})
        self.assertEqual(props["path"]["default"], ".")
        self.assertEqual(props["goal"]["default"], None)

        out_props = tool.output_schema["properties"]
        self.assertEqual(set(out_props), {"tool", "data", "processing"})
        self.assertEqual(
            set(tool.output_schema["required"]), {"tool", "data", "processing"}
        )
        defs = tool.output_schema.get("$defs", {})
        self.assertIn("SearchData", defs)
        self.assertEqual(
            set(defs["SearchData"]["required"]),
            {"stdout", "stderr", "exit_code"},
        )
        self.assertIn("ProcessingMetadata", defs)
        self.assertEqual(
            set(defs["ProcessingMetadata"]["required"]),
            {"status", "retention", "estimated_tokens", "recovery_path", "chunks"},
        )

    def test_tool_descriptions_distinguish_semantic_and_exact_search(self):
        srv = make_server()
        tools = self.run_async(srv.list_tools())
        retrieval = next(t for t in tools if t.name == "codebase_retrieval")
        search = next(t for t in tools if t.name == "codebase_search")
        self.assertIn("**FIRST CHOICE**", retrieval.description)
        self.assertIn("**ALWAYS** use `codebase-retrieval`", retrieval.description)
        self.assertIn("Absolute path", retrieval.description)
        self.assertIn("Fast regex search", search.description)
        self.assertIn("-F", search.description)

    # ---- 2. Effective-goal behavior -----------------------------------

    def test_effective_goal_forwarding_retrieval(self):
        class RecordingProcessor:
            def __init__(self):
                self.calls = []

            async def process_retrieval(self, query, result, goal=None):
                self.calls.append({"query": query, "result": result, "goal": goal})
                return {
                    "tool": "codebase_retrieval",
                    "data": {"result": result},
                    "processing": {
                        "status": "classified",
                        "retention": "below_threshold",
                        "estimated_tokens": 1,
                        "recovery_path": None,
                        "chunks": [],
                    },
                }

        rec = RecordingProcessor()
        srv = make_server(processor_factory=lambda _: rec)

        # Explicit goal provided
        self.run_async(
            srv.call_tool(
                "codebase_retrieval",
                {
                    "repository_path": str(self.repo),
                    "query": "query text",
                    "goal": "explicit goal",
                },
            )
        )
        self.assertEqual(rec.calls[-1]["goal"], "explicit goal")

        # Goal omitted (falls back to query)
        self.run_async(
            srv.call_tool(
                "codebase_retrieval",
                {"repository_path": str(self.repo), "query": "fallback query"},
            )
        )
        self.assertEqual(rec.calls[-1]["goal"], "fallback query")

    def test_effective_goal_forwarding_search(self):
        class RecordingProcessor:
            def __init__(self):
                self.calls = []

            async def process_search(
                self,
                pattern,
                stdout,
                stderr,
                exit_code,
                path=".",
                flags=None,
                goal=None,
                repository=None,
            ):
                self.calls.append({"pattern": pattern, "goal": goal})
                return {
                    "tool": "codebase_search",
                    "data": {
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                    },
                    "processing": {
                        "status": "classified",
                        "retention": "below_threshold",
                        "estimated_tokens": 1,
                        "recovery_path": None,
                        "chunks": [],
                    },
                }

        rec = RecordingProcessor()
        srv = make_server(self.repo, processor_factory=lambda _: rec)

        # Explicit goal provided
        self.run_async(
            srv.call_tool(
                "codebase_search",
                {"pattern": "pat", "goal": "explicit search goal"},
            )
        )
        self.assertEqual(rec.calls[-1]["goal"], "explicit search goal")

        # Goal omitted (falls back to pattern)
        self.run_async(srv.call_tool("codebase_search", {"pattern": "fallback_pat"}))
        self.assertEqual(rec.calls[-1]["goal"], "fallback_pat")

    # ---- 3. Success path and replacement envelopes --------------------

    def test_successful_retrieval_returns_envelope(self):
        srv = make_server()
        result = self.run_async(
            srv.call_tool(
                "codebase_retrieval",
                {"repository_path": str(self.repo), "query": "what is here?"},
            )
        )
        self.assertFalse(result.is_error)
        sc = result.structured_content
        self.assertEqual(sc["tool"], "codebase_retrieval")
        self.assertEqual(sc["data"]["result"], "Fake final answer.")
        self.assertIn("processing", sc)
        self.assertIn(sc["processing"]["status"], {"classified", "unclassified"})
        self.assertIn(
            sc["processing"]["retention"], {"below_threshold", "failed_open"}
        )
        # Content projection is serialized JSON
        parsed_content = json.loads(result.content[0].text)
        self.assertEqual(parsed_content["tool"], "codebase_retrieval")
        self.assertEqual(parsed_content["data"]["result"], "Fake final answer.")

    def test_search_call_returns_envelope(self):
        srv = make_server(self.repo)
        result = self.run_async(
            srv.call_tool(
                "codebase_search",
                {"pattern": "hello", "path": ".", "flags": ["-F"]},
            )
        )
        self.assertFalse(result.is_error)
        sc = result.structured_content
        self.assertEqual(sc["tool"], "codebase_search")
        self.assertEqual(sc["data"]["exit_code"], 0)
        self.assertIn("search warning", sc["data"]["stderr"])
        self.assertIn("hello", sc["data"]["stdout"])
        self.assertIn("processing", sc)
        parsed_content = json.loads(result.content[0].text)
        self.assertEqual(parsed_content["tool"], "codebase_search")
        self.assertEqual(parsed_content["data"]["exit_code"], 0)

    def test_search_no_match_is_not_a_tool_error(self):
        srv = make_server(self.repo)
        result = self.run_async(
            srv.call_tool("codebase_search", {"pattern": "nomatch"})
        )
        self.assertFalse(result.is_error)
        sc = result.structured_content
        self.assertEqual(sc["tool"], "codebase_search")
        self.assertEqual(sc["data"]["exit_code"], 1)
        self.assertIn("processing", sc)

    def test_search_exit_two_is_a_tool_error(self):
        srv = make_server(self.repo)
        with self.assertRaisesRegex(ToolError, "regex parse error"):
            self.run_async(srv.call_tool("codebase_search", {"pattern": "bad["}))

    # ---- 4. Missing API key fail-open ---------------------------------

    def test_missing_api_key_fail_open_retrieval_and_search(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}, clear=False):
            srv = make_server(self.repo)

            # Retrieval fails open gracefully
            res_r = self.run_async(
                srv.call_tool(
                    "codebase_retrieval",
                    {"repository_path": str(self.repo), "query": "q"},
                )
            )
            self.assertFalse(res_r.is_error)
            sc_r = res_r.structured_content
            self.assertEqual(sc_r["tool"], "codebase_retrieval")
            self.assertEqual(sc_r["data"]["result"], "Fake final answer.")
            self.assertEqual(sc_r["processing"]["status"], "unclassified")
            self.assertEqual(sc_r["processing"]["retention"], "failed_open")

            # Search fails open gracefully
            res_s = self.run_async(
                srv.call_tool("codebase_search", {"pattern": "hello"})
            )
            self.assertFalse(res_s.is_error)
            sc_s = res_s.structured_content
            self.assertEqual(sc_s["tool"], "codebase_search")
            self.assertEqual(sc_s["data"]["exit_code"], 0)
            self.assertEqual(sc_s["processing"]["status"], "unclassified")
            self.assertEqual(sc_s["processing"]["retention"], "failed_open")

    # ---- 5. Runner errors stay MCP ToolError ---------------------------

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

    def test_search_path_escape_returns_tool_error(self):
        srv = make_server(self.repo)
        with self.assertRaises(ToolError):
            self.run_async(
                srv.call_tool("codebase_search", {"pattern": "x", "path": ".."})
            )

    def test_unsupported_search_flag_returns_tool_error(self):
        srv = make_server(self.repo)
        with self.assertRaises(ToolError):
            self.run_async(
                srv.call_tool(
                    "codebase_search", {"pattern": "x", "flags": ["--exclude"]}
                )
            )

    def test_missing_tgrep_returns_tool_error(self):
        srv = server.create_server(
            executable=str(FAKE_PI),
            tgrep_executable="/definitely/missing/tgrep",
            repository_root=self.repo,
        )
        with self.assertRaises(ToolError):
            self.run_async(srv.call_tool("codebase_search", {"pattern": "x"}))

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

    # ---- 6. Cancellation propagation -----------------------------------

    def test_cancellation_propagates_during_runner(self):
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

    def test_cancellation_propagates_during_retrieval_processing(self):
        async def scenario():
            started = asyncio.Event()
            cancelled = asyncio.Event()

            class BlockingProcessor:
                async def process_retrieval(self, query, result, goal=None):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise

            srv = make_server(processor_factory=lambda _: BlockingProcessor())
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

    def test_cancellation_propagates_during_search_processing(self):
        async def scenario():
            started = asyncio.Event()
            cancelled = asyncio.Event()

            class BlockingProcessor:
                async def process_search(self, *args, **kwargs):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise

            srv = make_server(
                self.repo, processor_factory=lambda _: BlockingProcessor()
            )
            task = asyncio.create_task(
                srv.call_tool(
                    "codebase_search",
                    {"pattern": "hello"},
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
            self.run_async(
                srv.call_tool(
                    "nope", {"repository_path": str(self.repo), "query": "q"}
                )
            )

    # ---- 7. Lifespan and Client Reuse ---------------------------------

    def test_lifespan_creates_and_shares_single_client(self):
        async def scenario():
            fake_client = FakeClientWithCounters()
            factory_calls = 0

            def client_factory():
                nonlocal factory_calls
                factory_calls += 1
                fake_client.created += 1
                return fake_client

            srv = make_server(self.repo, client_factory=client_factory)
            self.assertEqual(fake_client.created, 0)

            async with srv.settings.lifespan(srv) as state:
                self.assertEqual(factory_calls, 1)
                self.assertEqual(fake_client.created, 1)
                self.assertEqual(fake_client.closed, 0)

                # First call: retrieval reuses client
                res1 = await srv.call_tool(
                    "codebase_retrieval",
                    {"repository_path": str(self.repo), "query": "q"},
                )
                self.assertFalse(res1.is_error)
                self.assertEqual(fake_client.closed, 0)

                # Second call: search reuses same client
                res2 = await srv.call_tool("codebase_search", {"pattern": "hello"})
                self.assertFalse(res2.is_error)
                self.assertEqual(fake_client.closed, 0)

                # Third call: another search reuses same client
                res3 = await srv.call_tool("codebase_search", {"pattern": "nomatch"})
                self.assertFalse(res3.is_error)
                self.assertEqual(fake_client.closed, 0)
                self.assertEqual(factory_calls, 1)

            # Shutdown closes client exactly once
            self.assertEqual(fake_client.closed, 1)

        self.run_async(scenario())

    def test_lifespan_closes_client_on_failure(self):
        async def scenario():
            fake_client = FakeClientWithCounters()
            srv = make_server(self.repo, client_factory=lambda: fake_client)
            async with srv.settings.lifespan(srv):
                with self.assertRaises(ToolError):
                    await srv.call_tool("codebase_search", {"pattern": "bad["})
                self.assertEqual(fake_client.closed, 0)
            self.assertEqual(fake_client.closed, 1)

        self.run_async(scenario())

    def test_no_production_global_client_exists(self):
        self.assertFalse(hasattr(server, "_client"))
        self.assertFalse(hasattr(server, "client"))
        self.assertFalse(hasattr(server, "CLIENT"))
        self.assertFalse(hasattr(server, "GLOBAL_CLIENT"))

    def test_client_creation_helper_disables_retries(self):
        from auggie_mcp.output_processing import JevConfig
        from typesafe_sdk import RetryPolicy

        cfg = JevConfig(api_key="sk-test-key-123")
        with patch("auggie_mcp.output_processing.AsyncTypeSafeClient") as mock_cls:
            create_typesafe_client(cfg)
            mock_cls.assert_called_once_with(
                api_key=cfg.api_key,
                model=cfg.model,
                retry=RetryPolicy(max_retries=0),
                timeout=cfg.timeout_seconds,
            )

    # ---- 8. Entry points and configuration -----------------------------

    def test_main_module_has_run_entry(self):
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
        import auggie_mcp.server as s

        self.assertTrue(hasattr(s, "run"))
        self.assertFalse(hasattr(s, "run_sse"))
        self.assertFalse(hasattr(s, "run_http"))

    # ---- 9. Contract test with fake Pi over stdio ---------------------

    def test_contract_stdio_discovery_and_call(self):
        """Start the real server over stdio and speak minimal JSON-RPC."""
        env = dict(os.environ)
        fake_bin_dir = tempfile.TemporaryDirectory()
        self.addCleanup(fake_bin_dir.cleanup)
        fake_pi = Path(fake_bin_dir.name) / "pi"
        fake_pi.symlink_to(FAKE_PI)
        fake_tgrep = Path(fake_bin_dir.name) / "tgrep"
        fake_tgrep.symlink_to(FAKE_TGREP)
        env["PATH"] = fake_bin_dir.name + os.pathsep + env.get("PATH", "")
        # Ensure fail-open for test determinism without real API key
        env.pop("TYPESAFE_API_KEY", None)

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
            # 1. initialize
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
            self.assertEqual(resp["result"]["serverInfo"]["version"], "0.2.0")

            # 2. notifications/initialized
            proc.stdin.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "method": "notifications/initialized"}
                    )
                    + "\n"
                ).encode()
            )
            proc.stdin.flush()

            # 3. tools/list
            resp = request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            tools = resp["result"]["tools"]
            names = [t["name"] for t in tools]
            self.assertEqual(names, ["codebase_retrieval", "codebase_search"])

            r_tool = next(t for t in tools if t["name"] == "codebase_retrieval")
            s_tool = next(t for t in tools if t["name"] == "codebase_search")

            self.assertIn("goal", r_tool["inputSchema"]["properties"])
            self.assertIn("goal", s_tool["inputSchema"]["properties"])
            self.assertEqual(
                set(r_tool["outputSchema"]["properties"]),
                {"tool", "data", "processing"},
            )
            self.assertEqual(
                set(s_tool["outputSchema"]["properties"]),
                {"tool", "data", "processing"},
            )

            # 4. tools/call: codebase_retrieval
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
            sc_r = resp["result"]["structuredContent"]
            self.assertEqual(sc_r["tool"], "codebase_retrieval")
            self.assertEqual(sc_r["data"]["result"], "Fake final answer.")
            self.assertIn("processing", sc_r)
            self.assertEqual(sc_r["processing"]["status"], "unclassified")
            self.assertEqual(sc_r["processing"]["retention"], "failed_open")

            # 5. tools/call: codebase_search
            resp = request(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "codebase_search",
                        "arguments": {
                            "pattern": "hello",
                        },
                    },
                }
            )
            self.assertFalse(resp["result"]["isError"])
            sc_s = resp["result"]["structuredContent"]
            self.assertEqual(sc_s["tool"], "codebase_search")
            self.assertEqual(sc_s["data"]["exit_code"], 0)
            self.assertIn("processing", sc_s)
            self.assertEqual(sc_s["processing"]["status"], "unclassified")
            self.assertEqual(sc_s["processing"]["retention"], "failed_open")

        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
