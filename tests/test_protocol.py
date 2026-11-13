"""Wire-level tests: the server driven by an in-process 2026-07-28 client."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client, MCPError
from mcp.types import UNSUPPORTED_PROTOCOL_VERSION, Implementation
from mcp.types.version import LATEST_HANDSHAKE_VERSION, MODERN_PROTOCOL_VERSIONS

from mcp.server.context import ServerRequestContext

from auggie_mcp import index_tools
from auggie_mcp.server import _refuse_initialize
from tests.test_server import (  # noqa: F401
    FakeClientWithCounters,
    make_server,
    setUpModule,
    tearDownModule,
)

PROTOCOL = "2026-07-28"


def connect(srv):
    return Client(
        srv,
        mode=PROTOCOL,
        client_info=Implementation(name="test", version="1"),
    )


def fake_server(repository_root):
    return make_server(repository_root, client_factory=FakeClientWithCounters)


def leaf_exceptions(exc):
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in leaf_exceptions(sub)]
    return [exc]


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def run_client(self, make_srv, fn):
        async def main():
            async with connect(make_srv()) as client:
                return await fn(client)

        return asyncio.run(main())

    def test_negotiates_2026_07_28_and_discover_reports_identity(self):
        async def fn(client):
            raw = await client.session.send_discover(PROTOCOL)
            return client.protocol_version, raw

        version, raw = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertEqual(version, PROTOCOL)
        self.assertIn(PROTOCOL, raw["supportedVersions"])
        info = raw["_meta"]["io.modelcontextprotocol/serverInfo"]
        self.assertEqual(info["name"], "auggie-mcp")
        self.assertEqual(info["version"], "0.5.0")

    def test_tools_list_order_schemas_and_annotations(self):
        srv = fake_server(self.repo)
        expected = asyncio.run(srv.list_tools())

        async def fn(client):
            return await client.list_tools()

        result = self.run_client(lambda: srv, fn)
        self.assertEqual(
            [t.name for t in result.tools],
            ["codebase_retrieval", "codebase_search", "index_status", "index_update"],
        )
        for wire, local in zip(result.tools, expected):
            self.assertEqual(wire.input_schema, local.input_schema)
            self.assertEqual(wire.annotations, local.annotations)

    def test_search_exact_returns_complete_envelope(self):
        async def fn(client):
            return await client.call_tool(
                "codebase_search", {"query": "hello", "path": ".", "flags": ["-F"]}
            )

        result = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertFalse(result.is_error)
        self.assertEqual(result.result_type, "complete")
        sc = result.structured_content
        self.assertEqual(sc["tool"], "codebase_search")
        self.assertEqual(sc["data"]["exit_code"], 0)
        self.assertIn("hello", sc["data"]["stdout"])
        self.assertIn("processing", sc)

    def test_failing_search_returns_tool_error_text(self):
        async def fn(client):
            return await client.call_tool(
                "codebase_search", {"query": "x", "flags": ["--exclude"]}
            )

        result = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertTrue(result.is_error)
        self.assertIn("--exclude", result.content[0].text)

    def test_index_update_delivers_progress(self):
        events = []

        async def on_progress(progress, total, message):
            events.append((progress, total, message))

        async def fn(client):
            return await client.call_tool(
                "index_update",
                {"repository_path": str(self.repo)},
                progress_callback=on_progress,
            )

        result = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertFalse(result.is_error)
        self.assertGreaterEqual(len(events), 1)

    def test_index_status_unexpected_exception_message_reaches_client(self):
        async def fn(client):
            return await client.call_tool(
                "index_status", {"repository_path": str(self.repo)}
            )

        with patch.object(index_tools, "index_status", side_effect=RuntimeError("boom")):
            result = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertTrue(result.is_error)
        self.assertIn("boom", result.content[0].text)

    def test_index_update_unexpected_exception_message_reaches_client(self):
        async def fn(client):
            return await client.call_tool(
                "index_update", {"repository_path": str(self.repo)}
            )

        with patch.object(
            index_tools, "run_index_update", side_effect=RuntimeError("boom")
        ):
            result = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertTrue(result.is_error)
        self.assertIn("boom", result.content[0].text)

    def telemetry_events(self, handler):
        tdir = Path(self.tmp.name) / "telemetry"
        env = patch.dict(os.environ, {"AUGGIE_TELEMETRY_DIR": str(tdir)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AUGGIE_TELEMETRY", None)
        handler()
        return [
            json.loads(line)
            for path in sorted(tdir.glob("events-*.jsonl"))
            for line in path.read_text().splitlines()
        ]

    def test_index_status_telemetry_records_original_exception_type(self):
        async def fn(client):
            return await client.call_tool(
                "index_status", {"repository_path": str(self.repo)}
            )

        def run():
            with patch.object(
                index_tools, "index_status", side_effect=RuntimeError("boom")
            ):
                self.run_client(lambda: fake_server(self.repo), fn)

        (event,) = self.telemetry_events(run)
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["error"]["type"], "RuntimeError")

    def test_index_update_telemetry_records_original_exception_type(self):
        async def fn(client):
            return await client.call_tool(
                "index_update", {"repository_path": str(self.repo)}
            )

        def run():
            with patch.object(
                index_tools, "run_index_update", side_effect=RuntimeError("boom")
            ):
                self.run_client(lambda: fake_server(self.repo), fn)

        (event,) = self.telemetry_events(run)
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["error"]["type"], "RuntimeError")

    def test_retrieval_processing_exception_message_reaches_client(self):
        class FailingProcessor:
            async def process_retrieval(self, **kwargs):
                raise RuntimeError("boom")

        async def fn(client):
            return await client.call_tool(
                "codebase_retrieval",
                {"repository_path": str(self.repo), "query": "what is here?"},
            )

        result = self.run_client(
            lambda: make_server(
                self.repo,
                client_factory=FakeClientWithCounters,
                processor_factory=lambda _: FailingProcessor(),
            ),
            fn,
        )
        self.assertTrue(result.is_error)
        self.assertIn("boom", result.content[0].text)

    def test_initialize_is_refused_with_supported_versions(self):
        async def main():
            async with Client(
                fake_server(self.repo),
                mode="legacy",
                client_info=Implementation(name="test", version="1"),
            ):
                pass

        with self.assertRaises(BaseExceptionGroup) as caught:
            asyncio.run(main())
        (error,) = leaf_exceptions(caught.exception)
        self.assertIsInstance(error, MCPError)
        self.assertEqual(error.code, UNSUPPORTED_PROTOCOL_VERSION)
        self.assertIn(PROTOCOL, error.message)
        self.assertEqual(error.data["supported"], list(MODERN_PROTOCOL_VERSIONS))
        self.assertEqual(error.data["supported"], ["2026-07-28"])
        self.assertEqual(error.data["requested"], LATEST_HANDSHAKE_VERSION)

    def test_tools_list_carries_one_hour_private_cache_hint(self):
        async def fn(client):
            return await client.list_tools()

        result = self.run_client(lambda: fake_server(self.repo), fn)
        self.assertEqual(result.ttl_ms, 3_600_000)
        self.assertEqual(result.cache_scope, "private")

    def test_discover_advertises_no_list_changed_or_subscribe(self):
        async def fn(client):
            return await client.session.send_discover(PROTOCOL)

        raw = self.run_client(lambda: fake_server(self.repo), fn)
        caps = raw["capabilities"]
        self.assertFalse(caps.get("tools", {}).get("listChanged", False))
        self.assertFalse(caps.get("resources", {}).get("subscribe", False))
        self.assertNotIn("subscriptions", caps)

    def test_initialize_notification_passes_through(self):
        ctx = ServerRequestContext(
            session=None,
            lifespan_context={},
            protocol_version=PROTOCOL,
            method="initialize",
            params={"protocolVersion": "2025-11-25"},
            request_id=None,
        )
        seen = []

        async def call_next(c):
            seen.append(c)
            return {"ok": True}

        result = asyncio.run(_refuse_initialize(ctx, call_next))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(seen, [ctx])

    def test_initialize_empty_protocol_version_reports_none(self):
        ctx = ServerRequestContext(
            session=None,
            lifespan_context={},
            protocol_version=PROTOCOL,
            method="initialize",
            params={"protocolVersion": ""},
            request_id=1,
        )

        async def call_next(c):
            raise AssertionError("call_next must not run")

        with self.assertRaises(MCPError) as caught:
            asyncio.run(_refuse_initialize(ctx, call_next))
        self.assertEqual(caught.exception.code, UNSUPPORTED_PROTOCOL_VERSION)
        self.assertIsNone(caught.exception.data["requested"])


if __name__ == "__main__":
    unittest.main()
