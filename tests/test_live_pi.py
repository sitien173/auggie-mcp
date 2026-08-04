"""Optional live-Pi contract test.

Skipped unless AUGGIE_LIVE_PI=1 is set, because it requires real Pi with model
credentials, network access, and model quota.
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from auggie_mcp import server

TESTS_DIR = Path(__file__).resolve().parent


@unittest.skipUnless(
    os.environ.get("AUGGIE_LIVE_PI") == "1",
    "set AUGGIE_LIVE_PI=1 to run the live-Pi test",
)
class LivePiServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_live_pi_codebase_retrieval(self):
        srv = server.create_server()
        result = self.run_async(
            srv.call_tool(
                "codebase_retrieval",
                {
                    "repository_path": str(self.repo),
                    "query": "What does hello.txt contain? Answer with exactly: FOUND",
                },
            )
        )
        self.assertFalse(result.is_error)
        self.assertIn("FOUND", result.content[0].text)


if __name__ == "__main__":
    unittest.main()
