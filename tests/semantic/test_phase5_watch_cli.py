from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie import cli
from auggie.commands import __all__ as command_exports


class Phase5WatchCliTests(unittest.TestCase):
    def test_watch_command_is_registered(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args(["watch", ".", "--quiet"])

        self.assertEqual(args.cmd, "watch")
        self.assertEqual(args.path, ".")
        self.assertTrue(args.quiet)
        self.assertIn("cmd_watch", command_exports)


if __name__ == "__main__":
    unittest.main()
