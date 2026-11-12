from __future__ import annotations

import io
import sys
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie.commands.watch import cmd_watch


class _FakeObserver:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:  # noqa: ARG002
        self.joined = True

    def is_alive(self) -> bool:
        return self.started and not self.stopped


class _FakeService:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:  # noqa: ARG002
        self.joined = True


class Phase5WatchCommandTests(unittest.TestCase):
    def test_cmd_watch_runs_initial_pass_and_stops_cleanly_on_ctrl_c(self) -> None:
        root = Path(mkdtemp(prefix="auggie-watch-command-"))
        db_path = root / ".auggie" / "index.sqlite"
        observer = _FakeObserver()
        services: list[_FakeService] = []
        index_calls: list[dict] = []

        def fake_index_path(**kwargs):
            index_calls.append(kwargs)
            return types.SimpleNamespace(files_indexed=1, files_skipped=2)

        def fake_service_factory(**kwargs):
            service = _FakeService(**kwargs)
            services.append(service)
            return service

        def fake_create_observer(root_arg, handler):  # noqa: ARG001
            raise KeyboardInterrupt

        args = Namespace(
            path=str(root),
            db=None,
            profile=None,
            global_cache=False,
            repo_root=root,
            model="voyage-code-3",
            dimensions=None,
            chunk_lines=None,
            overlap=None,
            max_bytes=2_000_000,
            exclude=[],
            include=[],
            ast=None,
            no_detect=False,
            quiet=True,
        )

        with (
            patch("auggie.commands.watch.require_embedding_config", return_value=True),
            patch("auggie.commands.watch._resolve_paths", return_value=(root, db_path)),
            patch("auggie.commands.watch._resolve_chunk_lines", return_value=30),
            patch("auggie.commands.watch._resolve_overlap", return_value=15),
            patch("auggie.commands.watch.index_path", side_effect=fake_index_path),
            patch("auggie.commands.watch.AuggieWatchService", side_effect=fake_service_factory),
            patch("auggie.commands.watch.create_observer", return_value=observer),
            patch("auggie.commands.watch._wait_for_interrupt", side_effect=KeyboardInterrupt),
        ):
            with redirect_stderr(io.StringIO()):
                exit_code = cmd_watch(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(index_calls), 1)
        self.assertEqual(index_calls[0]["root"], root)
        self.assertEqual(index_calls[0]["db_path"], db_path)
        self.assertEqual(len(services), 1)
        self.assertTrue(services[0].started)
        self.assertTrue(services[0].stopped)
        self.assertTrue(services[0].joined)
        self.assertTrue(observer.started)
        self.assertTrue(observer.stopped)
        self.assertTrue(observer.joined)


if __name__ == "__main__":
    unittest.main()
