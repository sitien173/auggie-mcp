from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", type("TqdmStub", (), {"tqdm": staticmethod(lambda iterable=None, **kwargs: iterable)})())

from auggie.watcher import AuggieWatchService, WatchEventHandler


class _FakeEvent:
    def __init__(self, src_path: str, dest_path: str | None = None, is_directory: bool = False):
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = is_directory


class Phase5WatcherTests(unittest.TestCase):
    def test_debounce_collapses_burst_into_single_pass(self) -> None:
        root = Path(mkdtemp(prefix="auggie-watch-debounce-"))
        first = root / "a.py"
        second = root / "b.py"
        first.write_text("a\n", encoding="utf-8")
        second.write_text("b\n", encoding="utf-8")

        calls: list[set[str]] = []
        done = threading.Event()

        def index_callback(paths: set[str]) -> None:
            calls.append(set(paths))
            done.set()

        service = AuggieWatchService(root=root, debounce_ms=60, index_callback=index_callback)
        service.start()
        try:
            self.assertTrue(service.mark_dirty(first))
            time.sleep(0.02)
            self.assertTrue(service.mark_dirty(second))
            self.assertTrue(done.wait(1.0))
        finally:
            service.stop()
            service.join()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {str(first.resolve()), str(second.resolve())})

    def test_filters_auggie_and_skip_dirs(self) -> None:
        root = Path(mkdtemp(prefix="auggie-watch-filter-"))
        source = root / "keep.py"
        source.write_text("x\n", encoding="utf-8")

        calls: list[set[str]] = []
        done = threading.Event()

        def index_callback(paths: set[str]) -> None:
            calls.append(set(paths))
            done.set()

        service = AuggieWatchService(root=root, debounce_ms=40, index_callback=index_callback)
        handler = WatchEventHandler(service)
        service.start()
        try:
            handler.on_modified(_FakeEvent(str(root / ".auggie" / "index.sqlite")))
            handler.on_modified(_FakeEvent(str(root / ".git" / "HEAD")))
            handler.on_modified(_FakeEvent(str(root / "__pycache__" / "x.pyc")))
            handler.on_modified(_FakeEvent(str(source)))
            self.assertTrue(done.wait(1.0))
        finally:
            service.stop()
            service.join()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {str(source.resolve())})

    def test_single_writer_requeues_mid_pass_events(self) -> None:
        root = Path(mkdtemp(prefix="auggie-watch-queue-"))
        first = root / "first.py"
        second = root / "second.py"
        first.write_text("a\n", encoding="utf-8")
        second.write_text("b\n", encoding="utf-8")

        calls: list[set[str]] = []
        active = 0
        max_active = 0
        gate = threading.Event()
        call_lock = threading.Lock()
        first_started = threading.Event()
        second_done = threading.Event()

        def index_callback(paths: set[str]) -> None:
            nonlocal active, max_active
            with call_lock:
                active += 1
                max_active = max(max_active, active)
            calls.append(set(paths))
            if len(calls) == 1:
                first_started.set()
                gate.wait(1.0)
            else:
                second_done.set()
            with call_lock:
                active -= 1

        service = AuggieWatchService(root=root, debounce_ms=30, index_callback=index_callback)
        service.start()
        try:
            self.assertTrue(service.mark_dirty(first))
            self.assertTrue(first_started.wait(1.0))
            self.assertTrue(service.mark_dirty(second))
            gate.set()
            self.assertTrue(second_done.wait(1.0))
        finally:
            service.stop()
            service.join()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], {str(first.resolve())})
        self.assertEqual(calls[1], {str(second.resolve())})
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
