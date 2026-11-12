from __future__ import annotations

import array
import json
import math
import sqlite3
import threading
import time
from argparse import Namespace
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

import pytest

from auggie.commands.context import cmd_context
from auggie.commands.reindex import cmd_reindex
from auggie.indexer import index_path
from auggie.search import query as search_query
from auggie.watcher import AuggieWatchService


class FakeLanceStore:
    registry: dict[str, dict[int, tuple[bytes, int]]]

    def __init__(self, root: Path):
        self.root = str(Path(root))
        if not hasattr(type(self), "registry"):
            type(self).registry = {}
        type(self).registry.setdefault(self.root, {})

    @property
    def _rows(self) -> dict[int, tuple[bytes, int]]:
        return type(self).registry[self.root]

    def upsert_chunk_vectors(self, rows, *, model):  # noqa: ARG002
        Path(self.root).mkdir(parents=True, exist_ok=True)
        for chunk_id, emb, dim in rows:
            self._rows[int(chunk_id)] = (emb, int(dim))

    def fetch_vectors_by_chunk_ids(self, chunk_ids):
        return {int(chunk_id): self._rows[int(chunk_id)] for chunk_id in chunk_ids if int(chunk_id) in self._rows}

    def delete_chunk_ids(self, chunk_ids):
        for chunk_id in chunk_ids:
            self._rows.pop(int(chunk_id), None)

    def count(self):
        return len(self._rows)

    def list_chunk_ids(self):
        return sorted(self._rows)

    def ann_query(self, vector, limit):
        query_vec = array.array("f")
        query_vec.frombytes(vector)
        q = list(query_vec)
        hits = []
        for chunk_id, (emb, _dim) in self._rows.items():
            doc_vec = array.array("f")
            doc_vec.frombytes(emb)
            dot = sum(float(a) * float(b) for a, b in zip(q, doc_vec, strict=True))
            distance = math.sqrt(max(0.0, 2.0 - 2.0 * dot))
            hits.append(type("Hit", (), {"chunk_id": chunk_id, "distance": distance})())
        hits.sort(key=lambda item: item.distance)
        return hits[:limit]


def fake_embed_texts_factory(call_log: list[int]):
    def fake_embed_texts(texts, model=None, dimensions=None):  # noqa: ARG001
        vectors = []
        for text in texts:
            normalized = text.replace("\r\n", "\n")
            if "RateLimiter acquire" in normalized or "query-rate-limiter" in normalized:
                vec = [1.0, 0.0]
            elif "cache" in normalized or "query-cache" in normalized:
                vec = [0.0, 1.0]
            elif "alpha" in normalized:
                vec = [1.0, 0.0]
            elif "delta" in normalized:
                vec = [0.0, 1.0]
            else:
                vec = [0.5, 0.5]
            vectors.append(array.array("f", vec).tobytes())
        call_log.append(len(texts))
        return vectors, 2

    return fake_embed_texts


def test_incremental_index_only_embeds_changed_chunks_and_zero_on_unchanged() -> None:
    root = Path(mkdtemp(prefix="auggie-phase7-incremental-"))
    db_path = root / ".auggie" / "index.sqlite"
    source = root / "sample.py"
    source.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    embed_calls: list[int] = []
    fake_embed = fake_embed_texts_factory(embed_calls)
    FakeLanceStore.registry = {}

    with (
        patch("auggie.indexer.embed_texts", side_effect=fake_embed),
        patch("auggie.indexer.LanceStore", FakeLanceStore),
    ):
        index_path(root, db_path, model="voyage-code-3", chunk_lines=2, overlap=0, detect=False, branch="main")

        embed_calls.clear()
        unchanged = index_path(
            root,
            db_path,
            model="voyage-code-3",
            chunk_lines=2,
            overlap=0,
            detect=False,
            branch="main",
        )

        source.write_text("alpha\nbeta\ngamma\nomega\n", encoding="utf-8")
        embed_calls.clear()
        changed = index_path(
            root,
            db_path,
            model="voyage-code-3",
            chunk_lines=2,
            overlap=0,
            detect=False,
            branch="main",
        )

        assert unchanged.files_indexed == 0
        assert embed_calls == [1]
        assert changed.chunks_embedded == 1


def test_reindex_wipes_sqlite_and_lance_then_rebuilds_with_parity() -> None:
    root = Path(mkdtemp(prefix="auggie-phase7-reindex-"))
    db_path = root / ".auggie" / "index.sqlite"
    lance_dir = root / ".auggie" / "lance"
    source = root / "sample.py"
    source.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    lance_dir.mkdir(parents=True, exist_ok=True)
    (lance_dir / "stale.txt").write_text("stale", encoding="utf-8")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("obsolete", encoding="utf-8")

    embed_calls: list[int] = []
    fake_embed = fake_embed_texts_factory(embed_calls)
    FakeLanceStore.registry = {}

    args = Namespace(
        path=str(root),
        db=str(db_path),
        profile=None,
        global_cache=False,
        repo_root=root,
        model="voyage-code-3",
        dimensions=None,
        chunk_lines=2,
        overlap=0,
        max_bytes=2_000_000,
        exclude=[],
        include=[],
        ast=None,
        json=True,
    )

    with (
        patch("auggie.commands.reindex.require_embedding_config", return_value=True),
        patch("auggie.indexer.embed_texts", side_effect=fake_embed),
        patch("auggie.indexer.LanceStore", FakeLanceStore),
    ):
        assert cmd_reindex(args) == 0

    con = sqlite3.connect(str(db_path))
    try:
        chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        con.close()

    store_count = FakeLanceStore(lance_dir).count()
    assert chunk_count == store_count
    assert not (lance_dir / "stale.txt").exists()


def test_watch_debounce_collapses_burst_into_single_pass() -> None:
    root = Path(mkdtemp(prefix="auggie-phase7-watch-"))
    target = root / "sample.py"
    target.write_text("x\n", encoding="utf-8")

    calls: list[set[str]] = []
    done = threading.Event()

    def index_callback(paths: set[str]) -> None:
        calls.append(set(paths))
        done.set()

    service = AuggieWatchService(root=root, debounce_ms=40, index_callback=index_callback)
    service.start()
    try:
        assert service.mark_dirty(target)
        assert service.mark_dirty(target)
        time.sleep(0.01)
        assert service.mark_dirty(target)
        assert done.wait(1.0)
    finally:
        service.stop()
        service.join()

    assert len(calls) == 1


def test_query_and_context_parity_on_small_fixture_repo() -> None:
    root = Path(mkdtemp(prefix="auggie-phase7-context-"))
    db_path = root / ".auggie" / "index.sqlite"
    source = root / "sample.py"
    source.write_text(
        "RateLimiter acquire token\nhelper line\ncache lookup result\nother line\n",
        encoding="utf-8",
    )

    embed_calls: list[int] = []
    fake_embed = fake_embed_texts_factory(embed_calls)
    FakeLanceStore.registry = {}

    with (
        patch("auggie.indexer.embed_texts", side_effect=fake_embed),
        patch("auggie.indexer.LanceStore", FakeLanceStore),
    ):
        index_path(root, db_path, model="voyage-code-3", chunk_lines=2, overlap=0, detect=False, branch="default")

    with (
        patch("auggie.search.embed_texts", side_effect=lambda texts, model=None, dimensions=None: ([array.array("f", [1.0, 0.0]).tobytes()], 2)),
        patch("auggie.search.LanceStore", FakeLanceStore),
    ):
        hits, _ = search_query(
            db_path=db_path,
            q="query-rate-limiter",
            top_k=3,
            model="voyage-code-3",
            dimensions=None,
            mode="semantic",
            no_cache=True,
            branch="default",
        )

    assert hits
    assert hits[0].path == str(source.resolve())

    args = Namespace(
        query="query-rate-limiter",
        top=3,
        context=1,
        db=str(db_path),
        profile=None,
        global_cache=False,
        repo_root=root,
        model="voyage-code-3",
        dimensions=None,
        json=True,
    )

    stdout_lines: list[str] = []

    def capture_print(text):
        stdout_lines.append(text)

    with (
        patch("auggie.commands.context.require_embedding_config", return_value=True),
        patch("auggie.commands.context.get_current_branch", return_value="default"),
        patch(
            "auggie.commands.context.query_db",
            return_value=(hits, True),
        ),
        patch("builtins.print", side_effect=capture_print),
    ):
        assert cmd_context(args) == 0

    payload = json.loads(stdout_lines[0])
    assert payload["results"][0]["path"] == str(source.resolve())
    assert payload["results"][0]["match_source"] == "semantic"
