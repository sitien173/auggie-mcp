from __future__ import annotations

import array
import sqlite3
from argparse import Namespace
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from auggie.commands.reindex import cmd_reindex


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


def test_old_sqlite_only_layout_rebuilds_cleanly() -> None:
    root = Path(mkdtemp(prefix="auggie-phase7-migration-"))
    db_path = root / ".auggie" / "index.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    source = root / "sample.py"
    source.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    old = sqlite3.connect(str(db_path))
    try:
        old.executescript(
            """
            CREATE TABLE files (
              id INTEGER PRIMARY KEY,
              path TEXT NOT NULL,
              mtime_ns INTEGER NOT NULL,
              size INTEGER NOT NULL,
              sha256 TEXT NOT NULL
            );

            CREATE TABLE chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_id INTEGER NOT NULL,
              chunk_index INTEGER NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding BLOB NOT NULL,
              dim INTEGER NOT NULL,
              model TEXT NOT NULL
            );
            """
        )
        old.commit()
    finally:
        old.close()

    def fake_embed_texts(texts, model=None, dimensions=None):  # noqa: ARG001
        return [array.array("f", [1.0, 0.0]).tobytes() for _ in texts], 2

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
        patch("auggie.indexer.embed_texts", side_effect=fake_embed_texts),
        patch("auggie.indexer.LanceStore", FakeLanceStore),
    ):
        assert cmd_reindex(args) == 0

    con = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(chunks)").fetchall()}
        sqlite_chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        con.close()

    assert "embedding" not in columns
    assert sqlite_chunk_count == FakeLanceStore(root / ".auggie" / "lance").count()
