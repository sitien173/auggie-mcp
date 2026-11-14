"""Tests for auggie_mcp.store."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from auggie_mcp import store


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.db = store.db_path(self.repo)


class SchemaTests(StoreTestCase):
    def test_open_write_creates_schema_with_user_version_1(self):
        conn = store.open_write(self.db)
        self.addCleanup(conn.close)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, 1)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        self.assertEqual(tables, {"meta", "files", "chunks", "embeddings"})
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        self.assertIn("chunks_hash", indexes)

    def test_delete_file_cascades_to_chunks(self):
        conn = store.open_write(self.db)
        self.addCleanup(conn.close)
        conn.execute(
            "INSERT INTO files(path, mtime_ns, size, sha256) VALUES (?, ?, ?, ?)",
            ("a.py", 1, 10, b"\x00" * 32),
        )
        file_id = conn.execute("SELECT id FROM files WHERE path = 'a.py'").fetchone()[0]
        conn.execute(
            "INSERT INTO chunks(file_id, ordinal, start_line, end_line, text, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (file_id, 0, 1, 5, "a.py\ncode", b"\x01" * 32),
        )
        conn.commit()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)

        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)

    def test_constraints_reject_bad_rows(self):
        conn = store.open_write(self.db)
        self.addCleanup(conn.close)
        conn.execute("INSERT INTO files(path) VALUES ('a.py')")
        file_id = conn.execute("SELECT id FROM files").fetchone()[0]

        conn.execute(
            "INSERT INTO chunks(file_id, ordinal, start_line, end_line, text, content_hash) "
            "VALUES (?, 0, 1, 5, 'x', ?)",
            (file_id, b"\x01" * 32),
        )
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chunks(file_id, ordinal, start_line, end_line, text, content_hash) "
                "VALUES (?, 0, 6, 10, 'y', ?)",
                (file_id, b"\x02" * 32),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chunks(file_id, ordinal, start_line, end_line, text, content_hash) "
                "VALUES (?, 1, 10, 5, 'bad', ?)",
                (file_id, b"\x03" * 32),
            )

        conn.execute(
            "INSERT INTO embeddings(content_hash, model_key, vector, last_seen) "
            "VALUES (?, 'm:1:f32', ?, 1)",
            (b"\x01" * 32, b"\x00\x00\x80\x3f"),
        )
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO embeddings(content_hash, model_key, vector, last_seen) "
                "VALUES (?, 'm:1:f32', ?, 2)",
                (b"\x01" * 32, b"\x00\x00\x00\x40"),
            )

    def test_mismatched_user_version_raises_naming_version(self):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(self.db)
        raw.execute("PRAGMA user_version = 7")
        raw.close()

        with self.assertRaises(store.StoreError) as write_ctx:
            store.open_write(self.db)
        self.assertIn("7", str(write_ctx.exception))

        with self.assertRaises(store.StoreError) as read_ctx:
            store.open_read(self.db)
        self.assertIn("7", str(read_ctx.exception))

    def test_read_connection_rejects_writes(self):
        wconn = store.open_write(self.db)
        wconn.close()

        rconn = store.open_read(self.db)
        self.addCleanup(rconn.close)
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            rconn.execute("INSERT INTO meta(key, value) VALUES ('k', 'v')")
        self.assertIn("readonly", str(ctx.exception).lower())

    def test_vector_round_trip(self):
        vec = [0.25, -1.5, 3.75, 0.0]
        blob = store.encode_vector(vec)
        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), 4 * len(vec))
        decoded = store.decode_vector(blob)
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_allclose(decoded, np.array(vec, dtype=np.float32))
