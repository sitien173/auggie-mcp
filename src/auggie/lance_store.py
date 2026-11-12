"""
LanceDB storage for chunk vectors.
"""

from __future__ import annotations

import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TABLE_NAME = "chunks"


def get_lance_dir(db_path: Path) -> Path:
    """Get the LanceDB directory for an index database."""
    return db_path.parent / "lance"


@dataclass(frozen=True)
class LanceHit:
    """Single ANN search hit from LanceDB."""

    chunk_id: int
    distance: float


class LanceStore:
    """Minimal LanceDB wrapper used by the indexer."""

    def __init__(self, root: Path):
        self.root = root
        self._db: Any | None = None

    def _connect(self):
        try:
            import lancedb
        except ImportError as e:
            raise ImportError(
                "LanceDB support requires the lancedb package. "
                "Install the dependencies from requirements.txt."
            ) from e

        if self._db is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.root))
        return self._db

    def _open_table(self):
        db = self._connect()
        try:
            return db.open_table(TABLE_NAME)
        except Exception:
            return None

    def _create_table(self, rows: list[dict[str, Any]], dim: int):
        import pyarrow as pa

        db = self._connect()
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("model", pa.utf8()),
                pa.field("dim", pa.int32()),
            ]
        )
        return db.create_table(TABLE_NAME, data=rows, schema=schema)

    def upsert_chunk_vectors(
        self,
        rows: list[tuple[int, bytes, int]],
        *,
        model: str,
    ) -> None:
        """Upsert vectors keyed by chunk_id."""
        if not rows:
            return

        payload = []
        for chunk_id, emb, dim in rows:
            vector = array.array("f")
            vector.frombytes(emb)
            payload.append(
                {
                    "chunk_id": chunk_id,
                    "vector": list(vector),
                    "model": model,
                    "dim": dim,
                }
            )
        dim = rows[0][2]

        table = self._open_table()
        if table is None:
            self._create_table(payload, dim)
            return

        (
            table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(payload)
        )

    def fetch_vectors_by_chunk_ids(self, chunk_ids: list[int]) -> dict[int, tuple[bytes, int]]:
        """Fetch vectors for chunk ids."""
        if not chunk_ids:
            return {}

        table = self._open_table()
        if table is None:
            return {}

        results: dict[int, tuple[bytes, int]] = {}
        batch_size = 256
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            predicate = f"chunk_id IN ({','.join(str(i) for i in batch)})"
            rows = (
                table.search()
                .where(predicate)
                .select(["chunk_id", "vector", "dim"])
                .limit(len(batch))
                .to_list()
            )
            for row in rows:
                results[int(row["chunk_id"])] = (
                    array.array("f", row["vector"]).tobytes(),
                    int(row["dim"]),
                )
        return results

    def ann_query(self, vector: bytes | list[float], limit: int) -> list[LanceHit]:
        """Run ANN search and return chunk ids with distances."""
        table = self._open_table()
        if table is None:
            return []

        if isinstance(vector, bytes):
            query_arr = array.array("f")
            query_arr.frombytes(vector)
            query_vector = list(query_arr)
        else:
            query_vector = vector

        rows = table.search(query_vector).select(["chunk_id"]).limit(limit).to_list()
        return [
            LanceHit(
                chunk_id=int(row["chunk_id"]),
                distance=float(row.get("_distance", 0.0)),
            )
            for row in rows
        ]

    def delete_chunk_ids(self, chunk_ids: list[int]) -> None:
        """Delete rows by chunk_id."""
        if not chunk_ids:
            return

        table = self._open_table()
        if table is None:
            return

        batch_size = 256
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            predicate = f"chunk_id IN ({','.join(str(i) for i in batch)})"
            table.delete(predicate)

    def count(self) -> int:
        """Count rows in the Lance table."""
        table = self._open_table()
        if table is None:
            return 0
        return int(table.count_rows())

    def list_chunk_ids(self) -> list[int]:
        """List all stored chunk ids."""
        table = self._open_table()
        if table is None:
            return []

        rows = table.search().select(["chunk_id"]).limit(self.count()).to_list()
        return [int(row["chunk_id"]) for row in rows]
