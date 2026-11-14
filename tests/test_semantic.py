from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
from pathlib import Path

import pytest

from auggie_mcp.config import Config, EmbeddingConfig
from auggie_mcp.indexer import index_update
from auggie_mcp.semantic import IndexMissingError, ModelMismatchError, semantic_search
from auggie_mcp.store import db_path
from tests.fake_embedder import FakeEmbedder


class CountingFakeEmbedder(FakeEmbedder):
    def __init__(self, model="voyage-code-3", dimensions=16):
        super().__init__(model, dimensions)
        self.lock = threading.Lock()

    def embed(self, texts):
        with self.lock:
            return super().embed(texts)


def make_repo(tmp_path: Path, files: dict[str, str]):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    config = Config(embedding=EmbeddingConfig(dimensions=16))
    embedder = CountingFakeEmbedder()
    index_update(repo, config=config, embedder=embedder)
    return repo, config, embedder


def test_search_orders_by_cosine_and_returns_raw_scores(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha", "b.py": "beta"})
    hits = semantic_search(repo, "alpha", config=config, embedder=embedder)["hits"]
    query = embedder.embed(["alpha"])[0]
    docs = {name: embedder.embed([f"{name}\n{text}"])[0] for name, text in (("a.py", "alpha"), ("b.py", "beta"))}
    expected = sorted(docs, key=lambda name: float(docs[name] @ query), reverse=True)
    assert [hit["path"] for hit in hits] == expected
    assert isinstance(hits[0]["score"], float)
    assert not hits[0]["text"].startswith("a.py\n")
    assert not ({"confidence", "label"} & hits[0].keys())


def test_path_filter_is_applied_before_top_k(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha", "pkg/b.py": "alpha"})
    hits = semantic_search(repo, "alpha", path="pkg", top_k=1, config=config, embedder=embedder)["hits"]
    assert [hit["path"] for hit in hits] == ["pkg/b.py"]


def test_path_outside_repo_returns_empty(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha"})
    assert semantic_search(repo, "alpha", path=tmp_path.parent, config=config, embedder=embedder)["hits"] == []


def test_generation_reload_is_single_under_concurrent_queries(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha"})
    semantic_search(repo, "alpha", config=config, embedder=embedder)
    (repo / "a.py").write_text("alpha updated")
    index_update(repo, config=config, embedder=embedder)
    from unittest.mock import patch

    with (
        patch("auggie_mcp.semantic._read_index", wraps=__import__("auggie_mcp.semantic", fromlist=["_read_index"])._read_index) as reload_index,
        concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,
    ):
        results = list(pool.map(lambda _: semantic_search(repo, "alpha", config=config, embedder=embedder), range(8)))
    assert all(result["hits"] for result in results)
    assert reload_index.call_count == 1


def test_cache_holds_at_most_two_repositories(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / str(i)).mkdir()
        repo, config, embedder = make_repo(tmp_path / str(i), {"a.py": f"value{i}"})
        semantic_search(repo, f"value{i}", config=config, embedder=embedder)
    from auggie_mcp.semantic import _CACHE
    assert len(_CACHE) <= 2


def test_changed_file_is_stale_without_context_but_context_added_when_fresh(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "one\ntwo\nthree\nfour\nfive\n"})
    fresh = semantic_search(repo, "two", context_lines=1, config=config, embedder=embedder)["hits"][0]
    assert fresh["text"]
    file_path = repo / "a.py"
    file_path.write_text("changed\nnow\n")
    stale = semantic_search(repo, "two", context_lines=1, config=config, embedder=embedder)["hits"][0]
    assert stale["stale"] is True
    assert stale["text"] == ""


def test_missing_index_and_model_mismatch_are_distinct(tmp_path: Path) -> None:
    with pytest.raises(IndexMissingError):
        semantic_search(tmp_path / "missing", "q", embedder=FakeEmbedder())
    repo, config, _embedder = make_repo(tmp_path, {"a.py": "alpha"})
    with pytest.raises(ModelMismatchError):
        semantic_search(repo, "alpha", config=config, embedder=FakeEmbedder(model="other"))


def test_search_database_access_is_read_only(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha"})
    db = db_path(repo)
    before = db.read_bytes()
    semantic_search(repo, "alpha", config=config, embedder=embedder)
    assert db.read_bytes() == before
    from auggie_mcp.store import open_read
    with open_read(db) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("UPDATE meta SET value = value WHERE key = 'generation'")


def test_result_contains_index_metadata(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha"})
    result = semantic_search(repo, "alpha", config=config, embedder=embedder)
    assert set(result) == {"hits", "index", "truncated", "archive_path", "hits_omitted"}
    assert result["truncated"] is False
    assert result["archive_path"] is None
    assert result["hits_omitted"] == 0
    assert isinstance(result["index"]["generation"], int)
    assert result["index"]["last_indexed"]


def test_response_limit_applies_to_complete_serialized_result(tmp_path: Path) -> None:
    import json

    repo, config, embedder = make_repo(
        tmp_path, {"a.py": "a" * 40_000, "b.py": "b" * 40_000}
    )

    result = semantic_search(repo, "q", top_k=2, config=config, embedder=embedder)

    assert len(json.dumps(result, ensure_ascii=False).encode()) < 64 * 1024
    assert result["truncated"] is True
    assert result["archive_path"] is not None
    assert len(result["hits"]) == 2
    assert all({"path", "start_line", "end_line", "score", "stale"} <= hit.keys() for hit in result["hits"])
    archived = json.loads(Path(result["archive_path"]).read_text())
    assert sorted(len(hit["text"]) for hit in archived["hits"]) == [40_000, 40_000]


def test_oversized_single_hit_keeps_metadata_and_archive_path(tmp_path: Path) -> None:
    import json

    repo, config, embedder = make_repo(tmp_path, {"a.py": "x" * 70_000})

    result = semantic_search(repo, "q", top_k=1, config=config, embedder=embedder)

    assert result["truncated"] is True
    assert result["archive_path"] is not None
    assert len(result["hits"]) == 1
    assert {"path", "start_line", "end_line", "score", "stale", "text"} <= result["hits"][0].keys()
    assert len(json.dumps(result, ensure_ascii=False).encode()) < 64 * 1024
    archived = json.loads(Path(result["archive_path"]).read_text())
    assert len(archived["hits"][0]["text"]) == 70_000


def test_metadata_over_cap_omits_low_scoring_tail_hits_and_archives_full_result(tmp_path: Path) -> None:
    import json

    shared_dirs = "/".join("d" * 180 for _ in range(8))
    files = {f"{shared_dirs}/{i:02}.py": "same" for i in range(50)}
    repo, config, embedder = make_repo(tmp_path, files)

    result = semantic_search(repo, "q", top_k=50, config=config, embedder=embedder)

    assert len(json.dumps(result).encode("utf-8")) < 64 * 1024
    assert result["truncated"] is True
    assert result["hits_omitted"] == 50 - len(result["hits"])
    assert result["hits_omitted"] > 0
    assert all({"path", "start_line", "end_line", "score", "stale"} <= hit.keys() for hit in result["hits"])
    archived = json.loads(Path(result["archive_path"]).read_text())
    assert len(archived["hits"]) == 50
    assert archived["hits_omitted"] == 0
    assert all(hit["text"] == "same" for hit in archived["hits"])
    assert [hit["score"] for hit in result["hits"]] == [
        hit["score"] for hit in archived["hits"][: len(result["hits"])]
    ]


def test_non_ascii_text_is_bounded_using_default_json_serialization(tmp_path: Path) -> None:
    import json

    repo, config, embedder = make_repo(tmp_path, {"unicode.txt": "é" * 25_000})

    result = semantic_search(repo, "q", top_k=1, config=config, embedder=embedder)

    assert len(json.dumps(result).encode("utf-8")) < 64 * 1024
    assert result["truncated"] is True
    assert result["archive_path"] is not None


def test_outside_repository_path_returns_complete_result_shape(tmp_path: Path) -> None:
    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha"})

    result = semantic_search(
        repo, "alpha", path=tmp_path.parent, config=config, embedder=embedder
    )

    assert set(result) == {"hits", "index", "truncated", "archive_path", "hits_omitted"}
    assert result["hits"] == []
    assert result["truncated"] is False
    assert result["archive_path"] is None
    assert result["hits_omitted"] == 0


def test_malformed_last_indexed_metadata_cannot_exceed_cap(tmp_path: Path) -> None:
    import json

    repo, config, embedder = make_repo(tmp_path, {"a.py": "alpha"})
    with sqlite3.connect(db_path(repo)) as conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'last_indexed'", ("x" * 70_000,))

    result = semantic_search(repo, "alpha", config=config, embedder=embedder)

    assert len(json.dumps(result).encode("utf-8")) < 64 * 1024
    assert result["index"]["last_indexed"] is None
    assert len(result["hits"]) == 1
