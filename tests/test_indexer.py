"""Unit tests for auggie_mcp.indexer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from auggie_mcp.config import ChunkingConfig, Config, EmbeddingConfig
from auggie_mcp.indexer import ModelMismatchError, index_status, index_update
from auggie_mcp.locking import IndexBusyError, index_lock
from auggie_mcp.store import db_path, open_read
from tests.fake_embedder import FakeEmbedder


def _make_config(*, model: str = "voyage-code-3", dimensions: int = 16) -> Config:
    return Config(
        embedding=EmbeddingConfig(model=model, dimensions=dimensions, batch_size=8),
        chunking=ChunkingConfig(lines=10, overlap_lines=2),
    )


def _write_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (repo / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    return repo


def test_first_run_embeds_all_chunks_and_records_meta(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    result = index_update(repo, config=cfg, embedder=embedder)

    assert result["files_scanned"] == 2
    assert result["files_changed"] == 2
    assert result["files_deleted"] == 0
    assert result["chunks_embedded"] == 2
    assert result["chunks_reused"] == 0
    assert isinstance(result["duration_ms"], int)
    assert embedder.call_count == 1
    assert embedder.text_count == 2

    conn = open_read(db_path(repo))
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        assert meta["model_key"] == embedder.model_key
        assert int(meta["generation"]) >= 1
        assert "last_indexed" in meta
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2
    finally:
        conn.close()


def test_second_run_no_changes_makes_zero_embed_calls(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    index_update(repo, config=cfg, embedder=embedder)
    calls_after_first = embedder.call_count

    result = index_update(repo, config=cfg, embedder=embedder)
    assert embedder.call_count == calls_after_first
    assert result["files_changed"] == 0
    assert result["chunks_embedded"] == 0


def test_edit_then_revert_makes_zero_embed_calls_on_revert(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    original = (repo / "a.py").read_text(encoding="utf-8")

    index_update(repo, config=cfg, embedder=embedder)
    (repo / "a.py").write_text("def alpha():\n    return 999\n", encoding="utf-8")
    r2 = index_update(repo, config=cfg, embedder=embedder)
    assert r2["chunks_embedded"] == 1
    calls_before_revert = embedder.call_count

    (repo / "a.py").write_text(original, encoding="utf-8")
    r3 = index_update(repo, config=cfg, embedder=embedder)
    assert embedder.call_count == calls_before_revert
    assert r3["chunks_embedded"] == 0
    assert r3["chunks_reused"] == 1


def test_touch_same_content_updates_mtime_only_and_embeds_nothing(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    index_update(repo, config=cfg, embedder=embedder)
    calls_before = embedder.call_count

    a_path = repo / "a.py"
    old_stat = a_path.stat()
    os.utime(a_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))

    result = index_update(repo, config=cfg, embedder=embedder)
    assert embedder.call_count == calls_before
    assert result["files_changed"] == 0
    assert result["chunks_embedded"] == 0

    conn = open_read(db_path(repo))
    try:
        stored_mtime = conn.execute(
            "SELECT mtime_ns FROM files WHERE path = 'a.py'"
        ).fetchone()[0]
        assert stored_mtime == a_path.stat().st_mtime_ns
    finally:
        conn.close()


def test_deleted_file_removes_its_rows(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    index_update(repo, config=cfg, embedder=embedder)
    (repo / "b.py").unlink()

    result = index_update(repo, config=cfg, embedder=embedder)
    assert result["files_deleted"] == 1

    conn = open_read(db_path(repo))
    try:
        paths = [r[0] for r in conn.execute("SELECT path FROM files").fetchall()]
        assert paths == ["a.py"]
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    finally:
        conn.close()


def test_crash_after_first_batch_keeps_batch_and_next_run_completes(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    (repo / "c.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    def boom(batch_idx: int) -> None:
        if batch_idx == 0:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        index_update(
            repo,
            config=cfg,
            embedder=embedder,
            file_batch_size=1,
            after_batch_hook=boom,
        )

    conn = open_read(db_path(repo))
    try:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
        gen_after_crash = int(
            conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()[0]
        )
        assert gen_after_crash >= 1
    finally:
        conn.close()

    # Next run completes the remaining files without re-embedding the committed one.
    # The committed file is rechunked because a partial first run records no
    # chunker_version, but its embedding is reused.
    calls_before_resume = embedder.call_count
    result = index_update(repo, config=cfg, embedder=embedder, file_batch_size=1)
    assert result["files_changed"] == 3
    assert result["chunks_embedded"] == 2
    assert embedder.call_count == calls_before_resume + 1

    conn = open_read(db_path(repo))
    try:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 3
        gen_after_resume = int(
            conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()[0]
        )
        assert gen_after_resume > gen_after_crash
    finally:
        conn.close()


def test_concurrent_run_while_lock_held_fails_with_busy_error(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    with index_lock(repo), pytest.raises(IndexBusyError):
        index_update(repo, config=cfg, embedder=embedder)


def test_model_mismatch_raises_without_rebuild_and_succeeds_with_rebuild(
    tmp_path: Path,
) -> None:
    repo = _write_repo(tmp_path)
    cfg1 = _make_config(model="voyage-code-3", dimensions=16)
    e1 = FakeEmbedder(model="voyage-code-3", dimensions=16)
    index_update(repo, config=cfg1, embedder=e1)

    cfg2 = _make_config(model="voyage-3", dimensions=16)
    e2 = FakeEmbedder(model="voyage-3", dimensions=16)
    with pytest.raises(ModelMismatchError):
        index_update(repo, config=cfg2, embedder=e2)

    result = index_update(repo, config=cfg2, embedder=e2, rebuild=True)
    assert result["files_changed"] == 2
    assert result["chunks_embedded"] == 2

    conn = open_read(db_path(repo))
    try:
        model_key = conn.execute(
            "SELECT value FROM meta WHERE key = 'model_key'"
        ).fetchone()[0]
        assert model_key == e2.model_key
    finally:
        conn.close()


def test_orphan_cleanup_deletes_only_old_unreferenced_embeddings(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    day0 = 1_700_000_000.0

    index_update(repo, config=cfg, embedder=embedder, now_fn=lambda: day0)

    # Delete b.py at day 10; retention starts at day 10 so at day 39 (< 30d) it stays.
    (repo / "b.py").unlink()
    index_update(
        repo, config=cfg, embedder=embedder, now_fn=lambda: day0 + 10 * 86400
    )
    index_update(
        repo, config=cfg, embedder=embedder, now_fn=lambda: day0 + 39 * 86400
    )
    conn = open_read(db_path(repo))
    try:
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2
    finally:
        conn.close()

    # At day 41 (> 30 days after deletion at day 10), unreferenced embedding is deleted.
    index_update(
        repo, config=cfg, embedder=embedder, now_fn=lambda: day0 + 41 * 86400
    )
    conn = open_read(db_path(repo))
    try:
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    finally:
        conn.close()


def test_content_hash_equals_sha256_of_exact_embedded_text(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    index_update(repo, config=cfg, embedder=embedder)

    embedded_texts = [t for batch in embedder.calls for t in batch]
    expected_hashes = {
        hashlib.sha256(t.encode("utf-8")).digest() for t in embedded_texts
    }

    conn = open_read(db_path(repo))
    try:
        stored_hashes = {
            bytes(r[0]) for r in conn.execute("SELECT content_hash FROM chunks").fetchall()
        }
        assert stored_hashes == expected_hashes
    finally:
        conn.close()


def test_chunker_version_change_crash_after_first_batch_rechunks_remaining_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auggie_mcp.indexer as indexer_mod

    repo = _write_repo(tmp_path)
    (repo / "c.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    index_update(repo, config=cfg, embedder=embedder)

    # Simulate chunker version bump.
    monkeypatch.setattr(indexer_mod, "CHUNKER_VERSION", 2)

    def boom(batch_idx: int) -> None:
        if batch_idx == 0:
            raise RuntimeError("crash during rechunk")

    with pytest.raises(RuntimeError, match="crash during rechunk"):
        index_update(
            repo,
            config=cfg,
            embedder=embedder,
            file_batch_size=1,
            after_batch_hook=boom,
        )

    # meta.chunker_version must still be old (1) so resume rechunks remaining files.
    conn = open_read(db_path(repo))
    try:
        cv = conn.execute(
            "SELECT value FROM meta WHERE key = 'chunker_version'"
        ).fetchone()[0]
        assert cv == "1"
    finally:
        conn.close()

    calls_before_resume = embedder.call_count
    result = index_update(repo, config=cfg, embedder=embedder, file_batch_size=1)
    assert result["files_changed"] == 3
    assert result["chunks_embedded"] == 0
    assert result["chunks_reused"] == 3
    assert embedder.call_count == calls_before_resume

    conn = open_read(db_path(repo))
    try:
        cv = conn.execute(
            "SELECT value FROM meta WHERE key = 'chunker_version'"
        ).fetchone()[0]
        assert cv == "2"
    finally:
        conn.close()


def test_index_status_opens_no_repository_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    real_open = Path.open
    opened_repo_files: list[Path] = []

    def guarded_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        resolved = self.resolve(strict=False)
        if resolved.is_relative_to(repo.resolve()) and ".auggie" not in resolved.parts:
            opened_repo_files.append(resolved)
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", guarded_open)
    status = index_status(repo, include_legacy=True)
    assert status["exists"] is True
    assert status["stale"] == {"modified": 0, "deleted": 0, "added": 0}
    assert opened_repo_files == []


def test_vanished_file_after_walk_is_treated_as_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auggie_mcp.indexer as indexer_mod

    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    real_walk = indexer_mod.walk_repo

    def vanishing_walk(root: Path | str, **kwargs: object):  # type: ignore[no-untyped-def]
        paths = list(real_walk(root, **kwargs))  # type: ignore[arg-type]
        (repo / "b.py").unlink()
        return iter(paths)

    monkeypatch.setattr(indexer_mod, "walk_repo", vanishing_walk)
    result = index_update(repo, config=cfg, embedder=embedder)
    assert result["files_deleted"] == 1

    conn = open_read(db_path(repo))
    try:
        paths = [r[0] for r in conn.execute("SELECT path FROM files").fetchall()]
        assert paths == ["a.py"]
    finally:
        conn.close()


def test_unreadable_existing_file_keeps_record_and_chunks_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    # Touch b.py so it isn't skipped by mtime, then make read_bytes fail with PermissionError.
    b_path = repo / "b.py"
    st = b_path.stat()
    os.utime(b_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    real_read_bytes = Path.read_bytes

    def failing_read_bytes(self: Path) -> bytes:
        if self.name == "b.py":
            raise PermissionError("permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    result = index_update(repo, config=cfg, embedder=embedder)
    assert result["files_deleted"] == 0
    assert result["files_changed"] == 0

    conn = open_read(db_path(repo))
    try:
        paths = sorted(r[0] for r in conn.execute("SELECT path FROM files").fetchall())
        assert paths == ["a.py", "b.py"]
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    finally:
        conn.close()


def test_deleted_file_after_30_days_refreshes_last_seen_so_restore_reuses(
    tmp_path: Path,
) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    day0 = 1_700_000_000.0
    b_content = (repo / "b.py").read_text(encoding="utf-8")

    index_update(repo, config=cfg, embedder=embedder, now_fn=lambda: day0)

    # Advance past 30 days with an unchanged run.
    index_update(
        repo, config=cfg, embedder=embedder, now_fn=lambda: day0 + 35 * 86400
    )

    # Delete b.py at day 35; its embedding's last_seen must be refreshed to day 35.
    (repo / "b.py").unlink()
    index_update(
        repo, config=cfg, embedder=embedder, now_fn=lambda: day0 + 35 * 86400
    )

    # Restore b.py at day 36; embedding must be reused with 0 embed calls.
    (repo / "b.py").write_text(b_content, encoding="utf-8")
    calls_before_restore = embedder.call_count
    result = index_update(
        repo, config=cfg, embedder=embedder, now_fn=lambda: day0 + 36 * 86400
    )
    assert embedder.call_count == calls_before_restore
    assert result["chunks_embedded"] == 0
    assert result["chunks_reused"] == 1


def test_generation_exact_counts_for_single_batch_no_change_and_delete(
    tmp_path: Path,
) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    def read_gen() -> int:
        conn = open_read(db_path(repo))
        try:
            return int(
                conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()[0]
            )
        finally:
            conn.close()

    # First run with 1 batch -> generation == 1.
    index_update(repo, config=cfg, embedder=embedder, file_batch_size=50)
    assert read_gen() == 1

    # No-change run -> generation stays 1.
    index_update(repo, config=cfg, embedder=embedder, file_batch_size=50)
    assert read_gen() == 1

    # Run with only a deletion -> generation == 2.
    (repo / "b.py").unlink()
    index_update(repo, config=cfg, embedder=embedder, file_batch_size=50)
    assert read_gen() == 2


def test_crashed_first_run_persists_model_key_in_first_batch(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    cfg1 = _make_config(model="voyage-code-3", dimensions=16)
    e1 = FakeEmbedder(model="voyage-code-3", dimensions=16)

    def boom(batch_idx: int) -> None:
        if batch_idx == 0:
            raise RuntimeError("crash after batch 0")

    with pytest.raises(RuntimeError, match="crash after batch 0"):
        index_update(
            repo,
            config=cfg1,
            embedder=e1,
            file_batch_size=1,
            after_batch_hook=boom,
        )

    cfg2 = _make_config(model="voyage-3", dimensions=16)
    e2 = FakeEmbedder(model="voyage-3", dimensions=16)
    with pytest.raises(ModelMismatchError):
        index_update(repo, config=cfg2, embedder=e2)


def test_chunker_version_not_updated_when_retained_file_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auggie_mcp.indexer as indexer_mod

    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    monkeypatch.setattr(indexer_mod, "CHUNKER_VERSION", 2)
    real_read_bytes = Path.read_bytes

    def fail_b(self: Path) -> bytes:
        if self.name == "b.py":
            raise PermissionError("cannot read b.py")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_b)
    index_update(repo, config=cfg, embedder=embedder)

    conn = open_read(db_path(repo))
    try:
        cv = conn.execute(
            "SELECT value FROM meta WHERE key = 'chunker_version'"
        ).fetchone()[0]
        assert cv == "1"
    finally:
        conn.close()

    # Next run with readable b.py retries and completes version bump.
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)
    result = index_update(repo, config=cfg, embedder=embedder)
    assert result["files_changed"] == 2

    conn = open_read(db_path(repo))
    try:
        cv = conn.execute(
            "SELECT value FROM meta WHERE key = 'chunker_version'"
        ).fetchone()[0]
        assert cv == "2"
    finally:
        conn.close()


def test_index_status_with_both_ignore_files_opens_no_candidate_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_repo(tmp_path)
    (repo / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    (repo / ".auggieignore").write_text("*.skip\n", encoding="utf-8")
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    real_open = Path.open
    opened_candidates: list[Path] = []
    allowed_names = {".gitignore", ".auggieignore"}

    def guarded_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        resolved = self.resolve(strict=False)
        if (
            resolved.is_relative_to(repo.resolve())
            and ".auggie" not in resolved.parts
            and resolved.name not in allowed_names
        ):
            opened_candidates.append(resolved)
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", guarded_open)
    status = index_status(repo)
    assert status["exists"] is True
    assert status["stale"] == {"modified": 0, "deleted": 0, "added": 0}
    assert opened_candidates == []


def test_index_status_clean_with_binary_file_reports_zero_added(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    status = index_status(repo)
    assert status["stale"] == {"modified": 0, "deleted": 0, "added": 0}
    assert status["next_step"] is None


def test_unreadable_file_at_walker_level_preserves_existing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    real_open = Path.open

    def fail_b_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self.name == "b.py":
            raise PermissionError("permission denied")
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", fail_b_open)
    result = index_update(repo, config=cfg, embedder=embedder)
    assert result["files_deleted"] == 0

    conn = open_read(db_path(repo))
    try:
        paths = sorted(r[0] for r in conn.execute("SELECT path FROM files").fetchall())
        assert paths == ["a.py", "b.py"]
    finally:
        conn.close()


def test_rebuild_clear_increments_generation_before_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auggie_mcp.indexer as indexer_mod

    repo = _write_repo(tmp_path)
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)
    index_update(repo, config=cfg, embedder=embedder)

    def fail_walk(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom after rebuild clear")

    monkeypatch.setattr(indexer_mod, "walk_repo", fail_walk)
    with pytest.raises(RuntimeError, match="boom after rebuild clear"):
        index_update(repo, config=cfg, embedder=embedder, rebuild=True)

    conn = open_read(db_path(repo))
    try:
        gen = int(
            conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()[0]
        )
        assert gen == 2
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    finally:
        conn.close()


def test_partial_first_index_then_chunker_change_rechunks_committed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auggie_mcp.indexer as indexer_mod

    repo = _write_repo(tmp_path)
    (repo / "c.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    cfg = _make_config()
    embedder = FakeEmbedder(dimensions=16)

    def boom(batch_idx: int) -> None:
        if batch_idx == 0:
            raise RuntimeError("crash during first index")

    with pytest.raises(RuntimeError, match="crash during first index"):
        index_update(
            repo,
            config=cfg,
            embedder=embedder,
            file_batch_size=1,
            after_batch_hook=boom,
        )

    monkeypatch.setattr(indexer_mod, "CHUNKER_VERSION", 2)
    result = index_update(repo, config=cfg, embedder=embedder, file_batch_size=1)
    assert result["files_changed"] == 3

    conn = open_read(db_path(repo))
    try:
        cv = conn.execute(
            "SELECT value FROM meta WHERE key = 'chunker_version'"
        ).fetchone()[0]
        assert cv == "2"
    finally:
        conn.close()
