from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from auggie_mcp.output_limits import cleanup_archives, truncate_output


def test_text_under_limit_is_unchanged_without_archive(tmp_path: Path) -> None:
    text = "short\ntext\n"

    result, archive_path = truncate_output(text, tmp_path)

    assert result == text
    assert archive_path is None
    assert not (tmp_path / ".auggie" / "archive").exists()


def test_text_over_limit_is_cut_at_line_boundary_and_archived(tmp_path: Path) -> None:
    text = "x" * 32767 + "\n" + "y" * 32767 + "\n" + "tail\n"

    result, archive_path = truncate_output(text, tmp_path)

    assert len(result.encode("utf-8")) < 64 * 1024
    assert result.endswith("\n")
    assert archive_path is not None
    assert Path(archive_path).read_text() == text
    assert Path(archive_path).is_relative_to(tmp_path / ".auggie" / "archive")


def test_archive_cleanup_runs_on_write_and_startup(tmp_path: Path) -> None:
    archive_dir = tmp_path / ".auggie" / "archive"
    archive_dir.mkdir(parents=True)
    old = archive_dir / "old.txt"
    recent = archive_dir / "recent.txt"
    old.write_text("old")
    recent.write_text("recent")
    now = datetime.now(UTC)
    old_time = (now - timedelta(days=8)).timestamp()
    recent_time = (now - timedelta(days=1)).timestamp()
    import os

    os.utime(old, (old_time, old_time))
    os.utime(recent, (recent_time, recent_time))

    cleanup_archives(tmp_path, now=now)

    assert not old.exists()
    assert recent.exists()
    old.write_text("old again")
    os.utime(old, (old_time, old_time))
    truncate_output("x" * (64 * 1024) + "\n", tmp_path, now=now)
    assert not old.exists()
    assert recent.exists()


def test_public_startup_cleanup_removes_expired_archives(tmp_path: Path) -> None:
    import os
    from datetime import datetime, timedelta

    from auggie_mcp import output_limits

    archive_dir = tmp_path / ".auggie" / "archive"
    archive_dir.mkdir(parents=True)
    expired = archive_dir / "expired.txt"
    recent = archive_dir / "recent.txt"
    expired.write_text("expired")
    recent.write_text("recent")
    now = datetime.now(UTC)
    expired_time = (now - timedelta(days=8)).timestamp()
    recent_time = (now - timedelta(days=1)).timestamp()
    os.utime(expired, (expired_time, expired_time))
    os.utime(recent, (recent_time, recent_time))

    startup_cleanup = getattr(
        output_limits, "cleanup_archives_on_startup", lambda _repository_path: None
    )
    startup_cleanup(tmp_path)

    assert not expired.exists()
    assert recent.exists()
