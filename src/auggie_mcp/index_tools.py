"""Semantic index inspection and update for the ``index_status`` and ``index_update`` tools."""

import asyncio
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote

from mcp.server.mcpserver.exceptions import ToolError

from auggie.commands._common import _detect_git_branch, find_git_root, resolve_db_path
from auggie.indexer import iter_files, load_auggieignore
from auggie_mcp import agent_runner

DEFAULT_TIMEOUT_S = 1800.0
# Mirrors the indexer's default --max-bytes, which `index_update` never overrides.
_MAX_BYTES = 2_000_000
# Upper bound (characters) on stderr kept in an indexer failure message.
_STDERR_TAIL = 2000
# Seconds between SIGTERM and SIGKILL when stopping the indexer's process group.
_KILL_GRACE_S = 0.5


def default_index_command() -> list[str]:
    """Indexer command prefix: this interpreter's ``auggie`` module, never ``PATH``."""
    return [sys.executable, "-m", "auggie"]


def _locate(repository_path) -> tuple[Path, Path, str]:
    """Return the repository root (Git root, else the directory), its index path, and branch.

    The branch is detected on every call because ``get_current_branch`` caches for the
    process lifetime and a checkout would leave it stale.
    """
    try:
        root = agent_runner.validate_repository(repository_path)
    except ValueError as exc:
        raise ToolError(f"{exc}. Pass an existing repository directory.") from exc
    repo_root = find_git_root(root) or root
    return repo_root, resolve_db_path(None, None, False, repo_root), _detect_git_branch(repo_root)


def open_readonly(db_path) -> sqlite3.Connection:
    """Open the index database read-only; never creates, migrates, or writes it."""
    uri = f"file:{quote(str(db_path))}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def stale_files(con, branch) -> tuple[list[str], list[str]]:
    """Return indexed paths of ``branch`` that were modified and that were deleted on disk."""
    modified, deleted = [], []
    for path, mtime_ns, size in con.execute(
        "SELECT path, mtime_ns, size FROM files WHERE branch = ?", (branch,)
    ):
        try:
            st = os.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            deleted.append(path)
        except OSError:
            continue
        else:
            if st.st_mtime_ns != mtime_ns or st.st_size != size:
                modified.append(path)
    return modified, deleted


def _candidate_added(root: Path, indexed: set[str]) -> int:
    """Count walkable files under the indexer's ignore and size rules that are not indexed.

    MIME detection is skipped, so binary files the indexer later rejects may be counted.
    """
    count = 0
    for path in iter_files(root, exclude=load_auggieignore(root)):
        try:
            if str(path.resolve()) in indexed or path.stat().st_size > _MAX_BYTES:
                continue
        except OSError:
            continue
        count += 1
    return count


def _stored_model(con) -> tuple[str | None, int | None]:
    row = con.execute("SELECT model, dim FROM chunks LIMIT 1").fetchone()
    return (row[0], row[1]) if row else (None, None)


def _update_hint(repository_path) -> str:
    return f"index_update(repository_path={str(repository_path)!r})"


def index_status(repository_path) -> dict:
    """Summarize the semantic index for ``repository_path`` without writing to it."""
    root, db, branch = _locate(repository_path)
    if not db.exists():
        return {
            "exists": False,
            "model": None,
            "dim": None,
            "files": 0,
            "chunks": 0,
            "last_indexed": None,
            "stale": None,
            "next_step": f"No index for {root}. Call {_update_hint(root)} to build it.",
        }
    try:
        con = open_readonly(db)
        try:
            model, dim = _stored_model(con)
            (files,) = con.execute(
                "SELECT COUNT(*) FROM files WHERE branch = ?", (branch,)
            ).fetchone()
            (chunks,) = con.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_id IN "
                "(SELECT id FROM files WHERE branch = ?)",
                (branch,),
            ).fetchone()
            (last_indexed,) = con.execute(
                "SELECT MAX(timestamp) FROM index_history"
            ).fetchone()
            modified, deleted = stale_files(con, branch)
            indexed = {p for (p,) in con.execute("SELECT path FROM files WHERE branch = ?", (branch,))}
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise ToolError(
            f"Index at {db} is unreadable ({exc}). "
            f"Call {_update_hint(root)[:-1]}, rebuild=True) to rebuild it."
        ) from exc
    candidate_added = _candidate_added(root, indexed)
    stale = {
        "modified": len(modified),
        "deleted": len(deleted),
        "candidate_added": candidate_added,
    }
    if any(stale.values()):
        next_step = f"Index is stale. Call {_update_hint(root)} to refresh it."
    else:
        next_step = "Index is current. No update is needed."
    return {
        "exists": True,
        "model": model,
        "dim": dim,
        "files": files,
        "chunks": chunks,
        "last_indexed": last_indexed,
        "stale": stale,
        "next_step": next_step,
    }


def _timeout_seconds() -> float:
    raw = os.environ.get("AUGGIE_INDEX_TIMEOUT_S")
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0:
        raise ToolError(
            f"AUGGIE_INDEX_TIMEOUT_S must be a positive number of seconds, got {raw!r}. "
            "Fix the variable in the MCP server environment and call index_update again."
        )
    return value


def _parse_json(text: str) -> dict | None:
    """Parse the indexer's JSON object from the whole output, else from its last line."""
    lines = [line for line in text.splitlines() if line.strip()]
    for candidate in (text, lines[-1] if lines else ""):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


async def _terminate_group(process) -> None:
    """SIGTERM then SIGKILL the indexer's process group and reap the leader.

    Unlike ``rg_runner._terminate_process_group``, this signals the group even when
    the leader has already exited, because its children can still hold the pipes.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        if sig is signal.SIGTERM:
            await asyncio.sleep(_KILL_GRACE_S)
    await process.wait()


async def run_index_update(
    repository_path,
    *,
    rebuild: bool = False,
    command=None,
    report_progress=None,
) -> dict:
    """Run ``<command> index|reindex <root>`` in a new session and return its JSON plus ``duration_ms``.

    ``report_progress(progress, total, message)`` is awaited at start and finish. Timeout
    and cancellation terminate the indexer's process group and reap it.
    """
    root, db, _ = _locate(repository_path)
    if not os.environ.get("VOYAGE_API_KEY"):
        raise ToolError(
            "VOYAGE_API_KEY is not set. Set it in the MCP server environment, "
            "then call index_update again."
        )
    timeout_s = _timeout_seconds()

    verb = "reindex" if rebuild else "index"
    args = [verb, str(root), "--json"]
    if not rebuild and db.exists():
        try:
            con = open_readonly(db)
            try:
                model, dim = _stored_model(con)
            finally:
                con.close()
        except sqlite3.Error:
            model = dim = None
        if model is not None:
            args += ["--model", model, "--dimensions", str(dim)]
    cmd = [*(default_index_command() if command is None else command), *args]

    async def progress(value, message):
        if report_progress is not None:
            await report_progress(value, 1, message)

    start = time.perf_counter()
    await progress(0, f"{verb} started")
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolError(
            f"could not start the indexer ({cmd[0]}): {exc}. "
            "Check that the server's Python environment includes the auggie package."
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
    except asyncio.TimeoutError:
        await _terminate_group(process)
        raise ToolError(
            f"index update timed out after {timeout_s:g}s and was stopped. "
            "Raise AUGGIE_INDEX_TIMEOUT_S, then call index_update again; "
            "an interrupted incremental run keeps its progress, "
            "and index_status shows what exists."
        ) from None
    except asyncio.CancelledError:
        await _terminate_group(process)
        raise

    out = stdout.decode("utf-8", errors="replace")
    parsed = _parse_json(out)
    if process.returncode != 0:
        if parsed and parsed.get("error_code") == "INDEX_BUSY":
            raise ToolError(
                "index update already running (another auggie process holds the index lock). "
                "Wait for it to finish, then call index_status."
            )
        detail = stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL:].strip()
        detail = detail or (parsed or {}).get("error") or out[-_STDERR_TAIL:].strip()
        raise ToolError(
            f"index update failed with exit status {process.returncode}: {detail}. "
            "Fix the cause, then call index_update again."
        )
    if parsed is None:
        raise ToolError(
            "index update finished but its output was not JSON. "
            "Call index_status to check the index."
        )
    await progress(1, f"{verb} finished")
    return {**parsed, "duration_ms": round((time.perf_counter() - start) * 1000)}
