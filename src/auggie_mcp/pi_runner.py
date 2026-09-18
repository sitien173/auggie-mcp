"""Ephemeral Pi subprocess adapter for codebase retrieval.

Starts one noninteractive Pi process per invocation against a caller-selected
directory and returns Pi's text-mode final response. Timeout and cancellation
terminate the entire Linux process group. No concurrency limit, no semaphore,
and no queue.
"""

import asyncio
import importlib.resources
import os
import signal
from pathlib import Path

# Thirty-minute per-invocation timeout (seconds).
TIMEOUT_SECONDS = 1800.0

# Grace period between SIGTERM and SIGKILL during shutdown.
_SHUTDOWN_GRACE_SECONDS = 0.5

# Upper bound (bytes) on retained standard error diagnostics.
STDERR_LIMIT = 8192

# Pi executable discovered on PATH.
_PI_EXECUTABLE = "pi"


class PiRunnerError(Exception):
    """Base class for pi_runner failures."""


class PiExecutionError(PiRunnerError):
    """Pi exited unsuccessfully."""

    def __init__(self, message, returncode=None, stderr=""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class PiParseError(PiRunnerError):
    """Pi produced no final assistant response."""


def load_system_prompt() -> str:
    """Return the packaged retrieval-only system prompt text."""
    return (importlib.resources.files("auggie_mcp") / "system_prompt.md").read_text(
        encoding="utf-8"
    )


def validate_repository(path) -> Path:
    """Resolve ``path`` to an absolute directory or raise ``ValueError``.

    Symlinks are followed so the returned path is the real target directory.
    Missing paths and non-directories fail before any process creation.
    """
    resolved = Path(os.path.realpath(os.fspath(path)))
    if not resolved.exists():
        raise ValueError(f"repository path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"repository path is not a directory: {resolved}")
    return resolved


def build_pi_command(repository, query, system_prompt) -> list:
    """Build the Pi argument list (without the executable).

    ``repository`` is a validated absolute path (resolved before use as cwd).
    ``query`` is passed as a single argument, never through a shell.
    """
    return [
        "--model",
        "gpt-5.6-luna",
        "--thinking",
        "high",
        "--approve",
        "--print",
        "--no-session",
        "--mode",
        "text",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--system-prompt",
        system_prompt,
        query,
    ]


async def _drain_bounded(stream, chunks, limit):
    """Drain ``stream`` while retaining at most ``limit`` bytes."""
    retained = sum(len(chunk) for chunk in chunks)
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        remaining = limit - retained
        if remaining > 0:
            kept = chunk[:remaining]
            chunks.append(kept)
            retained += len(kept)


def _join_bounded(chunks, limit):
    total = 0
    kept = []
    for chunk in chunks:
        if total >= limit:
            break
        kept.append(chunk)
        total += len(chunk)
    return b"".join(kept)[:limit]


async def _run_pi_process(cmd, repo, timeout):
    """Create Pi in a fresh process group and return its text output."""
    stdout = bytearray()
    stderr_chunks = []
    proc = None
    stderr_task = None

    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise FileNotFoundError(
                f"Pi executable not found: {cmd[0]} ({exc})"
            ) from exc

        stderr_task = asyncio.ensure_future(
            _drain_bounded(proc.stderr, stderr_chunks, STDERR_LIMIT)
        )

        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            stdout.extend(chunk)

        await proc.wait()
        returncode = proc.returncode

        # Let the stderr drainer reach EOF before reading diagnostics.
        if stderr_task is not None:
            try:
                await asyncio.wait_for(stderr_task, timeout=_SHUTDOWN_GRACE_SECONDS)
            except (asyncio.TimeoutError, Exception):
                stderr_task.cancel()
                try:
                    await stderr_task
                except (asyncio.CancelledError, Exception):
                    pass

        if returncode != 0:
            stderr_text = _join_bounded(stderr_chunks, STDERR_LIMIT).decode(
                "utf-8", errors="replace"
            )
            raise PiExecutionError(
                f"Pi exited with status {returncode}",
                returncode=returncode,
                stderr=stderr_text[:STDERR_LIMIT],
            )

        final_text = stdout.decode("utf-8", errors="replace").rstrip("\r\n")
        if not final_text.strip():
            raise PiParseError("Pi produced no final assistant response")
        return final_text
    except asyncio.CancelledError:
        await _terminate_group(proc)
        raise
    finally:
        if stderr_task is not None:
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass


async def run_pi(
    repository,
    query,
    system_prompt,
    *,
    executable=_PI_EXECUTABLE,
    timeout=TIMEOUT_SECONDS,
    extra_args=None,
):
    """Run Pi in ``repository`` with ``query`` and return the final answer.

    ``repository`` may be a path; it is validated and resolved before it becomes
    the subprocess working directory. Pi is invoked without a shell in a fresh
    Linux process group. On timeout or cancellation the whole group receives
    SIGTERM and then SIGKILL after a grace period. Concurrent invocations are
    unlimited.
    """
    repo = validate_repository(repository)
    cmd = [executable] + build_pi_command(repo, query, system_prompt)
    if extra_args:
        cmd.extend(extra_args)
    return await asyncio.wait_for(_run_pi_process(cmd, repo, timeout), timeout=timeout)


async def _terminate_group(proc):
    """Terminate the whole process group.

    Captures the process-group id while the leader is alive, then SIGTERMs the
    group, waits up to ``_SHUTDOWN_GRACE_SECONDS``, and SIGKILLs the group by
    that captured pgid even if the leader has already been reaped — a
    SIGTERM-ignoring grandchild cannot outlive the group.
    """
    if proc is None:
        return
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    await asyncio.sleep(_SHUTDOWN_GRACE_SECONDS)
    # SIGKILL the group by the captured pgid; the leader may already be reaped.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await proc.wait()
    except Exception:
        pass
