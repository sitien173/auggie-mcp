"""Incremental auggie semantic index refresh so searches see current files."""

import asyncio

from auggie_mcp import pi_runner, rg_runner

# Upper bound (characters) on stderr kept in a refresh failure message.
_STDERR_TAIL = 2000


class IndexRefreshError(RuntimeError):
    """An index refresh exited unsuccessfully."""


async def refresh_index(executable, root) -> None:
    """Run ``<executable> index <root>`` in ``root`` and wait for it.

    The auggie semantic index updates incrementally, so unchanged files are skipped.
    Cancellation terminates the whole process group.
    """
    repo = pi_runner.validate_repository(root)
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "index",
            str(repo),
            cwd=str(repo),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise FileNotFoundError(
            f"index executable not found: {executable} ({exc})"
        ) from exc

    try:
        _, stderr = await process.communicate()
    except asyncio.CancelledError:
        await rg_runner._terminate_process_group(process)
        raise

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL:].strip()
        raise IndexRefreshError(
            f"{executable} index exited with status {process.returncode}: {detail}"
        )
