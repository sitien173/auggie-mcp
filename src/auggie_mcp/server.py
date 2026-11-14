"""Stdio MCP server exposing the four SQLite rewrite tools."""

from __future__ import annotations

import asyncio
import functools
import importlib.metadata
import json
import os
import re
import signal
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server import CacheHint, MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from auggie_mcp import (
    config,
    embed,
    indexer,
    locking,
    output_limits,
    retrieval,
    semantic,
    telemetry,
)

TOOL_TIMEOUT_S = 1800

_INSTRUCTIONS = """\
1. Use your built-in grep tool for known literals, symbols, and regular expressions.
2. For concept queries, call `index_status`. If the index is missing or stale, call `index_update`, then `semantic_search`.
3. Use `codebase_retrieval` when you need a synthesized answer.
"""

_TOOL_DESCRIPTIONS = {
    "semantic_search": "Return ranked code chunks from the SQLite semantic index. Search never updates the index.",
    "index_status": "Inspect index counts and stat-only staleness without changing the index.",
    "index_update": "Build or refresh the repository's SQLite semantic index.",
    "codebase_retrieval": "Run the configured read-only agent CLI and return a synthesized answer.",
}

_ERROR_ACTIONS = {
    "INVALID_PATH": "Fix the path",
    "INVALID_ARGUMENT": "Fix the argument",
    "INDEX_MISSING": "Call index_update",
    "INDEX_MODEL_MISMATCH": "Call index_update with rebuild=true",
    "INDEX_BUSY": "Retry later",
    "MISSING_API_KEY": "Set the environment variable",
    "EMBED_FAILED": "Retry or check quota",
    "TIMEOUT": "Narrow the scope",
    "RETRIEVAL_BUSY": "Retry later",
    "REPO_MODIFIED": "The result is discarded. The message lists changed files",
    "BINARY_NOT_FOUND": "Install the binary",
    "INTERNAL_ERROR": "Inspect the server logs",
}


class InvalidPathError(ValueError):
    """Raised when a repository path does not identify a directory."""


def _client_info(ctx: Context | None) -> dict[str, str] | None:
    if ctx is None:
        return None
    try:
        info = ctx.session.client_params.client_info
        return {"name": info.name, "version": info.version}
    except (AttributeError, TypeError, ValueError):
        return None


def _error_code(exc: BaseException) -> tuple[str, str]:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if isinstance(exc, InvalidPathError):
        return "INVALID_PATH", message
    if "voyage_api_key" in lowered:
        return "MISSING_API_KEY", message
    if "model_key" in lowered and ("does not match" in lowered or "mismatch" in lowered):
        return "INDEX_MODEL_MISMATCH", message
    if re.search(r"\bhttp\s+\d{3}\b", message, re.IGNORECASE):
        return "EMBED_FAILED", message
    if "binary_not_found" in lowered or "missing_cli" in lowered:
        return "BINARY_NOT_FOUND", message
    if isinstance(exc, locking.IndexBusyError):
        return "INDEX_BUSY", message
    if isinstance(exc, semantic.IndexMissingError):
        return "INDEX_MISSING", message
    if isinstance(exc, (semantic.ModelMismatchError, indexer.ModelMismatchError)):
        return "INDEX_MODEL_MISMATCH", message
    if isinstance(exc, embed.MissingApiKeyError):
        return "MISSING_API_KEY", message
    if isinstance(exc, embed.EmbedError):
        status = exc.http_status or exc.status
        if status is not None and not re.search(rf"\b{status}\b", message):
            message = f"HTTP {status}: {message}"
        return "EMBED_FAILED", message
    if isinstance(exc, retrieval.RepoModifiedError):
        return "REPO_MODIFIED", message
    if isinstance(exc, retrieval.RetrievalBusyError):
        return "RETRIEVAL_BUSY", message
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timed out" in lowered or "timeout" in lowered:
        return "TIMEOUT", message
    if isinstance(exc, (FileNotFoundError, OSError)) and any(
        part in message.lower() for part in ("executable not found", "binary not found", "no such file or directory")
    ):
        return "BINARY_NOT_FOUND", message
    if isinstance(exc, (ValueError, FileNotFoundError)):
        if "path does not exist" in message.lower() or "not a directory" in message.lower():
            return "INVALID_PATH", message
        return "INVALID_ARGUMENT", message
    return "INTERNAL_ERROR", message


def _tool_error(exc: BaseException) -> ToolError:
    code, message = _error_code(exc)
    return ToolError(f"{code}: {message}. {_ERROR_ACTIONS[code]}")


def _instrument(tool_name: str, telemetry_config: config.TelemetryConfig) -> Callable:
    """Record one telemetry event and map every tool exception consistently."""
    def decorate(function: Callable) -> Callable:
        @functools.wraps(function)
        async def wrapped(*args, **kwargs):
            ctx = kwargs.get("ctx")
            tool_input = {key: value for key, value in kwargs.items() if key != "ctx"}
            record = telemetry.CallRecord(
                "mcp", tool_name, tool_input, client=_client_info(ctx),
                telemetry_enabled=telemetry_config.enabled,
                telemetry_directory=telemetry_config.dir,
            )
            outcome = "error"
            error: BaseException | None = None
            try:
                with record.stage("runner"):
                    result = await function(*args, **kwargs)
                if tool_name == "codebase_retrieval":
                    record.set_runner(
                        exit_code=0,
                        stdout_bytes=len(result["answer"].encode("utf-8")),
                        stderr_bytes=0,
                    )
                outcome = "ok"
                return result
            except asyncio.CancelledError as exc:
                outcome = "cancelled"
                error = exc
                if tool_name == "codebase_retrieval":
                    record.set_runner(exit_code=None, stdout_bytes=0, stderr_bytes=0)
                raise
            except Exception as exc:
                error = exc
                if tool_name == "codebase_retrieval":
                    record.set_runner(
                        exit_code=None, stdout_bytes=0,
                        stderr_bytes=len(str(exc).encode("utf-8")),
                    )
                raise _tool_error(exc) from exc
            finally:
                record.finish(outcome, error)
        return wrapped
    return decorate


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.returncode is None:
            await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.returncode is None:
        await process.wait()


def _repository_path(repository_path: str) -> Path:
    path = Path(repository_path).expanduser().resolve()
    if not path.exists():
        raise InvalidPathError(f"repository path does not exist: {path}")
    if not path.is_dir():
        raise InvalidPathError(f"repository path is not a directory: {path}")
    return path


async def _run_index_update(
    repository_path: str, rebuild: bool, timeout_s: float = TOOL_TIMEOUT_S
) -> dict[str, Any]:
    repository_path = str(_repository_path(repository_path))
    command = [sys.executable, "-m", "auggie_mcp.cli", "index"]
    if rebuild:
        command.append("--rebuild")
    command.extend(("--json", repository_path))
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise FileNotFoundError(f"index CLI binary not found: {sys.executable}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_s
        )
    except TimeoutError as exc:
        await _kill_process_group(process)
        raise TimeoutError(f"index update exceeded {timeout_s} seconds") from exc
    except asyncio.CancelledError:
        await _kill_process_group(process)
        raise
    from auggie_mcp.cli import EXIT_BUSY

    if process.returncode == EXIT_BUSY:
        raise locking.IndexBusyError("another process holds the index lock")
    if process.returncode:
        record = stdout.decode("utf-8", errors="replace").strip()
        try:
            error = json.loads(record).get("error")
        except (json.JSONDecodeError, AttributeError):
            error = None
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            status = error.get("http_status")
            if not isinstance(code, str) or not isinstance(message, str):
                raise TypeError("index CLI returned malformed error JSON")
            if code == "MISSING_API_KEY":
                raise embed.MissingApiKeyError(message)
            if code == "EMBED_FAILED":
                raise embed.EmbedError(message, status=status)
            if code == "INDEX_MODEL_MISMATCH":
                raise indexer.ModelMismatchError(message)
            if code == "INDEX_BUSY":
                raise locking.IndexBusyError(message)
            if code == "TIMEOUT":
                raise TimeoutError(message)
            if code == "INVALID_PATH":
                raise InvalidPathError(message)
            if code == "INVALID_ARGUMENT":
                raise ValueError(message)
            if code == "BINARY_NOT_FOUND":
                raise FileNotFoundError(message)
            raise RuntimeError(message)
        diagnostic = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(diagnostic or record or f"index CLI exited with status {process.returncode}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("index CLI returned invalid JSON") from exc
    keys = ("files_scanned", "files_changed", "files_deleted", "chunks_embedded", "chunks_reused", "duration_ms")
    return {key: result[key] for key in keys}


def _agent_config(agent: config.AgentConfig) -> retrieval.AgentConfig:
    return retrieval.AgentConfig(
        driver=agent.driver,
        model=agent.model,
        reasoning=agent.reasoning,
        args=agent.args,
        executable=agent.executable,
        timeout_s=agent.timeout_s,
    )


def create_server(
    *,
    app_config: config.Config | None = None,
    executable: str | None = None,
    repository_root: str | Path | None = None,
) -> MCPServer:
    """Build the server with configuration supplied by its launcher."""
    cfg = app_config or config.load()
    server_root = Path.cwd() if repository_root is None else Path(repository_root)

    @asynccontextmanager
    async def lifespan(app: MCPServer) -> AsyncGenerator[dict[str, Any], None]:
        output_limits.cleanup_archives_on_startup(server_root)
        telemetry.prune_old_files(
            enabled=cfg.telemetry.enabled,
            directory=cfg.telemetry.dir,
            retention=cfg.telemetry.retention_days,
        )
        yield {"config": cfg}

    mcp = MCPServer(
        name="auggie-mcp",
        title="Auggie Codebase Search",
        description="Semantic search, indexing, and read-only codebase retrieval.",
        version=importlib.metadata.version("auggie-mcp"),
        instructions=_INSTRUCTIONS,
        lifespan=lifespan,
        cache_hints={"tools/list": CacheHint(ttl_ms=3_600_000, scope="private")},
        subscriptions=False,
    )

    @mcp.tool(name="semantic_search", description=_TOOL_DESCRIPTIONS["semantic_search"], annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    @_instrument("semantic_search", cfg.telemetry)
    async def semantic_search_tool(
        repository_path: str,
        query: str,
        path: str = ".",
        top_k: int = 10,
        context_lines: int = 0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        if not 1 <= top_k <= 50 or not 0 <= context_lines <= 20:
            raise ValueError("top_k must be 1 to 50 and context_lines must be 0 to 20")
        repository_path = str(_repository_path(repository_path))
        return await asyncio.to_thread(
            semantic.semantic_search,
            repository_path,
            query,
            path=path,
            top_k=top_k,
            context_lines=context_lines,
            config=cfg,
        )

    @mcp.tool(description=_TOOL_DESCRIPTIONS["index_status"], annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False))
    @_instrument("index_status", cfg.telemetry)
    async def index_status(repository_path: str, ctx: Context | None = None) -> dict[str, Any]:
        repository_path = str(_repository_path(repository_path))
        return await asyncio.to_thread(indexer.index_status, repository_path)

    @mcp.tool(description=_TOOL_DESCRIPTIONS["index_update"], annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True))
    @_instrument("index_update", cfg.telemetry)
    async def index_update(
        repository_path: str,
        rebuild: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await _run_index_update(
            repository_path, rebuild, timeout_s=cfg.index.timeout_s
        )

    @mcp.tool(description=_TOOL_DESCRIPTIONS["codebase_retrieval"], annotations=ToolAnnotations(read_only_hint=False, open_world_hint=True))
    @_instrument("codebase_retrieval", cfg.telemetry)
    async def codebase_retrieval(
        repository_path: str,
        query: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        result = await retrieval.run_agent(
            repository_path,
            query,
            config=_agent_config(cfg.agent),
            app_config=cfg,
            executable=executable,
            max_concurrent=cfg.retrieval.max_concurrent,
        )
        response = {**result, "truncated": False, "archive_path": None}
        full_output = output_limits.serialize_json(response)
        if len(full_output.encode("utf-8")) <= output_limits.MAX_OUTPUT_BYTES:
            return response

        _, archive_path = output_limits.truncate_output(full_output, repository_path)
        response["truncated"] = True
        response["archive_path"] = archive_path
        lines = str(response["answer"]).splitlines(keepends=True)
        low, high = 0, len(lines)
        while low < high:
            middle = (low + high + 1) // 2
            response["answer"] = "".join(lines[:middle])
            size = len(output_limits.serialize_json(response).encode("utf-8"))
            if size <= output_limits.MAX_OUTPUT_BYTES:
                low = middle
            else:
                high = middle - 1
        response["answer"] = "".join(lines[:low])
        return response

    return mcp


def run(app_config: config.Config | None = None) -> None:
    """Run the server on stdio."""
    create_server(app_config=app_config).run("stdio")


def main() -> None:
    run()


if __name__ == "__main__":
    run()
