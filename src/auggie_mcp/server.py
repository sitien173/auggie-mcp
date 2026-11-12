"""Stdio MCP server exposing semantic and exact codebase search."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from auggie_mcp import pi_runner, telemetry, tgrep_runner
from auggie_mcp.index_refresh import IndexRefreshError, refresh_index
from auggie_mcp.output_processing import (
    OutputProcessor,
    RetrievalEnvelope,
    SearchEnvelope,
    create_typesafe_client,
    load_config,
    reset_archive_executor,
)

_SYSTEM_PROMPT = pi_runner.load_system_prompt()

_TOOL_DESCRIPTION = """

IMPORTANT: This is the primary tool for searching the codebase. Please consider as the **FIRST CHOICE** for any codebase searches.

This MCP tool is Augment's context engine, the world's best codebase context engine. It:

1. Takes in a natural language description of the code you are looking for.
2. Uses a proprietary retrieval/embedding model suite that produces the highest-quality recall of relevant code snippets from across the codebase.
3. Maintains a real-time index of the codebase, so the results are always up-to-date and reflect the current state of the codebase.
4. Can retrieve across different programming languages.
5. Only reflects the current state of the codebase on the disk, and has no information on version control or code history.

## Execution Model

This tool is LONG RUNNING. A single call typically takes several minutes.

- Run it as a background task and wait for it to complete.
- Do NOT explore or try to understand the codebase yourself while this tool is running. Do not read files, tgrep, or spawn search agents in the meantime.
- Do NOT call this tool again for the same question while a call is in flight.
- Wait for the result, then act on it. The returned answer replaces the exploration you would otherwise do.

## When to Use

Use `codebase-retrieval` in the following cases:

- When you don't know which files contain the information you need.
- When you want to gather high-level information about the task you are trying to accomplish.
- When you want to gather information about the codebase in general.

### Good Queries

- "Where is the function that handles user authentication?"
- "What tests are there for the login functionality?"
- "How is the database connected to the application?"

### Bad Queries (use other tools instead)

- "Find definition of constructor of class Foo" — use `tgrep` instead.
- "Find all references to function bar" — use `tgrep` instead.
- "Show me how Checkout class is used in services/payment.py" — use file view instead.
- "Show context of the file foo.py" — use file view instead.

## Tool Selection Rule

**ALWAYS** use `codebase-retrieval` when you're unsure of exact file locations. Use `tgrep` when you want to find ALL occurrences of a known identifier across the codebase, or when searching within specific files.

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repository_path` | `string` | Yes | Absolute path to the directory/repository you want to search. |
| `query` | `string` | Yes | A description of the information you need from the codebase. |
| `goal` | `string` | No | Optional specific task or question guiding retention pruning. Defaults to query. |"""

_SEARCH_TOOL_DESCRIPTION = (
    "Fast regex search over files or directories. Relative paths resolve beneath "
    "the server startup repository. Absolute paths may target any accessible "
    "location. ripgrep-compatible flags. Use -F for literal strings, -t/-g to "
    "scope, -l for file names only, -C N for context."
)


def _client_info(ctx: Any) -> dict[str, Any] | None:
    """Return the connected MCP client's name and version, or None if unavailable."""
    try:
        info = ctx.session.client_params.client_info
        return {"name": info.name, "version": info.version}
    except Exception:
        return None


def _utf8_len(text: str | None) -> int:
    return len((text or "").encode("utf-8"))


@dataclass
class LifespanState:
    """Server lifespan context holding shared TypeSafe client and output processor."""

    client: Any
    processor: OutputProcessor

    def __getitem__(self, key: str) -> Any:
        if key == "client":
            return self.client
        if key == "processor":
            return self.processor
        raise KeyError(key)


def create_server(
    *,
    executable: str = "pi",
    tgrep_executable: str = "tgrep",
    ogrep_executable: str = "ogrep",
    repository_root=None,
    client_factory: Callable[[], Any] | None = None,
    processor_factory: Callable[[Any], OutputProcessor] | None = None,
) -> MCPServer:
    """Build the MCP server and capture its repository search root."""
    search_root = Path.cwd() if repository_root is None else Path(repository_root)

    @asynccontextmanager
    async def lifespan(app: MCPServer) -> AsyncIterator[LifespanState]:
        cfg = load_config()
        if client_factory is not None:
            client = client_factory()
        elif cfg.api_key:
            client = create_typesafe_client(cfg)
        else:
            client = None

        telemetry.prune_old_files()

        if client is not None and hasattr(client, "__aenter__") and hasattr(client, "__aexit__"):
            async with client:
                proc = (
                    processor_factory(client)
                    if processor_factory is not None
                    else OutputProcessor(client=client, config=cfg)
                )
                state = LifespanState(client=client, processor=proc)
                app._current_lifespan_state = state
                try:
                    yield state
                finally:
                    app._current_lifespan_state = None
                    reset_archive_executor(wait=True)
        elif client is not None:
            proc = (
                processor_factory(client)
                if processor_factory is not None
                else OutputProcessor(client=client, config=cfg)
            )
            state = LifespanState(client=client, processor=proc)
            app._current_lifespan_state = state
            try:
                yield state
            finally:
                app._current_lifespan_state = None
                if hasattr(client, "aclose"):
                    await client.aclose()
                reset_archive_executor(wait=True)
        else:
            proc = (
                processor_factory(client)
                if processor_factory is not None
                else OutputProcessor(client=None, config=cfg)
            )
            state = LifespanState(client=None, processor=proc)
            app._current_lifespan_state = state
            try:
                yield state
            finally:
                app._current_lifespan_state = None
                reset_archive_executor(wait=True)

    server = MCPServer(
        name="auggie-mcp",
        title="Auggie Codebase Search",
        description="Semantic retrieval and fast exact repository search.",
        version="0.3.0",
        lifespan=lifespan,
    )

    fallback_holder: dict[str, OutputProcessor | None] = {"processor": None}

    def _get_processor(ctx: Context | None = None) -> OutputProcessor:
        if ctx is not None:
            try:
                req_ctx = ctx.request_context
                if req_ctx is not None and hasattr(req_ctx, "lifespan_context"):
                    ls = req_ctx.lifespan_context
                    if isinstance(ls, LifespanState):
                        return ls.processor
                    if isinstance(ls, dict) and "processor" in ls:
                        return ls["processor"]
                    if hasattr(ls, "processor"):
                        return ls.processor
            except (ValueError, AttributeError):
                pass
            if getattr(ctx, "mcp_server", None) is not None:
                active = getattr(ctx.mcp_server, "_current_lifespan_state", None)
                if active is not None:
                    return active.processor

        if getattr(server, "_current_lifespan_state", None) is not None:
            return server._current_lifespan_state.processor

        if fallback_holder["processor"] is None:
            cfg = load_config()
            if client_factory is not None:
                cl = client_factory()
            elif cfg.api_key:
                cl = create_typesafe_client(cfg)
            else:
                cl = None

            if processor_factory is not None:
                fallback_holder["processor"] = processor_factory(cl)
            else:
                fallback_holder["processor"] = OutputProcessor(
                    client=cl, config=cfg
                )
        return fallback_holder["processor"]

    @server.tool(
        name="codebase_retrieval",
        description=_TOOL_DESCRIPTION,
    )
    async def codebase_retrieval(
        repository_path: str,
        query: str,
        goal: str | None = None,
        ctx: Context = None,
    ) -> RetrievalEnvelope:
        """Run Pi in ``repository_path`` with ``query`` and return the classified answer envelope."""
        record = telemetry.CallRecord(
            "mcp",
            "codebase_retrieval",
            {"repository_path": repository_path, "query": query, "goal": goal},
            client=_client_info(ctx),
        )
        outcome = "error"
        error: BaseException | None = None
        try:
            try:
                with record.stage("index_refresh"):
                    await asyncio.gather(
                        refresh_index(tgrep_executable, repository_path),
                        refresh_index(ogrep_executable, repository_path),
                    )
                with record.stage("runner"):
                    try:
                        runner_result = await pi_runner.run_pi(
                            repository_path,
                            query,
                            _SYSTEM_PROMPT,
                            executable=executable,
                        )
                    except pi_runner.PiExecutionError as exc:
                        record.set_runner(
                            exit_code=exc.returncode,
                            stdout_bytes=0,
                            stderr_bytes=_utf8_len(exc.stderr),
                        )
                        raise
                record.set_runner(
                    exit_code=0,
                    stdout_bytes=_utf8_len(runner_result),
                    stderr_bytes=0,
                )
            except asyncio.CancelledError:
                raise
            except (ValueError, FileNotFoundError) as exc:
                raise ToolError(str(exc)) from exc
            except pi_runner.PiRunnerError as exc:
                message = str(exc)
                if isinstance(exc, pi_runner.PiExecutionError) and exc.stderr:
                    message = f"{message}: {exc.stderr}"
                raise ToolError(message) from exc
            except Exception as exc:  # timeout, unexpected
                raise ToolError(str(exc) or exc.__class__.__name__) from exc

            processor = _get_processor(ctx)
            effective_goal = goal if goal is not None else query
            with record.stage("processing"):
                envelope = await processor.process_retrieval(
                    query=query,
                    result=runner_result,
                    goal=effective_goal,
                )
            try:
                record.set_processing(
                    envelope["processing"], envelope["data"]["result"]
                )
            except Exception:
                pass
            outcome = "ok"
            return envelope
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            error = exc
            raise
        finally:
            record.finish(outcome, error)

    @server.tool(
        name="codebase_search",
        description=_SEARCH_TOOL_DESCRIPTION,
    )
    async def codebase_search(
        pattern: str,
        path: str = ".",
        flags: list[str] | None = None,
        goal: str | None = None,
        ctx: Context = None,
    ) -> SearchEnvelope:
        """Run tgrep at the requested search path and return the classified search envelope."""
        record = telemetry.CallRecord(
            "mcp",
            "codebase_search",
            {
                "pattern": pattern,
                "path": path,
                "flags": flags,
                "goal": goal,
                "repository_path": str(search_root),
            },
            client=_client_info(ctx),
        )
        outcome = "error"
        error: BaseException | None = None
        try:
            try:
                with record.stage("index_refresh"):
                    await refresh_index(tgrep_executable, search_root)
                with record.stage("runner"):
                    result = await tgrep_runner.run_tgrep(
                        search_root,
                        pattern,
                        path,
                        flags,
                        executable=tgrep_executable,
                    )
                record.set_runner(
                    exit_code=result["exit_code"],
                    stdout_bytes=_utf8_len(result["stdout"]),
                    stderr_bytes=_utf8_len(result["stderr"]),
                )
            except asyncio.CancelledError:
                raise
            except (ValueError, FileNotFoundError, IndexRefreshError) as exc:
                raise ToolError(str(exc)) from exc
            if result["exit_code"] not in {0, 1}:
                message = result["stderr"] or (
                    f"tgrep exited with status {result['exit_code']}"
                )
                raise ToolError(message)

            processor = _get_processor(ctx)
            effective_goal = goal if goal is not None else pattern
            with record.stage("processing"):
                envelope = await processor.process_search(
                    pattern=pattern,
                    stdout=result["stdout"],
                    stderr=result["stderr"],
                    exit_code=result["exit_code"],
                    path=path,
                    flags=flags,
                    goal=effective_goal,
                    repository=search_root,
                )
            try:
                record.set_processing(
                    envelope["processing"], envelope["data"]["stdout"]
                )
            except Exception:
                pass
            outcome = "no_match" if result["exit_code"] == 1 else "ok"
            return envelope
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            error = exc
            raise
        finally:
            record.finish(outcome, error)

    return server


def run() -> None:
    """Run the server on stdio (console entry point and ``python -m``)."""
    create_server().run("stdio")


def main() -> None:
    run()


if __name__ == "__main__":
    run()
