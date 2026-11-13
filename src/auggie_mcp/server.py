"""Stdio MCP server exposing semantic and exact codebase search."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from auggie_mcp import agent_runner, index_tools, rg_runner, semantic_search, telemetry
from auggie_mcp.output_processing import (
    OutputProcessor,
    RetrievalEnvelope,
    SearchEnvelope,
    create_typesafe_client,
    load_config,
    reset_archive_executor,
)

_TOOL_DESCRIPTION = """
Answer a question about a codebase by running a retrieval agent over it and returning a synthesized, pruned answer.

This tool is LONG RUNNING. A single call typically takes several minutes.

- Run it as a background task and wait for it to complete.
- Do NOT explore the codebase yourself while it runs. Do not read files, grep, or spawn search agents in the meantime.
- Do NOT call it again for the same question while a call is in flight.
- The agent can edit files, so this tool is not read-only.
- It reads the files on disk and has no information on version control or code history. It reads an existing semantic index for hints and never builds or updates it.

## When to Use

Use `codebase_retrieval` only when you need a synthesized answer and do not know which files hold the information, for example:

- "Where is the function that handles user authentication?"
- "What tests are there for the login functionality?"
- "How is the database connected to the application?"

For known literals and symbols, use `codebase_search` with `mode="exact"`. For a ranked list of code that matches a concept, use `index_status`, then `index_update` if needed, then `codebase_search` with `mode="semantic"`. To read a known file, use file view.

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repository_path` | `string` | Yes | Absolute path to the directory or repository you want to search. |
| `query` | `string` | Yes | A description of the information you need from the codebase. |
| `goal` | `string` | No | Optional specific task or question guiding retention pruning. Defaults to query. |"""

_INDEX_STATUS_DESCRIPTION = (
    "Inspect the semantic index of a repository without changing it. Returns whether "
    "the index exists, its model, dimensions, file and chunk counts, the last index "
    "time, stale counts (modified, deleted, candidate_added), and a next_step. "
    "candidate_added counts files the indexer may add and can overcount binary files."
)

_INDEX_UPDATE_DESCRIPTION = (
    "Build or refresh the semantic index of a repository. By default the update is "
    "incremental and reuses the indexed model; the first call builds the index. "
    "rebuild=True deletes the existing index first and then rebuilds it, so an "
    "interrupted rebuild can leave no index; index_status then reports exists=false. "
    "Calls Voyage and incurs charges; charges already incurred are not recovered "
    "if the call is cancelled or times out. Fails if another update is running. "
    "Requires VOYAGE_API_KEY in the server environment. The timeout is "
    "AUGGIE_INDEX_TIMEOUT_S seconds, default 1800."
)


_SEARCH_TOOL_DESCRIPTION = (
    "Search a repository in one of two modes. mode='exact' (default) runs ripgrep for "
    "literals, symbols, and regexes; it needs no index. Relative paths resolve beneath "
    "the server startup repository and absolute paths may target any accessible "
    "location. Accepts ripgrep-compatible flags: -F for literal strings, -t/-g to "
    "scope, -l for file names only, -C N for context. mode='semantic' returns the "
    "highest-ranked indexed code chunks for a concept query, rendered as "
    "'path:start-end score=<score>' followed by the code, with blocks separated by "
    "'--'. It requires an index: call index_status, then index_update if the index is "
    "missing or stale. Semantic search never updates the index, and data.index reports "
    "how many indexed files are stale. A path below the repository root limits results "
    "to that subtree and can return fewer than top_k hits. flags apply only to "
    "mode='exact'. top_k (1 to 50, default 10) and context_lines (0 to 20, default 0) "
    "apply only to mode='semantic'. An empty result is a normal result with exit_code 1."
)

_INSTRUCTIONS = """\
1. Use `codebase_search` with `mode=exact` for known literals and symbols. It needs no index.
2. For concept queries, call `index_status`. If the index is missing or stale, call `index_update`, then `codebase_search` with `mode=semantic`.
3. Use `codebase_retrieval` only for a synthesized answer, and do not know which files hold the information"""


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
    executable: str | None = None,
    rg_executable: str = "rg",
    index_command: Sequence[str] | None = None,
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
        version="0.5.0",
        instructions=_INSTRUCTIONS,
        lifespan=lifespan,
    )

    def _lifespan_processor(ctx: Context | None = None) -> OutputProcessor | None:
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
        return None

    @asynccontextmanager
    async def _processor_scope(
        ctx: Context | None = None,
    ) -> AsyncIterator[OutputProcessor]:
        """Yield the lifespan processor, or a per-call one whose client is closed on exit."""
        shared = _lifespan_processor(ctx)
        if shared is not None:
            yield shared
            return

        cfg = load_config()
        if client_factory is not None:
            cl = client_factory()
        elif cfg.api_key:
            cl = create_typesafe_client(cfg)
        else:
            cl = None
        try:
            if processor_factory is not None:
                yield processor_factory(cl)
            else:
                yield OutputProcessor(client=cl, config=cfg)
        finally:
            if cl is not None and hasattr(cl, "aclose"):
                try:
                    await cl.aclose()
                except Exception:
                    pass

    @server.tool(
        name="codebase_retrieval",
        description=_TOOL_DESCRIPTION,
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=True),
    )
    async def codebase_retrieval(
        repository_path: str,
        query: str,
        goal: str | None = None,
        ctx: Context = None,
    ) -> RetrievalEnvelope:
        """Run the agent CLI in ``repository_path`` with ``query`` and return the classified answer envelope."""
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
                with record.stage("runner"):
                    try:
                        runner_result = await agent_runner.run_agent(
                            repository_path,
                            query,
                            executable=executable,
                        )
                    except agent_runner.AgentError as exc:
                        record.set_runner(
                            exit_code=exc.returncode,
                            stdout_bytes=0,
                            stderr_bytes=_utf8_len(exc.detail),
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
            except agent_runner.AgentError as exc:
                raise ToolError(str(exc)) from exc
            except Exception as exc:  # timeout, unexpected
                raise ToolError(str(exc) or exc.__class__.__name__) from exc

            effective_goal = goal if goal is not None else query
            async with _processor_scope(ctx) as processor:
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
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    )
    async def codebase_search(
        query: str,
        path: str = ".",
        mode: Literal["exact", "semantic"] = "exact",
        flags: list[str] | None = None,
        top_k: int = 10,
        context_lines: int = 0,
        goal: str | None = None,
        ctx: Context = None,
    ) -> SearchEnvelope:
        """Search exactly with ripgrep or semantically through the index and return the classified search envelope."""
        record = telemetry.CallRecord(
            "mcp",
            "codebase_search",
            {
                "query": query,
                "mode": mode,
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
            semantic_search.validate_params(
                mode, flags=flags, top_k=top_k, context_lines=context_lines
            )
            index_info = None
            try:
                with record.stage("runner"):
                    if mode == "semantic":
                        result = await asyncio.to_thread(
                            semantic_search.run_semantic,
                            search_root,
                            query,
                            path,
                            top_k,
                            context_lines,
                        )
                        index_info = result["index"]
                    else:
                        result = await rg_runner.run_rg(
                            search_root,
                            query,
                            path,
                            flags,
                            executable=rg_executable,
                        )
                record.set_runner(
                    exit_code=result["exit_code"],
                    stdout_bytes=_utf8_len(result["stdout"]),
                    stderr_bytes=_utf8_len(result["stderr"]),
                )
            except asyncio.CancelledError:
                raise
            except ToolError:
                raise
            except (ValueError, FileNotFoundError) as exc:
                raise ToolError(str(exc)) from exc
            if result["exit_code"] not in {0, 1}:
                message = result["stderr"] or (
                    f"rg exited with status {result['exit_code']}"
                )
                raise ToolError(message)

            effective_goal = goal if goal is not None else query
            async with _processor_scope(ctx) as processor:
                with record.stage("processing"):
                    envelope = await processor.process_search(
                        query=query,
                        stdout=result["stdout"],
                        stderr=result["stderr"],
                        exit_code=result["exit_code"],
                        path=path,
                        flags=flags,
                        goal=effective_goal,
                        repository=search_root,
                        mode=mode,
                        index=index_info,
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

    @server.tool(
        name="index_status",
        description=_INDEX_STATUS_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    async def index_status(
        repository_path: str, ctx: Context = None
    ) -> dict[str, Any]:
        """Report whether the semantic index exists, its counts, and how stale it is."""
        record = telemetry.CallRecord(
            "mcp",
            "index_status",
            {"repository_path": repository_path},
            client=_client_info(ctx),
        )
        outcome = "error"
        error: BaseException | None = None
        try:
            try:
                with record.stage("status"):
                    result = await asyncio.to_thread(
                        index_tools.index_status, repository_path
                    )
            except asyncio.CancelledError:
                raise
            except ToolError:
                raise
            except (ValueError, FileNotFoundError) as exc:
                raise ToolError(str(exc)) from exc
            outcome = "ok"
            return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            error = exc
            raise
        finally:
            record.finish(outcome, error)

    @server.tool(
        name="index_update",
        description=_INDEX_UPDATE_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, open_world_hint=True
        ),
    )
    async def index_update(
        repository_path: str, rebuild: bool = False, ctx: Context = None
    ) -> dict[str, Any]:
        """Build or refresh the semantic index, or delete and rebuild it with ``rebuild=True``."""
        record = telemetry.CallRecord(
            "mcp",
            "index_update",
            {"repository_path": repository_path, "rebuild": rebuild},
            client=_client_info(ctx),
        )
        outcome = "error"
        error: BaseException | None = None

        async def report_progress(progress, total, message):
            if ctx is not None:
                await ctx.report_progress(progress, total, message)

        try:
            try:
                with record.stage("indexer"):
                    result = await index_tools.run_index_update(
                        repository_path,
                        rebuild=rebuild,
                        command=index_command,
                        report_progress=report_progress,
                    )
            except asyncio.CancelledError:
                raise
            except ToolError:
                raise
            except (ValueError, FileNotFoundError) as exc:
                raise ToolError(str(exc)) from exc
            outcome = "ok"
            return result
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
